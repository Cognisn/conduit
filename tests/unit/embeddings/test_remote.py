"""The remote embedding adapters, replayed from synthetic recordings through the registry."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conduit.core.catalogue import default_catalogue
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure, ResidencyRefused
from conduit.core.events import MemoryEventSink
from conduit.core.provider import EmbeddingRequest, embedding_usage
from conduit.core.registry import EmbeddingCallFinishedEvent, ProviderRegistry
from conduit.core.residency import ResidencyPolicy
from conduit.core.secrets import LiteralSecretResolver
from conduit.embeddings import (
    BedrockEmbeddingFactory,
    GeminiEmbeddingFactory,
    OllamaEmbeddingFactory,
    OpenAICompatEmbeddingFactory,
)
from conduit.embeddings.bedrock import BedrockEmbedder
from conduit.embeddings.gemini import read_embed_response
from conduit.embeddings.ollama import classify as classify_ollama
from conduit.embeddings.openai_compat import read_embeddings
from conduit.testing.replay import Recording, Replay, load_recording
from tests.unit.embeddings import recordings
from tests.unit.embeddings.recordings import DIRECTORY, TEXTS, all_recordings, synthetic_vector

SECRETS = LiteralSecretResolver({"openai/key": "not-a-real-key", "gemini/key": "not-a-real-key"})
BEDROCK = {"kind": "bedrock", "region": "eu-west-2"}
OPENAI = {"kind": "openai_compat", "service": {"name": "openai", "api_key": "secret://openai/key"}}
GEMINI = {"kind": "gemini", "api_key": "secret://gemini/key"}
OLLAMA = {"kind": "ollama"}


def _recording(name: str) -> Recording:
    return load_recording(DIRECTORY / f"{name}.json")


def _registry(sink: MemoryEventSink | None = None, **kwargs: Any) -> ProviderRegistry:
    return ProviderRegistry(
        secrets=SECRETS,
        events=sink,
        embedding_factories=[
            BedrockEmbeddingFactory(),
            OpenAICompatEmbeddingFactory(),
            GeminiEmbeddingFactory(),
            OllamaEmbeddingFactory(),
        ],
        **kwargs,
    )


def _clean(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (*TEXTS, "sk-synthetic", "example.net"):
        assert text not in str(error)
        assert text not in repr(error)
        assert all(text not in str(arg) for arg in error.args)


def test_the_committed_recordings_equal_the_builder() -> None:
    for recording in all_recordings():
        path = DIRECTORY / f"{recording.name}.json"
        assert load_recording(path) == recording, path.name


def test_every_recording_file_is_built_by_the_builder() -> None:
    built = {f"{r.name}.json" for r in all_recordings()}
    assert {p.name for p in Path(DIRECTORY).glob("*.json")} == built


# Bedrock: Titan Text Embeddings V2 ----------------------------------------------------


class FakeBedrock:
    """Answers InvokeModel from a recording, matched by request (calls run concurrently)."""

    def __init__(
        self, recording: Recording | None = None, *, error: Exception | None = None
    ) -> None:
        self.answers = {
            json.dumps(i.request, sort_keys=True): i.response
            for i in (recording.interactions if recording else ())
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.threads: set[str] = set()
        self.raw: bytes | None = None
        self.closed = False
        self.barrier: threading.Barrier | None = None

    def invoke_model(self, **params: Any) -> dict[str, Any]:
        self.threads.add(threading.current_thread().name)
        assert params["contentType"] == "application/json"
        assert params["accept"] == "application/json"
        request = {"modelId": params["modelId"], "body": json.loads(params["body"])}
        self.calls.append(request)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        if self.error is not None:
            raise self.error
        if self.raw is not None:
            return {"body": io.BytesIO(self.raw)}
        answer = self.answers[json.dumps(request, sort_keys=True)]
        return {"body": io.BytesIO(json.dumps(answer).encode())}

    def close(self) -> None:
        self.closed = True


class ClientError(Exception):
    """Shaped like botocore's ClientError: the classifier reads the response only."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(f"synthetic provider text for {TEXTS[0]}")
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


async def _titan(client: FakeBedrock, sink: MemoryEventSink | None = None, **kwargs: Any) -> Any:
    return await _registry(sink, **kwargs).build_embedder("titan", BEDROCK, client=client)


@pytest.mark.parametrize(("dimensions", "name"), [(None, "titan-v2-1024"), (256, "titan-v2-256")])
async def test_titan_v2_embeds_each_text_and_sums_its_tokens(
    dimensions: int | None, name: str
) -> None:
    sink = MemoryEventSink()
    client = FakeBedrock(_recording(name))
    embedder = await _titan(client, sink)
    response = await embedder.embed(
        EmbeddingRequest(model=recordings.TITAN, texts=TEXTS, dimensions=dimensions)
    )
    length = dimensions or 1024
    assert response.vectors == (
        tuple(synthetic_vector(0, length)),
        tuple(synthetic_vector(1, length)),
    )
    assert response.usage == embedding_usage(23)
    assert sorted(c["body"]["inputText"] for c in client.calls) == sorted(TEXTS)
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert (finished.dimensions, finished.usage) == (length, embedding_usage(23))


async def test_titan_v2_offers_256_and_512_only() -> None:
    embedder = await _titan(FakeBedrock())
    assert embedder.accepts_dimensions(recordings.TITAN, 256)
    assert embedder.accepts_dimensions(recordings.TITAN, 512)
    assert not embedder.accepts_dimensions(recordings.TITAN, 384)
    assert not embedder.accepts_dimensions("amazon.titan-embed-text-v1", 256)
    with pytest.raises(ConfigurationError, match="does not offer 384"):
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS, dimensions=384))


def test_the_catalogue_keeps_titan_v2_at_1024_dimensions() -> None:
    entry = default_catalogue().require("bedrock", recordings.TITAN)
    assert (entry.kind, entry.embedding_dimensions, entry.context_window) == (
        "embedding",
        1024,
        8192,
    )


def test_bedrock_serves_titan_text_models_only() -> None:
    embedder = BedrockEmbedder("titan", None, None, catalogue=default_catalogue())  # type: ignore[arg-type]
    assert embedder.serves("amazon.titan-embed-text-v2:0")
    assert embedder.serves("amazon.titan-embed-text-v1")
    assert embedder.serves("amazon.titan-embed-g1-text-02")
    assert not embedder.serves("cohere.embed-english-v3")
    assert not embedder.serves("amazon.titan-embed-image-v1")


async def test_bedrock_refuses_an_unserved_model_before_any_call() -> None:
    client = FakeBedrock()
    embedder = BedrockEmbedder("titan", None, client, catalogue=default_catalogue())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="cannot call"):
        await embedder.embed(EmbeddingRequest(model="cohere.embed-english-v3", texts=TEXTS))
    assert client.calls == []


async def test_bedrock_calls_run_concurrently_on_their_own_threads() -> None:
    client = FakeBedrock(_recording("titan-v2-1024"))
    client.barrier = threading.Barrier(2)  # both calls must be in flight at once
    embedder = await _titan(client)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.001)

    tick = asyncio.create_task(ticker())
    try:
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS))
    finally:
        tick.cancel()
    assert client.threads == {"conduit-bedrock-embed"}
    assert ticks > 0


async def test_bedrock_holds_to_its_concurrency() -> None:
    client = FakeBedrock(_recording("titan-v2-1024"))
    client.barrier = threading.Barrier(2)
    registry = ProviderRegistry(
        secrets=SECRETS, embedding_factories=[BedrockEmbeddingFactory(concurrency=1)]
    )
    embedder = await registry.build_embedder("titan", BEDROCK, client=client)
    # With one call at a time the barrier never fills: the call times out.
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS, timeout_s=0.5))
    assert caught.value.failure is ProviderFailure.TIMEOUT
    _clean(caught.value)
    client.barrier.abort()


def test_bedrock_concurrency_must_be_positive() -> None:
    with pytest.raises(ConfigurationError):
        BedrockEmbeddingFactory(concurrency=0)


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (ClientError("ThrottlingException", 429), ProviderFailure.UNAVAILABLE),
        (ClientError("ValidationException", 400), ProviderFailure.REJECTED),
        (ClientError("AccessDeniedException", 403), ProviderFailure.REJECTED),
        (ClientError("ModelTimeoutException", 408), ProviderFailure.TIMEOUT),
    ],
)
async def test_bedrock_errors_are_classified_and_carry_nothing(
    error: Exception, failure: ProviderFailure
) -> None:
    embedder = await _titan(FakeBedrock(error=error))
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS))
    assert caught.value.failure is failure
    _clean(caught.value)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b'{"embedding": [0.1, 0.2]}',
        b'{"embedding": [0.1, "x"], "inputTextTokenCount": 3}',
        b'{"embedding": [0.1, 0.2], "inputTextTokenCount": -1}',
        b'{"embedding": [0.1, 0.2], "inputTextTokenCount": true}',
        b'{"embedding": [], "inputTextTokenCount": 3}',
    ],
)
async def test_a_malformed_titan_body_is_a_protocol_failure(raw: bytes) -> None:
    client = FakeBedrock()
    client.raw = raw
    embedder = BedrockEmbedder("titan", None, client, catalogue=default_catalogue())  # type: ignore[arg-type]
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS[:1]))
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)


async def test_a_titan_vector_of_the_wrong_length_is_caught_by_the_funnel() -> None:
    client = FakeBedrock()
    client.raw = json.dumps({"embedding": [0.1] * 1000, "inputTextTokenCount": 3}).encode()
    embedder = await _titan(client)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS[:1]))
    assert caught.value.failure is ProviderFailure.PROTOCOL


async def test_bedrock_residency_is_held_on_an_unassigned_call() -> None:
    policy = ResidencyPolicy(regions=frozenset({"eu-central-1"}))
    client = FakeBedrock(_recording("titan-v2-1024"))
    embedder = await _titan(client, residency=policy)
    with pytest.raises(ResidencyRefused):
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS))
    assert client.calls == []


async def test_bedrock_cancellation_propagates() -> None:
    client = FakeBedrock(_recording("titan-v2-1024"))
    client.barrier = threading.Barrier(3)  # never fills: the calls wait
    embedder = await _titan(client)
    task = asyncio.create_task(
        embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS))
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.barrier.abort()


async def test_a_failure_stops_calls_not_yet_started() -> None:
    client = FakeBedrock(error=ClientError("ValidationException", 400))
    registry = ProviderRegistry(
        secrets=SECRETS, embedding_factories=[BedrockEmbeddingFactory(concurrency=1)]
    )
    embedder = await registry.build_embedder("titan", BEDROCK, client=client)
    with pytest.raises(ProviderError):
        await embedder.embed(EmbeddingRequest(model=recordings.TITAN, texts=TEXTS * 3))
    assert len(client.calls) == 1


async def test_bedrock_closes_its_client() -> None:
    client = FakeBedrock()
    embedder = await _titan(client)
    await embedder.aclose()
    assert client.closed


# OpenAI ------------------------------------------------------------------------------


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


class FakeOpenAI:
    def __init__(self, replay: Replay | None = None, *, error: Exception | None = None) -> None:
        self.replay = replay
        self.error = error
        self.timeouts: list[float] = []
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.embeddings = self
        self.answer: Any = None

    async def create(self, *, timeout: float, **params: Any) -> Any:
        self.timeouts.append(timeout)
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer
        assert self.replay is not None
        return _namespace(self.replay.next(params))

    async def close(self) -> None:
        self.closed = True


class APIStatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"synthetic provider text sk-synthetic for {TEXTS[0]}")
        self.status_code = status


async def test_openai_embeds_a_batch_in_index_order() -> None:
    replay = Replay(_recording("openai-small-256"))
    client = FakeOpenAI(replay)
    embedder = await _registry().build_embedder("openai", OPENAI, client=client)
    response = await embedder.embed(
        EmbeddingRequest(model=recordings.OPENAI, texts=TEXTS, dimensions=256, timeout_s=30)
    )
    replay.assert_exhausted()
    assert response.vectors[0] == tuple(synthetic_vector(0, 256))
    assert response.vectors[1] == tuple(synthetic_vector(1, 256))
    assert response.usage == embedding_usage(21)
    assert client.timeouts == [30]


async def test_openai_splits_a_batch_and_sums_usage() -> None:
    client = FakeOpenAI()

    async def create(*, timeout: float, **params: Any) -> Any:
        client.calls.append(params)
        return _namespace(
            {
                "data": [
                    {"index": i, "embedding": synthetic_vector(i, 1536)}
                    for i in range(len(params["input"]))
                ],
                "usage": {"prompt_tokens": 2 * len(params["input"])},
            }
        )

    client.create = create  # type: ignore[method-assign]
    factory = OpenAICompatEmbeddingFactory()
    registry = ProviderRegistry(secrets=SECRETS, embedding_factories=[factory])
    embedder = await registry.build_embedder("openai", OPENAI, client=client)
    embedder._inner._max_inputs = 2  # type: ignore[attr-defined]
    response = await embedder.embed(EmbeddingRequest(model=recordings.OPENAI, texts=TEXTS * 3))
    assert [len(c["input"]) for c in client.calls] == [2, 2, 2]
    assert "dimensions" not in client.calls[0]
    assert len(response.vectors) == 6
    assert response.usage == embedding_usage(12)


def test_openai_offers_shorter_vectors_for_text_embedding_3_only() -> None:
    from conduit.embeddings.openai_compat import OpenAICompatEmbedder

    embedder = OpenAICompatEmbedder("o", None, None, catalogue=default_catalogue())  # type: ignore[arg-type]
    assert embedder.accepts_dimensions("text-embedding-3-small", 256)
    assert embedder.accepts_dimensions("text-embedding-3-large", 1024)
    assert not embedder.accepts_dimensions("text-embedding-ada-002", 256)
    assert not embedder.accepts_dimensions("my-foundry-deployment", 256)


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (429, ProviderFailure.UNAVAILABLE),
        (500, ProviderFailure.UNAVAILABLE),
        (400, ProviderFailure.REJECTED),
        (401, ProviderFailure.REJECTED),
        (408, ProviderFailure.TIMEOUT),
    ],
)
async def test_openai_errors_are_classified_and_carry_nothing(
    status: int, failure: ProviderFailure
) -> None:
    embedder = await _registry().build_embedder(
        "openai", OPENAI, client=FakeOpenAI(error=APIStatusError(status))
    )
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.OPENAI, texts=TEXTS))
    assert caught.value.failure is failure
    _clean(caught.value)


@pytest.mark.parametrize(
    "answer",
    [
        {"data": [], "usage": {"prompt_tokens": 1}},
        {
            "data": [{"index": 0, "embedding": [0.1]}, {"index": 0, "embedding": [0.2]}],
            "usage": {"prompt_tokens": 1},
        },
        {
            "data": [{"index": 0, "embedding": [0.1]}, {"index": 5, "embedding": [0.2]}],
            "usage": {"prompt_tokens": 1},
        },
        {
            "data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": "AAAA"}],
            "usage": {"prompt_tokens": 1},
        },
        {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.2]}]},
        {
            "data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.2]}],
            "usage": {"prompt_tokens": "3"},
        },
        {"usage": {"prompt_tokens": 1}},
    ],
)
async def test_a_malformed_openai_response_is_a_protocol_failure(answer: dict[str, Any]) -> None:
    client = FakeOpenAI()
    client.answer = _namespace(answer)
    embedder = await _registry().build_embedder("openai", OPENAI, client=client)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.OPENAI, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)


def test_openai_usage_maps_prompt_tokens_to_input() -> None:
    read = read_embeddings(
        _namespace(
            {
                "data": [{"index": 0, "embedding": [0.5]}],
                "usage": {"prompt_tokens": 7, "total_tokens": 9},
            }
        ),
        1,
    )
    assert read == ([(0.5,)], 7)


async def test_openai_closes_its_client() -> None:
    client = FakeOpenAI()
    embedder = await _registry().build_embedder("openai", OPENAI, client=client)
    await embedder.aclose()
    assert client.closed


async def test_openai_through_the_real_sdk_client() -> None:
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    seen: list[Any] = []
    recording = _recording("openai-small-256")

    def handler(request: Any) -> Any:
        seen.append(request)
        return httpx.Response(200, json=recording.interactions[0].response)

    sdk = openai.AsyncOpenAI(
        api_key="not-a-real-key",
        base_url="https://api.example.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    embedder = await _registry().build_embedder("openai", OPENAI, client=sdk)
    response = await embedder.embed(
        EmbeddingRequest(model=recordings.OPENAI, texts=TEXTS, dimensions=256)
    )
    assert response.usage == embedding_usage(21)
    assert seen[0].url.path == "/v1/embeddings"
    assert json.loads(seen[0].content) == recording.interactions[0].request
    await embedder.aclose()


# Gemini ------------------------------------------------------------------------------


class FakeGemini:
    def __init__(self, replay: Replay | None = None, *, error: BaseException | None = None) -> None:
        self.replay = replay
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.answer: Any = None
        self.closed = False
        self.delay = 0.0
        self.aio = SimpleNamespace(models=self, aclose=self._aclose)

    async def embed_content(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer
        assert self.replay is not None
        return _namespace(self.replay.next(params))

    async def _aclose(self) -> None:
        self.closed = True


class GeminiClientError(Exception):
    __module__ = "google.genai.errors"

    def __init__(self, code: int) -> None:
        super().__init__(f"synthetic provider text for {TEXTS[0]}")
        self.code = code


async def test_gemini_embeds_with_output_dimensionality() -> None:
    replay = Replay(_recording("gemini-001-768"))
    sink = MemoryEventSink()
    embedder = await _registry(sink).build_embedder("gemini", GEMINI, client=FakeGemini(replay))
    response = await embedder.embed(
        EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS, dimensions=768)
    )
    replay.assert_exhausted()
    assert response.dimensions == 768
    assert response.usage is None  # the Gemini API reports no token count
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert (finished.outcome, finished.usage) == ("ok", None)


async def test_gemini_offers_128_dimensions_and_more() -> None:
    embedder = await _registry().build_embedder("gemini", GEMINI, client=FakeGemini())
    assert embedder.accepts_dimensions(recordings.GEMINI, 128)
    assert not embedder.accepts_dimensions(recordings.GEMINI, 127)


def test_gemini_usage_comes_from_token_counts_when_every_embedding_has_one() -> None:
    both = _namespace(
        {
            "embeddings": [
                {"values": [0.1], "statistics": {"token_count": 4.0}},
                {"values": [0.2], "statistics": {"token_count": 6}},
            ]
        }
    )
    assert read_embed_response(both, 2) == ([(0.1,), (0.2,)], 10)
    one = _namespace(
        {"embeddings": [{"values": [0.1], "statistics": {"token_count": 4}}, {"values": [0.2]}]}
    )
    assert read_embed_response(one, 2) == ([(0.1,), (0.2,)], None)
    bad = _namespace({"embeddings": [{"values": [0.1], "statistics": {"token_count": -2}}]})
    assert read_embed_response(bad, 1) is None


async def test_gemini_splits_a_batch() -> None:
    client = FakeGemini()

    async def embed_content(**params: Any) -> Any:
        client.calls.append(params)
        return _namespace(
            {"embeddings": [{"values": synthetic_vector(0, 3072)} for _ in params["contents"]]}
        )

    client.aio.models = SimpleNamespace(embed_content=embed_content)
    embedder = await _registry().build_embedder("gemini", GEMINI, client=client)
    embedder._inner._max_texts = 1  # type: ignore[attr-defined]
    response = await embedder.embed(EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS))
    assert [c["contents"] for c in client.calls] == [[TEXTS[0]], [TEXTS[1]]]
    assert "config" not in client.calls[0]
    assert response.dimensions == 3072


@pytest.mark.parametrize(
    ("code", "failure"),
    [
        (429, ProviderFailure.UNAVAILABLE),
        (400, ProviderFailure.REJECTED),
        (504, ProviderFailure.TIMEOUT),
    ],
)
async def test_gemini_errors_are_classified_and_carry_nothing(
    code: int, failure: ProviderFailure
) -> None:
    embedder = await _registry().build_embedder(
        "gemini", GEMINI, client=FakeGemini(error=GeminiClientError(code))
    )
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS))
    assert caught.value.failure is failure
    _clean(caught.value)


async def test_gemini_times_out_cleanly() -> None:
    client = FakeGemini()
    client.delay = 5
    embedder = await _registry().build_embedder("gemini", GEMINI, client=client)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS, timeout_s=0.05))
    assert caught.value.failure is ProviderFailure.TIMEOUT
    _clean(caught.value)


async def test_gemini_cancellation_propagates() -> None:
    client = FakeGemini()
    client.delay = 5
    embedder = await _registry().build_embedder("gemini", GEMINI, client=client)
    task = asyncio.create_task(
        embedder.embed(EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS))
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_malformed_gemini_response_is_a_protocol_failure() -> None:
    client = FakeGemini()
    client.answer = _namespace({"embeddings": [{"values": [0.1]}]})  # one vector for two texts
    embedder = await _registry().build_embedder("gemini", GEMINI, client=client)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)


async def test_gemini_closes_its_client() -> None:
    client = FakeGemini()
    embedder = await _registry().build_embedder("gemini", GEMINI, client=client)
    await embedder.aclose()
    assert client.closed


async def test_gemini_through_the_real_sdk_client() -> None:
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("google.genai")
    recording = _recording("gemini-001-768")
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        body = {"embeddings": recording.interactions[0].response["embeddings"]}
        return httpx.Response(200, json=body)

    registry = ProviderRegistry(
        secrets=SECRETS,
        embedding_factories=[
            GeminiEmbeddingFactory(async_client_args={"transport": httpx.MockTransport(handler)})
        ],
    )
    embedder = await registry.build_embedder("gemini", GEMINI)
    response = await embedder.embed(
        EmbeddingRequest(model=recordings.GEMINI, texts=TEXTS, dimensions=768)
    )
    assert response.dimensions == 768
    sent = json.loads(seen[0].content)
    assert seen[0].url.path.endswith(f"/models/{recordings.GEMINI}:batchEmbedContents")
    assert [r["content"]["parts"][0]["text"] for r in sent["requests"]] == list(TEXTS)
    assert {r["outputDimensionality"] for r in sent["requests"]} == {768}
    await registry.aclose()


# Ollama ------------------------------------------------------------------------------


async def test_ollama_embeds_through_the_real_client() -> None:
    httpx = pytest.importorskip("httpx")
    replay = Replay(_recording("nomic-embed-text"))
    timeouts: list[Any] = []

    def handler(request: Any) -> Any:
        timeouts.append(request.extensions.get("timeout"))
        answer = replay.next(
            {
                "method": request.method,
                "path": request.url.path,
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(answer["status"], json=answer["body"])

    registry = ProviderRegistry(
        secrets=SECRETS,
        embedding_factories=[OllamaEmbeddingFactory(transport=httpx.MockTransport(handler))],
    )
    embedder = await registry.build_embedder("ollama", OLLAMA)
    response = await embedder.embed(
        EmbeddingRequest(model=recordings.OLLAMA, texts=TEXTS, timeout_s=15)
    )
    replay.assert_exhausted()
    assert response.dimensions == 768
    assert response.usage == embedding_usage(23)
    assert timeouts[0]["read"] == 15
    await registry.aclose()


async def _ollama(handler: Any, **settings: Any) -> Any:
    httpx = pytest.importorskip("httpx")
    registry = ProviderRegistry(
        secrets=SECRETS,
        embedding_factories=[OllamaEmbeddingFactory(transport=httpx.MockTransport(handler))],
    )
    return await registry.build_embedder("ollama", {**OLLAMA, **settings})


async def test_ollama_sends_the_settings_num_ctx() -> None:
    httpx = pytest.importorskip("httpx")
    bodies: list[Any] = []

    def handler(request: Any) -> Any:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"embeddings": [synthetic_vector(0, 768)] * 2, "prompt_eval_count": 4},
        )

    embedder = await _ollama(handler, num_ctx=512)
    await embedder.embed(EmbeddingRequest(model=recordings.OLLAMA, texts=TEXTS))
    assert bodies[0]["options"] == {"num_ctx": 512}
    assert bodies[0]["truncate"] is False


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (400, ProviderFailure.REJECTED),
        (404, ProviderFailure.REJECTED),
        (302, ProviderFailure.REJECTED),
        (500, ProviderFailure.UNAVAILABLE),
        (503, ProviderFailure.UNAVAILABLE),
        (504, ProviderFailure.TIMEOUT),
    ],
)
async def test_ollama_statuses_are_classified_and_carry_nothing(
    status: int, failure: ProviderFailure
) -> None:
    httpx = pytest.importorskip("httpx")

    def handler(request: Any) -> Any:
        return httpx.Response(status, json={"error": f"synthetic text for {TEXTS[0]}"})

    embedder = await _ollama(handler)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.OLLAMA, texts=TEXTS))
    assert caught.value.failure is failure
    _clean(caught.value)


async def test_ollama_transport_errors_are_classified_and_carry_nothing() -> None:
    httpx = pytest.importorskip("httpx")

    def timeout(request: Any) -> Any:
        raise httpx.ReadTimeout(f"synthetic text for {TEXTS[0]}")

    def refused(request: Any) -> Any:
        raise httpx.ConnectError(f"synthetic text for {TEXTS[0]}")

    for handler, failure in (
        (timeout, ProviderFailure.TIMEOUT),
        (refused, ProviderFailure.UNAVAILABLE),
    ):
        embedder = await _ollama(handler)
        with pytest.raises(ProviderError) as caught:
            await embedder.embed(EmbeddingRequest(model=recordings.OLLAMA, texts=TEXTS))
        assert caught.value.failure is failure
        _clean(caught.value)


def test_ollama_classifies_by_class_name_only() -> None:
    assert classify_ollama(TimeoutError()) is ProviderFailure.TIMEOUT
    assert classify_ollama(RuntimeError()) is ProviderFailure.UNAVAILABLE

    class TimeoutException(Exception):  # not httpx's: its module is this test's
        pass

    assert classify_ollama(TimeoutException()) is ProviderFailure.UNAVAILABLE


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"embeddings": [[0.1]], "prompt_eval_count": 1}',
        b'{"embeddings": [[0.1], [0.2]]}',
        b'{"embeddings": [[0.1], ["x"]], "prompt_eval_count": 1}',
        b'{"embeddings": [[0.1], [0.2]], "prompt_eval_count": 1.5}',
        b"[]",
    ],
)
async def test_a_malformed_ollama_body_is_a_protocol_failure(body: bytes) -> None:
    httpx = pytest.importorskip("httpx")

    def handler(request: Any) -> Any:
        return httpx.Response(200, content=body)

    embedder = await _ollama(handler)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=recordings.OLLAMA, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)


async def test_ollama_offers_no_shorter_vectors() -> None:
    httpx = pytest.importorskip("httpx")
    embedder = await _ollama(lambda request: httpx.Response(500))
    assert not embedder.accepts_dimensions(recordings.OLLAMA, 256)


# Every remote adapter ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "model", "client"),
    [
        (BEDROCK, "anthropic.claude-sonnet-4-6", FakeBedrock()),
        (OPENAI, "gpt-5", FakeOpenAI()),
        (GEMINI, "gemini-3.8-flash", FakeGemini()),
        (OLLAMA, "llama3.1:8b", None),
    ],
)
async def test_no_remote_adapter_will_embed_with_a_chat_model(
    settings: dict[str, Any], model: str, client: Any
) -> None:
    entry = default_catalogue().get(
        "openai" if settings["kind"] == "openai_compat" else settings["kind"], model
    )
    if entry is None:
        pytest.skip(f"{model} is not in this catalogue")
    assert entry.kind == "chat"
    embedder = await _registry().build_embedder("probe", settings, client=client)
    with pytest.raises(ConfigurationError, match="chat model"):
        await embedder.embed(EmbeddingRequest(model=model, texts=TEXTS))
