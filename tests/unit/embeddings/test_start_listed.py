"""Embedding start-up on each provider's own listing (C18).

``start_embeddings`` lists every provider an embedding task class is assigned to, through
the embedding adapter (which inherits its chat adapter's discovery), and resolves each
assignment on that listing. Every listing here is a synthetic recording replayed through
the adapter's real discovery code, or a synthetic answer from a fake control-plane or SDK
client; nothing reaches a network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr

from conduit.core.errors import AssignmentRefused, ProviderError, ResidencyRefused
from conduit.core.events import DiscoveryFinishedEvent, MemoryEventSink
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.embeddings import (
    BedrockEmbeddingFactory,
    GeminiEmbeddingFactory,
    OllamaEmbeddingFactory,
    OpenAICompatEmbeddingFactory,
    start_embeddings,
)
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

httpx = pytest.importorskip("httpx", reason="needs the ollama and gemini extras")

EMBED = TaskClass(name="embed", floor=CapabilityFloor(kind="embedding"))
SECRETS = LiteralSecretResolver(
    {"gemini/key": "AIzaSyD-SYNTHETIC-EXAMPLE-KEY-000000000", "openai/key": "not-a-real-key"}
)
Handler = Callable[[Any], Any]


def _settings(provider: Mapping[str, Any], model: str, **extra: Any) -> ConduitSettings:
    return ConduitSettings.model_validate(
        {
            "providers": {"embedder": dict(provider)},
            "assignments": {"embed": {"provider": "embedder", "model": model}},
            **extra,
        }
    )


def _http(name: str, *, with_query: bool) -> tuple[Handler, Replay]:
    replay = Replay(load_recording(recordings.ROOT / name))

    def handle(request: Any) -> Any:
        sent: dict[str, Any] = {"method": request.method, "path": request.url.path}
        if with_query:
            sent["query"] = dict(sorted(request.url.params.items()))
        recorded = replay.next(sent)
        return httpx.Response(recorded["status"], json=recorded["body"])

    return handle, replay


def _never(request: Any) -> Any:
    raise AssertionError("no listing is made in catalogue mode")


async def _start(
    settings: ConduitSettings, factory: Any, sink: MemoryEventSink | None = None
) -> Any:
    return await start_embeddings(
        settings, task_classes=[EMBED], factories=[factory], secrets=SECRETS, events=sink
    )


OLLAMA = {"kind": "ollama"}
GEMINI = {"kind": "gemini", "api_key": "secret://gemini/key", "max_attempts": 1}


class TestOllama:
    async def test_a_listed_embedding_model_starts(self) -> None:
        handler, replay = _http("ollama/discovery-tags.json", with_query=False)
        factory = OllamaEmbeddingFactory(transport=httpx.MockTransport(handler))
        runtime = await _start(_settings(OLLAMA, "nomic-embed-text"), factory)
        try:
            replay.assert_exhausted()
            assert runtime.discovery["embedder"].supported
            entry = runtime.for_task("embed").assignment.entry
            assert entry.residency_scope is ResidencyScope.LOCAL
            assert entry.embedding_dimensions == 768
        finally:
            await runtime.aclose()

    async def test_a_model_the_server_does_not_list_is_refused(self) -> None:
        handler, _ = _http("ollama/discovery-empty.json", with_query=False)
        factory = OllamaEmbeddingFactory(transport=httpx.MockTransport(handler))
        with pytest.raises(AssignmentRefused, match="is not listed by provider 'embedder'"):
            await _start(_settings(OLLAMA, "nomic-embed-text"), factory)

    async def test_a_failed_listing_fails_closed(self) -> None:
        sink = MemoryEventSink()
        factory = OllamaEmbeddingFactory(
            transport=httpx.MockTransport(lambda request: httpx.Response(500, json={}))
        )
        with pytest.raises(ProviderError):
            await _start(_settings(OLLAMA, "nomic-embed-text"), factory, sink)
        [event] = [e for e in sink.events if isinstance(e, DiscoveryFinishedEvent)]
        assert event.outcome == "failed"

    async def test_catalogue_mode_makes_no_listing(self) -> None:
        factory = OllamaEmbeddingFactory(transport=httpx.MockTransport(_never))
        settings = _settings(OLLAMA, "nomic-embed-text", discovery={"embedder": "catalogue"})
        runtime = await _start(settings, factory)
        try:
            assert runtime.discovery == {}
            assert runtime.for_task("embed").assignment.model == "nomic-embed-text"
        finally:
            await runtime.aclose()


class TestGemini:
    async def test_the_embedding_model_is_listed_and_starts(self) -> None:
        pytest.importorskip("google.genai", reason="needs the gemini extra")
        handler, replay = _http("gemini/discovery-models.json", with_query=True)
        factory = GeminiEmbeddingFactory(
            async_client_args={"transport": httpx.MockTransport(handler)}
        )
        runtime = await _start(_settings(GEMINI, "gemini-embedding-001"), factory)
        try:
            replay.assert_exhausted()
            entry = runtime.for_task("embed").assignment.entry
            assert entry.kind == "embedding"
            assert entry.residency_scope is ResidencyScope.GLOBAL
        finally:
            await runtime.aclose()

    async def test_a_region_policy_refuses_the_global_api(self) -> None:
        pytest.importorskip("google.genai", reason="needs the gemini extra")
        handler, _ = _http("gemini/discovery-models.json", with_query=True)
        factory = GeminiEmbeddingFactory(
            async_client_args={"transport": httpx.MockTransport(handler)}
        )
        policy = ResidencyPolicy(regions=frozenset({"xx-home-1"})).model_dump()
        with pytest.raises(ResidencyRefused, match="global"):
            await _start(_settings(GEMINI, "gemini-embedding-001", residency=policy), factory)


class _FakeControl:
    """A Bedrock control-plane client answering from the synthetic listing recording."""

    def __init__(self) -> None:
        self.replay = Replay(load_recording(recordings.ROOT / "bedrock/discovery-listing.json"))

    def _answer(self, operation: str, params: dict[str, Any]) -> Any:
        return self.replay.next({"operation": operation, "params": params})["body"]

    def list_foundation_models(self, **params: Any) -> Any:
        return self._answer("ListFoundationModels", params)

    def list_inference_profiles(self, **params: Any) -> Any:
        return self._answer("ListInferenceProfiles", params)


class _ListedBedrock(BedrockEmbeddingFactory):
    """Bedrock embeddings with a fake runtime client and a recorded control plane."""

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        pytest.importorskip("boto3", reason="needs the bedrock extra")
        from conduit.providers.bedrock.factory import BedrockClients

        return BedrockClients(runtime=object(), control=_FakeControl())


class TestBedrock:
    SETTINGS: ClassVar[dict[str, Any]] = {"kind": "bedrock", "region": recordings.REGION}

    async def test_titan_is_processed_in_the_region_it_is_listed_in(self) -> None:
        policy = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({recordings.REGION})
        ).model_dump()
        settings = _settings(self.SETTINGS, "amazon.titan-embed-text-v2:0", residency=policy)
        runtime = await _start(settings, _ListedBedrock())
        try:
            entry = runtime.for_task("embed").assignment.entry
            assert entry.regions == frozenset({recordings.REGION})
            assert entry.embedding_dimensions is not None
        finally:
            await runtime.aclose()

    async def test_a_policy_without_the_listed_region_refuses_it(self) -> None:
        policy = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({"eu-west-2"})
        ).model_dump()
        settings = _settings(self.SETTINGS, "amazon.titan-embed-text-v2:0", residency=policy)
        with pytest.raises(ResidencyRefused, match=recordings.REGION):
            await _start(settings, _ListedBedrock())


class _ListedOpenAI(OpenAICompatEmbeddingFactory):
    """OpenAI embeddings whose SDK answers the Models API from the synthetic recording."""

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        openai = pytest.importorskip("openai", reason="needs the openai extra")
        httpx2 = pytest.importorskip("httpx2", reason="needs the openai extra")
        from conduit.providers.openai_compat.factory import OpenAICompatClient

        replay = Replay(load_recording(recordings.ROOT / "openai_compat/discovery-openai.json"))

        def handle(request: Any) -> Any:
            query = dict(sorted(request.url.params.items()))
            sent = {"method": request.method, "path": request.url.path, "query": query}
            recorded = replay.next(sent)
            return httpx2.Response(recorded["status"], json=recorded["body"])

        sdk = openai.AsyncOpenAI(
            api_key="not-a-real-key",
            base_url="https://api.openai.com/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        )
        return OpenAICompatClient(sdk=sdk)


class TestOpenAI:
    SETTINGS: ClassVar[dict[str, Any]] = {
        "kind": "openai_compat",
        "service": {"name": "openai", "api_key": "secret://openai/key"},
    }

    async def test_a_listed_embedding_model_starts(self) -> None:
        runtime = await _start(_settings(self.SETTINGS, "text-embedding-3-small"), _ListedOpenAI())
        try:
            assert runtime.discovery["embedder"].get("text-embedding-3-small") is not None
            assert runtime.for_task("embed").assignment.entry.kind == "embedding"
        finally:
            await runtime.aclose()

    async def test_a_model_the_api_does_not_list_is_refused(self) -> None:
        with pytest.raises(AssignmentRefused, match="is not listed"):
            await _start(_settings(self.SETTINGS, "text-embedding-3-large"), _ListedOpenAI())
