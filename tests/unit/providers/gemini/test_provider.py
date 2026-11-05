"""The Gemini provider end to end through the registry, over a fake google-genai client.

These tests need neither the gemini extra nor a network: the fake client answers from
recordings or scripted chunks, in the JSON form the provider converts SDK chunks to.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest

from conduit.core.catalogue import Catalogue, CatalogueOverlay, default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import MeteredProvider, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamStopped, TextDelta
from conduit.core.types import (
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.gemini import GeminiFactory, GeminiProvider
from conduit.providers.gemini.mapping import SKIP_SIGNATURE
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers.gemini import recordings
from tests.unit.providers.gemini.chains import assert_nothing_attached
from tests.unit.providers.gemini.recordings import FakeClient

SETTINGS = {"kind": "gemini", "api_key": "secret://gemini/key"}
TOOL = ToolSpec(
    name="lookup_host",
    description="Look a host up in the inventory.",
    input_schema=recordings.SCHEMA,
)


async def _build(
    client: Any, *, catalogue: Catalogue | None = None, **factory: Any
) -> MeteredProvider:
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}), catalogue=catalogue, factories=[GeminiFactory(**factory)]
    )
    return await registry.build("gemini", SETTINGS, client=client)


def _structured(supported: bool) -> Catalogue:
    entry = default_catalogue().require("gemini", recordings.MODEL)
    changed = entry.model_copy(
        update={
            "capabilities": entry.capabilities.model_copy(update={"structured_output": supported})
        }
    )
    return default_catalogue().apply(CatalogueOverlay(name="structured", override=(changed,)))


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    values: dict[str, Any] = {
        "model": recordings.MODEL,
        "messages": messages or (Message.user("Hi"),),
        "max_output_tokens": 512,
    }
    values.update(fields)
    return ModelRequest(**values)


def _done(text: str = "Hi.") -> list[dict[str, Any]]:
    return [
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finish_reason": "STOP"}
            ],
            "usage_metadata": {"prompt_token_count": 3, "candidates_token_count": 2},
        }
    ]


class TestRecordings:
    @pytest.mark.parametrize("filename", sorted(recordings.RECORDINGS))
    def test_committed_recordings_match_their_builders(self, filename: str) -> None:
        committed = load_recording(recordings.DIRECTORY / filename)
        assert committed == recordings.RECORDINGS[filename]()

    async def test_a_tool_round_trip_returns_the_thought_signature(self) -> None:
        replay = Replay(load_recording(recordings.DIRECTORY / "tool-round-trip.json"))
        provider = await _build(FakeClient(recordings.replaying(replay)))
        system = (TextBlock(text=recordings.SYSTEM),)
        first = _request(Message.user(recordings.QUESTION), system=system, tools=(TOOL,))

        response = await provider.complete(first)
        assert response.text == "Checking the inventory."
        assert response.stop_reason is StopReason.TOOL_USE
        assert response.usage == Usage(
            input_tokens=96, output_tokens=64, cache_read_tokens=0, cache_write_tokens=0
        )
        (call,) = response.tool_uses
        assert call.id == "fc_example0001"
        assert call.input == {"host": "host-a.example.com"}

        second = first.model_copy(
            update={
                "messages": (
                    *first.messages,
                    Message(role="assistant", content=response.content),
                    Message(
                        role="user",
                        content=(
                            ToolResultBlock(
                                tool_use_id=call.id, content=(TextBlock(text=recordings.RESULT),)
                            ),
                        ),
                    ),
                )
            }
        )
        # The replay refuses the second request unless it carries the signature.
        answer = await provider.complete(second)
        assert answer.text == "Yes: host-a.example.com is up."
        assert answer.usage == Usage(
            input_tokens=54, output_tokens=9, cache_read_tokens=96, cache_write_tokens=0
        )
        replay.assert_exhausted()

    async def test_a_reply_cut_short(self) -> None:
        replay = Replay(load_recording(recordings.DIRECTORY / "cut-short.json"))
        provider = await _build(FakeClient(recordings.replaying(replay)))
        request = _request(
            Message.user("Describe the example.com network."),
            system=(TextBlock(text=recordings.SYSTEM),),
            max_output_tokens=8,
        )
        events = [event async for event in provider.stream(request)]
        assert events[:-1] == [
            TextDelta(text="The example.com network "),
            TextDelta(text="has two"),
        ]
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is StopReason.MAX_TOKENS
        replay.assert_exhausted()


class TestSignatures:
    async def test_a_call_the_provider_never_saw_gets_the_skip_value(self) -> None:
        client = FakeClient(lambda params: _done())
        provider = await _build(client)
        call = ToolUseBlock(id="toolu_foreign", name="lookup_host", input={})
        await provider.complete(
            _request(
                Message.user("Go"),
                Message(role="assistant", content=(call,)),
                Message(
                    role="user", content=(ToolResultBlock(tool_use_id="toolu_foreign", content=()),)
                ),
            )
        )
        parts = client.aio.models.calls[0]["contents"][1]["parts"]
        assert parts[0]["thought_signature"] == SKIP_SIGNATURE

    async def test_signature_memory_is_bounded(self) -> None:
        def handle(params: dict[str, Any]) -> list[dict[str, Any]]:
            n = len(client.aio.models.calls)
            part = {
                "function_call": {"id": f"fc_{n}", "name": "lookup_host", "args": {}},
                "thought_signature": "c2ln",
            }
            return [
                {
                    "candidates": [
                        {"content": {"role": "model", "parts": [part]}, "finish_reason": "STOP"}
                    ],
                    "usage_metadata": {},
                }
            ]

        client = FakeClient(handle)
        factory = GeminiFactory(signature_memory=2)
        settings = factory.settings_model.model_validate(SETTINGS)
        provider = factory.create_provider("gemini", settings, client)
        assert isinstance(provider, GeminiProvider)
        for _ in range(3):
            await provider.complete(_request())
        assert dict(provider.signatures) == {"fc_2": b"sig", "fc_3": b"sig"}


class _Slow:
    """``client.aio.models`` whose stream (or whose start) never answers in time."""

    def __init__(self, *, at_start: bool) -> None:
        self.at_start = at_start

    async def generate_content_stream(self, **params: Any) -> AsyncIterator[Any]:
        if self.at_start:
            await asyncio.sleep(5)

        async def stream() -> AsyncIterator[Any]:
            await asyncio.sleep(5)
            yield {}

        return stream()


class TestCatalogue:
    SCHEMA: ClassVar[dict[str, Any]] = {"type": "object", "properties": {"a": {"type": "string"}}}

    async def test_structured_output_is_sent_where_the_catalogue_allows_it(self) -> None:
        seen: list[dict[str, Any]] = []

        def handle(params: dict[str, Any]) -> Any:
            seen.append(params)
            return _done("{}")

        provider = await _build(FakeClient(handle), catalogue=_structured(True))
        await provider.complete(_request(output_schema=self.SCHEMA))
        assert seen[0]["config"]["response_json_schema"] == self.SCHEMA

    async def test_structured_output_is_refused_where_the_catalogue_says_no(self) -> None:
        seen: list[dict[str, Any]] = []

        def handle(params: dict[str, Any]) -> Any:
            seen.append(params)
            return _done("{}")

        provider = await _build(FakeClient(handle), catalogue=_structured(False))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request(output_schema=self.SCHEMA))
        assert caught.value.failure is ProviderFailure.REJECTED
        assert seen == []  # refused before any call

    def test_without_the_registry_the_factory_uses_the_default_catalogue(self) -> None:
        factory = GeminiFactory()
        settings = factory.settings_model.model_validate(SETTINGS)
        provider = factory.create_provider("gemini", settings, FakeClient(lambda p: _done()))
        assert isinstance(provider, GeminiProvider)
        assert provider.structured_output(recordings.MODEL) is True


class TestFailures:
    async def test_an_error_at_the_start_carries_no_provider_text(self) -> None:
        def handle(params: dict[str, Any]) -> Any:
            raise RuntimeError("secret detail")

        provider = await _build(FakeClient(handle))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        assert "secret detail" not in str(caught.value)
        assert_nothing_attached(caught.value, "secret detail")

    async def test_an_error_mid_stream_carries_no_provider_text(self) -> None:
        provider = await _build(
            FakeClient(lambda params: [_done()[0], ValueError("secret detail")])
        )
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        assert "secret detail" not in str(caught.value)
        assert_nothing_attached(caught.value, "secret detail")

    @pytest.mark.parametrize(
        "chunk",
        [
            {"candidates": [{"content": ["secret detail"]}]},
            {
                "candidates": [{"content": {"parts": [{"text": "a"}]}, "finish_reason": "STOP"}],
                "usage_metadata": ["secret detail"],
            },
            {"candidates": "secret detail"},
            {"candidates": [{"content": {"parts": "secret detail"}}]},
        ],
    )
    async def test_a_malformed_chunk_is_a_protocol_failure_without_its_text(
        self, chunk: dict[str, Any]
    ) -> None:
        provider = await _build(FakeClient(lambda params: [chunk]))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.PROTOCOL
        assert "secret detail" not in str(caught.value)
        assert_nothing_attached(caught.value, "secret detail")

    @pytest.mark.parametrize("at_start", [True, False])
    async def test_the_call_timeout_applies(self, at_start: bool) -> None:
        client = FakeClient(lambda params: [])
        client.aio.models = _Slow(at_start=at_start)  # type: ignore[assignment]
        provider = await _build(client)
        began = time.perf_counter()
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request(timeout_s=0.05))
        assert caught.value.failure is ProviderFailure.TIMEOUT
        assert time.perf_counter() - began < 2  # the slow parts take 5 s each

    async def test_a_stream_without_a_finish_is_a_protocol_failure(self) -> None:
        provider = await _build(FakeClient(lambda params: [{"candidates": []}]))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.PROTOCOL


class TestLifecycle:
    async def test_stopping_early_closes_the_sdk_stream(self) -> None:
        client = FakeClient(lambda params: _done("one") + _done("two"))
        provider = await _build(client)
        stream = provider.stream(_request())
        assert await anext(stream) == TextDelta(text="one")
        await stream.aclose()
        assert client.aio.models.closed_streams == 1

    async def test_aclose_closes_the_async_client(self) -> None:
        client = FakeClient(lambda params: _done())
        provider = await _build(client)
        await provider.aclose()
        assert client.aio.closed
