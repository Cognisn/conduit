"""The provider end to end, through the registry, against recorded responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from conduit.core.catalogue import (
    Capabilities,
    Catalogue,
    CatalogueOverlay,
    ModelEntry,
    default_catalogue,
)
from conduit.core.errors import ProviderError, ProviderFailure, UndeclaredModel
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.registry import MeteredProvider, ProviderRegistry
from conduit.core.residency import ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamEvent, StreamStopped, TextDelta, ToolCallFinished
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers import builtin_factories
from conduit.providers.openai_compat import OpenAICompatFactory, OpenAICompatProvider
from tests.unit.providers.openai_compat import recordings
from tests.unit.providers.openai_compat.fakes import FakeClient

SYSTEM = (TextBlock(text="You are a concise assistant."), CacheBreakpoint())
WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
OPENAI = {"kind": "openai_compat", "service": {"name": "openai", "api_key": "secret://OPENAI"}}
XAI = {"kind": "openai_compat", "service": {"name": "xai", "api_key": "secret://XAI"}}
FOUNDRY = {
    "kind": "openai_compat",
    "region": "australiaeast",
    "service": {
        "name": "foundry",
        "endpoint": "https://example-resource.openai.azure.com/openai/v1/",
        "auth": {"mode": "entra_id"},
    },
}


def _deployment(model_id: str, *, structured_output: bool) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider="openai",
        residency_scope=ResidencyScope.IN_REGION,
        regions=frozenset({"australiaeast"}),
        capabilities=Capabilities(
            tools=True, vision=True, prompt_caching=True, structured_output=structured_output
        ),
        context_window=200_000,
        output_limit=32_000,
    )


def _catalogue() -> Catalogue:
    return default_catalogue().apply(
        CatalogueOverlay(
            name="foundry-deployments",
            add=(
                _deployment("example-gpt-deployment", structured_output=True),
                _deployment("example-plain-deployment", structured_output=False),
            ),
        )
    )


async def _build(
    settings: dict[str, Any], client: FakeClient, *, events: MemoryEventSink | None = None
) -> MeteredProvider:
    catalogue = _catalogue()
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}),
        catalogue=catalogue,
        events=events,
        factories=[OpenAICompatFactory(catalogue=catalogue)],
    )
    return await registry.build("test", settings, client=client)


def _request(model: str, *messages: Message, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model=model, system=SYSTEM, messages=messages, max_output_tokens=512, **fields
    )


class TestOpenAI:
    async def test_text_with_cached_usage(self) -> None:
        client = FakeClient(recordings.load("openai-text-cached"))
        provider = await _build(OPENAI, client)
        response = await provider.complete(_request("gpt-5", Message.user("Say hello.")))
        client.replay.assert_exhausted()
        assert response.text == "Hello, there."
        assert response.stop_reason is StopReason.END_TURN
        assert response.usage == Usage(
            input_tokens=128, output_tokens=24, cache_read_tokens=1920, cache_write_tokens=0
        )

    async def test_parallel_tool_calls(self) -> None:
        client = FakeClient(recordings.load("openai-parallel-tool-calls"))
        provider = await _build(OPENAI, client)
        response = await provider.complete(
            _request("gpt-5", Message.user("Weather in Sydney and Hobart?"), tools=(WEATHER,))
        )
        assert response.stop_reason is StopReason.TOOL_USE
        assert [(t.id, t.input) for t in response.tool_uses] == [
            ("call_example_1", {"city": "Sydney"}),
            ("call_example_2", {"city": "Hobart"}),
        ]
        assert response.usage.input_tokens == 310
        assert response.usage.cache_read_tokens == 0

    async def test_tool_results_and_images(self) -> None:
        client = FakeClient(recordings.load("openai-tool-results"))
        provider = await _build(OPENAI, client)
        request = _request(
            "gpt-5",
            Message.user("Weather in Sydney and Hobart?"),
            Message(
                role="assistant",
                content=(
                    ToolUseBlock(id="call_example_1", name="get_weather", input={"city": "Sydney"}),
                    ToolUseBlock(id="call_example_2", name="get_weather", input={"city": "Hobart"}),
                ),
            ),
            Message(
                role="user",
                content=(
                    ToolResultBlock(
                        tool_use_id="call_example_1",
                        content=(
                            TextBlock(text="24 C, sunny"),
                            ImageBlock(media_type="image/png", data=recordings.PNG),
                        ),
                    ),
                    ToolResultBlock(
                        tool_use_id="call_example_2",
                        content=(TextBlock(text="weather service unavailable"),),
                        is_error=True,
                    ),
                ),
            ),
            tools=(WEATHER,),
        )
        response = await provider.complete(request)
        client.replay.assert_exhausted()
        assert response.text == "Sydney is 24 C and sunny; Hobart is unavailable."
        assert response.usage.input_tokens == 420  # no prompt_tokens_details: nothing cached

    async def test_the_request_timeout_is_passed_to_the_sdk(self) -> None:
        client = FakeClient(recordings.load("openai-text-cached"))
        provider = await _build(OPENAI, client)
        await provider.complete(_request("gpt-5", Message.user("Say hello."), timeout_s=42))
        assert client.calls[0]["timeout"] == 42

    async def test_the_metered_event_carries_all_four_counts(self) -> None:
        sink = MemoryEventSink()
        client = FakeClient(recordings.load("openai-text-cached"))
        provider = await _build(OPENAI, client, events=sink)
        await provider.complete(_request("gpt-5", Message.user("Say hello.")))
        finished = [e for e in sink.events if isinstance(e, ModelCallFinishedEvent)]
        assert len(finished) == 1
        assert finished[0].usage == Usage(
            input_tokens=128, output_tokens=24, cache_read_tokens=1920, cache_write_tokens=0
        )
        assert finished[0].kind == "openai_compat"


class TestXAI:
    async def test_text_with_cached_usage(self) -> None:
        client = FakeClient(recordings.load("xai-text-cached"))
        provider = await _build(XAI, client)
        response = await provider.complete(_request("grok-4.3", Message.user("Say hello.")))
        assert response.text == "Hello."
        assert response.usage == Usage(
            input_tokens=27, output_tokens=2, cache_read_tokens=98, cache_write_tokens=0
        )

    async def test_models_are_looked_up_under_xai(self) -> None:
        provider = await _build(XAI, FakeClient(recordings.load("xai-text-cached")))
        with pytest.raises(UndeclaredModel):
            await provider.complete(_request("gpt-5", Message.user("Say hello.")))


class TestFoundry:
    SCHEMA = recordings.ANSWER_SCHEMA

    async def test_structured_output_where_the_catalogue_allows_it(self) -> None:
        client = FakeClient(recordings.load("foundry-structured-output"))
        provider = await _build(FOUNDRY, client)
        response = await provider.complete(
            _request(
                "example-gpt-deployment",
                Message.user("Weather in Hobart as JSON."),
                output_schema=self.SCHEMA,
            )
        )
        client.replay.assert_exhausted()
        assert response.text == '{"city": "Hobart", "celsius": 11}'
        assert response.usage.input_tokens == 76
        assert response.usage.cache_read_tokens == 1024

    async def test_structured_output_is_rejected_where_the_catalogue_says_no(self) -> None:
        client = FakeClient(recordings.load("foundry-structured-output"))
        provider = await _build(FOUNDRY, client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(
                _request(
                    "example-plain-deployment",
                    Message.user("Weather in Hobart as JSON."),
                    output_schema=self.SCHEMA,
                )
            )
        assert caught.value.failure is ProviderFailure.REJECTED
        assert client.calls == []  # refused before any call


class TestErrors:
    TEXT = "provider text that must never be repeated"

    def _error(self, name: str, **attributes: Any) -> Exception:
        error: Exception = type(name, (Exception,), {})(self.TEXT)
        for key, value in attributes.items():
            setattr(error, key, value)
        return error

    async def test_an_error_on_create_is_a_provider_error_without_text(self) -> None:
        error = self._error("RateLimitError", status_code=429)
        client = FakeClient(recordings.load("openai-text-cached"), create_error=error)
        provider = await _build(OPENAI, client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request("gpt-5", Message.user("Say hello.")))
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        assert self.TEXT not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    async def test_an_error_mid_stream_is_a_provider_error_and_the_stream_is_closed(
        self,
    ) -> None:
        error = self._error("APITimeoutError")
        client = FakeClient(
            recordings.load("openai-text-cached"), stream_error=error, stream_error_after=2
        )
        provider = await _build(OPENAI, client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request("gpt-5", Message.user("Say hello.")))
        assert caught.value.failure is ProviderFailure.TIMEOUT
        assert self.TEXT not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert client.streams[0].closed

    async def test_a_stream_left_early_is_closed(self) -> None:
        client = FakeClient(recordings.load("openai-text-cached"))
        # The adapter's own stream; the metered wrapper leaves closing to the loop.
        provider = OpenAICompatProvider(client, catalogue_provider="openai")
        stream = cast(
            AsyncGenerator[StreamEvent, None],
            provider.stream(_request("gpt-5", Message.user("Say hello."))),
        )
        async for event in stream:
            if isinstance(event, TextDelta):
                break
        await stream.aclose()
        assert client.streams[0].closed

    async def test_a_cancelled_call_is_not_swallowed(self) -> None:
        client = FakeClient(
            recordings.load("openai-text-cached"), create_error=asyncio.CancelledError()
        )
        provider = await _build(OPENAI, client)
        with pytest.raises(asyncio.CancelledError):
            await provider.complete(_request("gpt-5", Message.user("Say hello.")))


class TestLifecycle:
    async def test_closing_the_provider_closes_the_client(self) -> None:
        client = FakeClient(recordings.load("openai-text-cached"))
        provider = await _build(OPENAI, client)
        await provider.aclose()
        assert client.closed

    async def test_a_client_without_close_is_fine(self) -> None:
        provider = OpenAICompatProvider(object(), catalogue_provider="openai")
        await provider.aclose()

    def test_the_factory_is_built_in(self) -> None:
        kinds = [f.kind for f in builtin_factories()]
        assert "openai_compat" in kinds

    async def test_concurrent_streams_are_independent(self) -> None:
        one = FakeClient(recordings.load("openai-text-cached"))
        two = FakeClient(recordings.load("xai-text-cached"))
        first = await _build(OPENAI, one)
        second = await _build(XAI, two)
        a, b = await asyncio.gather(
            first.complete(_request("gpt-5", Message.user("Say hello."))),
            second.complete(_request("grok-4.3", Message.user("Say hello."))),
        )
        assert (a.text, b.text) == ("Hello, there.", "Hello.")


class TestSdkChunkObjects:
    """The real SDK yields pydantic chunk objects, not mappings."""

    async def test_chunk_objects_map_like_their_json(self) -> None:
        chat = pytest.importorskip("openai.types.chat")
        client = FakeClient(
            recordings.load("openai-parallel-tool-calls"),
            wrap=chat.ChatCompletionChunk.model_validate,
        )
        provider = await _build(OPENAI, client)
        events = [
            e
            async for e in provider.stream(
                _request("gpt-5", Message.user("Weather in Sydney and Hobart?"), tools=(WEATHER,))
            )
        ]
        finished = [e for e in events if isinstance(e, ToolCallFinished)]
        assert [e.input for e in finished] == [{"city": "Sydney"}, {"city": "Hobart"}]
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage.input_tokens == 310


class TestRecordings:
    @pytest.mark.parametrize("name", sorted(recordings.BUILDERS))
    def test_committed_recordings_equal_their_builders(self, name: str) -> None:
        assert recordings.load(name) == recordings.BUILDERS[name]()
