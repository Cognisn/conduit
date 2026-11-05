"""The Bedrock provider against recorded responses and fakes: no SDK, no network."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from conduit.core.catalogue import CatalogueOverlay, default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamStopped, TextDelta, ToolCallFinished, ToolCallStarted
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.bedrock import BedrockFactory, BedrockProvider, BedrockSettings
from conduit.providers.bedrock.provider import STREAM_BUFFER
from conduit.testing.replay import load_recording
from tests.unit.providers.bedrock import recordings
from tests.unit.providers.bedrock.recordings import NOTES, SYSTEM, RecordedRuntime

TOOL = ToolSpec(
    name="lookup_host",
    description="Look a host up in the inventory.",
    input_schema={
        "type": "object",
        "properties": {"host": {"type": "string"}},
        "required": ["host"],
    },
)


def turn_one(model: str) -> ModelRequest:
    return ModelRequest(
        model=model,
        system=(TextBlock(text=SYSTEM), CacheBreakpoint()),
        tools=(TOOL,),
        cache_tools=True,
        messages=(Message.user(NOTES),),
        max_output_tokens=1024,
    )


def turn_two(model: str) -> ModelRequest:
    first = turn_one(model)
    return first.model_copy(
        update={
            "messages": (
                *first.messages,
                Message(
                    role="assistant",
                    content=(
                        TextBlock(text="Checking the inventory."),
                        ToolUseBlock(
                            id="tooluse_example0001",
                            name="lookup_host",
                            input={"host": "host-a.example.com"},
                        ),
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        ToolResultBlock(
                            tool_use_id="tooluse_example0001",
                            content=(TextBlock(text="host-a.example.com: web tier, owner team-a"),),
                        ),
                    ),
                ),
            )
        }
    )


def settings(region: str = "eu-west-2", **fields: Any) -> BedrockSettings:
    return BedrockSettings(region=region, **fields)


async def built(
    runtime: Any, *, region: str = "eu-west-2", factory: BedrockFactory | None = None, **fields: Any
) -> tuple[Any, MemoryEventSink]:
    sink = MemoryEventSink()
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}), events=sink, factories=[factory or BedrockFactory()]
    )
    provider = await registry.build("bedrock_main", settings(region, **fields), client=runtime)
    return provider, sink


class TestRecordings:
    @pytest.mark.parametrize("build", recordings.ALL, ids=lambda b: b.__name__)
    def test_the_committed_recordings_are_what_the_module_builds(self, build: Any) -> None:
        recording = build()
        assert load_recording(recordings.path(recording)) == recording


class TestInRegionSonnet:
    """Q0004: bare model ids called in-region, request shape proved by the recordings."""

    async def test_sonnet_4_6_streams_a_tool_call_then_answers_from_the_result(self) -> None:
        recording = load_recording(recordings.path(recordings.sonnet_4_6_in_region()))
        runtime = RecordedRuntime(recording, region="eu-west-2")
        provider, sink = await built(runtime)
        model = "anthropic.claude-sonnet-4-6"

        events = [e async for e in provider.stream(turn_one(model))]
        assert events == [
            TextDelta(text="Checking "),
            TextDelta(text="the inventory."),
            ToolCallStarted(id="tooluse_example0001", name="lookup_host"),
            ToolCallFinished(
                id="tooluse_example0001", name="lookup_host", input={"host": "host-a.example.com"}
            ),
            StreamStopped(
                stop_reason=StopReason.TOOL_USE,
                usage=Usage(
                    input_tokens=52, output_tokens=31, cache_read_tokens=0, cache_write_tokens=1200
                ),
            ),
        ]
        answer = await provider.complete(turn_two(model))
        assert answer.text == "host-a.example.com (web tier, team-a) stopped answering at 09:00."
        assert answer.stop_reason is StopReason.END_TURN
        assert answer.usage == Usage(
            input_tokens=61, output_tokens=17, cache_read_tokens=1200, cache_write_tokens=0
        )
        runtime.replay.assert_exhausted()
        finished = [e for e in sink.events if isinstance(e, ModelCallFinishedEvent)]
        assert [e.usage for e in finished] == [events[-1].usage, answer.usage]

    async def test_sonnet_5_through_converse(self) -> None:
        recording = load_recording(recordings.path(recordings.sonnet_5_in_region()))
        runtime = RecordedRuntime(recording, region="ap-southeast-1")
        provider, _ = await built(runtime, region="ap-southeast-1", use_converse_stream=False)
        response = await provider.complete(turn_one("anthropic.claude-sonnet-5"))
        assert response.text == "Checking the inventory."
        assert response.tool_uses == (
            ToolUseBlock(
                id="tooluse_example0001", name="lookup_host", input={"host": "host-a.example.com"}
            ),
        )
        assert response.usage.cache_write_tokens == 1200
        runtime.replay.assert_exhausted()


class TestCacheGating:
    async def test_no_cache_points_where_the_catalogue_says_the_model_does_not_cache(
        self,
    ) -> None:
        model = "eu.anthropic.claude-sonnet-4-6"
        entry = default_catalogue().require("bedrock", model)
        uncached = entry.model_copy(
            update={"capabilities": entry.capabilities.model_copy(update={"prompt_caching": False})}
        )
        catalogue = default_catalogue().apply(
            CatalogueOverlay(name="no-cache", override=(uncached,))
        )
        recording = load_recording(recordings.path(recordings.uncached_profile()))
        runtime = RecordedRuntime(recording, region="eu-west-2")
        provider, _ = await built(runtime, factory=BedrockFactory(catalogue=catalogue))
        response = await provider.complete(turn_one(model))
        assert response.stop_reason is StopReason.TOOL_USE
        runtime.replay.assert_exhausted()

    def test_the_provider_reads_prompt_caching_from_its_catalogue(self) -> None:
        provider = BedrockProvider("b", settings(), object(), catalogue=default_catalogue())
        assert provider.prompt_caching("anthropic.claude-opus-5-5") is True
        assert provider.prompt_caching("amazon.titan-embed-text-v2:0") is False
        assert provider.prompt_caching("not-declared") is False


class ClientError(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(f"{code}: echoed prompt text {NOTES}")
        self.response = {
            "Error": {"Code": code, "Message": f"echoed prompt text {NOTES}"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class ScriptedRuntime:
    """A fake client whose stream is a generator the test controls."""

    def __init__(
        self,
        stream: Iterator[Any] | None = None,
        error: Exception | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        self.stream = stream
        self.error = error
        self.on_close = on_close
        self.threads: list[int] = []
        self.thread_objects: list[threading.Thread] = []
        self.closed_streams = 0

    def converse(self, **params: Any) -> Any:
        self.threads.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        raise AssertionError("not scripted")

    def converse_stream(self, **params: Any) -> Any:
        self.threads.append(threading.get_ident())
        self.thread_objects.append(threading.current_thread())
        if self.error is not None:
            raise self.error
        runtime = self

        class Stream:
            def __iter__(self) -> Iterator[Any]:
                assert runtime.stream is not None
                return runtime.stream

            def close(self) -> None:
                runtime.closed_streams += 1
                if runtime.on_close is not None:
                    runtime.on_close()

        return {"stream": Stream()}


STOP = [
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
]
REQUEST = ModelRequest(
    model="anthropic.claude-sonnet-4-6", messages=(Message.user("Hi"),), max_output_tokens=16
)


class TestErrors:
    @pytest.mark.parametrize("streaming", [True, False])
    @pytest.mark.parametrize(
        ("code", "status", "failure"),
        [
            ("ThrottlingException", 429, ProviderFailure.UNAVAILABLE),
            ("ValidationException", 400, ProviderFailure.REJECTED),
            ("AccessDeniedException", 403, ProviderFailure.REJECTED),
            ("ModelTimeoutException", 408, ProviderFailure.TIMEOUT),
        ],
    )
    async def test_a_call_error_carries_its_class_and_no_provider_text(
        self, streaming: bool, code: str, status: int, failure: ProviderFailure
    ) -> None:
        runtime = ScriptedRuntime(error=ClientError(code, status))
        provider, sink = await built(runtime, use_converse_stream=streaming)
        with pytest.raises(ProviderError) as raised:
            await provider.complete(REQUEST)
        assert raised.value.failure is failure
        assert NOTES not in str(raised.value)
        assert raised.value.__cause__ is None and raised.value.__suppress_context__
        assert NOTES not in repr(sink.events)

    async def test_an_error_part_way_through_a_stream(self) -> None:
        def stream() -> Iterator[Any]:
            yield {"contentBlockDelta": {"delta": {"text": "par"}, "contentBlockIndex": 0}}
            raise ClientError("throttlingException", 429)

        provider, _ = await built(ScriptedRuntime(stream()))
        seen: list[Any] = []
        with pytest.raises(ProviderError) as raised:
            async for event in provider.stream(REQUEST):
                seen.append(event)
        assert seen == [TextDelta(text="par")]
        assert raised.value.failure is ProviderFailure.UNAVAILABLE

    async def test_a_stalled_stream_times_out(self) -> None:
        release = threading.Event()

        def stream() -> Iterator[Any]:
            release.wait(5)
            yield from STOP

        provider, _ = await built(ScriptedRuntime(stream()))
        try:
            with pytest.raises(ProviderError) as raised:
                await provider.complete(REQUEST.model_copy(update={"timeout_s": 0.2}))
            assert raised.value.failure is ProviderFailure.TIMEOUT
        finally:
            release.set()


class TestWorkerThreads:
    async def test_streams_run_off_the_loop_and_concurrently(self) -> None:
        # Each stream blocks its thread until both have started. Run on the loop, or one
        # after the other, they would deadlock and time out.
        both = threading.Barrier(2, timeout=5)

        def stream() -> Iterator[Any]:
            both.wait()
            yield {"contentBlockDelta": {"delta": {"text": "ok"}, "contentBlockIndex": 0}}
            yield from STOP

        runtimes = [ScriptedRuntime(stream()), ScriptedRuntime(stream())]
        providers = [(await built(r))[0] for r in runtimes]
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        tick = asyncio.create_task(ticker())
        try:
            answers = await asyncio.gather(*(p.complete(REQUEST) for p in providers))
        finally:
            tick.cancel()
        assert [a.text for a in answers] == ["ok", "ok"]
        loop_thread = threading.get_ident()
        workers = [r.threads[0] for r in runtimes]
        assert loop_thread not in workers
        assert workers[0] != workers[1]
        assert ticks > 0

    async def test_an_abandoned_stream_is_closed_and_its_thread_exits(self) -> None:
        # The stream stays blocked until it is closed, as a real network read does;
        # once closed it would keep producing events, so only the worker's own check
        # of abandonment ends the thread.
        closed = threading.Event()

        def stream() -> Iterator[Any]:
            yield {"contentBlockDelta": {"delta": {"text": "first"}, "contentBlockIndex": 0}}
            closed.wait(10)
            while True:
                yield {"contentBlockDelta": {"delta": {"text": "more"}, "contentBlockIndex": 0}}
                time.sleep(0.005)

        runtime = ScriptedRuntime(stream(), on_close=closed.set)
        provider, _ = await built(runtime)
        events = provider.stream(REQUEST)
        assert await anext(events) == TextDelta(text="first")
        await events.aclose()
        assert runtime.closed_streams >= 1  # closed from the loop side, at once
        worker = runtime.thread_objects[0]
        await asyncio.to_thread(worker.join, 2)
        assert not worker.is_alive()

    async def test_the_worker_reads_only_a_bounded_way_ahead(self) -> None:
        pulled = 0

        def stream() -> Iterator[Any]:
            nonlocal pulled
            for _ in range(2_000):
                pulled += 1
                yield {"contentBlockDelta": {"delta": {"text": "x"}, "contentBlockIndex": 0}}
            yield from STOP

        provider, _ = await built(ScriptedRuntime(stream()))
        events = provider.stream(REQUEST)
        assert await anext(events) == TextDelta(text="x")
        await asyncio.sleep(0.3)  # the worker runs ahead as far as it may
        assert pulled <= STREAM_BUFFER + 2
        await events.aclose()

    async def test_aclose_closes_the_client(self) -> None:
        runtime = RecordedRuntime(
            load_recording(recordings.path(recordings.sonnet_5_in_region())), region="eu-west-2"
        )
        provider, _ = await built(runtime)
        await provider.aclose()
        assert runtime.closed
