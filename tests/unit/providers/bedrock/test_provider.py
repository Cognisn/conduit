"""The Bedrock provider against recorded responses and fakes: no SDK, no network."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from conduit.core.catalogue import Catalogue, CatalogueOverlay, default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamStopped, TextDelta, ToolCallFinished, ToolCallStarted
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    ProviderStateBlock,
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
    runtime: Any,
    *,
    region: str = "eu-west-2",
    factory: BedrockFactory | None = None,
    catalogue: Catalogue | None = None,
    **fields: Any,
) -> tuple[Any, MemoryEventSink]:
    sink = MemoryEventSink()
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}),
        catalogue=catalogue,
        events=sink,
        factories=[factory or BedrockFactory()],
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


def plain(model: str, *, tools: bool = False) -> ModelRequest:
    """The vendor recordings' request: a cached system prompt, and the tool if asked."""
    return ModelRequest(
        model=model,
        system=(TextBlock(text=SYSTEM), CacheBreakpoint()),
        tools=(TOOL,) if tools else (),
        cache_tools=tools,
        messages=(Message.user(NOTES),),
        max_output_tokens=1024,
    )


ANSWER = "host-a.example.com stopped answering at 09:00."


class TestOtherVendors:
    """Converse chat models from vendors other than Anthropic, against recordings."""

    async def replay(self, build: Any, region: str) -> tuple[Any, RecordedRuntime]:
        recording = load_recording(recordings.path(build()))
        runtime = RecordedRuntime(recording, region=region)
        provider, _ = await built(runtime, region=region)
        return provider, runtime

    @pytest.mark.parametrize(
        ("build", "model", "region"),
        [
            (recordings.nova_lite_tools, "amazon.nova-lite-v1:0", "us-east-1"),
            (recordings.qwen_tools, "qwen.qwen3-235b-a22b-2507-v1:0", "eu-west-2"),
        ],
        ids=["amazon", "qwen"],
    )
    async def test_a_streamed_tool_call(self, build: Any, model: str, region: str) -> None:
        provider, runtime = await self.replay(build, region)
        events = [e async for e in provider.stream(plain(model, tools=True))]
        assert events[-3:-1] == [
            ToolCallStarted(id="tooluse_example0001", name="lookup_host"),
            ToolCallFinished(
                id="tooluse_example0001", name="lookup_host", input={"host": "host-a.example.com"}
            ),
        ]
        stopped = events[-1]
        assert isinstance(stopped, StreamStopped)
        assert stopped.stop_reason is StopReason.TOOL_USE
        assert stopped.usage.input_tokens == 52
        assert stopped.usage.output_tokens == 31
        runtime.replay.assert_exhausted()

    async def test_nova_writes_its_cache_from_the_system_point(self) -> None:
        provider, runtime = await self.replay(recordings.nova_lite_tools, "us-east-1")
        response = await provider.complete(plain("amazon.nova-lite-v1:0", tools=True))
        assert response.usage.cache_write_tokens == 1200
        runtime.replay.assert_exhausted()

    async def test_tools_go_through_converse_where_the_stream_refuses_them(self) -> None:
        # The provider streams by default; Mistral Large (24.02) refuses tools on
        # ConverseStream, so the recording expects a Converse call.
        provider, runtime = await self.replay(recordings.mistral_large_tools, "eu-west-2")
        assert provider.settings.use_converse_stream
        response = await provider.complete(plain("mistral.mistral-large-2402-v1:0", tools=True))
        assert response.tool_uses == (
            ToolUseBlock(
                id="tooluse_example0002", name="lookup_host", input={"host": "host-a.example.com"}
            ),
        )
        assert response.usage == Usage(
            input_tokens=64, output_tokens=22, cache_read_tokens=0, cache_write_tokens=0
        )
        runtime.replay.assert_exhausted()

    async def test_without_tools_the_same_model_streams(self) -> None:
        # ScriptedRuntime fails any Converse call, so this passes only by streaming.
        runtime = ScriptedRuntime(stream=iter(STOP))
        provider, _ = await built(runtime)
        response = await provider.complete(plain("mistral.mistral-large-2402-v1:0"))
        assert response.stop_reason is StopReason.END_TURN
        assert len(runtime.threads) == 1

    @pytest.mark.parametrize(
        ("build", "model", "region"),
        [
            (recordings.gpt_oss_reasoning, "openai.gpt-oss-120b-1:0", "us-west-2"),
            (recordings.nemotron_reasoning, "nvidia.nemotron-super-3-120b", "eu-west-2"),
        ],
        ids=["openai-stream", "nvidia-converse"],
    )
    async def test_reasoning_without_a_signature_is_state_not_text(
        self, build: Any, model: str, region: str
    ) -> None:
        recording = load_recording(recordings.path(build()))
        runtime = RecordedRuntime(recording, region=region)
        streaming = recording.interactions[0].request["operation"] == "ConverseStream"
        provider, _ = await built(runtime, region=region, use_converse_stream=streaming)
        response = await provider.complete(plain(model))
        assert response.text == ANSWER
        states = [b for b in response.content if isinstance(b, ProviderStateBlock)]
        assert len(states) == 1
        assert states[0].provider_kind == "bedrock"
        assert json.loads(states[0].payload) == {"reasoningText": {"text": recordings.REASONING}}
        runtime.replay.assert_exhausted()

    @pytest.mark.parametrize(
        ("build", "model", "region"),
        [
            (recordings.gpt_profile_text, "in.openai.gpt-5.6-terra", "ap-south-1"),
            (recordings.deepseek_text, "deepseek.v3.2", "ap-southeast-2"),
            (recordings.minimax_text, "minimax.minimax-m2.5", "eu-west-2"),
            (recordings.kimi_text, "moonshotai.kimi-k2.5", "ap-southeast-2"),
            (recordings.palmyra_text, "writer.palmyra-vision-7b", "us-east-1"),
            (recordings.glm_text, "zai.glm-5", "us-west-2"),
        ],
        ids=["openai-profile", "deepseek", "minimax", "moonshot", "writer", "zai"],
    )
    async def test_a_streamed_answer_and_its_usage(
        self, build: Any, model: str, region: str
    ) -> None:
        provider, runtime = await self.replay(build, region)
        events = [e async for e in provider.stream(plain(model))]
        assert [e.text for e in events if isinstance(e, TextDelta)] == [
            "host-a.example.com ",
            "stopped answering at 09:00.",
        ]
        stopped = events[-1]
        assert isinstance(stopped, StreamStopped)
        assert stopped.stop_reason is StopReason.END_TURN
        assert stopped.usage.input_tokens > 0 and stopped.usage.output_tokens > 0
        assert stopped.usage.cache_read_tokens == stopped.usage.cache_write_tokens == 0
        runtime.replay.assert_exhausted()

    @pytest.mark.parametrize("build", recordings.VENDORS, ids=lambda b: b.__name__)
    def test_each_vendor_model_is_declared_for_its_recorded_region(self, build: Any) -> None:
        request = build().interactions[0].request
        entry = default_catalogue().require("bedrock", request["params"]["modelId"])
        assert entry.residency_scope.value in {"in_region", "cross_region"}
        assert request["region"] in entry.regions


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
        provider, _ = await built(runtime, catalogue=catalogue)
        response = await provider.complete(turn_one(model))
        assert response.stop_reason is StopReason.TOOL_USE
        runtime.replay.assert_exhausted()

    def test_without_the_registry_the_factory_uses_the_default_catalogue(self) -> None:
        provider = BedrockFactory().create_provider("b", settings(), object())
        assert isinstance(provider, BedrockProvider)
        assert provider.prompt_caching("anthropic.claude-opus-5-5") is True

    def test_the_factory_hands_its_provider_the_catalogue_it_is_given(self) -> None:
        model = "anthropic.claude-opus-5-5"
        entry = default_catalogue().require("bedrock", model)
        uncached = entry.model_copy(
            update={"capabilities": entry.capabilities.model_copy(update={"prompt_caching": False})}
        )
        catalogue = default_catalogue().apply(
            CatalogueOverlay(name="no-cache", override=(uncached,))
        )
        provider = BedrockFactory().create_provider("b", settings(), object(), catalogue=catalogue)
        assert isinstance(provider, BedrockProvider)
        assert provider.prompt_caching(model) is False

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
        assert raised.value.__cause__ is None and raised.value.__context__ is None
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
            assert raised.value.__cause__ is None and raised.value.__context__ is None
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


CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1},
}


class ConverseRuntime:
    """A fake client whose Converse call waits on ``release`` (if given), then answers."""

    def __init__(
        self, *, release: threading.Event | None = None, error: Exception | None = None
    ) -> None:
        self.release = release
        self.error = error
        self.threads: list[threading.Thread] = []
        self.returned = threading.Event()

    def converse(self, **params: Any) -> Any:
        self.threads.append(threading.current_thread())
        try:
            if self.release is not None:
                self.release.wait(10)
            if self.error is not None:
                raise self.error
            return CONVERSE_RESPONSE
        finally:
            self.returned.set()


class TestConverseThread:
    """Q0020: each Converse call runs on its own daemon thread, as ConverseStream does."""

    async def test_converse_runs_on_its_own_daemon_thread(self) -> None:
        runtime = ConverseRuntime()
        provider, _ = await built(runtime, use_converse_stream=False)
        assert (await provider.complete(REQUEST)).text == "ok"
        (thread,) = runtime.threads
        assert thread.daemon
        assert thread.name == "conduit-bedrock-converse"
        assert thread is not threading.current_thread()

    async def test_a_timed_out_call_holds_no_executor_thread(self) -> None:
        # With one executor worker, a call left waiting in the executor would block
        # the next piece of executor work until the stuck call returned.
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        release = threading.Event()
        runtime = ConverseRuntime(release=release)
        provider, _ = await built(runtime, use_converse_stream=False)
        try:
            with pytest.raises(ProviderError) as raised:
                await provider.complete(REQUEST.model_copy(update={"timeout_s": 0.2}))
            assert raised.value.failure is ProviderFailure.TIMEOUT
            assert raised.value.__cause__ is None and raised.value.__context__ is None
            assert await asyncio.wait_for(loop.run_in_executor(None, lambda: 7), 2) == 7
        finally:
            release.set()
            await asyncio.to_thread(runtime.returned.wait, 5)

    async def test_a_late_answer_after_the_timeout_is_dropped_quietly(self) -> None:
        release = threading.Event()
        runtime = ConverseRuntime(release=release)
        provider, _ = await built(runtime, use_converse_stream=False)
        loop = asyncio.get_running_loop()
        reported: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        try:
            with pytest.raises(ProviderError):
                await provider.complete(REQUEST.model_copy(update={"timeout_s": 0.1}))
            release.set()
            await asyncio.to_thread(runtime.returned.wait, 5)
            (thread,) = runtime.threads
            await asyncio.to_thread(thread.join, 2)
            assert not thread.is_alive()
            for _ in range(5):
                await asyncio.sleep(0)  # the late answer's callback runs on the loop
            assert reported == []  # nothing failed on the loop when it arrived
        finally:
            loop.set_exception_handler(None)

    async def test_an_error_on_the_thread_carries_its_class_and_no_provider_text(self) -> None:
        runtime = ConverseRuntime(error=ClientError("ThrottlingException", 429))
        provider, _ = await built(runtime, use_converse_stream=False)
        with pytest.raises(ProviderError) as raised:
            await provider.complete(REQUEST)
        assert raised.value.failure is ProviderFailure.UNAVAILABLE
        assert raised.value.__cause__ is None and raised.value.__context__ is None
        assert NOTES not in str(raised.value)

    async def test_cancelling_the_caller_propagates_unchanged(self) -> None:
        release = threading.Event()
        runtime = ConverseRuntime(release=release)
        provider, _ = await built(runtime, use_converse_stream=False)
        task = asyncio.create_task(provider.complete(REQUEST))
        try:
            while not runtime.threads:
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.to_thread(runtime.returned.wait, 5)


class TestStalledConsumer:
    """A consumer that stops iterating without closing: the worker gives up and closes."""

    @staticmethod
    def _endless() -> Iterator[Any]:
        while True:
            yield {"contentBlockDelta": {"delta": {"text": "x"}, "contentBlockIndex": 0}}

    async def test_the_worker_gives_up_after_the_timeout_and_closes_the_stream(self) -> None:
        runtime = ScriptedRuntime(self._endless())
        provider, _ = await built(runtime)
        events = provider.stream(REQUEST.model_copy(update={"timeout_s": 0.3}))
        assert await anext(events) == TextDelta(text="x")
        # Neither iterated nor closed from here: the worker waits at the bound.
        worker = runtime.thread_objects[0]
        await asyncio.to_thread(worker.join, 5)
        assert not worker.is_alive()
        assert runtime.closed_streams >= 1
        await events.aclose()

    async def test_the_stream_closes_at_the_give_up_even_if_the_model_then_pauses(self) -> None:
        # After the event that fills the queue the model goes quiet. The worker must
        # close the stream as it gives up, not wait for another upstream event.
        paused = threading.Event()

        def stream() -> Iterator[Any]:
            for _ in range(STREAM_BUFFER + 2):
                yield {"contentBlockDelta": {"delta": {"text": "x"}, "contentBlockIndex": 0}}
            paused.wait(10)
            yield from STOP

        runtime = ScriptedRuntime(stream())
        provider, _ = await built(runtime)
        events = provider.stream(REQUEST.model_copy(update={"timeout_s": 0.3}))
        try:
            assert await anext(events) == TextDelta(text="x")
            started = time.monotonic()
            while runtime.closed_streams == 0 and time.monotonic() - started < 3:
                await asyncio.sleep(0.05)
            assert runtime.closed_streams >= 1
            assert time.monotonic() - started < 2  # about the timeout, not the pause
        finally:
            paused.set()
            await events.aclose()

    async def test_a_consumer_that_comes_back_after_the_give_up_times_out(self) -> None:
        runtime = ScriptedRuntime(self._endless())
        provider, _ = await built(runtime)
        events = provider.stream(REQUEST.model_copy(update={"timeout_s": 0.3}))
        await anext(events)
        await asyncio.to_thread(runtime.thread_objects[0].join, 5)
        read = 0
        try:
            with pytest.raises(ProviderError) as raised:
                # Only what was buffered before the give-up remains to be read.
                while read <= STREAM_BUFFER + 2:
                    await anext(events)
                    read += 1
            assert raised.value.failure is ProviderFailure.TIMEOUT
            assert raised.value.__cause__ is None and raised.value.__context__ is None
        finally:
            await events.aclose()

    async def test_a_pause_shorter_than_the_timeout_does_not_give_up(self) -> None:
        def stream() -> Iterator[Any]:
            for _ in range(STREAM_BUFFER * 2):
                yield {"contentBlockDelta": {"delta": {"text": "x"}, "contentBlockIndex": 0}}
            yield from STOP

        runtime = ScriptedRuntime(stream())
        provider, _ = await built(runtime)
        events = provider.stream(REQUEST.model_copy(update={"timeout_s": 1.0}))
        await anext(events)
        await asyncio.sleep(0.4)  # the worker waits at the bound, within the timeout
        rest = [event async for event in events]
        assert isinstance(rest[-1], StreamStopped)
        assert sum(isinstance(e, TextDelta) for e in rest) == STREAM_BUFFER * 2 - 1
