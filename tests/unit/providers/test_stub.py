"""The stub provider: deterministic, scriptable, streaming, and never for production."""

from __future__ import annotations

import logging

import pytest

from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.registry import ProviderRegistry
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import (
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    UsageReported,
)
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from conduit.providers import builtin_factories
from conduit.providers.stub import (
    StubFactory,
    StubProvider,
    StubSettings,
    StubToolCall,
    StubTurn,
    answer_tool_results,
    echo,
    scripted,
)


def _request(*messages: Message, **fields: object) -> ModelRequest:
    return ModelRequest(
        model="stub-echo",
        messages=messages or (Message.user("hello there"),),
        max_output_tokens=256,
        **fields,  # type: ignore[arg-type]
    )


def _stub(script: object = echo) -> StubProvider:
    return StubProvider(script, not_for_production=True)  # type: ignore[arg-type]


class TestNotForProduction:
    def test_construction_needs_the_acknowledgement(self) -> None:
        with pytest.raises(ConfigurationError, match="not for production"):
            StubProvider(echo, not_for_production=False)

    def test_settings_need_the_acknowledgement(self) -> None:
        with pytest.raises(ValueError, match="not_for_production"):
            StubSettings.model_validate({"kind": "stub"})
        StubSettings.model_validate({"kind": "stub", "not_for_production": True})

    def test_construction_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="conduit"):
            _stub()
        assert "not for production" in caplog.text


class TestStreaming:
    async def test_echo_streams_text_and_ends_with_usage(self) -> None:
        events = [e async for e in _stub().stream(_request())]
        assert isinstance(events[0], UsageReported)
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "hello there"
        assert len([e for e in events if isinstance(e, TextDelta)]) > 1  # really streamed
        assert isinstance(events[-1], StreamStopped)
        assert events[-1].stop_reason is StopReason.END_TURN

    async def test_it_is_deterministic(self) -> None:
        first = [e async for e in _stub().stream(_request())]
        second = [e async for e in _stub().stream(_request())]
        assert first == second

    async def test_complete_is_the_collected_stream(self) -> None:
        response = await _stub().complete(_request())
        assert response.text == "hello there"
        assert response.model == "stub-echo"

    async def test_requests_are_recorded(self) -> None:
        stub = _stub()
        await stub.complete(_request())
        assert len(stub.requests) == 1


class TestUsage:
    async def test_all_four_counts_are_emitted(self) -> None:
        usage = (await _stub().complete(_request())).usage
        assert usage.input_tokens > 0 and usage.output_tokens > 0
        assert usage.cache_read_tokens == 0 and usage.cache_write_tokens == 0

    async def test_a_cache_breakpoint_writes_then_reads(self) -> None:
        stub = _stub()
        request = _request(
            system=(TextBlock(text="You are a careful analyst. " * 20), CacheBreakpoint())
        )
        first = (await stub.complete(request)).usage
        second = (await stub.complete(request)).usage
        assert first.cache_write_tokens > 0 and first.cache_read_tokens == 0
        assert second.cache_read_tokens == first.cache_write_tokens
        assert second.cache_write_tokens == 0
        assert first.input_tokens == second.input_tokens  # the uncached part is the same

    async def test_scripted_usage_is_used_as_given(self) -> None:
        given = Usage(input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4)
        stub = _stub(scripted([StubTurn(text="x", usage=given)]))
        assert (await stub.complete(_request())).usage == given


class TestScripts:
    async def test_scripted_turns_play_in_order_then_run_out(self) -> None:
        stub = _stub(
            scripted(
                [
                    StubTurn(
                        tool_calls=(StubToolCall(name="lookup", input={"host": "example.com"}),)
                    ),
                    StubTurn(text="Done."),
                ]
            )
        )
        first = await stub.complete(_request())
        assert first.stop_reason is StopReason.TOOL_USE
        assert first.tool_uses[0].name == "lookup"
        assert first.tool_uses[0].id == "toolu_stub_0_0"
        assert (await stub.complete(_request())).text == "Done."
        with pytest.raises(AssertionError, match="exhausted"):
            await stub.complete(_request())

    async def test_tool_calls_stream_as_started_and_finished(self) -> None:
        stub = _stub(scripted([StubTurn(tool_calls=(StubToolCall(name="lookup", input={}),))]))
        events = [e async for e in stub.stream(_request())]
        kinds = [type(e) for e in events]
        assert kinds.index(ToolCallStarted) < kinds.index(ToolCallFinished)

    async def test_a_scripted_failure_raises_a_provider_error(self) -> None:
        stub = _stub(scripted([ProviderError(ProviderFailure.TIMEOUT)]))
        with pytest.raises(ProviderError) as raised:
            await stub.complete(_request())
        assert raised.value.failure is ProviderFailure.TIMEOUT

    async def test_a_reactive_script_answers_tool_results(self) -> None:
        script = answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="lookup", input={"host": "example.org"}),)),
            lambda results: StubTurn(text=f"The host is {results[0].content[0].text}."),  # type: ignore[union-attr]
        )
        stub = _stub(script)
        ask = Message.user("Where is example.org?")
        first = await stub.complete(_request(ask))
        call = first.tool_uses[0]
        reply = Message(role="assistant", content=(call,))
        result = Message(
            role="user",
            content=(ToolResultBlock(tool_use_id=call.id, content=(TextBlock(text="192.0.2.7"),)),),
        )
        second = await stub.complete(_request(ask, reply, result))
        assert second.text == "The host is 192.0.2.7."
        assert second.stop_reason is StopReason.END_TURN

    def test_tool_use_blocks_can_seed_a_turn(self) -> None:
        call = StubToolCall(name="lookup", input={})
        assert isinstance(call.block("toolu_1"), ToolUseBlock)


class TestThroughTheRegistry:
    async def test_the_builtin_factories_include_the_stub(self) -> None:
        assert any(f.kind == "stub" for f in builtin_factories())

    async def test_start_builds_a_stub_from_settings(self) -> None:
        sink = MemoryEventSink()
        settings = ConduitSettings.model_validate(
            {
                "providers": {"test": {"kind": "stub", "not_for_production": True}},
                "assignments": {"chat": {"provider": "test", "model": "stub-echo"}},
            }
        )
        runtime = await start(
            settings,
            task_classes=[TaskClass(name="chat", floor=CapabilityFloor(tools=True))],
            factories=builtin_factories(),
            secrets=LiteralSecretResolver({}),
            events=sink,
        )
        binding = runtime.for_task("chat")
        response = await binding.provider.complete(binding.request(messages=(Message.user("hi"),)))
        assert response.text == "hi"
        finished = sink.events[-1]
        assert isinstance(finished, ModelCallFinishedEvent)
        assert finished.usage == response.usage
        await runtime.aclose()

    async def test_named_scripts_can_be_supplied_to_the_factory(self) -> None:
        factory = StubFactory(scripts={"fixed": scripted([StubTurn(text="fixed")])})
        registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[factory])
        provider = await registry.build(
            "p", {"kind": "stub", "script": "fixed", "not_for_production": True}
        )
        assert (await provider.complete(_request())).text == "fixed"
        with pytest.raises(ConfigurationError, match="no stub script"):
            await registry.build(
                "q", {"kind": "stub", "script": "absent", "not_for_production": True}
            )
