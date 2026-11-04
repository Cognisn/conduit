"""Events, the content-capture policy and fan-out to several sinks."""

from __future__ import annotations

import asyncio
import hashlib
import logging

import pytest
from pydantic import ValidationError

from conduit.core.events import (
    CapturedContent,
    CapturePolicy,
    ContentKind,
    EventSink,
    FanOutSink,
    MemoryEventSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
    ToolCallEvent,
)
from conduit.core.types import StopReason, Usage

SECRET_PROMPT = "Summarise the incident at Example Pty Ltd"


def _started(policy: CapturePolicy, text: str = SECRET_PROMPT) -> ModelCallStartedEvent:
    return ModelCallStartedEvent(
        call_id="call_1",
        provider="primary",
        kind="stub",
        model="stub-echo",
        request=policy.capture(ContentKind.PROMPT, text),
        message_count=1,
        tool_count=0,
        max_output_tokens=256,
    )


class TestPolicy:
    def test_by_default_only_hash_and_size_are_kept(self) -> None:
        captured = CapturePolicy().capture(ContentKind.PROMPT, SECRET_PROMPT)
        assert captured.text is None
        assert captured.size == len(SECRET_PROMPT.encode("utf-8"))
        assert captured.sha256 == hashlib.sha256(SECRET_PROMPT.encode("utf-8")).hexdigest()

    def test_opting_in_keeps_each_kind_separately(self) -> None:
        prompts = CapturePolicy(prompts=True)
        assert prompts.capture(ContentKind.PROMPT, "p").text == "p"
        assert prompts.capture(ContentKind.TOOL, "t").text is None
        tools = CapturePolicy(tool_content=True)
        assert tools.capture(ContentKind.TOOL, "t").text == "t"
        assert tools.capture(ContentKind.PROMPT, "p").text is None

    def test_from_a_mapping(self) -> None:
        assert CapturePolicy.model_validate({"prompts": True}).allows(ContentKind.PROMPT)


class TestEventModels:
    def test_identifiers_refuse_prose(self) -> None:
        with pytest.raises(ValidationError):
            ModelCallStartedEvent(
                call_id="call 1 with words",
                provider="p",
                kind="stub",
                model="m",
                request=CapturePolicy().capture(ContentKind.PROMPT, ""),
                message_count=0,
                tool_count=0,
                max_output_tokens=1,
            )

    def test_events_get_an_id_and_a_time(self) -> None:
        event = _started(CapturePolicy())
        assert event.event_id.startswith("evt_")
        assert event.occurred_at.tzinfo is not None
        assert event.type == "model_call_started"

    def test_finished_event_carries_all_four_usage_counts(self) -> None:
        usage = Usage(input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4)
        event = ModelCallFinishedEvent(
            call_id="call_1",
            provider="p",
            kind="stub",
            model="m",
            outcome="ok",
            stop_reason=StopReason.END_TURN,
            usage=usage,
            duration_ms=1.5,
        )
        dumped = event.model_dump()
        assert dumped["usage"] == {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_read_tokens": 3,
            "cache_write_tokens": 4,
        }


class _Failing:
    async def emit(self, event: object) -> None:
        raise RuntimeError(SECRET_PROMPT)


class TestFanOut:
    async def test_every_sink_receives_every_event(self) -> None:
        first, second = MemoryEventSink(), MemoryEventSink()
        bus = FanOutSink([first, second])
        event = _started(CapturePolicy())
        await bus.emit(event)
        assert first.events == [event] and second.events == [event]
        assert isinstance(bus, EventSink)

    async def test_a_failing_sink_does_not_stop_the_others_or_leak_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        good = MemoryEventSink()
        bus = FanOutSink([_Failing(), good])
        with caplog.at_level(logging.WARNING, logger="conduit"):
            await bus.emit(_started(CapturePolicy()))
        assert len(good.events) == 1
        assert "_Failing" in caplog.text and "RuntimeError" in caplog.text
        assert SECRET_PROMPT not in caplog.text

    async def test_sinks_are_called_concurrently(self) -> None:
        started = asyncio.Event()
        order: list[str] = []

        class Waiting:
            async def emit(self, event: object) -> None:
                await asyncio.wait_for(started.wait(), timeout=2)
                order.append("waiting")

        class Releasing:
            async def emit(self, event: object) -> None:
                order.append("releasing")
                started.set()

        await FanOutSink([Waiting(), Releasing()]).emit(_started(CapturePolicy()))
        assert order == ["releasing", "waiting"]


class TestEnforcement:
    async def test_content_the_policy_forbids_never_reaches_a_sink(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = MemoryEventSink()
        bus = FanOutSink([sink], policy=CapturePolicy())
        # An emitter that captured under a looser policy than the bus's.
        leaky = _started(CapturePolicy(prompts=True))
        assert leaky.request.text == SECRET_PROMPT
        with caplog.at_level(logging.WARNING, logger="conduit"):
            await bus.emit(leaky)
        delivered = sink.events[0]
        assert isinstance(delivered, ModelCallStartedEvent)
        assert delivered.request.text is None
        assert delivered.request.sha256 == leaky.request.sha256
        assert SECRET_PROMPT not in delivered.model_dump_json()
        assert "removed" in caplog.text

    async def test_nested_content_is_enforced_too(self) -> None:
        sink = MemoryEventSink()
        bus = FanOutSink([sink], policy=CapturePolicy(prompts=True))
        loose = CapturePolicy(prompts=True, tool_content=True)
        event = ToolCallEvent(
            call_id="toolu_1",
            tool="lookup",
            outcome="ok",
            arguments=loose.capture(ContentKind.TOOL, '{"host": "example.com"}'),
            result=loose.capture(ContentKind.TOOL, "result body"),
            duration_ms=2.0,
        )
        await bus.emit(event)
        delivered = sink.events[0]
        assert isinstance(delivered, ToolCallEvent)
        assert delivered.arguments.text is None
        assert delivered.result is not None and delivered.result.text is None

    async def test_allowed_content_is_kept(self) -> None:
        sink = MemoryEventSink()
        bus = FanOutSink([sink], policy=CapturePolicy(prompts=True))
        await bus.emit(_started(CapturePolicy(prompts=True)))
        delivered = sink.events[0]
        assert isinstance(delivered, ModelCallStartedEvent)
        assert delivered.request.text == SECRET_PROMPT

    def test_captured_content_hash_must_be_hex(self) -> None:
        with pytest.raises(ValidationError):
            CapturedContent(kind=ContentKind.PROMPT, sha256="not hex", size=0)
