"""The event sink conformance suite.

Each scenario states one thing conduit relies on of an ``EventSink``. conduit awaits a
sink on the call's path, from concurrent tasks on one event loop, and adds event types
as its layers grow, so a sink must record every event it keeps once and in the order it
was emitted (per task, when tasks emit at once), accept an event type it does not know,
and return promptly. conduit delivers events to product sinks through a ``FanOutSink``
under the deployment's capture policy, so the content that policy withholds never
reaches a sink's record.

A sink is write-only, so the suite reads back what it kept through ``recorded`` (each
event it recorded, as a mapping holding at least ``event_id``) and ``recorded_text``
(everything it kept or sent, as text). A sink that keeps no record per event (an
exporter of aggregates) opts out with ``keeps_records = False`` and overrides
``recorded_text``; its two order scenarios are then skipped with that reason, and the
others still run. A suite that has not opted out and reads back nothing fails.
"""

# Suites are test code shipped for products to run under pytest: assertions are the point.
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, ClassVar, cast

from conduit.agent.events import RunFinishedEvent, RunStartedEvent
from conduit.core.events import (
    CapturePolicy,
    ContentKind,
    Event,
    EventSink,
    FanOutSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
    ToolCallEvent,
)
from conduit.core.types import StopReason, Usage
from conduit.testing.conformance.base import ConformanceSuite

#: The type of the event the suite sends as one no sink can know.
UNKNOWN_EVENT_TYPE = "conformance_probe_unknown"

_EVERYTHING = CapturePolicy(prompts=True, tool_content=True)
_NOTHING = CapturePolicy()


def _marker(run: int, field: str) -> str:
    return f"withheld-{field}-text-run-{run:03d}-zq"


def run_events(run: int, *, content: bool = False) -> list[Event]:
    """One synthetic run's events, in the order conduit emits them: a run start, a model
    call, a tool call, an event of a type no sink knows, and a run finish.

    With ``content``, every content field carries its text (marked so a leak is easy to
    find); otherwise each carries only its hash and size, as under the default policy.
    """
    policy = _EVERYTHING if content else _NOTHING
    run_id = f"run_suite_{run:03d}"
    common: dict[str, Any] = {"run_id": run_id, "task_class": "conformance"}
    usage = Usage(input_tokens=12, output_tokens=5, cache_read_tokens=3, cache_write_tokens=0)
    return [
        RunStartedEvent(
            **common,
            provider="stub",
            kind="stub",
            model="stub-echo",
            message_count=1,
            tool_count=1,
            hand_off_count=0,
        ),
        ModelCallStartedEvent(
            **common,
            call_id=f"call_{run:03d}",
            provider="stub",
            kind="stub",
            model="stub-echo",
            request=policy.capture(ContentKind.PROMPT, _marker(run, "request")),
            message_count=1,
            tool_count=1,
            max_output_tokens=256,
        ),
        ModelCallFinishedEvent(
            **common,
            call_id=f"call_{run:03d}",
            provider="stub",
            kind="stub",
            model="stub-echo",
            outcome="ok",
            stop_reason=StopReason.TOOL_USE,
            usage=usage,
            duration_ms=4.0,
            response=policy.capture(ContentKind.PROMPT, _marker(run, "response")),
            tool_call_count=1,
        ),
        ToolCallEvent(
            **common,
            call_id=f"tool_{run:03d}",
            tool="lookup_host",
            outcome="ok",
            arguments=policy.capture(ContentKind.TOOL, _marker(run, "arguments")),
            result=policy.capture(ContentKind.TOOL, _marker(run, "result")),
            duration_ms=2.0,
        ),
        Event(type=UNKNOWN_EVENT_TYPE, **common),
        RunFinishedEvent(
            **common,
            outcome="completed",
            usage=usage,
            model_calls=1,
            tool_calls=1,
            hand_offs=0,
            duration_ms=9.0,
            output=policy.capture(ContentKind.PROMPT, _marker(run, "output")),
        ),
    ]


class EventSinkSuite(ConformanceSuite):
    """Order, unknown types, concurrency, withheld content and prompt return."""

    interface: ClassVar[str] = "event_sink"
    #: Tasks emitting at once in the concurrency scenario.
    concurrent_runs: ClassVar[int] = 8
    #: How long one emit may take before the suite calls it stuck.
    emit_deadline_s: ClassVar[float] = 5.0
    #: Whether the sink keeps a record per event that ``recorded`` reads back. Set it
    #: false for a sink that does not (an exporter of aggregates), and override
    #: ``recorded_text``: the order scenarios are then skipped, saying why.
    keeps_records: ClassVar[bool] = True

    async def make_sink(self) -> EventSink:
        """A new sink with an empty record, for one scenario. Products override this."""
        raise NotImplementedError("override make_sink to return your event sink")

    async def recorded(self, sink: EventSink) -> Sequence[Mapping[str, Any]]:
        """Each event ``sink`` has recorded, in the order it recorded them, as a mapping
        holding at least the event's ``event_id`` (``event.model_dump(mode="json")`` is
        one). Products override this, unless they set ``keeps_records`` false."""
        raise NotImplementedError("override recorded to read back what your sink kept")

    async def recorded_text(self, sink: EventSink) -> str:
        """Everything ``sink`` has kept or sent, as text, for the content scenario. The
        default is ``recorded`` as JSON; a sink that keeps no record per event (with
        ``keeps_records`` false) must override it."""
        if not type(self).keeps_records:
            raise NotImplementedError(
                "override recorded_text: keeps_records is false, so the suite cannot read "
                "what your sink kept through recorded"
            )
        return json.dumps(list(await self._records(sink)), default=str, sort_keys=True)

    def records(self, event: Event) -> bool:
        """Whether the sink records ``event``. Override for a sink that keeps only some
        events (an audit trail of chosen actions); the default is every event."""
        return True

    async def close_sink(self, sink: EventSink) -> None:
        """Release a sink after a scenario. Override if it needs closing."""
        closer = getattr(sink, "aclose", None)
        if closer is not None:
            await closer()

    def withheld_markers(self) -> list[str]:
        """The texts the content scenario sends and the capture policy withholds."""
        return [
            text
            for run in range(2)
            for event in run_events(run, content=True)
            for text in _texts(event)
        ]

    def _scenario(self, body: Callable[[EventSink], Awaitable[None]]) -> None:
        async def scenario() -> None:
            sink = await self.make_sink()
            try:
                await body(sink)
            finally:
                await self.close_sink(sink)

        self.run(scenario)

    async def _records(self, sink: EventSink) -> Sequence[Mapping[str, Any]]:
        records: Any = await self.recorded(sink)
        assert records is not None, (
            "recorded returned None: read back what the sink kept, or, for a sink that "
            "keeps no record per event, set keeps_records = False and override recorded_text"
        )
        return cast(Sequence[Mapping[str, Any]], records)

    async def _ids(self, sink: EventSink) -> list[str]:
        if not type(self).keeps_records:
            raise unittest.SkipTest(
                "the sink keeps no record per event (keeps_records is false), so the order "
                "it records events in cannot be read back"
            )
        ids: list[str] = []
        for record in await self._records(sink):
            event_id = record.get("event_id")
            assert isinstance(event_id, str), "a recorded event has no event_id"
            ids.append(event_id)
        return ids

    def _expected(self, events: Sequence[Event]) -> list[str]:
        expected = [event.event_id for event in events if self.records(event)]
        assert expected, "records() keeps none of the suite's events, so nothing is checked"
        return expected

    def test_every_event_is_recorded_in_order(self) -> None:
        events = run_events(0) + run_events(1)

        async def body(sink: EventSink) -> None:
            for event in events:
                await sink.emit(event)
            assert await self._ids(sink) == self._expected(events)

        self._scenario(body)

    def test_an_unknown_event_type_is_accepted(self) -> None:
        unknown = Event(type=UNKNOWN_EVENT_TYPE, run_id="run_suite_unknown")
        after = run_events(2)

        async def body(sink: EventSink) -> None:
            raised: BaseException | None = None
            try:
                await sink.emit(unknown)
            except Exception as error:  # the suite judges whatever was raised
                raised = error
            assert raised is None, (
                f"an event of a type the sink does not know raised {type(raised).__name__}"
            )
            for event in after:  # and the sink still works afterwards
                await sink.emit(event)

        self._scenario(body)

    def test_concurrent_emits_are_each_recorded_once_in_order(self) -> None:
        runs = [run_events(n) for n in range(self.concurrent_runs)]

        async def emit_run(sink: EventSink, events: Sequence[Event]) -> None:
            for event in events:
                await sink.emit(event)
                await asyncio.sleep(0)

        async def body(sink: EventSink) -> None:
            await asyncio.gather(*(emit_run(sink, events) for events in runs))
            ids = await self._ids(sink)
            expected = [self._expected(events) for events in runs]
            assert sorted(ids) == sorted(i for run in expected for i in run), (
                "the record is not every event exactly once"
            )
            for run, wanted in enumerate(expected):
                kept = [i for i in ids if i in set(wanted)]
                assert kept == wanted, f"run {run}'s events were recorded out of order"

        self._scenario(body)

    def test_withheld_content_never_reaches_the_sink(self) -> None:
        events = run_events(0, content=True) + run_events(1, content=True)

        async def body(sink: EventSink) -> None:
            # As conduit delivers to a product's sink: through a fan-out under the
            # deployment's capture policy, here the default, which withholds all content.
            delivering = FanOutSink([sink], policy=CapturePolicy())
            for event in events:
                await delivering.emit(event)
            text = await self.recorded_text(sink)
            leaked = [marker for marker in self.withheld_markers() if marker in text]
            assert not leaked, f"the sink's record holds withheld content: {leaked}"

        self._scenario(body)

    def test_an_emit_returns_promptly(self) -> None:
        events = run_events(3)
        deadline = self.emit_deadline_s

        async def body(sink: EventSink) -> None:
            for event in events:
                returned = True
                try:
                    async with asyncio.timeout(deadline):
                        await sink.emit(event)
                except TimeoutError:
                    returned = False
                assert returned, f"emit did not return within {deadline} s for {event.type}"

        self._scenario(body)


def _texts(event: Event) -> list[str]:
    """Every content text an event carries."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("kind") in {k.value for k in ContentKind} and "sha256" in value:
                if isinstance(value.get("text"), str):
                    found.append(value["text"])
                return
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(event.model_dump(mode="json"))
    return found
