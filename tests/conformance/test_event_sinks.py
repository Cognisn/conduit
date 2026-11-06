"""The event sink conformance suite, against conduit's sinks, and the suite's own checks:
it must fail a sink that breaks the contract."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from conduit.core.events import Event, EventSink, FanOutSink, MemoryEventSink
from conduit.testing.conformance import EventSinkSuite


def _memory_records(sink: MemoryEventSink) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json") for event in sink.events]


class TestMemoryEventSink(EventSinkSuite):
    async def make_sink(self) -> EventSink:
        return MemoryEventSink()

    async def recorded(self, sink: EventSink) -> Sequence[Mapping[str, Any]]:
        assert isinstance(sink, MemoryEventSink)
        return _memory_records(sink)


class _FanOutOverMemory(FanOutSink):
    def __init__(self, *sinks: EventSink) -> None:
        self.memory = MemoryEventSink()
        super().__init__([*sinks, self.memory])


class TestFanOutSink(EventSinkSuite):
    async def make_sink(self) -> EventSink:
        return _FanOutOverMemory()

    async def recorded(self, sink: EventSink) -> Sequence[Mapping[str, Any]]:
        assert isinstance(sink, _FanOutOverMemory)
        return _memory_records(sink.memory)


class _AlwaysFails:
    async def emit(self, event: Event) -> None:
        raise RuntimeError("this sink always fails")


class TestFanOutSinkBesideAFailingSink(EventSinkSuite):
    """A sink that raises never stops the fan-out delivering to the others."""

    async def make_sink(self) -> EventSink:
        return _FanOutOverMemory(_AlwaysFails())

    async def recorded(self, sink: EventSink) -> Sequence[Mapping[str, Any]]:
        assert isinstance(sink, _FanOutOverMemory)
        return _memory_records(sink.memory)


class TestOtelEventSink(EventSinkSuite):
    """Spans and metrics in memory (the otel extra and the SDK, a dev dependency)."""

    keeps_records = False  # spans and metrics, not a record per event

    @pytest.fixture(autouse=True)
    def _telemetry(self) -> Iterator[None]:
        pytest.importorskip("opentelemetry.trace")
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(metric_readers=[self.reader])
        yield
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()

    async def make_sink(self) -> EventSink:
        from conduit.integrations.otel import OtelEventSink

        self.exporter.clear()
        return OtelEventSink(
            tracer_provider=self.tracer_provider, meter_provider=self.meter_provider
        )

    async def close_sink(self, sink: EventSink) -> None:
        from conduit.integrations.otel import OtelEventSink

        assert isinstance(sink, OtelEventSink)
        sink.close()

    async def recorded_text(self, sink: EventSink) -> str:
        from conduit.integrations.otel import OtelEventSink

        assert isinstance(sink, OtelEventSink)
        sink.close()  # end open spans, so every span is exported
        spans = [span.to_json() for span in self.exporter.get_finished_spans()]
        metrics = self.reader.get_metrics_data()
        return "\n".join([*spans, "" if metrics is None else metrics.to_json()])


class _Recorder:
    """warden's recorder interface, recording to a list."""

    def __init__(self) -> None:
        from warden.audit import ActionRegistry

        from conduit.integrations.warden import register_conduit_actions

        self._registry = ActionRegistry()
        register_conduit_actions(self._registry)
        self.events: list[Any] = []

    @property
    def registry(self) -> Any:
        return self._registry

    async def record(self, event: Any) -> int:
        self._registry.get(event.action)
        self.events.append(event)
        return len(self.events)


class TestWardenAuditSink(EventSinkSuite):
    """The audit trail records conduit's audited actions only (the warden extra)."""

    @pytest.fixture(autouse=True)
    def _warden(self) -> None:
        pytest.importorskip("warden.audit")

    async def make_sink(self) -> EventSink:
        from conduit.integrations.warden import WardenAuditSink

        self.recorder = _Recorder()
        return WardenAuditSink(self.recorder)

    def records(self, event: Event) -> bool:
        from conduit.integrations.warden import CONDUIT_ACTIONS

        return event.type in CONDUIT_ACTIONS

    async def recorded(self, sink: EventSink) -> Sequence[Mapping[str, Any]]:
        return [
            {**record.details, "action": record.action, "outcome": record.outcome}
            for record in self.recorder.events
        ]


# --- The suite's own checks: each broken sink fails the scenario it breaks. --------------


class _Kept(MemoryEventSink):
    pass


class _DropsEveryThird(_Kept):
    async def emit(self, event: Event) -> None:
        self.seen = getattr(self, "seen", 0) + 1
        if self.seen % 3:
            await super().emit(event)


class _Reorders(_Kept):
    """Holds each event back until the next arrives, then records them newest first."""

    async def emit(self, event: Event) -> None:
        held = getattr(self, "held", None)
        if held is None:
            self.held: Event | None = event
            return
        self.events += [event, held]
        self.held = None


class _RefusesUnknownTypes(_Kept):
    async def emit(self, event: Event) -> None:
        if type(event) is Event:
            raise ValueError(f"unknown event type {event.type}")
        await super().emit(event)


class _LosesConcurrentEmits(_Kept):
    """Reads, yields, then writes: concurrent emits overwrite one another."""

    async def emit(self, event: Event) -> None:
        current = list(self.events)
        await asyncio.sleep(0)
        self.events = [*current, event]


class _DuplicatesEvents(_Kept):
    async def emit(self, event: Event) -> None:
        await super().emit(event)
        await super().emit(event)


class _NeverReturns(_Kept):
    async def emit(self, event: Event) -> None:
        await asyncio.Event().wait()


def _suite(sink: type[_Kept], *, slow: bool = False) -> EventSinkSuite:
    class Suite(EventSinkSuite):
        emit_deadline_s = 0.2 if slow else EventSinkSuite.emit_deadline_s

        async def make_sink(self) -> EventSink:
            return sink()

        async def recorded(self, made: EventSink) -> Sequence[Mapping[str, Any]]:
            assert isinstance(made, _Kept)
            return _memory_records(made)

    return Suite()


@pytest.mark.parametrize(
    ("broken", "scenario"),
    [
        (_DropsEveryThird, "test_every_event_is_recorded_in_order"),
        (_Reorders, "test_every_event_is_recorded_in_order"),
        (_DuplicatesEvents, "test_every_event_is_recorded_in_order"),
        (_RefusesUnknownTypes, "test_an_unknown_event_type_is_accepted"),
        (_LosesConcurrentEmits, "test_concurrent_emits_are_each_recorded_once_in_order"),
        (_DuplicatesEvents, "test_concurrent_emits_are_each_recorded_once_in_order"),
    ],
)
def test_the_suite_fails_a_broken_sink(broken: type[_Kept], scenario: str) -> None:
    with pytest.raises(AssertionError):
        getattr(_suite(broken), scenario)()


def test_the_suite_fails_a_sink_that_never_returns() -> None:
    with pytest.raises(AssertionError, match="did not return"):
        _suite(_NeverReturns, slow=True).test_an_emit_returns_promptly()


def test_the_suite_fails_a_sink_whose_record_holds_withheld_content() -> None:
    class Suite(EventSinkSuite):
        async def make_sink(self) -> EventSink:
            return MemoryEventSink()

        async def recorded(self, made: EventSink) -> Sequence[Mapping[str, Any]]:
            return []

        async def recorded_text(self, made: EventSink) -> str:
            # A record that kept the content: what an unenforced path would deliver.
            return json.dumps(self.withheld_markers())

    with pytest.raises(AssertionError, match="withheld"):
        Suite().test_withheld_content_never_reaches_the_sink()


def test_a_sink_that_opts_out_of_records_skips_the_order_scenarios_with_a_reason() -> None:
    class Suite(EventSinkSuite):
        keeps_records = False

        async def make_sink(self) -> EventSink:
            return MemoryEventSink()

        async def recorded_text(self, made: EventSink) -> str:
            return ""

    for scenario in (
        Suite().test_every_event_is_recorded_in_order,
        Suite().test_concurrent_emits_are_each_recorded_once_in_order,
    ):
        with pytest.raises(unittest.SkipTest, match="keeps no record per event"):
            scenario()


def test_a_sink_that_keeps_no_record_fails_unless_it_opts_out() -> None:
    class Suite(EventSinkSuite):
        async def make_sink(self) -> EventSink:
            return MemoryEventSink()

        async def recorded(self, made: EventSink) -> Sequence[Mapping[str, Any]]:
            return None  # type: ignore[return-value]  # what a no-op sink's suite returns

    with pytest.raises(AssertionError, match="keeps_records = False"):
        Suite().test_every_event_is_recorded_in_order()


def test_a_sink_that_opts_out_must_say_what_it_kept() -> None:
    class Suite(EventSinkSuite):
        keeps_records = False

        async def make_sink(self) -> EventSink:
            return MemoryEventSink()

    with pytest.raises(NotImplementedError, match="recorded_text"):
        Suite().test_withheld_content_never_reaches_the_sink()


class _ReordersWhenEmitsOverlap(_Kept):
    """Records in order when emits come one at a time, newest first when they overlap."""

    async def emit(self, event: Event) -> None:
        self.inflight = getattr(self, "inflight", 0) + 1
        await asyncio.sleep(0)
        if self.inflight > 1:
            self.events.insert(0, event)
        else:
            self.events.append(event)
        self.inflight -= 1


def test_the_suite_fails_a_sink_that_reorders_only_under_concurrency() -> None:
    suite = _suite(_ReordersWhenEmitsOverlap)
    suite.test_every_event_is_recorded_in_order()  # sound one emit at a time
    with pytest.raises(AssertionError, match="out of order"):
        suite.test_concurrent_emits_are_each_recorded_once_in_order()


def test_a_records_filter_that_keeps_nothing_is_refused() -> None:
    class Suite(EventSinkSuite):
        async def make_sink(self) -> EventSink:
            return MemoryEventSink()

        async def recorded(self, made: EventSink) -> Sequence[Mapping[str, Any]]:
            return []

        def records(self, event: Event) -> bool:
            return False

    for scenario in (
        Suite().test_every_event_is_recorded_in_order,
        Suite().test_concurrent_emits_are_each_recorded_once_in_order,
    ):
        with pytest.raises(AssertionError, match="keeps none of the suite's events"):
            scenario()


@pytest.mark.parametrize(
    "scenario",
    [name for name in dir(EventSinkSuite) if name.startswith("test_")],
)
def test_every_scenario_passes_a_sound_sink(scenario: str) -> None:
    getattr(_suite(_Kept), scenario)()


def test_an_unconfigured_suite_says_what_to_override() -> None:
    with pytest.raises(NotImplementedError, match="make_sink"):
        EventSinkSuite().test_an_unknown_event_type_is_accepted()
