"""``OtelEventSink``: an event sink producing OpenTelemetry spans and metrics.

Spans
    One trace per run. ``conduit.run`` spans a run, from ``RunStartedEvent`` to
    ``RunFinishedEvent``; its children are ``conduit.model_call`` (from the registry's
    started and finished events), ``conduit.tool_call`` (from ``ToolCallStartedEvent`` to
    the core ``ToolCallEvent``), ``conduit.skill_load``, ``conduit.compaction`` and
    ``conduit.hand_off``. ``conduit.approval`` and ``conduit.mcp_call`` are children of
    the tool call they belong to. ``conduit.embedding_call`` spans an embedding call. A
    finished event whose start was never seen still gets its span, ending at the event
    and starting its ``duration_ms`` earlier. Output-contract failures, skill use,
    unloads and refusals, and budget exhaustion are span events on the run span. Text
    deltas make no span. A failure sets the span's status to error, described by an
    identifier (the error type or code) only.

Metrics
    ``conduit.model.tokens`` counts all four token kinds apart (``conduit.token.type``
    is ``input``, ``output``, ``cache_read`` or ``cache_write``), for chat and embedding
    calls (``conduit.operation``). ``conduit.model.calls`` counts calls and
    ``conduit.model.cache_hits`` the calls that read from the prompt cache.
    ``conduit.operation.duration`` records the latency of runs, model, embedding, tool
    and MCP calls. ``conduit.errors`` counts failures by operation and error type.
    ``conduit.budget.exhausted`` counts runs stopped by a budget, and
    ``conduit.run.tokens``, ``conduit.run.model_calls`` and ``conduit.run.tool_calls``
    record what each run used of the budgets that limit it. ``conduit.hand_offs`` counts
    hand-offs by outcome and task classes. Metric attributes are low-cardinality
    identifiers only, never a run, call or event id.

Every attribute is metadata (``conduit.integrations.otel.attributes``): content is
never exported, whatever the capture policy let an event carry. The sink never raises
into the emitter: a failure in the OpenTelemetry providers is logged by exception class
only. Spans left open are bounded (``max_open_spans``); the oldest is ended, marked
``conduit.incomplete``, when the bound is reached, and finishing a run ends the run's
spans still open.

The sink uses the tracer and meter providers it is given, or OpenTelemetry's global
ones. conduit configures no exporter and reads no ``OTEL_*`` variable itself: those
configure the product's OpenTelemetry SDK, which owns where telemetry goes. None of them
can make the sink export content, which it never reads.

Spans are ended after the sink's lock is released, so a slow span processor never holds
up another emitter; but ``emit`` still runs on the event loop, and a span processor that
exports synchronously (``SimpleSpanProcessor`` with a network exporter) blocks the loop
while it does. Products should export through ``BatchSpanProcessor`` and a periodic
metric reader, which export on their own threads.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from conduit import __version__
from conduit.core.errors import ConfigurationError
from conduit.core.events import Event
from conduit.core.types import Usage
from conduit.integrations.otel.attributes import (
    AttributeValue,
    identifier,
    metadata_attributes,
    pick,
)

__all__ = ["DEFAULT_MAX_OPEN_SPANS", "METRIC_NAMES", "SPAN_NAMES", "OtelEventSink"]

logger = logging.getLogger("conduit.integrations.otel")

SCOPE = "conduit"
DEFAULT_MAX_OPEN_SPANS = 10_000

RUN = "conduit.run"
MODEL_CALL = "conduit.model_call"
EMBEDDING_CALL = "conduit.embedding_call"
TOOL_CALL = "conduit.tool_call"
MCP_CALL = "conduit.mcp_call"
APPROVAL = "conduit.approval"
SKILL_LOAD = "conduit.skill_load"
COMPACTION = "conduit.compaction"
HAND_OFF = "conduit.hand_off"

SPAN_NAMES: tuple[str, ...] = (
    RUN,
    MODEL_CALL,
    EMBEDDING_CALL,
    TOOL_CALL,
    MCP_CALL,
    APPROVAL,
    SKILL_LOAD,
    COMPACTION,
    HAND_OFF,
)

METRIC_NAMES: tuple[str, ...] = (
    "conduit.model.tokens",
    "conduit.model.calls",
    "conduit.model.cache_hits",
    "conduit.operation.duration",
    "conduit.errors",
    "conduit.budget.exhausted",
    "conduit.run.tokens",
    "conduit.run.model_calls",
    "conduit.run.tool_calls",
    "conduit.hand_offs",
)

#: Run-level events recorded as span events on the run span.
_RUN_SPAN_EVENTS = frozenset(
    {"output_invalid", "skill_used", "skill_unloaded", "skill_refused", "budget_exhausted"}
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_Key = tuple[str, str]


def _ns(when: datetime) -> int:
    """``when`` in nanoseconds since the epoch, exactly (no float rounding)."""
    return ((when - _EPOCH) // timedelta(microseconds=1)) * 1000


class _Open:
    __slots__ = ("run_id", "span")

    def __init__(self, span: Span, run_id: str | None) -> None:
        self.span = span
        self.run_id = run_id


def _key(kind: str, value: object) -> _Key | None:
    return (kind, value) if isinstance(value, str) else None


def _token_counts(usage: Usage) -> dict[str, int]:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_write": usage.cache_write_tokens,
    }


class OtelEventSink:
    """Spans and metrics from conduit events. Pass it to a ``FanOutSink`` or use it alone."""

    def __init__(
        self,
        *,
        tracer_provider: trace.TracerProvider | None = None,
        meter_provider: metrics.MeterProvider | None = None,
        max_open_spans: int = DEFAULT_MAX_OPEN_SPANS,
    ) -> None:
        if max_open_spans < 1:
            raise ConfigurationError("max_open_spans must be positive")
        self._max_open = max_open_spans
        tracers = tracer_provider if tracer_provider is not None else trace.get_tracer_provider()
        meters = meter_provider if meter_provider is not None else metrics.get_meter_provider()
        self._tracer = tracers.get_tracer(SCOPE, __version__)
        meter = meters.get_meter(SCOPE, __version__)
        self._tokens = meter.create_counter(
            "conduit.model.tokens", unit="{token}", description="Tokens, by kind"
        )
        self._calls = meter.create_counter(
            "conduit.model.calls", unit="{call}", description="Model and embedding calls"
        )
        self._cache_hits = meter.create_counter(
            "conduit.model.cache_hits", unit="{call}", description="Calls that read the cache"
        )
        self._duration = meter.create_histogram(
            "conduit.operation.duration", unit="ms", description="Latency, by operation"
        )
        self._errors = meter.create_counter(
            "conduit.errors", unit="{error}", description="Failures, by operation and type"
        )
        self._exhausted = meter.create_counter(
            "conduit.budget.exhausted", unit="{run}", description="Runs stopped by a budget"
        )
        self._run_tokens = meter.create_histogram(
            "conduit.run.tokens", unit="{token}", description="Tokens used by a run"
        )
        self._run_model_calls = meter.create_histogram(
            "conduit.run.model_calls", unit="{call}", description="Model calls made by a run"
        )
        self._run_tool_calls = meter.create_histogram(
            "conduit.run.tool_calls", unit="{call}", description="Tool calls made by a run"
        )
        self._hand_offs = meter.create_counter(
            "conduit.hand_offs", unit="{hand_off}", description="Hand-offs, by outcome"
        )
        self._open: OrderedDict[_Key, _Open] = OrderedDict()
        self._lock = threading.Lock()
        #: Spans to end once the lock is released: ending one runs the span processors.
        self._ending: list[tuple[Span, int | None]] = []
        self._handlers: Mapping[str, Callable[[Event, dict[str, AttributeValue]], None]] = {
            "run_started": self._run_started,
            "run_finished": self._run_finished,
            "model_call_started": self._model_call_started,
            "model_call_finished": self._model_call_finished,
            "embedding_call_started": self._embedding_call_started,
            "embedding_call_finished": self._embedding_call_finished,
            "tool_call_started": self._tool_call_started,
            "tool_call": self._tool_call_finished,
            "mcp_call": self._mcp_call,
            "approval_requested": self._approval_requested,
            "approval_decided": self._approval_decided,
            "skill_loaded": self._skill_loaded,
            "conversation_compacted": self._compacted,
            "hand_off": self._hand_off,
        }

    def __repr__(self) -> str:
        return f"OtelEventSink(open_spans={self.open_spans})"

    @property
    def open_spans(self) -> int:
        """Spans started and not yet ended."""
        return len(self._open)

    async def emit(self, event: Event) -> None:
        failed: str | None = None
        ending: list[tuple[Span, int | None]] = []
        ended_failed: str | None = None
        try:
            try:
                with self._lock:
                    try:
                        self._handle(event)
                    finally:
                        ending, self._ending = self._ending, []
            except Exception as error:
                failed = type(error).__name__
        finally:
            # Spans the handler already finished are ended even when it failed or was
            # cancelled part-way: they are no longer tracked as open.
            ended_failed = self._end_all(ending)
        failed = failed or ended_failed
        if failed is not None:
            logger.warning("OpenTelemetry sink failed on %s: %s", event.type, failed)

    def close(self) -> None:
        """End every span still open, marked incomplete."""
        with self._lock:
            while self._open:
                _, entry = self._open.popitem(last=False)
                self._end_incomplete(entry.span, None)
            ending, self._ending = self._ending, []
        failed = self._end_all(ending)
        if failed is not None:
            logger.warning("OpenTelemetry sink failed on close: %s", failed)

    @staticmethod
    def _end_all(ending: list[tuple[Span, int | None]]) -> str | None:
        """End each span, outside the lock; the first failure's class name, if any."""
        failed: str | None = None
        for span, end in ending:
            try:
                span.end(end_time=end)
            except Exception as error:
                failed = failed or type(error).__name__
        return failed

    # Dispatch.

    def _handle(self, event: Event) -> None:
        attributes = metadata_attributes(event)
        handler = self._handlers.get(event.type)
        if handler is not None:
            handler(event, attributes)
        elif event.type in _RUN_SPAN_EVENTS:
            self._run_span_event(event, attributes)

    # Spans.

    def _context(self, *keys: _Key | None) -> Any:
        for key in keys:
            entry = self._open.get(key) if key is not None else None
            if entry is not None:
                return trace.set_span_in_context(entry.span)
        return None

    def _start(
        self,
        name: str,
        event: Event,
        attributes: dict[str, AttributeValue],
        parents: tuple[_Key | None, ...],
        start_ns: int,
        kind: SpanKind,
    ) -> Span:
        return self._tracer.start_span(
            name,
            context=self._context(*parents),
            kind=kind,
            attributes=attributes,
            start_time=start_ns,
        )

    def _open_span(
        self,
        key: _Key | None,
        name: str,
        event: Event,
        attributes: dict[str, AttributeValue],
        parents: tuple[_Key | None, ...],
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> None:
        if key is None:
            return
        span = self._start(name, event, attributes, parents, _ns(event.occurred_at), kind)
        previous = self._open.pop(key, None)
        if previous is not None:
            self._end_incomplete(previous.span, None)
        self._open[key] = _Open(span, event.run_id)
        while len(self._open) > self._max_open:
            _, oldest = self._open.popitem(last=False)
            self._end_incomplete(oldest.span, None)

    def _close_span(
        self,
        key: _Key | None,
        name: str,
        event: Event,
        attributes: dict[str, AttributeValue],
        parents: tuple[_Key | None, ...],
        duration_ms: float,
        failure: object | None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> None:
        end = _ns(event.occurred_at)
        entry = self._open.pop(key, None) if key is not None else None
        if entry is None:
            start = end - int(duration_ms * 1_000_000)
            span = self._start(name, event, attributes, parents, start, kind)
        else:
            span = entry.span
            span.set_attributes(attributes)
        if failure is not None:
            self._fail(span, failure)
        self._ending.append((span, end))

    def _one_shot(
        self,
        name: str,
        event: Event,
        attributes: dict[str, AttributeValue],
        parents: tuple[_Key | None, ...],
        duration_ms: float = 0.0,
        failure: object | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> None:
        self._close_span(None, name, event, attributes, parents, duration_ms, failure, kind)

    @staticmethod
    def _fail(span: Span, why: object) -> None:
        span.set_status(Status(StatusCode.ERROR, identifier(why)))

    def _end_incomplete(self, span: Span, end_ns: int | None) -> None:
        span.set_attribute("conduit.incomplete", True)
        span.set_status(Status(StatusCode.ERROR, "incomplete"))
        self._ending.append((span, end_ns))

    def _run_span_event(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        key = _key("run", event.run_id)
        entry = self._open.get(key) if key is not None else None
        if entry is not None:
            entry.span.add_event(
                f"conduit.{event.type}", attributes=attributes, timestamp=_ns(event.occurred_at)
            )
        if event.type == "budget_exhausted":
            self._exhausted.add(1, pick(attributes, "budget", "task_class"))
        elif event.type == "skill_refused":
            self._error("skill", getattr(event, "reason", None))

    # Metrics.

    def _error(self, operation: str, why: object) -> None:
        self._errors.add(
            1,
            {
                "conduit.operation": operation,
                "conduit.error.type": identifier(why) or "unknown",
            },
        )

    def _latency(self, operation: str, event: Event, attributes: dict[str, AttributeValue]) -> None:
        duration = getattr(event, "duration_ms", None)
        if isinstance(duration, int | float):
            self._duration.record(
                duration, {"conduit.operation": operation, **pick(attributes, "outcome")}
            )

    def _usage(self, operation: str, usage: object, attributes: dict[str, AttributeValue]) -> None:
        if not isinstance(usage, Usage):
            return
        common = {
            "conduit.operation": operation,
            **pick(attributes, "provider", "kind", "model", "task_class"),
        }
        for kind, count in _token_counts(usage).items():
            if count:
                self._tokens.add(count, {**common, "conduit.token.type": kind})

    # Handlers.

    def _run_started(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._open_span(_key("run", event.run_id), RUN, event, attributes, ())

    def _run_finished(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        end = _ns(event.occurred_at)
        run_key = _key("run", event.run_id)
        if event.run_id is not None:
            for key in [k for k, v in self._open.items() if v.run_id == event.run_id]:
                if key != run_key:
                    self._end_incomplete(self._open.pop(key).span, end)
        failed = getattr(event, "outcome", None) == "failed"
        why = getattr(event, "error_type", None) or getattr(event, "failure", None) or "failed"
        self._close_span(
            run_key,
            RUN,
            event,
            attributes,
            (),
            float(getattr(event, "duration_ms", 0.0)),
            why if failed else None,
        )
        self._latency("run", event, attributes)
        if failed:
            self._error("run", why)
        run_attributes = pick(attributes, "task_class", "outcome")
        usage = getattr(event, "usage", None)
        if isinstance(usage, Usage):
            self._run_tokens.record(usage.total_tokens, run_attributes)
        self._run_model_calls.record(int(getattr(event, "model_calls", 0)), run_attributes)
        self._run_tool_calls.record(int(getattr(event, "tool_calls", 0)), run_attributes)

    def _call_started(
        self,
        kind: str,
        name: str,
        operation: str,
        event: Event,
        attributes: dict[str, AttributeValue],
    ) -> None:
        gen_ai: dict[str, AttributeValue] = {"gen_ai.operation.name": operation}
        if "conduit.model" in attributes:
            gen_ai["gen_ai.request.model"] = attributes["conduit.model"]
        self._open_span(
            _key(kind, getattr(event, "call_id", None)),
            name,
            event,
            {**attributes, **gen_ai},
            (_key("run", event.run_id),),
            SpanKind.CLIENT,
        )

    def _call_finished(
        self,
        kind: str,
        name: str,
        operation: str,
        latency: str,
        event: Event,
        attributes: dict[str, AttributeValue],
    ) -> None:
        usage = getattr(event, "usage", None)
        extra: dict[str, AttributeValue] = {"gen_ai.operation.name": operation}
        if "conduit.model" in attributes:
            extra["gen_ai.request.model"] = attributes["conduit.model"]
        if isinstance(usage, Usage):
            extra["gen_ai.usage.input_tokens"] = usage.total_input_tokens
            extra["gen_ai.usage.output_tokens"] = usage.output_tokens
        failed = getattr(event, "outcome", None) == "failed"
        why = getattr(event, "error_type", None) or getattr(event, "failure", None) or "failed"
        self._close_span(
            _key(kind, getattr(event, "call_id", None)),
            name,
            event,
            {**attributes, **extra},
            (_key("run", event.run_id),),
            float(getattr(event, "duration_ms", 0.0)),
            why if failed else None,
            SpanKind.CLIENT,
        )
        self._latency(latency, event, attributes)
        call_attributes = {
            "conduit.operation": operation,
            **pick(attributes, "provider", "kind", "model", "task_class", "outcome"),
        }
        self._calls.add(1, call_attributes)
        if isinstance(usage, Usage) and usage.cache_read_tokens > 0:
            self._cache_hits.add(1, call_attributes)
        self._usage(operation, usage, attributes)
        if failed:
            self._error(latency, why)

    def _model_call_started(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._call_started("model", MODEL_CALL, "chat", event, attributes)

    def _model_call_finished(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._call_finished("model", MODEL_CALL, "chat", "model_call", event, attributes)

    def _embedding_call_started(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._call_started("embedding", EMBEDDING_CALL, "embedding", event, attributes)

    def _embedding_call_finished(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._call_finished(
            "embedding", EMBEDDING_CALL, "embedding", "embedding_call", event, attributes
        )

    def _tool_call_started(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._open_span(
            _key("tool", getattr(event, "call_id", None)),
            TOOL_CALL,
            event,
            attributes,
            (_key("run", event.run_id),),
        )

    def _tool_call_finished(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        failed = getattr(event, "outcome", None) == "failed"
        why = getattr(event, "error_code", None) or "failed"
        self._close_span(
            _key("tool", getattr(event, "call_id", None)),
            TOOL_CALL,
            event,
            attributes,
            (_key("run", event.run_id),),
            float(getattr(event, "duration_ms", 0.0)),
            why if failed else None,
        )
        self._latency("tool_call", event, attributes)
        if failed:
            self._error("tool_call", why)

    def _mcp_call(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        failed = getattr(event, "outcome", None) == "failed"
        why = getattr(event, "error_code", None) or "failed"
        self._one_shot(
            MCP_CALL,
            event,
            attributes,
            (_key("tool", getattr(event, "call_id", None)), _key("run", event.run_id)),
            float(getattr(event, "duration_ms", 0.0)),
            why if failed else None,
            SpanKind.CLIENT,
        )
        self._latency("mcp_call", event, attributes)
        if failed:
            self._error("mcp_call", why)

    def _approval_requested(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._open_span(
            _key("approval", getattr(event, "request_id", None)),
            APPROVAL,
            event,
            attributes,
            (_key("tool", getattr(event, "call_id", None)), _key("run", event.run_id)),
        )

    def _approval_decided(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        failed = getattr(event, "outcome", None) == "failed"
        self._close_span(
            _key("approval", getattr(event, "request_id", None)),
            APPROVAL,
            event,
            attributes,
            (_key("tool", getattr(event, "call_id", None)), _key("run", event.run_id)),
            0.0,
            "failed" if failed else None,
        )
        if failed:
            self._error("approval", "failed")

    def _skill_loaded(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._one_shot(SKILL_LOAD, event, attributes, (_key("run", event.run_id),))

    def _compacted(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._one_shot(COMPACTION, event, attributes, (_key("run", event.run_id),))

    def _hand_off(self, event: Event, attributes: dict[str, AttributeValue]) -> None:
        self._one_shot(
            HAND_OFF,
            event,
            attributes,
            (_key("tool", getattr(event, "call_id", None)), _key("run", event.run_id)),
        )
        self._hand_offs.add(1, pick(attributes, "outcome", "from_task_class", "to_task_class"))
