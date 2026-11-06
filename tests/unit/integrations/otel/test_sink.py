"""The OpenTelemetry event sink: spans and metrics from conduit events, metadata only.

Events are built by hand here, so each span's parent, timing and attributes can be
checked exactly; ``test_runs`` drives real runs through the sink. Every exporter is in
memory.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode

from conduit.agent.budgets import BudgetKind
from conduit.agent.events import (
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    BudgetExhaustedEvent,
    ConversationCompactedEvent,
    HandOffEvent,
    OutputInvalidEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextDeltaEvent,
    ToolCallStartedEvent,
)
from conduit.agent.results import RunFailure
from conduit.core.events import (
    CapturePolicy,
    ContentKind,
    Event,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
    ToolCallEvent,
)
from conduit.core.registry import EmbeddingCallFinishedEvent, EmbeddingCallStartedEvent
from conduit.core.types import StopReason, Usage
from conduit.integrations.otel import (
    METRIC_NAMES,
    SPAN_NAMES,
    OtelEventSink,
    metadata_attributes,
)
from conduit.mcp.events import McpCallEvent
from conduit.skills.events import (
    SkillLoadedEvent,
    SkillRefusedEvent,
    SkillUnloadedEvent,
    SkillUsedEvent,
)
from tests.unit.integrations.otel.conftest import Telemetry

MARKER = "zz synthetic otel content marker 6181 zz"
CAPTURE_ALL = CapturePolicy(prompts=True, tool_content=True)
T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
#: The only attribute keys a metric data point may carry: low-cardinality identifiers.
METRIC_ATTRIBUTES = frozenset(
    {
        "conduit.operation",
        "conduit.provider",
        "conduit.kind",
        "conduit.model",
        "conduit.task_class",
        "conduit.outcome",
        "conduit.token.type",
        "conduit.error.type",
        "conduit.budget",
        "conduit.from_task_class",
        "conduit.to_task_class",
    }
)
#: Values that would make a metric unbounded: event and call ids, and content hashes.
UNBOUNDED = re.compile(r"^(?:evt|call|tool|run|corr|appr|emb)_[0-9a-f]+$|^[0-9a-f]{64}$")
RUN: dict[str, Any] = {"run_id": "run_1", "correlation_id": "corr_1", "task_class": "chat"}


class RaisingOnEnd(SpanProcessor):
    """A span processor that fails on every span it is given to end."""

    def on_end(self, span: ReadableSpan) -> None:
        raise RuntimeError(f"exporter said {MARKER}")


def at(ms: float) -> datetime:
    return T0 + timedelta(milliseconds=ms)


def ns(when: datetime) -> int:
    return ((when - EPOCH) // timedelta(microseconds=1)) * 1000


def usage(
    input_tokens: int = 10, output: int = 5, cache_read: int = 0, cache_write: int = 0
) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def prompt() -> Any:
    return CAPTURE_ALL.capture(ContentKind.PROMPT, f"Prompt: {MARKER}")


def tool_text() -> Any:
    return CAPTURE_ALL.capture(ContentKind.TOOL, f"Tool: {MARKER}")


def sink(telemetry: Telemetry, **kwargs: Any) -> OtelEventSink:
    return OtelEventSink(
        tracer_provider=telemetry.tracer_provider,
        meter_provider=telemetry.meter_provider,
        **kwargs,
    )


def run_started(ms: float = 0) -> RunStartedEvent:
    return RunStartedEvent(
        **RUN,
        occurred_at=at(ms),
        provider="primary",
        kind="stub",
        model="stub-echo",
        message_count=1,
        tool_count=2,
        hand_off_count=1,
    )


def run_finished(ms: float, *, failed: bool = False) -> RunFinishedEvent:
    return RunFinishedEvent(
        **RUN,
        occurred_at=at(ms),
        outcome="failed" if failed else "completed",
        failure=RunFailure.TOOL_FAILED if failed else None,
        error_type="ToolFailed" if failed else None,
        stop_reason=None if failed else StopReason.END_TURN,
        usage=usage(30, 12, 7, 3),
        model_calls=2,
        tool_calls=1,
        hand_offs=1,
        duration_ms=ms,
        output=prompt(),
    )


def model_started(ms: float, call_id: str = "call_1") -> ModelCallStartedEvent:
    return ModelCallStartedEvent(
        **RUN,
        occurred_at=at(ms),
        call_id=call_id,
        provider="primary",
        kind="stub",
        model="stub-echo",
        request=prompt(),
        message_count=1,
        tool_count=2,
        max_output_tokens=1024,
    )


def model_finished(
    ms: float,
    call_id: str = "call_1",
    *,
    duration: float = 40,
    failed: bool = False,
    used: Usage | None = None,
    run: dict[str, Any] | None = None,
) -> ModelCallFinishedEvent:
    return ModelCallFinishedEvent(
        **(RUN if run is None else run),
        occurred_at=at(ms),
        call_id=call_id,
        provider="primary",
        kind="stub",
        model="stub-echo",
        outcome="failed" if failed else "ok",
        stop_reason=None if failed else StopReason.TOOL_USE,
        usage=used if used is not None else usage(10, 5, 4, 2),
        error_type="ProviderError" if failed else None,
        duration_ms=duration,
        response=prompt(),
        tool_call_count=1,
    )


def tool_started(ms: float, call_id: str = "tool_1") -> ToolCallStartedEvent:
    return ToolCallStartedEvent(
        **RUN, occurred_at=at(ms), call_id=call_id, tool="docs__echo", arguments=tool_text()
    )


def tool_finished(
    ms: float, call_id: str = "tool_1", *, failed: bool = False, duration: float = 30
) -> ToolCallEvent:
    return ToolCallEvent(
        **RUN,
        occurred_at=at(ms),
        call_id=call_id,
        tool="docs__echo",
        outcome="failed" if failed else "ok",
        error_code="tool_error" if failed else None,
        arguments=tool_text(),
        result=tool_text(),
        duration_ms=duration,
    )


def mcp_call(ms: float, call_id: str = "tool_1", *, failed: bool = False) -> McpCallEvent:
    return McpCallEvent(
        **RUN,
        occurred_at=at(ms),
        call_id=call_id,
        server="docs",
        tool="docs__echo",
        remote_tool="echo",
        outcome="failed" if failed else "ok",
        error_code="tool_error" if failed else None,
        arguments=tool_text(),
        result=tool_text(),
        duration_ms=20,
    )


def full_run() -> list[Event]:
    """One run with a model call, an approved MCP tool call, a skill, a compaction and a
    hand-off, in the order the runner and its layers emit them."""
    return [
        run_started(0),
        model_started(1),
        TextDeltaEvent(**RUN, occurred_at=at(2), model_call=1, text=prompt()),
        model_finished(41),
        tool_started(42),
        ApprovalRequestedEvent(
            **RUN, occurred_at=at(43), request_id="appr_1", call_id="tool_1", tool="docs__echo"
        ),
        ApprovalDecidedEvent(
            **RUN,
            occurred_at=at(48),
            request_id="appr_1",
            call_id="tool_1",
            tool="docs__echo",
            outcome="approved",
        ),
        mcp_call(70),
        tool_finished(72),
        SkillLoadedEvent(
            **RUN,
            occurred_at=at(73),
            skill="incident-triage",
            version="2",
            body_size=1200,
            resource_count=2,
            loaded_count=1,
            loaded_size=1200,
        ),
        ConversationCompactedEvent(**RUN, occurred_at=at(74), messages_before=9, messages_after=2),
        HandOffEvent(
            **RUN,
            occurred_at=at(75),
            call_id="tool_2",
            outcome="ok",
            from_task_class="chat",
            to_task_class="review",
            provider="specialist",
            model="stub-large",
        ),
        model_started(76, "call_2"),
        model_finished(100, "call_2", duration=24, used=usage(20, 7, 3, 1)),
        run_finished(101),
    ]


async def emit_all(target: OtelEventSink, events: list[Event]) -> None:
    for event in events:
        await target.emit(event)


def every_exported_text(telemetry: Telemetry) -> str:
    """Everything the exporters hold, as text: names, attributes, span events and status."""
    parts: list[str] = []
    for span in telemetry.spans():
        parts += [span.name, repr(dict(span.attributes or {}))]
        parts += [repr((e.name, dict(e.attributes or {}))) for e in span.events]
        parts.append(repr(span.status.description))
    data = telemetry.reader.get_metrics_data()
    if data is not None:
        parts.append(data.to_json())
    return "\n".join(parts)


class TestSpans:
    async def test_a_run_becomes_one_trace_with_a_span_per_operation(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(sink(telemetry), full_run())
        names = sorted(s.name for s in telemetry.spans())
        assert names == sorted(
            [
                "conduit.run",
                "conduit.model_call",
                "conduit.model_call",
                "conduit.tool_call",
                "conduit.approval",
                "conduit.mcp_call",
                "conduit.skill_load",
                "conduit.compaction",
                "conduit.hand_off",
            ]
        )
        run = telemetry.span("conduit.run")
        assert run.parent is None
        assert {s.context.trace_id for s in telemetry.spans()} == {run.context.trace_id}

    async def test_each_span_has_the_right_parent(self, telemetry: Telemetry) -> None:
        await emit_all(sink(telemetry), full_run())
        run = telemetry.span("conduit.run")
        tool = telemetry.span("conduit.tool_call")
        parents = {s.name: s.parent.span_id for s in telemetry.spans() if s.parent is not None}
        assert parents["conduit.tool_call"] == run.context.span_id
        assert parents["conduit.approval"] == tool.context.span_id
        assert parents["conduit.mcp_call"] == tool.context.span_id
        for name in ("conduit.model_call", "conduit.skill_load", "conduit.compaction"):
            assert parents[name] == run.context.span_id
        assert parents["conduit.hand_off"] == run.context.span_id  # its tool span is not open

    async def test_paired_spans_run_from_the_started_event_to_the_finished_one(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(sink(telemetry), full_run())
        run = telemetry.span("conduit.run")
        assert (run.start_time, run.end_time) == (ns(at(0)), ns(at(101)))
        tool = telemetry.span("conduit.tool_call")
        assert (tool.start_time, tool.end_time) == (ns(at(42)), ns(at(72)))
        approval = telemetry.span("conduit.approval")
        assert (approval.start_time, approval.end_time) == (ns(at(43)), ns(at(48)))

    async def test_a_single_event_span_ends_at_the_event_and_spans_its_duration(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(sink(telemetry), full_run())
        mcp = telemetry.span("conduit.mcp_call")
        assert (mcp.start_time, mcp.end_time) == (ns(at(50)), ns(at(70)))
        skill = telemetry.span("conduit.skill_load")
        assert skill.start_time == skill.end_time == ns(at(73))

    async def test_a_finished_event_without_its_start_still_gets_a_span(
        self, telemetry: Telemetry
    ) -> None:
        target = sink(telemetry)
        await target.emit(model_finished(500, "call_9", duration=120, run={}))
        await target.emit(tool_finished(600, "tool_9", duration=15))
        model = telemetry.span("conduit.model_call")
        assert (model.start_time, model.end_time) == (ns(at(380)), ns(at(500)))
        assert model.parent is None
        tool = telemetry.span("conduit.tool_call")
        assert (tool.start_time, tool.end_time) == (ns(at(585)), ns(at(600)))

    async def test_failures_set_an_error_status_named_by_an_identifier_only(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(
            sink(telemetry),
            [
                run_started(0),
                model_started(1),
                model_finished(5, failed=True),
                tool_started(6),
                mcp_call(8, failed=True),
                tool_finished(9, failed=True),
                run_finished(10, failed=True),
            ],
        )
        for name, why in [
            ("conduit.model_call", "ProviderError"),
            ("conduit.mcp_call", "tool_error"),
            ("conduit.tool_call", "tool_error"),
            ("conduit.run", "ToolFailed"),
        ]:
            status = telemetry.span(name).status
            assert (status.status_code, status.description) == (StatusCode.ERROR, why)

    async def test_successful_spans_are_left_unset_not_marked_ok(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(sink(telemetry), full_run())
        assert {s.status.status_code for s in telemetry.spans()} == {StatusCode.UNSET}

    async def test_a_failed_approval_is_an_error_and_a_refusal_is_not(
        self, telemetry: Telemetry
    ) -> None:
        target = sink(telemetry)
        outcomes: list[tuple[str, Literal["refused", "failed"]]] = [
            ("appr_1", "refused"),
            ("appr_2", "failed"),
        ]
        for request, outcome in outcomes:
            common = {**RUN, "request_id": request, "call_id": "tool_1", "tool": "lookup"}
            await target.emit(ApprovalRequestedEvent(**common, occurred_at=at(0)))
            await target.emit(ApprovalDecidedEvent(**common, occurred_at=at(1), outcome=outcome))
        codes = [s.status.status_code for s in telemetry.spans()]
        assert codes == [StatusCode.UNSET, StatusCode.ERROR]

    async def test_run_level_events_become_span_events_on_the_run(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(
            sink(telemetry),
            [
                run_started(0),
                OutputInvalidEvent(
                    **RUN,
                    occurred_at=at(1),
                    contract="answer",
                    attempt=1,
                    error_count=2,
                    retrying=True,
                ),
                SkillUsedEvent(**RUN, occurred_at=at(2), skill="incident-triage", resource="r.md"),
                SkillUnloadedEvent(
                    **RUN,
                    occurred_at=at(3),
                    skill="incident-triage",
                    body_size=10,
                    loaded_count=0,
                    loaded_size=0,
                ),
                SkillRefusedEvent(**RUN, occurred_at=at(4), reason="skill_unknown"),
                BudgetExhaustedEvent(
                    **RUN, occurred_at=at(5), budget=BudgetKind.TOKENS, limit=100, used=120
                ),
                run_finished(6, failed=True),
            ],
        )
        run = telemetry.span("conduit.run")
        assert [e.name for e in run.events] == [
            "conduit.output_invalid",
            "conduit.skill_used",
            "conduit.skill_unloaded",
            "conduit.skill_refused",
            "conduit.budget_exhausted",
        ]
        budget = run.events[-1]
        assert budget.attributes is not None
        assert budget.attributes["conduit.budget"] == "tokens"
        assert budget.attributes["conduit.used"] == 120

    async def test_text_deltas_and_unknown_events_make_no_span(self, telemetry: Telemetry) -> None:
        target = sink(telemetry)
        await target.emit(TextDeltaEvent(**RUN, model_call=1, text=prompt()))
        await target.emit(Event(type="product_defined"))
        assert telemetry.spans() == []

    async def test_embedding_calls_get_spans(self, telemetry: Telemetry) -> None:
        common: dict[str, Any] = {
            "call_id": "emb_1",
            "provider": "embed",
            "kind": "stub",
            "model": "stub-embed",
        }
        target = sink(telemetry)
        await target.emit(
            EmbeddingCallStartedEvent(
                **common, occurred_at=at(0), input=prompt(), text_count=3, dimensions=8
            )
        )
        await target.emit(
            EmbeddingCallFinishedEvent(
                **common,
                occurred_at=at(9),
                outcome="ok",
                usage=usage(12, 0),
                duration_ms=9,
                vector_count=3,
                dimensions=8,
            )
        )
        span = telemetry.span("conduit.embedding_call")
        assert (span.start_time, span.end_time) == (ns(at(0)), ns(at(9)))
        assert span.attributes is not None
        assert span.attributes["conduit.model"] == "stub-embed"

    def test_the_span_names_are_published(self) -> None:
        assert set(SPAN_NAMES) == {
            "conduit.run",
            "conduit.model_call",
            "conduit.embedding_call",
            "conduit.tool_call",
            "conduit.mcp_call",
            "conduit.approval",
            "conduit.skill_load",
            "conduit.compaction",
            "conduit.hand_off",
        }


class TestAttributes:
    async def test_spans_carry_identifiers_counts_hashes_and_sizes(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(sink(telemetry), full_run())
        model = next(
            s
            for s in telemetry.spans()
            if s.name == "conduit.model_call"
            and (s.attributes or {})["conduit.call_id"] == "call_1"
        )
        attributes = dict(model.attributes or {})
        expected = prompt()
        assert attributes["conduit.provider"] == "primary"
        assert attributes["conduit.model"] == "stub-echo"
        assert attributes["conduit.run_id"] == "run_1"
        assert attributes["conduit.task_class"] == "chat"
        assert attributes["conduit.max_output_tokens"] == 1024
        assert attributes["conduit.request.sha256"] == expected.sha256
        assert attributes["conduit.request.size"] == expected.size
        assert attributes["conduit.response.sha256"] == expected.sha256
        assert attributes["conduit.usage.cache_read_tokens"] == 4
        assert attributes["conduit.usage.cache_write_tokens"] == 2
        assert attributes["conduit.stop_reason"] == "tool_use"
        assert attributes["gen_ai.request.model"] == "stub-echo"
        assert attributes["gen_ai.usage.input_tokens"] == 16  # every input token, cached or not
        assert attributes["gen_ai.usage.output_tokens"] == 5

    async def test_no_content_is_exported_even_when_the_events_carry_it(
        self, telemetry: Telemetry
    ) -> None:
        events = full_run()
        assert sum(MARKER in e.model_dump_json() for e in events) >= 8  # not vacuous
        await emit_all(sink(telemetry), events)
        exported = every_exported_text(telemetry)
        assert telemetry.spans()
        assert MARKER not in exported
        assert "Prompt:" not in exported and "Tool:" not in exported

    def test_metadata_attributes_never_include_content_text(self) -> None:
        attributes = metadata_attributes(tool_finished(1))
        assert attributes["conduit.arguments.sha256"] == tool_text().sha256
        assert attributes["conduit.arguments.size"] == tool_text().size
        assert attributes["conduit.arguments.kind"] == "tool"
        assert not any(MARKER in str(v) for v in attributes.values())
        assert not any(k.endswith(".text") for k in attributes)

    def test_a_string_that_is_not_an_identifier_is_dropped(self) -> None:
        # Validation would refuse it; an emitter bypassing validation must not leak it.
        event = ToolCallEvent.model_construct(
            type="tool_call",
            event_id="evt_1",
            occurred_at=at(0),
            run_id=None,
            correlation_id=None,
            task_class=None,
            call_id="tool_1",
            tool=f"lookup {MARKER}",
            outcome="ok",
            error_code=None,
            arguments=tool_text(),
            result=None,
            duration_ms=1.0,
        )
        attributes = metadata_attributes(event)
        assert "conduit.tool" not in attributes
        assert attributes["conduit.call_id"] == "tool_1"
        assert not any(MARKER in str(v) for v in attributes.values())

    async def test_a_non_identifier_status_or_name_never_reaches_the_exporter(
        self, telemetry: Telemetry
    ) -> None:
        event = ToolCallEvent.model_construct(
            **ToolCallEvent(
                call_id="tool_1",
                tool="lookup",
                outcome="failed",
                arguments=tool_text(),
                duration_ms=1.0,
            ).__dict__
            | {"error_code": f"bad {MARKER}", "tool": f"lookup {MARKER}"}
        )
        target = sink(telemetry)
        await target.emit(event)
        exported = every_exported_text(telemetry)
        assert telemetry.spans()
        assert MARKER not in exported
        assert telemetry.span("conduit.tool_call").status.status_code is StatusCode.ERROR


class TestMetrics:
    async def test_all_four_token_counts_are_counted_apart(self, telemetry: Telemetry) -> None:
        await emit_all(sink(telemetry), full_run())
        totals: dict[str, int] = {}
        for point in telemetry.points("conduit.model.tokens"):
            kind = point.attributes["conduit.token.type"]
            totals[kind] = totals.get(kind, 0) + point.value
            assert point.attributes["conduit.model"] == "stub-echo"
            assert point.attributes["conduit.operation"] == "chat"
        assert totals == {"input": 30, "output": 12, "cache_read": 7, "cache_write": 3}

    async def test_cache_hits_count_calls_that_read_from_the_cache(
        self, telemetry: Telemetry
    ) -> None:
        target = sink(telemetry)
        await target.emit(model_finished(1, "c1", used=usage(cache_read=0)))
        await target.emit(model_finished(2, "c2", used=usage(cache_read=9)))
        await target.emit(model_finished(3, "c3", used=usage(cache_read=1)))
        hits = sum(p.value for p in telemetry.points("conduit.model.cache_hits"))
        calls = sum(p.value for p in telemetry.points("conduit.model.calls"))
        assert (hits, calls) == (2, 3)

    async def test_latency_is_recorded_per_operation(self, telemetry: Telemetry) -> None:
        await emit_all(sink(telemetry), full_run())
        by_operation = {
            p.attributes["conduit.operation"]: (p.count, p.sum)
            for p in telemetry.points("conduit.operation.duration")
        }
        assert by_operation == {
            "run": (1, 101),
            "model_call": (2, 64),
            "tool_call": (1, 30),
            "mcp_call": (1, 20),
        }

    async def test_errors_are_counted_by_operation_and_type(self, telemetry: Telemetry) -> None:
        await emit_all(
            sink(telemetry),
            [
                model_finished(1, failed=True),
                mcp_call(2, failed=True),
                tool_finished(3, failed=True),
                SkillRefusedEvent(**RUN, reason="skill_unknown"),
                run_finished(4, failed=True),
            ],
        )
        errors = {
            (p.attributes["conduit.operation"], p.attributes["conduit.error.type"]): p.value
            for p in telemetry.points("conduit.errors")
        }
        assert errors == {
            ("model_call", "ProviderError"): 1,
            ("mcp_call", "tool_error"): 1,
            ("tool_call", "tool_error"): 1,
            ("skill", "skill_unknown"): 1,
            ("run", "ToolFailed"): 1,
        }

    async def test_budget_use_is_recorded_per_run_and_exhaustion_counted(
        self, telemetry: Telemetry
    ) -> None:
        target = sink(telemetry)
        await target.emit(
            BudgetExhaustedEvent(**RUN, budget=BudgetKind.TOOL_CALLS, limit=4, used=4)
        )
        await target.emit(run_finished(9))
        exhausted = telemetry.points("conduit.budget.exhausted")
        assert [(p.attributes["conduit.budget"], p.value) for p in exhausted] == [("tool_calls", 1)]
        [tokens] = telemetry.points("conduit.run.tokens")
        assert (tokens.count, tokens.sum) == (1, 30 + 12 + 7 + 3)
        [model_calls] = telemetry.points("conduit.run.model_calls")
        assert model_calls.sum == 2
        [tool_calls] = telemetry.points("conduit.run.tool_calls")
        assert tool_calls.sum == 1

    async def test_hand_offs_are_counted(self, telemetry: Telemetry) -> None:
        await emit_all(sink(telemetry), full_run())
        [point] = telemetry.points("conduit.hand_offs")
        assert point.value == 1
        assert point.attributes == {
            "conduit.outcome": "ok",
            "conduit.from_task_class": "chat",
            "conduit.to_task_class": "review",
        }

    async def test_embedding_tokens_are_counted_as_embedding(self, telemetry: Telemetry) -> None:
        await sink(telemetry).emit(
            EmbeddingCallFinishedEvent(
                call_id="emb_1",
                provider="embed",
                kind="stub",
                model="stub-embed",
                outcome="ok",
                usage=usage(12, 0),
                duration_ms=3,
                dimensions=8,
            )
        )
        points = telemetry.points("conduit.model.tokens")
        assert {p.attributes["conduit.operation"] for p in points} == {"embedding"}
        assert sum(p.value for p in points if p.attributes["conduit.token.type"] == "input") == 12

    async def test_metric_attributes_never_carry_run_or_call_ids(
        self, telemetry: Telemetry
    ) -> None:
        await emit_all(sink(telemetry), full_run())
        data = telemetry.reader.get_metrics_data()
        assert data is not None
        text = data.to_json()
        for identifier in ("run_1", "corr_1", "call_1", "tool_1", "appr_1"):
            assert identifier not in text

    async def test_metric_attributes_come_only_from_the_low_cardinality_set(
        self, telemetry: Telemetry
    ) -> None:
        # Every kind of event that records a metric, succeeding and failing.
        events = [
            *full_run(),
            model_finished(200, "call_8", failed=True),
            mcp_call(201, "tool_8", failed=True),
            tool_finished(202, "tool_8", failed=True),
            SkillRefusedEvent(**RUN, reason="skill_unknown"),
            BudgetExhaustedEvent(**RUN, budget=BudgetKind.TOKENS, limit=1, used=2),
            ApprovalDecidedEvent(
                **RUN, request_id="appr_8", call_id="tool_8", tool="lookup", outcome="failed"
            ),
            EmbeddingCallFinishedEvent(
                call_id="emb_1",
                provider="embed",
                kind="stub",
                model="stub-embed",
                outcome="ok",
                usage=usage(12, 0),
                duration_ms=3,
                dimensions=8,
            ),
            run_finished(300, failed=True),
        ]
        await emit_all(sink(telemetry), events)
        data = telemetry.reader.get_metrics_data()
        assert data is not None
        points = [
            (metric.name, point)
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            for point in metric.data.data_points
        ]
        assert {name for name, _ in points} == set(METRIC_NAMES)
        for name, point in points:
            attributes = dict(point.attributes or {})
            assert set(attributes) <= METRIC_ATTRIBUTES, (name, sorted(attributes))
            for value in attributes.values():
                assert not UNBOUNDED.search(str(value)), (name, value)

    async def test_the_metric_names_are_published(self, telemetry: Telemetry) -> None:
        await emit_all(sink(telemetry), full_run())
        assert telemetry.metric_names() <= set(METRIC_NAMES)
        assert set(METRIC_NAMES) == {
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
        }


class TestLifecycle:
    async def test_finishing_a_run_ends_its_spans_left_open(self, telemetry: Telemetry) -> None:
        await emit_all(sink(telemetry), [run_started(0), tool_started(1), run_finished(5)])
        tool = telemetry.span("conduit.tool_call")
        assert tool.end_time == ns(at(5))
        assert (tool.attributes or {})["conduit.incomplete"] is True
        assert tool.status.status_code is StatusCode.ERROR

    async def test_open_spans_are_bounded(self, telemetry: Telemetry) -> None:
        target = sink(telemetry, max_open_spans=3)
        for index in range(5):
            await target.emit(tool_started(index, f"tool_{index}"))
        evicted = telemetry.spans()
        assert [(s.attributes or {})["conduit.call_id"] for s in evicted] == ["tool_0", "tool_1"]
        assert all((s.attributes or {})["conduit.incomplete"] is True for s in evicted)
        assert target.open_spans == 3

    def test_the_bound_must_be_positive(self, telemetry: Telemetry) -> None:
        with pytest.raises(ValueError, match="max_open_spans"):
            sink(telemetry, max_open_spans=0)

    async def test_close_ends_every_open_span(self, telemetry: Telemetry) -> None:
        target = sink(telemetry)
        await emit_all(target, [run_started(0), model_started(1)])
        target.close()
        assert sorted(s.name for s in telemetry.spans()) == ["conduit.model_call", "conduit.run"]
        assert target.open_spans == 0

    async def test_a_failing_tracer_is_logged_by_class_and_never_raised(
        self, telemetry: Telemetry, caplog: pytest.LogCaptureFixture
    ) -> None:
        class Broken:
            def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
                return self

            def start_span(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(f"collector said {MARKER}")

        target = OtelEventSink(
            tracer_provider=Broken(),  # type: ignore[arg-type]
            meter_provider=telemetry.meter_provider,
        )
        with caplog.at_level(logging.WARNING, logger="conduit.integrations.otel"):
            await target.emit(run_started(0))
        assert "RuntimeError" in caplog.text
        assert MARKER not in caplog.text

    async def test_cancellation_is_not_swallowed(self, telemetry: Telemetry) -> None:
        class Cancelling:
            def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
                return self

            def start_span(self, *args: Any, **kwargs: Any) -> Any:
                raise asyncio.CancelledError

        target = OtelEventSink(
            tracer_provider=Cancelling(),  # type: ignore[arg-type]
            meter_provider=telemetry.meter_provider,
        )
        with pytest.raises(asyncio.CancelledError):
            await target.emit(run_started(0))

    async def test_spans_are_ended_outside_the_sinks_lock(self, telemetry: Telemetry) -> None:
        # Ending a span runs the span processors, which may export over the network: the
        # sink's lock must not be held then, or a slow exporter stalls every emitter.
        target = sink(telemetry)
        held: list[bool] = []

        class Probe(SpanProcessor):
            def on_end(self, span: ReadableSpan) -> None:
                held.append(target._lock.locked())

        telemetry.tracer_provider.add_span_processor(Probe())
        await emit_all(target, full_run())
        await emit_all(target, [run_started(500), tool_started(501), tool_started(502, "tool_9")])
        target.close()
        assert len(held) == len(telemetry.spans()) >= 12
        assert not any(held)

    async def test_spans_already_finished_are_ended_when_a_handler_fails_part_way(
        self, telemetry: Telemetry, caplog: pytest.LogCaptureFixture
    ) -> None:
        target = sink(telemetry)

        class Failing:
            def record(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("histogram unavailable")

        await emit_all(target, [run_started(0), tool_started(1)])
        target._duration = Failing()  # type: ignore[assignment]
        with caplog.at_level(logging.WARNING, logger="conduit.integrations.otel"):
            await target.emit(run_finished(5))  # ends both spans, then records latency
        assert "RuntimeError" in caplog.text
        tool = telemetry.span("conduit.tool_call")
        assert tool.end_time == ns(at(5))
        assert tool.status.status_code is StatusCode.ERROR  # left open: incomplete
        run = telemetry.span("conduit.run")
        assert run.end_time == ns(at(5))
        assert run.status.status_code is StatusCode.UNSET
        assert target.open_spans == 0

    async def test_a_span_processor_failing_to_end_a_span_never_escapes(
        self, telemetry: Telemetry, caplog: pytest.LogCaptureFixture
    ) -> None:
        telemetry.tracer_provider.add_span_processor(RaisingOnEnd())
        target = sink(telemetry)
        with caplog.at_level(logging.WARNING, logger="conduit.integrations.otel"):
            await target.emit(run_started(0))
            await target.emit(run_finished(5))  # ending the run span raises in on_end
        assert "RuntimeError" in caplog.text
        assert MARKER not in caplog.text
        assert target.open_spans == 0

    async def test_a_failing_handler_is_logged_over_a_failing_span_end(
        self, telemetry: Telemetry, caplog: pytest.LogCaptureFixture
    ) -> None:
        class Failing:
            def record(self, *args: Any, **kwargs: Any) -> None:
                raise ValueError(f"histogram said {MARKER}")

        telemetry.tracer_provider.add_span_processor(RaisingOnEnd())
        target = sink(telemetry)
        await target.emit(run_started(0))
        target._duration = Failing()  # type: ignore[assignment]
        with caplog.at_level(logging.WARNING, logger="conduit.integrations.otel"):
            await target.emit(run_finished(5))  # the run span's end and the latency both fail
        assert "run_finished: ValueError" in caplog.text
        assert MARKER not in caplog.text
        assert target.open_spans == 0

    async def test_concurrent_runs_keep_their_spans_apart(self, telemetry: Telemetry) -> None:
        target = sink(telemetry)

        async def one(run_id: str) -> None:
            ids = {"run_id": run_id, "correlation_id": run_id, "task_class": "chat"}
            await target.emit(run_started(0).model_copy(update=ids))
            await asyncio.sleep(0)
            await target.emit(model_started(1, f"call_{run_id}").model_copy(update=ids))
            await asyncio.sleep(0)
            await target.emit(model_finished(5, f"call_{run_id}").model_copy(update=ids))
            await asyncio.sleep(0)
            await target.emit(run_finished(6).model_copy(update=ids))

        await asyncio.gather(*(one(f"run_{n}") for n in range(8)))
        runs = {
            (s.attributes or {})["conduit.run_id"]: s
            for s in telemetry.spans()
            if s.name == "conduit.run"
        }
        assert len(runs) == 8
        for span in telemetry.spans():
            if span.name == "conduit.model_call":
                run_id = (span.attributes or {})["conduit.run_id"]
                assert span.parent is not None
                assert span.parent.span_id == runs[run_id].context.span_id
        assert target.open_spans == 0

    def test_uses_the_global_providers_when_none_are_given(self) -> None:
        target = OtelEventSink()
        assert target.open_spans == 0
