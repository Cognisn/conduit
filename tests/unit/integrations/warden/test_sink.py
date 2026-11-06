"""The warden audit sink: conduit events as registered actions in warden's audit trail.

warden's own ``ActionRegistry`` and ``AuditEvent`` are used throughout. Its recorder
writes to Postgres, so a stand-in records instead: like warden's ``AuditRecorder`` it
refuses an action its registry does not hold. Nothing reaches a database or the network.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("warden.audit")

from warden.audit import ActionRegistry, AuditEvent, Severity

from conduit.agent.budgets import BudgetKind
from conduit.agent.events import (
    ApprovalDecidedEvent,
    BudgetExhaustedEvent,
    HandOffEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextDeltaEvent,
    ToolCallStartedEvent,
)
from conduit.core.errors import ConduitError
from conduit.core.events import (
    CapturePolicy,
    ContentKind,
    Event,
    FanOutSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
    ToolCallEvent,
)
from conduit.core.types import StopReason, Usage
from conduit.integrations.warden import (
    CONDUIT_ACTIONS,
    AuditRecordFailed,
    WardenAuditSink,
    action_for,
    register_conduit_actions,
)
from conduit.mcp.events import McpCallEvent
from conduit.skills.events import SkillLoadedEvent, SkillRefusedEvent

MARKER = "zz synthetic audit content marker 3390 zz"
CAPTURE_ALL = CapturePolicy(prompts=True, tool_content=True)
RUN: dict[str, Any] = {"run_id": "run_1", "correlation_id": "corr_1", "task_class": "chat"}
WHEN = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)


class Recorder:
    """Records to a list. Like warden's recorder, an unregistered action raises."""

    def __init__(self, registry: ActionRegistry, *, fail_with: BaseException | None = None):
        self._registry = registry
        self.fail_with = fail_with
        self.events: list[AuditEvent] = []

    @property
    def registry(self) -> ActionRegistry:
        return self._registry

    async def record(self, event: AuditEvent) -> int:
        self._registry.get(event.action)
        if self.fail_with is not None:
            raise self.fail_with
        self.events.append(event)
        return len(self.events)


def registered() -> ActionRegistry:
    registry = ActionRegistry()
    register_conduit_actions(registry)
    return registry


def tool_text() -> Any:
    return CAPTURE_ALL.capture(ContentKind.TOOL, f"Tool: {MARKER}")


def prompt() -> Any:
    return CAPTURE_ALL.capture(ContentKind.PROMPT, f"Prompt: {MARKER}")


def usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3, cache_write_tokens=1)


def tool_call(*, failed: bool = False) -> ToolCallEvent:
    return ToolCallEvent(
        **RUN,
        occurred_at=WHEN,
        call_id="tool_1",
        tool="lookup",
        outcome="failed" if failed else "ok",
        error_code="tool_error" if failed else None,
        arguments=tool_text(),
        result=tool_text(),
        duration_ms=12.5,
    )


def model_finished(*, failed: bool = False) -> ModelCallFinishedEvent:
    return ModelCallFinishedEvent(
        **RUN,
        occurred_at=WHEN,
        call_id="call_1",
        provider="primary",
        kind="stub",
        model="stub-echo",
        outcome="failed" if failed else "ok",
        stop_reason=None if failed else StopReason.END_TURN,
        usage=usage(),
        error_type="ProviderError" if failed else None,
        duration_ms=40,
        response=prompt(),
    )


def everything() -> list[Event]:
    return [
        RunStartedEvent(
            **RUN,
            provider="primary",
            kind="stub",
            model="stub-echo",
            message_count=1,
            tool_count=1,
            hand_off_count=0,
        ),
        ModelCallStartedEvent(
            **RUN,
            call_id="call_1",
            provider="primary",
            kind="stub",
            model="stub-echo",
            request=prompt(),
            message_count=1,
            tool_count=1,
            max_output_tokens=256,
        ),
        TextDeltaEvent(**RUN, model_call=1, text=prompt()),
        model_finished(),
        ToolCallStartedEvent(**RUN, call_id="tool_1", tool="lookup", arguments=tool_text()),
        ApprovalDecidedEvent(
            **RUN, request_id="appr_1", call_id="tool_1", tool="lookup", outcome="refused"
        ),
        McpCallEvent(
            **RUN,
            call_id="tool_1",
            server="docs",
            tool="docs__echo",
            remote_tool="echo",
            outcome="ok",
            arguments=tool_text(),
            result=tool_text(),
            duration_ms=4,
        ),
        tool_call(),
        SkillLoadedEvent(
            **RUN,
            skill="incident-triage",
            body_size=10,
            resource_count=0,
            loaded_count=1,
            loaded_size=10,
        ),
        HandOffEvent(**RUN, call_id="tool_2", outcome="ok", from_task_class="chat"),
        BudgetExhaustedEvent(**RUN, budget=BudgetKind.TOKENS, limit=10, used=12),
        RunFinishedEvent(
            **RUN,
            outcome="completed",
            usage=usage(),
            model_calls=1,
            tool_calls=1,
            hand_offs=0,
            duration_ms=90,
            output=prompt(),
        ),
    ]


class TestRegistration:
    def test_every_conduit_action_is_registered_under_the_conduit_namespace(self) -> None:
        registry = ActionRegistry()
        specs = register_conduit_actions(registry)
        assert {s.action for s in specs} == {f"conduit.{t}" for t in CONDUIT_ACTIONS}
        assert all(s.action in registry for s in specs)
        assert all(s.action.startswith("conduit.") for s in registry.actions())

    def test_registration_is_idempotent(self) -> None:
        registry = registered()
        before = registry.actions()
        register_conduit_actions(registry)
        assert registry.actions() == before

    def test_a_subset_can_be_registered(self) -> None:
        registry = ActionRegistry()
        register_conduit_actions(registry, ["run_finished", "tool_call"])
        assert [s.action for s in registry.actions()] == [
            "conduit.run_finished",
            "conduit.tool_call",
        ]

    def test_an_unknown_event_type_cannot_be_registered(self) -> None:
        with pytest.raises(ValueError, match="not a conduit audit action"):
            register_conduit_actions(ActionRegistry(), ["text_delta"])

    def test_summaries_render_from_the_event_fields_warden_allows(self) -> None:
        registry = registered()
        spec = registry.get("conduit.tool_call")
        rendered = spec.render(
            AuditEvent(
                action="conduit.tool_call",
                outcome="failure",
                resource_type="conduit.tool_call",
                resource_id="tool_1",
            )
        )
        assert "tool_1" in rendered and "failure" in rendered

    def test_the_audited_types_are_the_outcomes_not_the_stream(self) -> None:
        assert "text_delta" not in CONDUIT_ACTIONS
        assert "model_call_started" not in CONDUIT_ACTIONS
        assert "tool_call_started" not in CONDUIT_ACTIONS
        assert {
            "run_started",
            "run_finished",
            "model_call_finished",
            "embedding_call_finished",
            "tool_call",
            "mcp_call",
            "approval_requested",
            "approval_decided",
            "hand_off",
            "skill_loaded",
            "skill_unloaded",
            "skill_refused",
            "budget_exhausted",
            "output_invalid",
            "conversation_compacted",
        } <= set(CONDUIT_ACTIONS)


class TestRecording:
    async def test_registered_events_are_recorded_as_their_actions(self) -> None:
        recorder = Recorder(registered())
        sink = WardenAuditSink(recorder)
        for event in everything():
            await sink.emit(event)
        assert [e.action for e in recorder.events] == [
            "conduit.run_started",
            "conduit.model_call_finished",
            "conduit.approval_decided",
            "conduit.mcp_call",
            "conduit.tool_call",
            "conduit.skill_loaded",
            "conduit.hand_off",
            "conduit.budget_exhausted",
            "conduit.run_finished",
        ]

    async def test_only_registered_actions_are_recorded(self) -> None:
        registry = ActionRegistry()
        register_conduit_actions(registry, ["tool_call"])
        recorder = Recorder(registry)
        sink = WardenAuditSink(recorder)
        for event in everything():
            await sink.emit(event)
        assert [e.action for e in recorder.events] == ["conduit.tool_call"]

    async def test_a_product_event_type_is_never_recorded(self) -> None:
        registry = registered()
        registry.register("conduit_product.thing", severity=Severity.INFO, summary="{action}")
        recorder = Recorder(registry)
        await WardenAuditSink(recorder).emit(Event(type="thing"))
        await WardenAuditSink(recorder).emit(Event(type="product.thing"))
        assert recorder.events == []

    async def test_the_sink_can_record_a_subset(self) -> None:
        recorder = Recorder(registered())
        sink = WardenAuditSink(recorder, actions=["run_finished"])
        for event in everything():
            await sink.emit(event)
        assert [e.action for e in recorder.events] == ["conduit.run_finished"]

    def test_a_subset_naming_an_unknown_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a conduit audit action"):
            WardenAuditSink(Recorder(registered()), actions=["text_delta"])

    async def test_each_record_names_its_resource_outcome_time_and_actor(self) -> None:
        recorder = Recorder(registered())
        sink = WardenAuditSink(recorder, actor="svc-assistant")
        await sink.emit(tool_call(failed=True))
        await sink.emit(model_finished())
        failed, model = recorder.events
        assert (failed.outcome, failed.resource_type, failed.resource_id) == (
            "failure",
            "conduit.tool_call",
            "tool_1",
        )
        assert failed.occurred_at == WHEN
        assert failed.actor == "svc-assistant"
        assert (model.outcome, model.resource_type, model.resource_id) == (
            "success",
            "conduit.model_call",
            "call_1",
        )

    @pytest.mark.parametrize(
        ("event", "outcome"),
        [
            (
                ApprovalDecidedEvent(request_id="a", call_id="c", tool="t", outcome="approved"),
                "success",
            ),
            (
                ApprovalDecidedEvent(request_id="a", call_id="c", tool="t", outcome="refused"),
                "failure",
            ),
            (HandOffEvent(call_id="c", outcome="refused", from_task_class="chat"), "failure"),
            (SkillRefusedEvent(reason="skill_unknown"), "failure"),
            (BudgetExhaustedEvent(budget=BudgetKind.TOKENS, limit=1, used=2), "failure"),
        ],
    )
    async def test_refusals_and_exhaustion_are_failures(self, event: Event, outcome: str) -> None:
        recorder = Recorder(registered())
        await WardenAuditSink(recorder).emit(event)
        assert [e.outcome for e in recorder.events] == [outcome]

    async def test_details_hold_metadata_and_never_content(self) -> None:
        recorder = Recorder(registered())
        sink = WardenAuditSink(recorder)
        events = everything()
        assert sum(MARKER in e.model_dump_json() for e in events) >= 6  # not vacuous
        for event in events:
            await sink.emit(event)
        assert recorder.events
        for record in recorder.events:
            assert MARKER not in record.model_dump_json()
        tool = next(e for e in recorder.events if e.action == "conduit.tool_call")
        assert tool.details["arguments.sha256"] == tool_text().sha256
        assert tool.details["arguments.size"] == tool_text().size
        assert tool.details["duration_ms"] == 12.5
        assert tool.details["run_id"] == "run_1"
        assert not any(key.endswith(".text") for key in tool.details)
        model = next(e for e in recorder.events if e.action == "conduit.model_call_finished")
        assert model.details["usage.cache_read_tokens"] == 3
        assert model.details["usage.cache_write_tokens"] == 1

    async def test_a_string_that_is_not_an_identifier_never_reaches_the_trail(self) -> None:
        event = ToolCallEvent.model_construct(
            **tool_call().__dict__ | {"tool": f"lookup {MARKER}", "call_id": f"x {MARKER}"}
        )
        recorder = Recorder(registered())
        await WardenAuditSink(recorder).emit(event)
        (record,) = recorder.events
        assert MARKER not in record.model_dump_json()
        assert record.resource_id is None

    def test_action_for_maps_an_event_type_to_its_action(self) -> None:
        assert action_for("tool_call") == "conduit.tool_call"
        assert action_for("text_delta") is None


class TestFailures:
    async def test_a_failing_recorder_raises_a_conduit_error_without_its_cause(self) -> None:
        recorder = Recorder(registered(), fail_with=RuntimeError(f"db said {MARKER}"))
        with pytest.raises(AuditRecordFailed) as raised:
            await WardenAuditSink(recorder).emit(tool_call())
        error = raised.value
        assert isinstance(error, ConduitError)
        assert error.__cause__ is None and error.__context__ is None
        for shown in (str(error), repr(error), *map(str, error.args)):
            assert MARKER not in shown
            assert "run_1" not in shown and "tool_1" not in shown
        assert "conduit.tool_call" in str(error)

    async def test_through_a_fan_out_the_failure_is_logged_and_others_still_receive(
        self,
    ) -> None:
        recorder = Recorder(registered(), fail_with=RuntimeError("down"))
        good = Recorder(registered())
        bus = FanOutSink([WardenAuditSink(recorder), WardenAuditSink(good)])
        await bus.emit(tool_call())
        assert [e.action for e in good.events] == ["conduit.tool_call"]

    async def test_cancellation_propagates_unchanged(self) -> None:
        recorder = Recorder(registered(), fail_with=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await WardenAuditSink(recorder).emit(tool_call())

    async def test_an_action_missing_from_the_recorders_registry_is_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = Recorder(ActionRegistry())
        await WardenAuditSink(recorder).emit(tool_call())
        assert recorder.events == []
        assert "conduit.tool_call" in caplog.text
        assert "register_conduit_actions" in caplog.text
