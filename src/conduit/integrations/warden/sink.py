"""``WardenAuditSink``: conduit events recorded in warden's hash-chained audit trail.

Each audited conduit event type is a warden action, ``conduit.<event type>`` (for
example ``conduit.tool_call``), registered with a severity and a summary template by
``register_conduit_actions``. The sink records an event only when its type is one of
``CONDUIT_ACTIONS``, the sink was asked to audit it, and the recorder's registry holds
its action: nothing else ever reaches the trail. The stream's own events (text deltas,
and the started events whose finished event is audited) are not audit actions.

A record names its resource (``conduit.run``, ``conduit.model_call``,
``conduit.tool_call`` and so on, with the run, call, request or skill identifier), its
outcome (``failure`` for a failed call or run, a refusal, an exhausted budget or an
invalid output; ``success`` otherwise), when it happened, and an actor the product
gives the sink. Its details are the event's metadata (identifiers, counts, durations,
usage, and the hash and size of each piece of content), never content, whatever the
capture policy allowed the event to carry: warden's trail is append-only and cannot
forget, so prompt and tool text stays in the product's own sinks.

A recorder that fails raises ``AuditRecordFailed``, naming the action only, with no
cause or context kept; through a ``FanOutSink`` that is logged and the other sinks
still receive the event. Cancellation propagates unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from warden.audit import ActionRegistry, ActionSpec, AuditEvent, Outcome, Severity

from conduit.core.errors import ConduitError, ConfigurationError
from conduit.core.events import Event
from conduit.integrations._metadata import identifier, metadata_fields

__all__ = [
    "ACTION_NAMESPACE",
    "CONDUIT_ACTIONS",
    "AuditRecordFailed",
    "AuditRecording",
    "ConduitAction",
    "WardenAuditSink",
    "action_for",
    "register_conduit_actions",
]

logger = logging.getLogger("conduit.integrations.warden")

ACTION_NAMESPACE = "conduit."


@dataclass(frozen=True, slots=True)
class ConduitAction:
    """How one conduit event type is audited.

    ``resource_type`` names what the record is about, and ``resource_field`` the event
    field holding its identifier. ``failed_when`` maps an event field to the values
    that make the outcome ``failure``; ``always_fails`` marks an event that is a
    failure by nature (an exhausted budget, a refusal).
    """

    severity: Severity
    summary: str
    resource_type: str
    resource_field: str
    failed_when: tuple[str, frozenset[str]] | None = None
    always_fails: bool = False


_FAILED = ("outcome", frozenset({"failed"}))

#: Every conduit event type that is an audit action, and how it is recorded.
CONDUIT_ACTIONS: Mapping[str, ConduitAction] = {
    "run_started": ConduitAction(
        Severity.INFO, "conduit run {resource_id} started", "conduit.run", "run_id"
    ),
    "run_finished": ConduitAction(
        Severity.INFO,
        "conduit run {resource_id} finished: {outcome}",
        "conduit.run",
        "run_id",
        _FAILED,
    ),
    "model_call_finished": ConduitAction(
        Severity.INFO,
        "conduit model call {resource_id}: {outcome}",
        "conduit.model_call",
        "call_id",
        _FAILED,
    ),
    "embedding_call_finished": ConduitAction(
        Severity.INFO,
        "conduit embedding call {resource_id}: {outcome}",
        "conduit.embedding_call",
        "call_id",
        _FAILED,
    ),
    "tool_call": ConduitAction(
        Severity.INFO,
        "conduit tool call {resource_id}: {outcome}",
        "conduit.tool_call",
        "call_id",
        _FAILED,
    ),
    "mcp_call": ConduitAction(
        Severity.INFO,
        "conduit MCP call {resource_id}: {outcome}",
        "conduit.mcp_call",
        "call_id",
        _FAILED,
    ),
    "approval_requested": ConduitAction(
        Severity.NOTICE,
        "conduit approval {resource_id} requested",
        "conduit.approval",
        "request_id",
    ),
    "approval_decided": ConduitAction(
        Severity.NOTICE,
        "conduit approval {resource_id} decided: {outcome}",
        "conduit.approval",
        "request_id",
        ("outcome", frozenset({"refused", "unavailable", "failed"})),
    ),
    "hand_off": ConduitAction(
        Severity.NOTICE,
        "conduit hand-off in tool call {resource_id}: {outcome}",
        "conduit.tool_call",
        "call_id",
        ("outcome", frozenset({"refused"})),
    ),
    "skill_loaded": ConduitAction(
        Severity.INFO, "conduit skill {resource_id} loaded", "conduit.skill", "skill"
    ),
    "skill_unloaded": ConduitAction(
        Severity.INFO, "conduit skill {resource_id} unloaded", "conduit.skill", "skill"
    ),
    "skill_refused": ConduitAction(
        Severity.NOTICE,
        "conduit skill request refused",
        "conduit.skill",
        "skill",
        always_fails=True,
    ),
    "budget_exhausted": ConduitAction(
        Severity.NOTICE,
        "conduit run {resource_id} stopped by a budget",
        "conduit.run",
        "run_id",
        always_fails=True,
    ),
    "output_invalid": ConduitAction(
        Severity.INFO,
        "conduit run {resource_id} answer failed its output contract",
        "conduit.run",
        "run_id",
        always_fails=True,
    ),
    "conversation_compacted": ConduitAction(
        Severity.INFO, "conduit run {resource_id} conversation compacted", "conduit.run", "run_id"
    ),
}


def action_for(event_type: str) -> str | None:
    """The warden action for a conduit event type, or ``None`` if it is not audited."""
    return f"{ACTION_NAMESPACE}{event_type}" if event_type in CONDUIT_ACTIONS else None


def _audited(event_types: Iterable[str] | None) -> tuple[str, ...]:
    chosen = tuple(CONDUIT_ACTIONS) if event_types is None else tuple(event_types)
    unknown = sorted(t for t in chosen if t not in CONDUIT_ACTIONS)
    if unknown:
        raise ConfigurationError(f"{', '.join(unknown)}: not a conduit audit action")
    return chosen


def register_conduit_actions(
    registry: ActionRegistry, event_types: Iterable[str] | None = None
) -> list[ActionSpec]:
    """Register conduit's audit actions (all, or those of ``event_types``). Idempotent."""
    return [
        registry.register(
            f"{ACTION_NAMESPACE}{event_type}",
            severity=CONDUIT_ACTIONS[event_type].severity,
            summary=CONDUIT_ACTIONS[event_type].summary,
        )
        for event_type in _audited(event_types)
    ]


class AuditRecording(Protocol):
    """What the sink needs of warden's ``AuditRecorder``: its registry and ``record``."""

    @property
    def registry(self) -> ActionRegistry: ...

    async def record(self, event: AuditEvent) -> int: ...


class AuditRecordFailed(ConduitError):
    """warden could not record an audit event. Names the action only."""

    def __init__(self, action: str) -> None:
        super().__init__(f"the audit record for {action} could not be written")
        self.action = action


def _outcome(event: Event, spec: ConduitAction) -> Outcome:
    if spec.always_fails:
        return "failure"
    if spec.failed_when is not None:
        field, failing = spec.failed_when
        value = getattr(event, field, None)
        if isinstance(value, str) and value in failing:
            return "failure"
    return "success"


class WardenAuditSink:
    """Records conduit events in warden's audit trail, as registered actions only."""

    def __init__(
        self,
        recorder: AuditRecording,
        *,
        actions: Iterable[str] | None = None,
        actor: str | None = None,
    ) -> None:
        self._recorder = recorder
        self._audited = frozenset(_audited(actions))
        self._actor = actor
        self._unregistered_logged: set[str] = set()

    def __repr__(self) -> str:
        return f"WardenAuditSink(actions={len(self._audited)})"

    async def emit(self, event: Event) -> None:
        spec = CONDUIT_ACTIONS.get(event.type)
        if spec is None or event.type not in self._audited:
            return
        action = f"{ACTION_NAMESPACE}{event.type}"
        if action not in self._recorder.registry:
            if action not in self._unregistered_logged:
                self._unregistered_logged.add(action)
                logger.warning(
                    "audit action %s is not registered; call register_conduit_actions", action
                )
            return
        failed = False
        try:
            # Built inside the guard: warden refusing the record (a naive time, say)
            # fails like a failed write, without the record's values.
            record = AuditEvent(
                action=action,
                outcome=_outcome(event, spec),
                occurred_at=event.occurred_at,
                actor=self._actor,
                resource_type=spec.resource_type,
                resource_id=identifier(getattr(event, spec.resource_field, None)),
                details=dict(metadata_fields(event)),
            )
            await self._recorder.record(record)
        except Exception:
            failed = True
        if failed:
            raise AuditRecordFailed(action)
