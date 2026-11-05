"""Run events: what a run streams, and what it records in the event sink.

Each is an ``Event`` carrying identifiers, sizes, hashes and outcomes only. Content
(model output, tool arguments and results) appears only as ``CapturedContent``, whose
text the capture policy controls; tool fields are declared ``ToolContent`` so they are
held to the tool-content setting whatever an emitter labelled them.

A run streams, in order: ``RunStartedEvent``; for each model call its ``TextDeltaEvent``s;
for each tool call ``ToolCallStartedEvent``, any ``ApprovalRequestedEvent`` and
``ApprovalDecidedEvent``, any ``HandOffEvent``, and the core ``ToolCallEvent`` as the
finished event; ``OutputInvalidEvent`` for an answer that failed its contract;
``ConversationCompactedEvent`` when the compaction hook replaced the conversation;
``BudgetExhaustedEvent`` when a limit stopped the run; and last ``RunFinishedEvent``,
with the run's final usage. Every event but the text deltas also goes to the event sink.
Model calls are recorded by the provider registry's own events, as for any call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt

from conduit.agent.budgets import BudgetKind
from conduit.agent.results import RunFailure
from conduit.core.errors import ProviderFailure
from conduit.core.events import CapturedContent, Event, ToolCallEvent, ToolContent
from conduit.core.identifiers import Identifier
from conduit.core.types import StopReason, Usage

RunOutcome = Literal["completed", "failed"]
ApprovalOutcome = Literal["approved", "refused", "unavailable", "failed"]


class RunStartedEvent(Event):
    type: Literal["run_started"] = "run_started"
    provider: Identifier
    kind: Identifier
    model: Identifier
    message_count: NonNegativeInt
    tool_count: NonNegativeInt
    hand_off_count: NonNegativeInt
    contract: Identifier | None = None


class TextDeltaEvent(Event):
    """A piece of model output as it streams. Stream only; never sent to the sink."""

    type: Literal["text_delta"] = "text_delta"
    model_call: PositiveInt
    text: CapturedContent


class ToolCallStartedEvent(Event):
    """A tool call the model asked for. ``tool`` is ``_unknown`` if no such tool exists."""

    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: Identifier
    tool: Identifier
    arguments: ToolContent


class ApprovalRequestedEvent(Event):
    type: Literal["approval_requested"] = "approval_requested"
    request_id: Identifier
    call_id: Identifier
    tool: Identifier


class ApprovalDecidedEvent(Event):
    """``unavailable``: no handler was supplied; ``failed``: the handler raised. Both refuse."""

    type: Literal["approval_decided"] = "approval_decided"
    request_id: Identifier
    call_id: Identifier
    tool: Identifier
    outcome: ApprovalOutcome


class HandOffEvent(Event):
    """A hand-off taken (``ok``) or refused. A refused target is not named: it came from
    the model, and need not be a task class at all."""

    type: Literal["hand_off"] = "hand_off"
    call_id: Identifier
    outcome: Literal["ok", "refused"]
    from_task_class: Identifier
    to_task_class: Identifier | None = None
    provider: Identifier | None = None
    model: Identifier | None = None


class BudgetExhaustedEvent(Event):
    type: Literal["budget_exhausted"] = "budget_exhausted"
    budget: BudgetKind
    limit: NonNegativeFloat
    used: NonNegativeFloat


class OutputInvalidEvent(Event):
    type: Literal["output_invalid"] = "output_invalid"
    contract: Identifier
    attempt: PositiveInt
    error_count: NonNegativeInt
    retrying: bool


class ConversationCompactedEvent(Event):
    type: Literal["conversation_compacted"] = "conversation_compacted"
    messages_before: NonNegativeInt
    messages_after: NonNegativeInt


class RunFinishedEvent(Event):
    """The run's outcome and final usage. ``error_type`` names an exception the run ended
    on (its class name, or ``cancelled`` or ``abandoned``); such a run is never streamed
    to its end, so this event then reaches the sink only. ``filtered``: the product's
    output filter changed the answer."""

    type: Literal["run_finished"] = "run_finished"
    outcome: RunOutcome
    failure: RunFailure | None = None
    budget: BudgetKind | None = None
    provider_failure: ProviderFailure | None = None
    error_type: Identifier | None = None
    stop_reason: StopReason | None = None
    usage: Usage
    model_calls: NonNegativeInt
    tool_calls: NonNegativeInt
    hand_offs: NonNegativeInt
    duration_ms: NonNegativeFloat
    output: CapturedContent | None = None
    filtered: bool = False


RunEvent = (
    RunStartedEvent
    | TextDeltaEvent
    | ToolCallStartedEvent
    | ApprovalRequestedEvent
    | ApprovalDecidedEvent
    | HandOffEvent
    | ToolCallEvent
    | BudgetExhaustedEvent
    | OutputInvalidEvent
    | ConversationCompactedEvent
    | RunFinishedEvent
)

__all__ = [
    "ApprovalDecidedEvent",
    "ApprovalRequestedEvent",
    "BudgetExhaustedEvent",
    "ConversationCompactedEvent",
    "HandOffEvent",
    "OutputInvalidEvent",
    "RunEvent",
    "RunFinishedEvent",
    "RunStartedEvent",
    "TextDeltaEvent",
    "ToolCallEvent",
    "ToolCallStartedEvent",
]
