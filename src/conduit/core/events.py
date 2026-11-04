"""Structured events, the event sink protocol, fan-out and the content-capture policy.

Events describe what happened: identifiers, hashes, sizes and outcomes (C19). Prompt
and tool content is recorded only when the product opts in through ``CapturePolicy``,
and even then only inside a ``CapturedContent``: every other string field of an event
is an ``Identifier``, which cannot hold prose. The event-content tripwire in
``tests/tripwires`` checks that rule for every event class in conduit.

``FanOutSink`` delivers each event to several product-supplied sinks concurrently.
It enforces its own capture policy on the way through: content the policy does not
allow is removed before any sink sees it, whatever policy the emitter used. A failing
sink is logged (its class and the exception class only) and never stops the others.

Upper layers define their own event classes by subclassing ``Event``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from conduit.core.errors import ProviderFailure
from conduit.core.identifiers import Identifier, new_id
from conduit.core.settings import SettingsModel
from conduit.core.types import StopReason, Usage

logger = logging.getLogger("conduit.events")


class ContentKind(StrEnum):
    PROMPT = "prompt"  # system prompt, messages and model output
    TOOL = "tool"  # tool call arguments and tool results


class CapturedContent(BaseModel):
    """Content as an event may carry it: always its hash and size, its text only if allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ContentKind
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: NonNegativeInt  # bytes, UTF-8
    text: str | None = None


class CapturePolicy(SettingsModel):
    """What content events may carry. Nothing, unless the product opts in."""

    prompts: bool = False
    tool_content: bool = False

    def allows(self, kind: ContentKind) -> bool:
        return self.prompts if kind is ContentKind.PROMPT else self.tool_content

    def capture(self, kind: ContentKind, text: str) -> CapturedContent:
        data = text.encode("utf-8")
        return CapturedContent(
            kind=kind,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            text=text if self.allows(kind) else None,
        )


def _now() -> datetime:
    return datetime.now(UTC)


class Event(BaseModel):
    """Base for every event. Subclasses set ``type`` and use only metadata fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Identifier
    event_id: Identifier = Field(default_factory=lambda: new_id("evt"))
    occurred_at: datetime = Field(default_factory=_now)
    run_id: Identifier | None = None
    correlation_id: Identifier | None = None
    task_class: Identifier | None = None


CallOutcome = Literal["ok", "failed"]


class ModelCallStartedEvent(Event):
    type: Literal["model_call_started"] = "model_call_started"
    call_id: Identifier
    provider: Identifier  # the configured provider's name
    kind: Identifier
    model: Identifier
    request: CapturedContent
    message_count: NonNegativeInt
    tool_count: NonNegativeInt
    max_output_tokens: NonNegativeInt


class ModelCallFinishedEvent(Event):
    """One model call's outcome, with its final usage (the single metering point)."""

    type: Literal["model_call_finished"] = "model_call_finished"
    call_id: Identifier
    provider: Identifier
    kind: Identifier
    model: Identifier
    outcome: CallOutcome
    stop_reason: StopReason | None = None
    usage: Usage | None = None
    failure: ProviderFailure | None = None
    error_type: Identifier | None = None
    duration_ms: NonNegativeFloat
    response: CapturedContent | None = None
    tool_call_count: NonNegativeInt = 0


class ToolCallEvent(Event):
    type: Literal["tool_call"] = "tool_call"
    call_id: Identifier
    tool: Identifier
    outcome: CallOutcome
    error_code: Identifier | None = None
    arguments: CapturedContent
    result: CapturedContent | None = None
    duration_ms: NonNegativeFloat


@runtime_checkable
class EventSink(Protocol):
    """Receives events. Supplied by the product (a log, an audit trail, a ledger)."""

    async def emit(self, event: Event) -> None: ...


class MemoryEventSink:
    """Keeps every event in a list. For tests and the terminal interface."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class NullEventSink:
    async def emit(self, event: Event) -> None:
        return None


def _enforce(value: Any, policy: CapturePolicy, removed: list[ContentKind]) -> Any:
    if isinstance(value, CapturedContent):
        if value.text is not None and not policy.allows(value.kind):
            removed.append(value.kind)
            return value.model_copy(update={"text": None})
        return value
    if isinstance(value, BaseModel):
        updates: dict[str, Any] = {}
        for name in type(value).model_fields:
            current = getattr(value, name)
            enforced = _enforce(current, policy, removed)
            if enforced is not current:
                updates[name] = enforced
        return value.model_copy(update=updates) if updates else value
    if isinstance(value, tuple | list):
        items = [_enforce(item, policy, removed) for item in value]
        if any(a is not b for a, b in zip(items, value, strict=True)):
            return type(value)(items)
        return value
    if isinstance(value, dict):
        enforced_dict = {k: _enforce(v, policy, removed) for k, v in value.items()}
        if any(enforced_dict[k] is not value[k] for k in value):
            return enforced_dict
        return value
    return value


def enforce_capture(event: Event, policy: CapturePolicy) -> Event:
    """``event`` with every piece of content ``policy`` does not allow removed."""
    removed: list[ContentKind] = []
    enforced = _enforce(event, policy, removed)
    if removed:
        logger.warning(
            "event %s carried %s content the capture policy does not allow; removed",
            event.type,
            ", ".join(sorted({k.value for k in removed})),
        )
    if not isinstance(enforced, Event):  # pragma: no cover - _enforce keeps the type
        raise TypeError("capture enforcement changed the event's type")
    return enforced


class FanOutSink:
    """Delivers each event to every sink, concurrently, under one capture policy."""

    def __init__(self, sinks: Iterable[EventSink], *, policy: CapturePolicy | None = None) -> None:
        self._sinks = tuple(sinks)
        self.policy = policy or CapturePolicy()

    async def emit(self, event: Event) -> None:
        event = enforce_capture(event, self.policy)
        results = await asyncio.gather(
            *(sink.emit(event) for sink in self._sinks), return_exceptions=True
        )
        for sink, result in zip(self._sinks, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "event sink %s failed on %s: %s",
                    type(sink).__name__,
                    event.type,
                    type(result).__name__,
                )
