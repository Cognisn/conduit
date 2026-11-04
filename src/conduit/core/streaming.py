"""Typed stream events, and the collection of a stream into a normalised response.

Every provider streams (C19). A stream is a sequence of events ending with exactly
one ``StreamStopped``, which carries the call's final usage: usage is taken from the
final stream event, and ``UsageReported`` events before it are progress reports only,
never summed. A non-streaming call is defined as the collected stream (``collect``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.types import (
    TOOL_NAME_PATTERN,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextDelta(_Event):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallStarted(_Event):
    type: Literal["tool_call_started"] = "tool_call_started"
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(pattern=TOOL_NAME_PATTERN)


class ToolCallFinished(_Event):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    input: dict[str, Any]


class UsageReported(_Event):
    """A progress report of usage so far. Never summed; the final event's usage wins."""

    type: Literal["usage"] = "usage"
    usage: Usage


class StreamStopped(_Event):
    """The last event of every stream, with the call's final usage."""

    type: Literal["stop"] = "stop"
    stop_reason: StopReason
    usage: Usage


StreamEvent = Annotated[
    TextDelta | ToolCallStarted | ToolCallFinished | UsageReported | StreamStopped,
    Field(discriminator="type"),
]


def _broken() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


async def collect(events: AsyncIterator[StreamEvent], *, model: str) -> ModelResponse:
    """The normalised response a stream describes.

    Raises ``ProviderError(PROTOCOL)`` if the stream has no final ``StreamStopped``,
    carries events after it, or finishes a tool call it never started (or under
    another name), or starts one it never finishes.
    """
    blocks: list[TextBlock | ToolUseBlock] = []
    text: list[str] = []
    open_calls: dict[str, str] = {}
    stopped: StreamStopped | None = None

    def flush_text() -> None:
        if text:
            blocks.append(TextBlock(text="".join(text)))
            text.clear()

    async for event in events:
        if stopped is not None:
            raise _broken()
        if isinstance(event, TextDelta):
            text.append(event.text)
        elif isinstance(event, ToolCallStarted):
            flush_text()
            if event.id in open_calls:
                raise _broken()
            open_calls[event.id] = event.name
        elif isinstance(event, ToolCallFinished):
            if open_calls.pop(event.id, None) != event.name:
                raise _broken()
            blocks.append(ToolUseBlock(id=event.id, name=event.name, input=event.input))
        elif isinstance(event, StreamStopped):
            stopped = event
        # UsageReported: progress only.

    if stopped is None or open_calls:
        raise _broken()
    flush_text()
    return ModelResponse(
        content=tuple(blocks),
        stop_reason=stopped.stop_reason,
        usage=stopped.usage,
        model=model,
    )
