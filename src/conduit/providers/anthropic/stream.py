"""Translating the Messages API's stream events into conduit's stream events.

Events arrive as the SDK's event models (read through ``model_dump``) or as mappings in
the API's wire format (recorded responses). Text deltas stream as they come. A tool
call is reported when its block stops, with its input assembled from the JSON deltas;
a call cut off by ``max_tokens`` is dropped, since its input is incomplete. Thinking,
server-tool and unknown blocks are not streamed.

Usage: ``message_start`` carries the input counts (uncached input, cache reads and
cache writes) and ``message_delta`` the final, cumulative counts. Each count the final
delta states replaces the earlier value; a count it leaves out keeps it. Nothing is
summed, so no count is ever doubled. The one ``StreamStopped`` carries the result.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    UsageReported,
)
from conduit.core.types import StopReason, Usage
from conduit.providers.anthropic.errors import failure_for_stream_error

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.REFUSAL,
}

#: The API's usage names, mapped to conduit's four counts.
USAGE_FIELDS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
    "cache_creation_input_tokens": "cache_write_tokens",
}


def _broken() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


def as_mapping(event: Any) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        return event
    dump = getattr(event, "model_dump", None)
    if dump is None:
        raise _broken()
    data = dump()
    if not isinstance(data, Mapping):
        raise _broken()
    return data


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _broken()
    return value


@dataclass
class _Counts:
    values: dict[str, int] = field(default_factory=lambda: dict.fromkeys(USAGE_FIELDS.values(), 0))

    def update(self, usage: Any) -> None:
        if not isinstance(usage, Mapping):
            raise _broken()
        for wire, name in USAGE_FIELDS.items():
            value = _count(usage.get(wire))
            if value is not None:
                self.values[name] = value

    def usage(self) -> Usage:
        return Usage(**self.values)


@dataclass
class _Block:
    kind: str
    id: str = ""
    name: str = ""
    json: list[str] = field(default_factory=list)
    stopped: bool = False


class _Translator:
    def __init__(self) -> None:
        self.counts = _Counts()
        self.blocks: dict[int, _Block] = {}
        self.started = False
        self.stop_reason: StopReason | None = None
        self.truncated = 0
        self.finished = False

    def _block(self, event: Mapping[str, Any]) -> _Block:
        block = self.blocks.get(event.get("index"))  # type: ignore[arg-type]
        if block is None or block.stopped:
            raise _broken()
        return block

    def handle(self, event: Mapping[str, Any]) -> list[StreamEvent]:
        kind = event.get("type")
        if self.finished:
            raise _broken()
        if kind == "ping":
            return []
        if kind == "error":
            error = event.get("error")
            error_type = error.get("type") if isinstance(error, Mapping) else None
            raise ProviderError(failure_for_stream_error(error_type))
        if kind == "message_start":
            if self.started:
                raise _broken()
            self.started = True
            message = event.get("message")
            if not isinstance(message, Mapping):
                raise _broken()
            self.counts.update(message.get("usage"))
            return [UsageReported(usage=self.counts.usage())]
        if not self.started:
            raise _broken()
        if kind == "content_block_start":
            return self._start(event)
        if kind == "content_block_delta":
            return self._delta(event)
        if kind == "content_block_stop":
            return self._stop(event)
        if kind == "message_delta":
            delta = event.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, Mapping) else None
            self.stop_reason = _STOP_REASONS.get(reason, StopReason.OTHER)  # type: ignore[arg-type]
            self.counts.update(event.get("usage") or {})
            return []
        if kind == "message_stop":
            return self._finish()
        return []  # an event type added after this adapter was written

    def _start(self, event: Mapping[str, Any]) -> list[StreamEvent]:
        index = event.get("index")
        block = event.get("content_block")
        if not isinstance(index, int) or index in self.blocks or not isinstance(block, Mapping):
            raise _broken()
        kind = block.get("type")
        if kind == "tool_use":
            call_id, name = block.get("id"), block.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise _broken()
            self.blocks[index] = _Block("tool_use", id=call_id, name=name)
            return []
        self.blocks[index] = _Block(str(kind))
        if kind == "text" and block.get("text"):
            return [TextDelta(text=str(block["text"]))]
        return []

    def _delta(self, event: Mapping[str, Any]) -> list[StreamEvent]:
        block = self._block(event)
        delta = event.get("delta")
        if not isinstance(delta, Mapping):
            raise _broken()
        kind = delta.get("type")
        if kind == "text_delta" and block.kind == "text":
            text = delta.get("text")
            return [TextDelta(text=text)] if isinstance(text, str) and text else []
        if kind == "input_json_delta" and block.kind == "tool_use":
            block.json.append(str(delta.get("partial_json") or ""))
        return []

    def _stop(self, event: Mapping[str, Any]) -> list[StreamEvent]:
        block = self._block(event)
        block.stopped = True
        if block.kind != "tool_use":
            return []
        text = "".join(block.json).strip() or "{}"
        try:
            arguments = json.loads(text)
        except ValueError:
            self.truncated += 1  # acceptable only if the call stops on max_tokens
            return []
        if not isinstance(arguments, dict):
            raise _broken()
        return [
            ToolCallStarted(id=block.id, name=block.name),
            ToolCallFinished(id=block.id, name=block.name, input=arguments),
        ]

    def _finish(self) -> list[StreamEvent]:
        if self.stop_reason is None or any(not b.stopped for b in self.blocks.values()):
            raise _broken()
        if self.truncated and self.stop_reason is not StopReason.MAX_TOKENS:
            raise _broken()
        self.finished = True
        return [StreamStopped(stop_reason=self.stop_reason, usage=self.counts.usage())]


async def translate(events: AsyncIterator[Any]) -> AsyncIterator[StreamEvent]:
    """conduit's events for one Messages API stream, ending with one ``StreamStopped``."""
    translator = _Translator()
    async for event in events:
        for out in translator.handle(as_mapping(event)):
            yield out
    if not translator.finished:
        raise _broken()
