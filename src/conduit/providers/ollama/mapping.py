"""Mapping between conduit's types and Ollama's /api/chat wire format.

Requests become /api/chat bodies with ``stream: true``. The response is NDJSON: one
JSON object per line, each carrying a piece of the assistant message, the last with
``done: true``, the ``done_reason`` and the token counts. ``StreamMapper`` turns the
lines into conduit stream events ending with one ``StreamStopped``.

- Cache breakpoints are ignored: Ollama reuses its own key-value cache for a repeated
  prefix without being asked.
- Tool results become ``tool`` messages named by the tool (Ollama pairs results by
  name, not by call id); an error result is prefixed ``Error:`` because Ollama has no
  error flag. Ollama gives no call ids, so the mapper makes them.
- Usage: ``prompt_eval_count`` counts the whole prompt, and Ollama versions that
  report ``prompt_eval_cached_count`` count the part read from its cache, so input is
  the difference and the cached part is ``cache_read_tokens``. Without the cached
  count, the cache counts are zero. Nothing is ever written to a cache that conduit
  could count, so ``cache_write_tokens`` is always zero. A ``done`` line without
  counts (some Ollama versions omit ``prompt_eval_count`` when the whole prompt was
  cached) reports zero for each missing count; a negative count is a protocol failure.

Every failure is ``ProviderError(failure)`` raised outside any exception handler, so
no decoding or validation error is its cause or its context: Ollama's text never
reaches a log, an exception or an event.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.identifiers import new_id
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

CHAT_PATH = "/api/chat"
_PARAGRAPH = "\n\n"
_STOP_REASONS = {"stop": StopReason.END_TURN, "length": StopReason.MAX_TOKENS}


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


def _image(block: ImageBlock) -> str:
    return base64.b64encode(block.data).decode("ascii")


def _texts(blocks: tuple[Any, ...]) -> str:
    return _PARAGRAPH.join(b.text for b in blocks if isinstance(b, TextBlock))


def _tool_names(request: ModelRequest) -> dict[str, str]:
    return {
        block.id: block.name
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolUseBlock)
    }


def _tool_message(result: ToolResultBlock, names: dict[str, str]) -> dict[str, Any]:
    name = names.get(result.tool_use_id)
    if name is None:
        # A result for a call the history does not hold: refused before any call.
        raise ProviderError(ProviderFailure.REJECTED)
    text = _texts(result.content)
    entry: dict[str, Any] = {
        "role": "tool",
        "tool_name": name,
        "content": f"Error: {text}" if result.is_error else text,
    }
    images = [_image(b) for b in result.content if isinstance(b, ImageBlock)]
    if images:
        entry["images"] = images
    return entry


def _messages(message: Message, names: dict[str, str]) -> list[dict[str, Any]]:
    out = [_tool_message(b, names) for b in message.content if isinstance(b, ToolResultBlock)]
    content = [b for b in message.content if not isinstance(b, ToolResultBlock | CacheBreakpoint)]
    if not content and out:
        return out
    entry: dict[str, Any] = {"role": message.role, "content": _texts(tuple(content))}
    images = [_image(b) for b in content if isinstance(b, ImageBlock)]
    if images:
        entry["images"] = images
    calls = [b for b in content if isinstance(b, ToolUseBlock)]
    if calls:
        entry["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.input}} for call in calls
        ]
    return [*out, entry]


def build_chat_body(request: ModelRequest, *, num_ctx: int | None) -> dict[str, Any]:
    """The /api/chat body for ``request``, streaming, with ``num_ctx`` when given."""
    messages: list[dict[str, Any]] = []
    system = _texts(request.system)
    if system:
        messages.append({"role": "system", "content": system})
    names = _tool_names(request)
    for message in request.messages:
        messages += _messages(message, names)
    options: dict[str, Any] = {"num_predict": request.max_output_tokens}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.stop_sequences:
        options["stop"] = list(request.stop_sequences)
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    body: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]
    if request.output_schema is not None:
        body["format"] = request.output_schema
    return body


def failure_for_status(status: int) -> ProviderFailure:
    """The failure class for an HTTP error status from Ollama."""
    if status in (408, 504):
        return ProviderFailure.TIMEOUT
    if status == 429 or status >= 500:
        return ProviderFailure.UNAVAILABLE
    return ProviderFailure.REJECTED


def _count(data: dict[str, Any], key: str) -> int:
    value = data.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _protocol()
    return value


def _usage(data: dict[str, Any]) -> Usage:
    prompt = _count(data, "prompt_eval_count")
    cached = _count(data, "prompt_eval_cached_count")
    if cached > prompt:
        raise _protocol()
    return Usage(
        input_tokens=prompt - cached,
        output_tokens=_count(data, "eval_count"),
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )


_INVALID: Any = object()


def _loads(text: str) -> Any:
    """The JSON value of ``text``, or ``_INVALID``.

    The decoding error is not raised from here: the caller raises its own error after
    this handler has finished, so the decoder's message (which quotes the text) is
    neither its cause nor its context.
    """
    try:
        return json.loads(text)
    except ValueError:
        return _INVALID


def _call_events(call_id: str, name: str, arguments: Any) -> list[StreamEvent] | None:
    """A tool call's started and finished events, or ``None`` if they are invalid."""
    try:
        return [
            ToolCallStarted(id=call_id, name=name),
            ToolCallFinished(id=call_id, name=name, input=arguments),
        ]
    except ValidationError:
        return None


class StreamMapper:
    """Turns /api/chat NDJSON lines into conduit stream events."""

    def __init__(self, *, new_call_id: Callable[[], str] = lambda: new_id("toolu")) -> None:
        self._new_call_id = new_call_id
        self._called = False
        self._stopped: StreamStopped | None = None

    def _calls(self, calls: Any) -> list[StreamEvent]:
        if not isinstance(calls, list):
            raise _protocol()
        events: list[StreamEvent] = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                raise _protocol()
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = _loads(arguments)
                if arguments is _INVALID:
                    raise _protocol()
            name = function.get("name")
            if not isinstance(name, str):
                raise _protocol()
            given = call.get("id")
            call_id = given if isinstance(given, str) and given else self._new_call_id()
            pair = _call_events(call_id, name, arguments)
            if pair is None:
                raise _protocol()
            events += pair
            self._called = True
        return events

    def feed(self, line: str) -> list[StreamEvent]:
        """The events one NDJSON line carries (none for a blank line)."""
        if not line.strip():
            return []
        if self._stopped is not None:
            raise _protocol()
        data = _loads(line)
        if not isinstance(data, dict):
            raise _protocol()
        if "error" in data:
            raise ProviderError(ProviderFailure.UNAVAILABLE)
        events: list[StreamEvent] = []
        message = data.get("message") or {}
        if not isinstance(message, dict):
            raise _protocol()
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise _protocol()
        if content:
            events.append(TextDelta(text=content))
        if message.get("tool_calls"):
            events += self._calls(message["tool_calls"])
        if data.get("done") is True:
            reason = _STOP_REASONS.get(str(data.get("done_reason", "stop")), StopReason.OTHER)
            if reason is StopReason.END_TURN and self._called:
                reason = StopReason.TOOL_USE
            self._stopped = StreamStopped(stop_reason=reason, usage=_usage(data))
            events.append(self._stopped)
        return events

    def finish(self) -> list[StreamEvent]:
        """Called when the body ends: a stream that never said ``done`` is broken."""
        if self._stopped is None:
            raise _protocol()
        return []
