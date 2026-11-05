"""Chat Completions wire format: requests out, stream chunks in.

Requests
    System blocks become one ``system`` message. User text and images become content
    parts (images as base64 data URLs). Assistant tool use becomes ``tool_calls`` with
    JSON-encoded arguments. Tool results become ``tool`` messages placed straight after
    the assistant turn; Chat Completions gives tool messages text only and no error
    flag, so an error result is prefixed ``Error: `` and a result's images follow in a
    user message. ``max_output_tokens`` is sent as ``max_completion_tokens``. Usage is
    asked for on the final chunk (``stream_options.include_usage``).

    ``CacheBreakpoint`` (and ``cache_tools``) are ignored: OpenAI, X.AI and Foundry
    cache long prompt prefixes automatically and take no cache markers. Cached tokens
    still come back in usage, as ``cache_read_tokens``.

    ``output_schema`` becomes a strict ``json_schema`` response format, only for a
    model the catalogue marks ``structured_output``; otherwise the call is refused as
    ``ProviderError(REJECTED)`` before it is made. Strict mode has the provider's own
    schema rules (every property required, ``additionalProperties: false``).

Stream chunks
    Chunks are read as mappings (SDK chunk objects are dumped to JSON first). Text
    deltas (and refusal deltas) are passed on as they arrive; tool calls start when
    their id and name arrive and finish, with arguments assembled from every delta,
    when the stream ends. The final ``StreamStopped`` carries the usage chunk's counts.

Usage
    ``prompt_tokens`` includes cached tokens, so ``input_tokens`` is prompt tokens minus
    ``prompt_tokens_details.cached_tokens``, ``cache_read_tokens`` is the cached count,
    and ``cache_write_tokens`` is zero (these services do not bill or report writes).
    ``completion_tokens`` (reasoning included) is ``output_tokens``.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import (
    TOOL_NAME_PATTERN,
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

_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "content_filter": StopReason.REFUSAL,
}


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


# Requests -------------------------------------------------------------------------


def _image_part(block: ImageBlock) -> dict[str, Any]:
    data = base64.b64encode(block.data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{block.media_type};base64,{data}"}}


def _arguments(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _user_messages(message: Message) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    tool_images: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(_image_part(block))
        elif isinstance(block, ToolResultBlock):
            text = "".join(b.text for b in block.content if isinstance(b, TextBlock))
            tools.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": f"Error: {text}" if block.is_error else text,
                }
            )
            images = [b for b in block.content if isinstance(b, ImageBlock)]
            if images:
                tool_images.append(
                    {"type": "text", "text": f"Image from tool result {block.tool_use_id}:"}
                )
                tool_images.extend(_image_part(image) for image in images)
        elif isinstance(block, ToolUseBlock):
            raise ProviderError(ProviderFailure.REJECTED)  # a user cannot call tools
        # CacheBreakpoint: ignored (automatic caching).
    messages = tools
    if tool_images or parts:
        messages.append({"role": "user", "content": tool_images + parts})
    return messages


def _assistant_message(message: Message) -> dict[str, Any]:
    text: list[str] = []
    calls: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text.append(block.text)
        elif isinstance(block, ToolUseBlock):
            calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": _arguments(block.input)},
                }
            )
        elif isinstance(block, CacheBreakpoint):
            continue
        else:
            raise ProviderError(ProviderFailure.REJECTED)  # images or results from the model
    body: dict[str, Any] = {"role": "assistant", "content": "".join(text) if text else None}
    if calls:
        body["tool_calls"] = calls
    return body


def build_request(request: ModelRequest, *, structured_output: bool) -> dict[str, Any]:
    """The Chat Completions body for ``request`` (without the SDK's ``timeout``)."""
    messages: list[dict[str, Any]] = []
    system = "".join(b.text for b in request.system if isinstance(b, TextBlock))
    if system:
        messages.append({"role": "system", "content": system})
    for message in request.messages:
        if message.role == "user":
            messages.extend(_user_messages(message))
        else:
            messages.append(_assistant_message(message))
    body: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_completion_tokens": request.max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
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
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.stop_sequences:
        body["stop"] = list(request.stop_sequences)
    if request.output_schema is not None:
        if not structured_output:
            raise ProviderError(ProviderFailure.REJECTED)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": request.output_schema, "strict": True},
        }
    return body


# Usage ----------------------------------------------------------------------------


def _count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _protocol()
    return value


def usage_from_openai(usage: Mapping[str, Any]) -> Usage:
    """Usage from a Chat Completions ``usage`` object, cached tokens split out."""
    prompt = _count(usage.get("prompt_tokens"))
    completion = _count(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    cached_value = details.get("cached_tokens") if isinstance(details, Mapping) else None
    cached = 0 if cached_value is None else _count(cached_value)
    if cached > prompt:
        raise _protocol()
    return Usage(
        input_tokens=prompt - cached,
        output_tokens=completion,
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )


# Stream ---------------------------------------------------------------------------


@dataclass
class _ToolCall:
    id: str
    name: str
    arguments: list[str] = field(default_factory=list)

    def finished(self) -> ToolCallFinished:
        text = "".join(self.arguments).strip()
        try:
            value = json.loads(text) if text else {}
        except json.JSONDecodeError:
            raise _protocol() from None
        if not isinstance(value, dict):
            raise _protocol()
        return ToolCallFinished(id=self.id, name=self.name, input=value)


class StreamMapper:
    """Chat Completions chunks, one at a time, to conduit stream events."""

    def __init__(self) -> None:
        self._calls: dict[int, _ToolCall] = {}
        self._finish: str | None = None
        self._usage: Usage | None = None
        self._refused = False

    def feed(self, chunk: Mapping[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        usage = chunk.get("usage")
        if usage:
            self._usage = usage_from_openai(usage)  # the last report wins; never summed
        for choice in chunk.get("choices") or ():
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                events.append(TextDelta(text=content))
            refusal = delta.get("refusal")
            if refusal:
                self._refused = True
                events.append(TextDelta(text=refusal))
            for call in delta.get("tool_calls") or ():
                events.extend(self._tool_delta(call))
            finish = choice.get("finish_reason")
            if finish:
                self._finish = finish
        return events

    def _tool_delta(self, call: Mapping[str, Any]) -> list[StreamEvent]:
        index = call.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise _protocol()
        function = call.get("function") or {}
        call_id = call.get("id")
        events: list[StreamEvent] = []
        state = self._calls.get(index)
        if state is None:
            name = function.get("name")
            if not isinstance(call_id, str) or not call_id or len(call_id) > 256:
                raise _protocol()
            if not isinstance(name, str) or not re.fullmatch(TOOL_NAME_PATTERN, name):
                raise _protocol()
            state = self._calls[index] = _ToolCall(id=call_id, name=name)
            events.append(ToolCallStarted(id=call_id, name=name))
        elif call_id and call_id != state.id:
            raise _protocol()
        arguments = function.get("arguments")
        if arguments:
            if not isinstance(arguments, str):
                raise _protocol()
            state.arguments.append(arguments)
        return events

    def finish(self) -> list[StreamEvent]:
        """The closing events, once the stream has ended."""
        if self._finish is None or self._usage is None:
            raise _protocol()
        events: list[StreamEvent] = [self._calls[i].finished() for i in sorted(self._calls)]
        if self._refused:
            reason = StopReason.REFUSAL
        else:
            reason = _STOP_REASONS.get(self._finish, StopReason.OTHER)
        events.append(StreamStopped(stop_reason=reason, usage=self._usage))
        return events
