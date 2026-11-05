"""Mapping between conduit's types and the Bedrock Converse wire format.

Requests: system blocks, messages (text, images, tool use and tool results) and tool
specifications become Converse parameters. A ``CacheBreakpoint`` becomes a
``cachePoint`` only when the catalogue says the model supports prompt caching, and at
most four are sent (Bedrock's limit for Claude models): earlier breakpoints are
dropped first, since a cache read still finds the longest prefix ending at a later
one. Bedrock processes cache points in the order tools, system, messages.

Responses: ConverseStream events, or a whole Converse response, become conduit stream
events ending with one ``StreamStopped`` whose usage comes from the final metadata.
Bedrock's ``inputTokens`` counts uncached input only, so the four counts map one to
one: ``inputTokens``, ``outputTokens``, ``cacheReadInputTokens``,
``cacheWriteInputTokens``. Reasoning content is not surfaced.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
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

CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}
MAX_CACHE_POINTS = 4

_IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "guardrail_intervened": StopReason.REFUSAL,
    "content_filtered": StopReason.REFUSAL,
    "refusal": StopReason.REFUSAL,
}


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


class _CachePoints:
    """Hands out cache points in processing order, keeping only the last four."""

    def __init__(self, request: ModelRequest, *, enabled: bool) -> None:
        total = 0
        if enabled:
            total += int(request.cache_tools and bool(request.tools))
            total += sum(isinstance(b, CacheBreakpoint) for b in request.system)
            total += sum(
                isinstance(b, CacheBreakpoint) for m in request.messages for b in m.content
            )
        self._skip = max(0, total - MAX_CACHE_POINTS)
        self._enabled = enabled

    def take(self) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        if self._skip:
            self._skip -= 1
            return []
        return [CACHE_POINT]


def _image(block: ImageBlock) -> dict[str, Any]:
    return {"image": {"format": _IMAGE_FORMATS[block.media_type], "source": {"bytes": block.data}}}


def _content(block: Any, points: _CachePoints) -> list[dict[str, Any]]:
    if isinstance(block, TextBlock):
        return [{"text": block.text}]
    if isinstance(block, ImageBlock):
        return [_image(block)]
    if isinstance(block, CacheBreakpoint):
        return points.take()
    if isinstance(block, ToolUseBlock):
        return [{"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.input}}]
    if isinstance(block, ToolResultBlock):
        return [
            {
                "toolResult": {
                    "toolUseId": block.tool_use_id,
                    "content": [
                        {"text": b.text} if isinstance(b, TextBlock) else _image(b)
                        for b in block.content
                    ],
                    "status": "error" if block.is_error else "success",
                }
            }
        ]
    raise TypeError(f"unsupported content block {type(block).__name__}")


def _message(message: Message, points: _CachePoints) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in message.content:
        content.extend(_content(block, points))
    return {"role": message.role, "content": content}


def build_converse_request(request: ModelRequest, *, prompt_caching: bool) -> dict[str, Any]:
    """Converse (and ConverseStream) parameters for ``request``."""
    if request.output_schema is not None:
        # Structured output is not mapped to Converse yet; refuse rather than ignore it.
        raise ProviderError(ProviderFailure.REJECTED)
    points = _CachePoints(request, enabled=prompt_caching)
    params: dict[str, Any] = {"modelId": request.model}
    # Processing order is tools, system, messages, so cache points are taken in that order.
    tools: list[dict[str, Any]] = [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.input_schema},
            }
        }
        for tool in request.tools
    ]
    if tools and request.cache_tools:
        tools.extend(points.take())
    system: list[dict[str, Any]] = []
    for block in request.system:
        system.extend(_content(block, points))
    if system:
        params["system"] = system
    params["messages"] = [_message(m, points) for m in request.messages]
    if tools:
        params["toolConfig"] = {"tools": tools}
    inference: dict[str, Any] = {"maxTokens": request.max_output_tokens}
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    if request.stop_sequences:
        inference["stopSequences"] = list(request.stop_sequences)
    params["inferenceConfig"] = inference
    return params


def _count(raw: Mapping[str, Any], key: str, *, required: bool) -> int:
    if key not in raw:
        if required:
            raise _protocol()
        return 0
    value = raw[key]
    if type(value) is not int or value < 0:
        raise _protocol()
    return value


def usage_from_bedrock(raw: Mapping[str, Any]) -> Usage:
    """conduit's four usage counts from a Converse ``usage`` block."""
    return Usage(
        input_tokens=_count(raw, "inputTokens", required=True),
        output_tokens=_count(raw, "outputTokens", required=True),
        cache_read_tokens=_count(raw, "cacheReadInputTokens", required=False),
        cache_write_tokens=_count(raw, "cacheWriteInputTokens", required=False),
    )


def stop_reason_from_bedrock(raw: str) -> StopReason:
    return _STOP_REASONS.get(raw, StopReason.OTHER)


def _tool_input(text: str) -> dict[str, Any]:
    if not text:
        return {}
    value: Any = None
    with contextlib.suppress(json.JSONDecodeError):  # its message quotes the input
        value = json.loads(text)
    if not isinstance(value, dict):
        raise _protocol()
    return value


@dataclass
class _OpenCall:
    id: str
    name: str
    input: list[str] = field(default_factory=list)


class StreamMapper:
    """Turns ConverseStream events into conduit stream events, one event at a time."""

    def __init__(self) -> None:
        self._calls: dict[int, _OpenCall] = {}
        self._stop: StopReason | None = None
        self._usage: Usage | None = None

    def feed(self, raw: Mapping[str, Any]) -> Iterator[StreamEvent]:
        # ValueError covers pydantic's ValidationError, whose text quotes the input.
        broken = False
        try:
            yield from self._feed(raw)
        except (
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
        ):
            broken = True
        if broken:
            raise _protocol()

    def _feed(self, raw: Mapping[str, Any]) -> Iterator[StreamEvent]:
        if "contentBlockStart" in raw:
            body = raw["contentBlockStart"]
            use = body.get("start", {}).get("toolUse")
            if use is not None:
                index = body["contentBlockIndex"]
                if index in self._calls:
                    raise _protocol()
                call = _OpenCall(id=use["toolUseId"], name=use["name"])
                self._calls[index] = call
                yield ToolCallStarted(id=call.id, name=call.name)
        elif "contentBlockDelta" in raw:
            body = raw["contentBlockDelta"]
            delta = body["delta"]
            if "text" in delta:
                if delta["text"]:
                    yield TextDelta(text=delta["text"])
            elif "toolUse" in delta:
                open_call = self._calls.get(body["contentBlockIndex"])
                if open_call is None:
                    raise _protocol()
                open_call.input.append(delta["toolUse"].get("input", ""))
            # reasoningContent and other deltas are not surfaced.
        elif "contentBlockStop" in raw:
            finished = self._calls.pop(raw["contentBlockStop"]["contentBlockIndex"], None)
            if finished is not None:
                yield ToolCallFinished(
                    id=finished.id, name=finished.name, input=_tool_input("".join(finished.input))
                )
        elif "messageStop" in raw:
            self._stop = stop_reason_from_bedrock(raw["messageStop"]["stopReason"])
        elif "metadata" in raw:
            if self._usage is not None:
                raise _protocol()  # one metadata event per stream; never replace usage
            self._usage = usage_from_bedrock(raw["metadata"]["usage"])

    def finish(self) -> Iterator[StreamEvent]:
        """The final ``StreamStopped``, once the stream has ended."""
        if self._stop is None or self._usage is None or self._calls:
            raise _protocol()
        yield StreamStopped(stop_reason=self._stop, usage=self._usage)


def events_from_converse_response(response: Mapping[str, Any]) -> list[StreamEvent]:
    """The stream events a whole Converse response describes."""
    try:
        content: Sequence[Mapping[str, Any]] = response["output"]["message"]["content"]
        events: list[StreamEvent] = []
        for block in content:
            if "text" in block:
                if block["text"]:
                    events.append(TextDelta(text=block["text"]))
            elif "toolUse" in block:
                use = block["toolUse"]
                if not isinstance(use["input"], dict):
                    raise _protocol()
                events.append(ToolCallStarted(id=use["toolUseId"], name=use["name"]))
                events.append(
                    ToolCallFinished(id=use["toolUseId"], name=use["name"], input=use["input"])
                )
        events.append(
            StreamStopped(
                stop_reason=stop_reason_from_bedrock(response["stopReason"]),
                usage=usage_from_bedrock(response["usage"]),
            )
        )
    except (
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
    ):
        broken = True
    else:
        broken = False
    if broken:
        raise _protocol()
    return events
