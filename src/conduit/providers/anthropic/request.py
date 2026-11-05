"""Mapping a conduit ``ModelRequest`` to the Messages API's parameters.

Cache breakpoints become ``cache_control`` on the block before them in prompt order
(tools, then system, then messages), so a breakpoint opening a message marks the end of
the one before it. ``cache_tools`` marks the last tool. A breakpoint with nothing before
it is ignored, and consecutive breakpoints count once.

The Messages API takes at most four ``cache_control`` blocks per request. Beyond that
the breakpoints are merged, never refused: the first three and the last are kept, and
each dropped breakpoint's prefix is still covered by the next kept one. The first
breakpoints mark the most stable prefix (tools, system prompt, loaded skills) and the
last marks the conversation so far, which is what a following call reads back.

Sampling parameters are never sent: API version 2026-10-21 rejects ``temperature``,
``top_p`` and ``top_k`` on every model, so ``ModelRequest.temperature`` is not used.

Provider state: an assistant message's ``anthropic`` state blocks are the thinking and
redacted thinking blocks of the response it came from (see
``conduit.providers.anthropic.stream``), and each goes back unchanged in its place, as
the API requires for tool use with thinking. The API refuses ``cache_control`` on a
thinking block, so a breakpoint after one marks the block before it. Every other
kind's state is left out, so a conversation can move between providers. Anthropic
state that is not a whole thinking block is refused as ``REJECTED`` before any call,
with nothing of the payload in the error.
"""

from __future__ import annotations

import base64
import contextlib
import json
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    ModelRequest,
    ProviderStateBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

#: The Messages API's limit on ``cache_control`` blocks in one request.
MAX_CACHE_BREAKPOINTS = 4
_EPHEMERAL = {"type": "ephemeral"}
#: The provider kind of this adapter's state blocks.
KIND = "anthropic"
#: Each thinking block type and the string fields it carries, exactly.
THINKING_FIELDS: dict[str, frozenset[str]] = {
    "thinking": frozenset({"type", "thinking", "signature"}),
    "redacted_thinking": frozenset({"type", "data"}),
}

Block = dict[str, Any]


def is_thinking_block(block: Any) -> bool:
    """Whether ``block`` is a whole thinking or redacted thinking block, as the API sends."""
    if not isinstance(block, dict):
        return False
    fields = THINKING_FIELDS.get(block.get("type"))  # type: ignore[arg-type]
    return (
        fields is not None
        and set(block) == fields
        and all(isinstance(block[name], str) for name in fields)
    )


def _thinking(state: ProviderStateBlock) -> Block:
    """The thinking block an ``anthropic`` state block holds, or REJECTED."""
    block: Any = None
    with contextlib.suppress(ValueError):  # UnicodeDecodeError and JSONDecodeError
        block = json.loads(state.payload.decode("utf-8"))
    # Raised here, after the handler: the decoding error quotes the payload.
    if not is_thinking_block(block) or block["type"] != state.state_type:
        raise ProviderError(ProviderFailure.REJECTED)
    result: Block = block
    return result


def _image(block: ImageBlock) -> Block:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": block.media_type,
            "data": base64.standard_b64encode(block.data).decode("ascii"),
        },
    }


def _block(block: TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock) -> Block:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return _image(block)
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": [_block(item) for item in block.content],
        "is_error": block.is_error,
    }


class _Marks:
    """The blocks breakpoints fall on, in prompt order."""

    def __init__(self) -> None:
        self.last: Block | None = None
        self.marked: list[Block] = []

    def emitted(self, block: Block) -> Block:
        self.last = block
        return block

    def breakpoint(self) -> None:
        if self.last is not None and (not self.marked or self.marked[-1] is not self.last):
            self.marked.append(self.last)

    def apply(self) -> None:
        kept = self.marked
        if len(kept) > MAX_CACHE_BREAKPOINTS:
            kept = kept[: MAX_CACHE_BREAKPOINTS - 1] + kept[-1:]
        for block in kept:
            block["cache_control"] = dict(_EPHEMERAL)


def build_params(request: ModelRequest, *, structured_output: bool) -> dict[str, Any]:
    """The keyword arguments for ``client.messages.create``, streaming."""
    marks = _Marks()
    params: dict[str, Any] = {"model": request.model, "max_tokens": request.max_output_tokens}

    if request.tools:
        params["tools"] = [
            marks.emitted(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": dict(tool.input_schema),
                }
            )
            for tool in request.tools
        ]
        if request.cache_tools:
            marks.breakpoint()

    system: list[Block] = []
    for item in request.system:
        if isinstance(item, CacheBreakpoint):
            marks.breakpoint()
        else:
            system.append(marks.emitted(_block(item)))
    if system:
        params["system"] = system

    messages: list[Block] = []
    for message in request.messages:
        content: list[Block] = []
        for block in message.content:
            if isinstance(block, ProviderStateBlock):
                if block.provider_kind == KIND:
                    content.append(_thinking(block))  # never a cache_control target
                continue  # another kind's state: left out
            if isinstance(block, CacheBreakpoint):
                marks.breakpoint()
            else:
                content.append(marks.emitted(_block(block)))
        messages.append({"role": message.role, "content": content})
    params["messages"] = messages

    if request.stop_sequences:
        params["stop_sequences"] = list(request.stop_sequences)
    if request.output_schema is not None and structured_output:
        params["output_config"] = {
            "format": {"type": "json_schema", "schema": dict(request.output_schema)}
        }
    marks.apply()
    params["stream"] = True
    params["timeout"] = float(request.timeout_s)
    return params
