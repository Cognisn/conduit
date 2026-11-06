"""Reading a conversation for compaction: token estimates, the cut, and a text rendering.

``estimate_tokens`` counts about four characters a token over the prompt as a provider
would see it (tool definitions, system blocks and message blocks as JSON), the same rule
the stub provider uses, so thresholds behave the same against the stub and in tests.
A product with a better count passes its own estimator to the ``Compactor``.

``find_cut`` chooses where the kept tail begins. The tail always begins with an
assistant message and holds the last ``keep_turns`` assistant messages and everything
after them, so a tool use and its result are never split, and the summary (a user
message) followed by the tail keeps the conversation's roles alternating.

``render_messages`` writes messages as plain text for a summary: tool calls by name,
tool results trimmed or left out, images named, cache markers and provider state left
out. It is deterministic, so the same messages always give the same text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from conduit.core.errors import ConfigurationError
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ProviderStateBlock,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

__all__ = [
    "DEFAULT_MAX_MESSAGE_CHARS",
    "TOOL_RESULT_OMITTED",
    "TRUNCATED",
    "estimate_tokens",
    "find_cut",
    "render_messages",
]

TOOL_RESULT_OMITTED = "[tool result omitted]"
TRUNCATED = "\n... [truncated]"
DEFAULT_MAX_MESSAGE_CHARS = 3_000


def _tokens(characters: int) -> int:
    return (characters + 3) // 4


def estimate_tokens(
    *,
    system: Sequence[SystemBlock] = (),
    tools: Sequence[ToolSpec] = (),
    messages: Sequence[Message] = (),
) -> int:
    """About how many input tokens a prompt of these parts takes (four characters a token)."""
    characters = 0
    if tools:
        characters += len(json.dumps([t.model_dump(mode="json") for t in tools]))
    for block in system:
        if isinstance(block, TextBlock):
            characters += len(block.text)
    for message in messages:
        for content in message.content:
            if not isinstance(content, CacheBreakpoint):
                characters += len(message.role) + 1 + len(content.model_dump_json())
    return _tokens(characters)


def find_cut(messages: Sequence[Message], keep_turns: int) -> int | None:
    """The index the kept tail starts at, or ``None`` when fewer than two messages would go.

    The tail starts at the ``keep_turns``-th assistant message from the end.
    """
    if keep_turns < 1:
        raise ConfigurationError("keep_turns must be at least 1")
    seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            seen += 1
            if seen == keep_turns:
                return index if index >= 2 else None
    return None


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - len(TRUNCATED))] + TRUNCATED


def _result_text(block: ToolResultBlock, limit: int | None) -> str:
    if limit is None:
        return TOOL_RESULT_OMITTED
    parts = [c.text if isinstance(c, TextBlock) else "[image]" for c in block.content]
    marker = "error" if block.is_error else "result"
    return f"[tool {marker}: {_trim(' '.join(parts), limit)}]"


def render_messages(
    messages: Sequence[Message],
    *,
    tool_result_chars: int | None = 500,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> str:
    """``messages`` as plain text, one paragraph each.

    ``tool_result_chars`` trims each tool result to that many characters, or leaves every
    result out when ``None``; ``max_message_chars`` trims each message.
    """
    paragraphs: list[str] = []
    for message in messages:
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"[tool call: {block.name}]")
            elif isinstance(block, ToolResultBlock):
                parts.append(_result_text(block, tool_result_chars))
            elif isinstance(block, ImageBlock):
                parts.append("[image]")
            elif isinstance(block, CacheBreakpoint | ProviderStateBlock):
                continue
        text = _trim("\n".join(parts), max_message_chars)
        paragraphs.append(f"[{message.role.upper()}]: {text}")
    return "\n\n".join(paragraphs)
