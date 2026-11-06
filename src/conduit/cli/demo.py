"""The demonstration's stub script and tools: tool use without a live model.

The stub provider's ``demo`` script answers like a model that uses tools on request:

- a message ``call <tool> <JSON arguments>`` naming a tool the request offers is
  answered with that tool call (``call add {"a": 2, "b": 3}``,
  ``call use_skill {"name": "release-notes"}``, ``call docs__echo {"text": "hi"}``);
- once the tool results come back, it answers with their text;
- anything else is echoed back, as the stub's ``echo`` script does.

``add`` and ``word_count`` are the interface's own tools, run by ``demo_executor``.
Everything here is the interface's policy, a product's in miniature, built on conduit's
public API.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from conduit import ModelRequest, TextBlock, ToolResultBlock
from conduit.core.tools import ToolCallContext, ToolOutcome
from conduit.providers.stub import StubToolCall, StubTurn, echo

_CALL = re.compile(r"^call\s+([A-Za-z][A-Za-z0-9_-]{0,63})\s*(.*)$", re.DOTALL)


def _latest_results(request: ModelRequest) -> tuple[ToolResultBlock, ...]:
    if not request.messages or request.messages[-1].role != "user":
        return ()
    return tuple(b for b in request.messages[-1].content if isinstance(b, ToolResultBlock))


def _last_user_text(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return "".join(b.text for b in message.content if isinstance(b, TextBlock))
    return ""


def demo_script(request: ModelRequest, index: int) -> StubTurn:
    """The ``demo`` stub script (see the module docstring)."""
    results = _latest_results(request)
    if results:
        texts = [
            "".join(part.text for part in result.content if isinstance(part, TextBlock))
            for result in results
        ]
        return StubTurn(text="Tool results:\n" + "\n".join(texts))
    found = _CALL.match(_last_user_text(request).strip())
    offered = {spec.name for spec in request.tools}
    if found is None or found.group(1) not in offered:
        return echo(request, index)
    raw = found.group(2).strip() or "{}"
    arguments: Any = None
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        arguments = None
    if not isinstance(arguments, dict):
        return StubTurn(text="The tool arguments must be a JSON object.")
    return StubTurn(tool_calls=(StubToolCall(name=found.group(1), input=arguments),))


@dataclass(frozen=True, slots=True)
class DemoTool:
    """One of the interface's own tools, declared as a product declares a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    task_classes: frozenset[str] = field(default_factory=lambda: frozenset({"chat"}))


ADD = DemoTool(
    name="add",
    description="Add two numbers and return the sum.",
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
)
WORD_COUNT = DemoTool(
    name="word_count",
    description="Count the words in a text.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
DEMO_TOOLS = (ADD, WORD_COUNT)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


async def demo_executor(
    tool: Any, arguments: dict[str, Any], context: ToolCallContext
) -> ToolOutcome:
    """Runs ``add`` and ``word_count``; any other tool is refused."""
    name = getattr(tool, "name", None)
    if name == ADD.name:
        a, b = _number(arguments.get("a")), _number(arguments.get("b"))
        if a is None or b is None:
            return ToolOutcome(
                ok=False, content="add needs two numbers.", error_code="input_invalid"
            )
        total = a + b
        return ToolOutcome(ok=True, content=f"{total:g}")
    if name == WORD_COUNT.name:
        text = arguments.get("text")
        if not isinstance(text, str):
            return ToolOutcome(
                ok=False, content="word_count needs a text.", error_code="input_invalid"
            )
        return ToolOutcome(ok=True, content=str(len(text.split())))
    return ToolOutcome(ok=False, content="No such tool.", error_code="unknown_tool")
