"""Results a tool's executor already frames as untrusted are not framed a second time.

The declaration comes from the trusted side (the tool object, or the product's predicate),
never from the content: text that merely looks framed is framed again like any other.
The MCP tool bridge frames its results itself, so a product running MCP tools passes
``prefamed=lambda tool: isinstance(tool, McpTool)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from conduit.agent import AgentRunner, AgentTask, tool_results_prefamed
from conduit.core.types import Message, ToolResultBlock
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.unit.agent.helpers import (
    Executor,
    SampleTool,
    ScriptLog,
    result_text,
    runtime,
    task_tools,
)

FRAMED = '<untrusted-content source="mcp" server="docs" tool="search">\nhits\n</untrusted-content>'


@dataclass
class FramingTool(SampleTool):
    untrusted_framed: Any = True


async def _sent(tool: SampleTool, **runner: Any) -> str:
    log = ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name=tool.name),)),
            lambda results: StubTurn(text="ok"),
        )
    )
    rt = await runtime(log)
    tools = {"chat": task_tools([tool], Executor(answers={tool.name: FRAMED}))}
    await AgentRunner(rt, tools=tools, **runner).run(
        AgentTask(task_class="chat", messages=(Message.user("go"),))
    )
    block = log.requests[1].messages[2].content[0]
    assert isinstance(block, ToolResultBlock)
    return result_text(block)


async def test_a_tool_declaring_its_results_framed_is_sent_as_it_is() -> None:
    assert await _sent(FramingTool("search")) == FRAMED


async def test_content_that_only_looks_framed_is_framed_again() -> None:
    sent = await _sent(SampleTool("search"))
    assert sent.startswith('<<UNTRUSTED source="search"')
    assert FRAMED.split("\n")[1] in sent


@pytest.mark.parametrize("mark", [1, "yes", None, False])
async def test_only_a_literal_true_declares_results_framed(mark: object) -> None:
    sent = await _sent(FramingTool("search", untrusted_framed=mark))
    assert sent.startswith("<<UNTRUSTED ")


async def test_a_product_predicate_can_declare_results_framed() -> None:
    sent = await _sent(SampleTool("search"), prefamed=lambda tool: tool.name == "search")
    assert sent == FRAMED


def test_the_default_predicate_reads_the_tools_declaration() -> None:
    assert tool_results_prefamed(FramingTool("search"))
    assert not tool_results_prefamed(SampleTool("search"))
    assert not tool_results_prefamed(FramingTool("search", untrusted_framed="true"))
