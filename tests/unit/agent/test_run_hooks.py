"""Per-run hooks: the system-prompt hook and the tool-executor wrapper.

They are the seam the skills layer uses without the runner depending on it:
``system_hook=session.system`` rebuilds the system blocks (base, listing, loaded skills,
cache breakpoint) before every model call, and ``tool_wrapper=with_use_skill`` answers
``use_skill`` itself and hands every other tool to the product's executor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from conduit.agent import AgentRunner, AgentTask, SystemPromptRefused
from conduit.agent.events import RunFinishedEvent
from conduit.agent.hooks import CompactionRequest
from conduit.core.tools import ToolCallContext, ToolExecutor, ToolOutcome
from conduit.core.types import CacheBreakpoint, Message, SystemBlock, TextBlock
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results, scripted
from tests.unit.agent.helpers import (
    Executor,
    SampleTool,
    ScriptLog,
    Sink,
    result_text,
    runtime,
    task_tools,
)

BASE = (TextBlock(text="You are a careful assistant."),)


def _script() -> ScriptLog:
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="use_skill", input={"name": "triage"}),)),
            lambda results: StubTurn(text="done"),
        )
    )


class SystemHook:
    """Stands in for SkillSession.system: grows once a skill is loaded."""

    def __init__(self) -> None:
        self.bases: list[tuple[SystemBlock, ...]] = []
        self.loaded = False

    async def __call__(self, base: tuple[SystemBlock, ...]) -> tuple[SystemBlock, ...]:
        self.bases.append(base)
        extra: tuple[SystemBlock, ...] = (TextBlock(text="Skills: triage"),)
        if self.loaded:
            extra += (TextBlock(text="Skill triage: its instructions"),)
        return (*base, *extra, CacheBreakpoint())


def _wrapper(hook: SystemHook) -> Any:
    """Stands in for with_use_skill: runs use_skill itself, delegates everything else."""

    def wrap(executor: ToolExecutor) -> ToolExecutor:
        async def wrapped(
            tool: Any, arguments: dict[str, Any], context: ToolCallContext
        ) -> ToolOutcome:
            if tool.name == "use_skill":
                hook.loaded = True
                return ToolOutcome(ok=True, content="skill triage loaded")
            return await executor(tool, arguments, context)

        return wrapped

    return wrap


def _tools(executor: Executor) -> dict[str, Any]:
    return {"chat": task_tools([SampleTool("use_skill"), SampleTool("lookup")], executor)}


async def test_the_system_hook_rebuilds_the_system_before_every_call() -> None:
    log = _script()
    hook = SystemHook()
    executor = Executor()
    rt = await runtime(log)
    task = AgentTask(
        task_class="chat",
        system=BASE,
        messages=(Message.user("go"),),
        system_hook=hook,
        tool_wrapper=_wrapper(hook),
    )
    result = await AgentRunner(rt, tools=_tools(executor)).run(task)
    assert result.text == "done"
    assert hook.bases == [BASE, BASE]
    first, second = (r.system for r in log.requests)
    assert first == (*BASE, TextBlock(text="Skills: triage"), CacheBreakpoint())
    assert second == (
        *BASE,
        TextBlock(text="Skills: triage"),
        TextBlock(text="Skill triage: its instructions"),
        CacheBreakpoint(),
    )
    # use_skill was answered by the wrapper, never by the product's executor.
    assert executor.calls == []
    tool_result = log.requests[1].messages[2].content[0]
    assert "skill triage loaded" in result_text(tool_result)  # type: ignore[arg-type]


async def test_the_wrapper_hands_other_tools_to_the_products_executor() -> None:
    log = ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="lookup", input={"q": "x"}),)),
            lambda results: StubTurn(text="done"),
        )
    )
    hook = SystemHook()
    executor = Executor(answers={"lookup": "found it"})
    rt = await runtime(log)
    task = AgentTask(task_class="chat", messages=(Message.user("go"),), tool_wrapper=_wrapper(hook))
    result = await AgentRunner(rt, tools=_tools(executor)).run(task)
    assert result.text == "done"
    assert [c[0] for c in executor.calls] == ["lookup"]
    assert executor.calls[0][2].run_id == result.run_id


async def test_without_hooks_the_task_system_is_sent_as_it_is() -> None:
    log = ScriptLog(scripted([StubTurn(text="ok")]))
    rt = await runtime(log)
    await AgentRunner(rt).run(
        AgentTask(task_class="chat", system=BASE, messages=(Message.user("go"),))
    )
    assert log.requests[0].system == BASE


async def test_the_compaction_hook_sees_the_system_the_call_will_send() -> None:
    seen: list[tuple[SystemBlock, ...]] = []

    class Hook:
        async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
            seen.append(request.system)
            return None

    hook = SystemHook()
    rt = await runtime(ScriptLog(scripted([StubTurn(text="ok")])))
    await AgentRunner(rt, compaction=Hook()).run(
        AgentTask(task_class="chat", system=BASE, messages=(Message.user("go"),), system_hook=hook)
    )
    assert seen == [(*BASE, TextBlock(text="Skills: triage"), CacheBreakpoint())]


@pytest.mark.parametrize("returned", ["text", [object()], (Message.user("x"),), None])
async def test_a_system_hook_returning_anything_but_system_blocks_is_refused(
    returned: object,
) -> None:
    async def bad(base: tuple[SystemBlock, ...]) -> Any:
        return returned

    sink = Sink()
    log = ScriptLog(scripted([StubTurn(text="ok")]))
    rt = await runtime(log)
    with pytest.raises(SystemPromptRefused):
        await AgentRunner(rt, events=sink).run(
            AgentTask(task_class="chat", messages=(Message.user("go"),), system_hook=bad)
        )
    assert log.requests == []
    assert [e.error_type for e in sink.of(RunFinishedEvent)] == ["SystemPromptRefused"]


async def test_an_empty_system_from_the_hook_is_allowed() -> None:
    async def empty(base: tuple[SystemBlock, ...]) -> tuple[SystemBlock, ...]:
        return ()

    log = ScriptLog(scripted([StubTurn(text="ok")]))
    rt = await runtime(log)
    await AgentRunner(rt).run(
        AgentTask(task_class="chat", system=BASE, messages=(Message.user("go"),), system_hook=empty)
    )
    assert log.requests[0].system == ()
