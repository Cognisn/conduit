"""Concurrent runs on one runner keep their own state: budgets, tools, transcripts, events.

The runs are genuinely concurrent: every tool call waits on a barrier that only opens once
every run has reached it, so the test fails (times out) unless the runs interleave.
"""

from __future__ import annotations

import asyncio
from typing import Any

from conduit.agent import AgentRunner, AgentTask, BudgetKind, RunBudget, RunStatus
from conduit.agent.events import RunFinishedEvent
from conduit.core.tools import ToolCallContext, ToolOutcome
from conduit.core.transcript import MessageEntry
from conduit.core.types import Message, TextBlock, ToolResultBlock
from conduit.providers.stub import StubToolCall, StubTurn
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.agent.helpers import SampleTool, ScriptLog, Sink, result_text, runtime, task_tools

RUNS = 8


def _script(request: Any, index: int) -> StubTurn:
    """Look up the run's own tag twice, then answer with what came back."""
    last = request.messages[-1]
    results = [b for b in last.content if isinstance(b, ToolResultBlock)]
    first_text = request.messages[0].content[0]
    assert isinstance(first_text, TextBlock)
    tag = first_text.text
    seen = sum(1 for m in request.messages for b in m.content if isinstance(b, ToolResultBlock))
    if seen < 2:
        return StubTurn(tool_calls=(StubToolCall(name="lookup", input={"tag": tag}),))
    body = result_text(results[0]).split("\n")[1]
    return StubTurn(text=f"{tag}:{body}")


class BarrierExecutor:
    def __init__(self, parties: int) -> None:
        self.barrier = asyncio.Barrier(parties)
        self.contexts: list[ToolCallContext] = []

    async def __call__(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome:
        self.contexts.append(context)
        await self.barrier.wait()
        return ToolOutcome(ok=True, content=f"value-for-{arguments['tag']}")


async def test_concurrent_runs_interleave_and_keep_their_own_state() -> None:
    sink = Sink()
    store = MemoryTranscriptStore()
    rt = await runtime(ScriptLog(_script))
    executor = BarrierExecutor(RUNS)
    runner = AgentRunner(
        rt,
        tools={"chat": task_tools([SampleTool("lookup")], executor)},
        events=sink,
        transcripts=store,
    )
    tasks = [
        AgentTask(
            task_class="chat",
            messages=(Message.user(f"tag{i}"),),
            session_id=f"session-{i}",
            budget=RunBudget(max_model_calls=3, max_tool_calls=2),
        )
        for i in range(RUNS)
    ]
    results = await asyncio.wait_for(asyncio.gather(*(runner.run(t) for t in tasks)), 20)
    for i, result in enumerate(results):
        assert result.status is RunStatus.COMPLETED
        assert result.text == f"tag{i}:value-for-tag{i}"
        assert (result.model_calls, result.tool_calls) == (3, 2)
        transcript = await store.load(f"session-{i}")
        assert [e.message for e in transcript.entries if isinstance(e, MessageEntry)] == list(
            result.messages
        )
    assert len({r.run_id for r in results}) == RUNS
    # Each tool call was made under its own run.
    by_run: dict[str, int] = {}
    for context in executor.contexts:
        by_run[context.run_id] = by_run.get(context.run_id, 0) + 1
    assert by_run == {r.run_id: 2 for r in results}
    finished = sink.of(RunFinishedEvent)
    assert {e.run_id for e in finished} == {r.run_id for r in results}


async def test_one_runs_exhausted_budget_never_touches_another() -> None:
    rt = await runtime(ScriptLog(_script))
    executor = BarrierExecutor(2)
    runner = AgentRunner(rt, tools={"chat": task_tools([SampleTool("lookup")], executor)})
    tight = AgentTask(
        task_class="chat",
        messages=(Message.user("tight"),),
        budget=RunBudget(max_model_calls=2),
    )
    loose = AgentTask(
        task_class="chat",
        messages=(Message.user("loose"),),
        budget=RunBudget(max_model_calls=5),
    )
    # Both runs meet at the barrier on each of their two tool calls, so they interleave;
    # the tight run then stops on its third model call, and the loose run completes.
    a, b = await asyncio.wait_for(asyncio.gather(runner.run(tight), runner.run(loose)), 20)
    assert a.budget is BudgetKind.MODEL_CALLS
    assert a.model_calls == 2
    assert b.status is RunStatus.COMPLETED
    assert b.budget is None
    assert b.model_calls == 3
