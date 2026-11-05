"""Budgets in a run: each is refused cleanly when exhausted, with a budget event."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from conduit.agent import AgentRunner, AgentTask, BudgetKind, RunBudget, RunFailure, RunStatus
from conduit.agent.budgets import TokenCount
from conduit.agent.events import BudgetExhaustedEvent, RunFinishedEvent
from conduit.core.provider import Provider
from conduit.core.runtime import start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.streaming import StreamEvent
from conduit.core.types import Message, ModelRequest, ToolResultBlock, ToolUseBlock, Usage
from conduit.providers.stub import (
    StubFactory,
    StubProvider,
    StubToolCall,
    StubTurn,
    echo,
    scripted,
)
from tests.unit.agent.helpers import (
    CHAT,
    REVIEW,
    Executor,
    SampleTool,
    ScriptLog,
    Sink,
    result_text,
    runtime,
    settings,
    task_tools,
)


def _looping() -> ScriptLog:
    """A model that calls the lookup tool on every turn, forever."""
    return ScriptLog(lambda request, index: StubTurn(tool_calls=(StubToolCall(name="lookup"),)))


def _task(budget: RunBudget) -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("go"),), budget=budget)


def _tools(executor: Executor | None = None) -> dict[str, object]:
    return {"chat": task_tools([SampleTool("lookup")], executor or Executor())}


async def _run(log: ScriptLog, budget: RunBudget, executor: Executor | None = None, **kw: object):  # type: ignore[no-untyped-def]
    sink = Sink()
    rt = await runtime(log)
    runner = AgentRunner(rt, tools=_tools(executor), events=sink, **kw)  # type: ignore[arg-type]
    run = runner.start(_task(budget))
    streamed = [e async for e in run]
    return run.result, streamed, sink


async def test_the_model_call_budget_stops_a_looping_run() -> None:
    log = _looping()
    result, streamed, sink = await _run(log, RunBudget(max_model_calls=3, max_tool_calls=100))
    assert result.status is RunStatus.FAILED
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.MODEL_CALLS
    assert len(log.requests) == 3
    assert result.model_calls == 3
    exhausted = [e for e in streamed if isinstance(e, BudgetExhaustedEvent)]
    assert len(exhausted) == 1
    assert (exhausted[0].budget, exhausted[0].limit, exhausted[0].used) == (
        BudgetKind.MODEL_CALLS,
        3,
        3,
    )
    assert sink.of(BudgetExhaustedEvent) == exhausted
    finished = streamed[-1]
    assert isinstance(finished, RunFinishedEvent)
    assert (finished.outcome, finished.failure, finished.budget) == (
        "failed",
        RunFailure.BUDGET_EXHAUSTED,
        BudgetKind.MODEL_CALLS,
    )


async def test_the_tool_call_budget_stops_before_the_next_tool_runs() -> None:
    log = _looping()
    executor = Executor()
    result, streamed, _ = await _run(log, RunBudget(max_model_calls=50, max_tool_calls=2), executor)
    assert result.budget is BudgetKind.TOOL_CALLS
    assert len(executor.calls) == 2
    assert result.tool_calls == 2
    assert len(log.requests) == 3
    exhausted = [e for e in streamed if isinstance(e, BudgetExhaustedEvent)]
    assert [(e.budget, e.limit, e.used) for e in exhausted] == [(BudgetKind.TOOL_CALLS, 2, 2)]


async def test_a_tool_use_left_by_a_budget_still_gets_an_error_result() -> None:
    log = ScriptLog(
        scripted([StubTurn(tool_calls=(StubToolCall(name="lookup"), StubToolCall(name="lookup")))])
    )
    executor = Executor()
    result, _, _ = await _run(log, RunBudget(max_tool_calls=1), executor)
    assert result.budget is BudgetKind.TOOL_CALLS
    assert len(executor.calls) == 1
    uses = [b for b in result.messages[-2].content if isinstance(b, ToolUseBlock)]
    results = [b for b in result.messages[-1].content if isinstance(b, ToolResultBlock)]
    assert [r.tool_use_id for r in results] == [u.id for u in uses]
    assert not results[0].is_error
    assert results[1].is_error
    assert "budget" in result_text(results[1])


async def test_the_token_budget_stops_once_usage_passes_it() -> None:
    usage = Usage(input_tokens=40, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0)
    log = ScriptLog(
        lambda request, index: StubTurn(tool_calls=(StubToolCall(name="lookup"),), usage=usage)
    )
    executor = Executor()
    result, streamed, _ = await _run(log, RunBudget(max_tokens=120, max_model_calls=50), executor)
    assert result.budget is BudgetKind.TOKENS
    assert len(log.requests) == 3  # 50, 100, then 150 passes 120
    exhausted = next(e for e in streamed if isinstance(e, BudgetExhaustedEvent))
    assert (exhausted.limit, exhausted.used) == (120, 150)
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 30
    # The turn that passed the budget never runs its tools, and each gets an error result.
    assert len(executor.calls) == 2
    last = result.messages[-1].content[0]
    assert isinstance(last, ToolResultBlock) and last.is_error
    assert "budget" in result_text(last)


async def test_a_tool_call_after_the_deadline_is_refused_before_it_runs() -> None:
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()

    def script(request: object, index: int) -> StubTurn:
        clock.now = 100.0  # the model call ends past the deadline
        return StubTurn(tool_calls=(StubToolCall(name="lookup"),))

    executor = Executor()
    result, _, _ = await _run(
        ScriptLog(script), RunBudget(max_wall_time_s=10), executor, clock=clock
    )
    assert result.budget is BudgetKind.WALL_TIME
    assert executor.calls == []
    assert result.tool_calls == 0


async def test_a_slow_model_call_is_cut_off_at_the_wall_time_deadline() -> None:
    class SlowStub(StubProvider):
        async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
            await asyncio.sleep(10)
            async for event in super().stream(request):
                yield event

    class SlowFactory(StubFactory):
        def create_provider(
            self, name: str, settings: ProviderSettings, client: Any, *, catalogue: Any = None
        ) -> Provider:
            return SlowStub(echo, not_for_production=True)

    rt = await start(
        settings(),
        task_classes=[CHAT, REVIEW],
        factories=[SlowFactory()],
        secrets=LiteralSecretResolver({}),
    )
    began = time.monotonic()
    result = await AgentRunner(rt).run(_task(RunBudget(max_wall_time_s=0.2)))
    assert time.monotonic() - began < 5
    assert result.budget is BudgetKind.WALL_TIME
    assert result.model_calls == 1


async def test_the_token_budget_charges_only_the_counts_the_product_chose() -> None:
    usage = Usage(input_tokens=1000, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0)
    log = ScriptLog(
        lambda request, index: StubTurn(tool_calls=(StubToolCall(name="lookup"),), usage=usage)
    )
    budget = RunBudget(
        max_tokens=25, max_model_calls=50, counted_tokens=frozenset({TokenCount.OUTPUT})
    )
    result, _, _ = await _run(log, budget)
    assert result.budget is BudgetKind.TOKENS
    assert len(log.requests) == 3  # 10, 20, then 30 passes 25


async def test_the_wall_time_budget_is_checked_against_the_runners_clock() -> None:
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            Clock.now += 4.0  # every reading moves time on
            return Clock.now

    log = _looping()
    result, streamed, _ = await _run(
        log, RunBudget(max_wall_time_s=10, max_model_calls=50), clock=Clock()
    )
    assert result.budget is BudgetKind.WALL_TIME
    assert len(log.requests) < 50
    exhausted = next(e for e in streamed if isinstance(e, BudgetExhaustedEvent))
    assert exhausted.budget is BudgetKind.WALL_TIME
    assert exhausted.limit == 10


async def test_a_slow_tool_is_cut_off_at_the_wall_time_deadline() -> None:
    log = _looping()
    executor = Executor(delay_s=5)
    result, _, _ = await _run(log, RunBudget(max_wall_time_s=0.2), executor)
    assert result.budget is BudgetKind.WALL_TIME
    assert len(executor.calls) == 1
    last = result.messages[-1].content[0]
    assert isinstance(last, ToolResultBlock) and last.is_error


async def test_a_wall_time_budget_bounds_each_model_requests_timeout() -> None:
    log = ScriptLog(scripted([StubTurn(text="ok")]))
    await _run(log, RunBudget(max_wall_time_s=7))
    assert 0 < log.requests[0].timeout_s <= 7


async def test_a_run_within_every_budget_completes() -> None:
    log = ScriptLog(
        scripted([StubTurn(tool_calls=(StubToolCall(name="lookup"),)), StubTurn(text="ok")])
    )
    result, streamed, _ = await _run(
        log,
        RunBudget(max_model_calls=2, max_tool_calls=1, max_tokens=10_000, max_wall_time_s=60),
    )
    assert result.status is RunStatus.COMPLETED
    assert not [e for e in streamed if isinstance(e, BudgetExhaustedEvent)]


async def test_a_slow_system_hook_is_cut_off_at_the_wall_time_deadline() -> None:
    async def slow(base: tuple[Any, ...]) -> tuple[Any, ...]:
        await asyncio.sleep(5)
        return base

    log = ScriptLog(scripted([StubTurn(text="ok")]))
    rt = await runtime(log)
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_wall_time_s=0.1),
        system_hook=slow,
    )
    began = time.monotonic()
    result = await AgentRunner(rt).run(task)
    assert time.monotonic() - began < 2  # cut off at the deadline, not after the hook
    assert result.budget is BudgetKind.WALL_TIME
    assert log.requests == []


async def test_a_slow_compaction_hook_is_cut_off_at_the_wall_time_deadline() -> None:
    class Slow:
        async def before_model_call(self, request: Any) -> None:
            await asyncio.sleep(5)

    log = ScriptLog(scripted([StubTurn(text="ok")]))
    rt = await runtime(log)
    began = time.monotonic()
    result = await AgentRunner(rt, compaction=Slow()).run(_task(RunBudget(max_wall_time_s=0.1)))
    assert time.monotonic() - began < 2  # cut off at the deadline, not after the hook
    assert result.budget is BudgetKind.WALL_TIME
    assert log.requests == []


async def test_the_deadline_is_checked_again_after_the_hooks() -> None:
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()

    async def hook(base: tuple[Any, ...]) -> tuple[Any, ...]:
        clock.now = 100.0  # the hook returns at once, but past the deadline
        return base

    log = ScriptLog(scripted([StubTurn(text="ok")]))
    rt = await runtime(log)
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_wall_time_s=10),
        system_hook=hook,
    )
    result = await AgentRunner(rt, clock=clock).run(task)
    assert result.budget is BudgetKind.WALL_TIME
    assert log.requests == []


async def test_charged_output_caps_each_requests_output_limit_at_the_tokens_left() -> None:
    usage = Usage(input_tokens=0, output_tokens=30, cache_read_tokens=0, cache_write_tokens=0)
    log = ScriptLog(
        lambda request, index: StubTurn(tool_calls=(StubToolCall(name="lookup"),), usage=usage)
    )
    await _run(log, RunBudget(max_tokens=100, max_model_calls=50))
    assert [r.max_output_tokens for r in log.requests] == [100, 70, 40, 10]


async def test_output_not_charged_leaves_the_assignments_output_limit() -> None:
    usage = Usage(input_tokens=30, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    log = ScriptLog(
        lambda request, index: StubTurn(tool_calls=(StubToolCall(name="lookup"),), usage=usage)
    )
    budget = RunBudget(
        max_tokens=100, max_model_calls=50, counted_tokens=frozenset({TokenCount.INPUT})
    )
    await _run(log, budget)
    assert {r.max_output_tokens for r in log.requests} == {1024}
