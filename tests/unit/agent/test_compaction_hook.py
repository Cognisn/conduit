"""The compaction hook: called before every model call, it may replace the conversation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import pytest

from conduit.agent import (
    AgentRunner,
    AgentTask,
    BudgetKind,
    CompactionRefused,
    CompactionRequest,
    ModelCallCharge,
    RunBudget,
    RunFailed,
    RunFailure,
    RunStatus,
    TokenCount,
)
from conduit.agent.events import BudgetExhaustedEvent, ConversationCompactedEvent, RunFinishedEvent
from conduit.core.types import Message, Usage
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, Sink, runtime, task_tools


def _script() -> ScriptLog:
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="lookup"),)),
            lambda results: StubTurn(text="done"),
        )
    )


class Hook:
    def __init__(self, replace_on: int | None = None) -> None:
        self.requests: list[CompactionRequest] = []
        self.replace_on = replace_on

    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        self.requests.append(request)
        if request.model_calls == self.replace_on:
            return (Message.user("summary of everything so far"),)
        return None


def _task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("go"),), session_id="s-1")


async def test_the_hook_sees_every_model_call_before_it_is_made() -> None:
    hook = Hook()
    rt = await runtime(_script())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    result = await AgentRunner(rt, tools=tools, compaction=hook).run(_task())
    assert [r.model_calls for r in hook.requests] == [0, 1]
    first, second = hook.requests
    assert first.task_class == "chat"
    assert first.run_id == result.run_id
    assert first.session_id == "s-1"
    assert first.assignment.model == "stub-echo"
    assert first.assignment.entry.context_window == 1_000_000
    assert first.messages == (Message.user("go"),)
    assert first.last_usage is None
    assert len(second.messages) == 3
    assert second.last_usage is not None
    assert [t.name for t in second.tools] == ["lookup"]


async def test_a_replacement_conversation_is_used_for_the_call() -> None:
    hook = Hook(replace_on=1)
    log = _script()
    sink = Sink()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    run = AgentRunner(rt, tools=tools, compaction=hook, events=sink).start(_task())
    events = [e async for e in run]
    assert log.requests[1].messages == (Message.user("summary of everything so far"),)
    compacted = [e for e in events if isinstance(e, ConversationCompactedEvent)]
    assert [(e.messages_before, e.messages_after) for e in compacted] == [(3, 1)]
    assert sink.of(ConversationCompactedEvent) == compacted
    assert run.result.messages[0] == Message.user("summary of everything so far")


@pytest.mark.parametrize("replacement", [(), [object()], "text"])
async def test_an_unusable_replacement_is_refused(replacement: object) -> None:
    class Bad:
        async def before_model_call(self, request: CompactionRequest) -> object:
            return replacement

    sink = Sink()
    rt = await runtime(_script())
    with pytest.raises(CompactionRefused):
        await AgentRunner(rt, compaction=Bad(), events=sink).run(_task())  # type: ignore[arg-type]
    finished = sink.of(RunFinishedEvent)
    assert [(e.outcome, e.error_type) for e in finished] == [("failed", "CompactionRefused")]


async def test_a_replacement_ending_with_the_assistant_is_refused() -> None:
    class Bad:
        async def before_model_call(self, request: CompactionRequest) -> Sequence[Message]:
            return (Message.user("a"), Message.assistant("b"))

    rt = await runtime(_script())
    with pytest.raises(CompactionRefused):
        await AgentRunner(rt, compaction=Bad()).run(_task())


# A raising hook ends the run as a typed failure ----------------------------------------

_HOOK_SECRET = "zz-compaction-hook-secret-0000zz"


class Raising:
    """A hook that raises on the call numbered ``on``, with content in its message."""

    def __init__(self, error: BaseException, on: int = 0) -> None:
        self.error = error
        self.on = on
        self.calls = 0

    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        self.calls += 1
        if request.model_calls == self.on:
            raise self.error
        return None


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(_HOOK_SECRET),
        ValueError(_HOOK_SECRET),
        TimeoutError(_HOOK_SECRET),
        CompactionRefused(_HOOK_SECRET),
    ],
)
async def test_a_raising_hook_ends_the_run_as_compaction_failed(error: Exception) -> None:
    log = _script()
    sink = Sink()
    rt = await runtime(log)
    run = AgentRunner(rt, compaction=Raising(error), events=sink).start(_task())
    events = [e async for e in run]  # the stream itself never raises
    result = run.result
    assert result.status is RunStatus.FAILED
    assert result.failure is RunFailure.COMPACTION_FAILED
    assert result.text == ""
    assert log.requests == []  # the call the hook came before was never made
    [finished] = [e for e in events if isinstance(e, RunFinishedEvent)]
    assert (finished.outcome, finished.failure, finished.error_type) == (
        "failed",
        RunFailure.COMPACTION_FAILED,
        None,
    )
    assert sink.of(RunFinishedEvent) == [finished]
    assert _HOOK_SECRET not in repr(events)
    assert _HOOK_SECRET not in repr(result)


async def test_a_compaction_failure_raised_for_the_product_keeps_no_chain() -> None:
    rt = await runtime(_script())
    result = await AgentRunner(rt, compaction=Raising(RuntimeError(_HOOK_SECRET))).run(_task())
    with pytest.raises(RunFailed) as caught:
        result.raise_for_failure()
    failed = caught.value
    assert failed.failure is RunFailure.COMPACTION_FAILED
    assert failed.__cause__ is None
    assert failed.__context__ is None
    for shown in (str(failed), repr(failed), repr(failed.args)):
        assert _HOOK_SECRET not in shown
        assert "RuntimeError" not in shown


async def test_a_hook_failing_after_a_tool_turn_leaves_the_conversation_well_formed() -> None:
    log = _script()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    hook = Raising(RuntimeError(_HOOK_SECRET), on=1)
    result = await AgentRunner(rt, tools=tools, compaction=hook).run(_task())
    assert result.failure is RunFailure.COMPACTION_FAILED
    assert len(log.requests) == 1
    assert result.model_calls == 2  # the second call was counted, then never made
    roles = [m.role for m in result.messages]
    assert roles == ["user", "assistant", "user"]
    assert hook.calls == 2


async def test_a_hook_failure_is_logged_by_class_name_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rt = await runtime(_script())
    with caplog.at_level(logging.WARNING, logger="conduit.agent"):
        await AgentRunner(rt, compaction=Raising(RuntimeError(_HOOK_SECRET))).run(_task())
    assert "RuntimeError" in caplog.text
    assert _HOOK_SECRET not in caplog.text


async def test_cancellation_in_the_hook_still_propagates() -> None:
    sink = Sink()
    rt = await runtime(_script())
    runner = AgentRunner(rt, compaction=Raising(asyncio.CancelledError()), events=sink)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(_task())
    assert [(e.outcome, e.error_type) for e in sink.of(RunFinishedEvent)] == [
        ("failed", "cancelled")
    ]


async def test_a_hook_cancelled_from_outside_propagates() -> None:
    started = asyncio.Event()

    class Waiting:
        async def before_model_call(self, request: CompactionRequest) -> None:
            started.set()
            await asyncio.sleep(3600)

    rt = await runtime(_script())
    task = asyncio.create_task(AgentRunner(rt, compaction=Waiting()).run(_task()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_hook_past_the_wall_time_budget_is_still_a_budget_failure() -> None:
    class Slow:
        async def before_model_call(self, request: CompactionRequest) -> None:
            await asyncio.sleep(3600)

    rt = await runtime(_script())
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_wall_time_s=0.2),
    )
    result = await AgentRunner(rt, compaction=Slow()).run(task)
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.WALL_TIME


async def test_a_hook_swallowing_its_deadline_is_still_a_budget_failure() -> None:
    # A hook that catches the cancellation the deadline sends, and raises instead, is
    # out of time: the run ends on the wall-time budget, not as a compaction failure.
    class Swallowing:
        async def before_model_call(self, request: CompactionRequest) -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise RuntimeError(_HOOK_SECRET) from None

    rt = await runtime(_script())
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_wall_time_s=0.2),
    )
    # The meter's clock stands still, so only the deadline itself can tell the runner.
    result = await AgentRunner(rt, compaction=Swallowing(), clock=lambda: 0.0).run(task)
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.WALL_TIME


# The run's budget charged for the hook's own model calls (D0085) ------------------------


class Charging:
    """A hook that makes its own model call through the run's charge before call ``on``."""

    def __init__(
        self, usage: Usage, on: int = 0, *, output: int = 300, record: bool = True
    ) -> None:
        self.usage = usage
        self.on = on
        self.output = output
        self.record = record
        self.allowed: list[int | None] = []
        self.recorded: list[bool] = []

    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        if request.model_calls != self.on:
            return None
        charge = request.charge
        assert charge is not None
        allowed = charge.begin(self.output)
        self.allowed.append(allowed)
        if allowed is not None and self.record:
            self.recorded.append(charge.record(self.usage))
        return None


_SUMMARY_USAGE = Usage(input_tokens=40, output_tokens=7, cache_read_tokens=3, cache_write_tokens=2)


async def test_the_request_carries_a_charge_for_the_run() -> None:
    hook = Hook()
    rt = await runtime(_script())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    await AgentRunner(rt, tools=tools, compaction=hook).run(_task())
    assert all(isinstance(r.charge, ModelCallCharge) for r in hook.requests)


async def test_a_charged_call_counts_against_the_run() -> None:
    hook = Charging(_SUMMARY_USAGE)
    log = _script()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    sink = Sink()
    result = await AgentRunner(rt, tools=tools, compaction=hook, events=sink).run(_task())
    assert result.failure is None
    assert hook.allowed == [300]
    assert hook.recorded == [True]
    assert result.model_calls == len(log.requests) + 1
    without = await AgentRunner(rt, tools=tools).run(_task())
    assert result.usage == without.usage + _SUMMARY_USAGE
    [finished] = sink.of(RunFinishedEvent)
    assert finished.model_calls == result.model_calls
    assert finished.usage == result.usage


async def test_a_charged_call_past_the_model_call_budget_ends_the_run() -> None:
    hook = Charging(_SUMMARY_USAGE, on=1)
    log = _script()
    sink = Sink()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_model_calls=2),
    )
    result = await AgentRunner(rt, tools=tools, compaction=hook, events=sink).run(task)
    assert hook.allowed == [None]
    assert hook.recorded == []
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.MODEL_CALLS
    assert len(log.requests) == 1  # the call the hook came before was not made either
    [exhausted] = sink.of(BudgetExhaustedEvent)
    assert (exhausted.budget, exhausted.limit, exhausted.used) == (BudgetKind.MODEL_CALLS, 2, 2)


async def test_a_charged_call_on_a_spent_token_budget_ends_the_run() -> None:
    class Twice(Charging):
        async def before_model_call(self, request: CompactionRequest) -> None:
            await super().before_model_call(request)
            assert request.charge is not None
            self.allowed.append(request.charge.begin(self.output))

    # The first charged call spends the budget exactly, so the second is refused.
    hook = Twice(
        Usage(input_tokens=5_000, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    )
    log = _script()
    rt = await runtime(log)
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_tokens=5_000),
    )
    result = await AgentRunner(rt, compaction=hook).run(task)
    assert hook.allowed == [300, None]
    assert hook.recorded == [True]
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.TOKENS
    assert log.requests == []


async def test_charged_usage_past_the_token_budget_ends_the_run() -> None:
    hook = Charging(
        Usage(input_tokens=10_000, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0)
    )
    log = _script()
    sink = Sink()
    rt = await runtime(log)
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_tokens=5_000),
    )
    result = await AgentRunner(rt, compaction=hook, events=sink).run(task)
    assert hook.allowed == [300]
    assert hook.recorded == [False]
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.TOKENS
    assert log.requests == []
    [exhausted] = sink.of(BudgetExhaustedEvent)
    assert (exhausted.budget, exhausted.limit, exhausted.used) == (
        BudgetKind.TOKENS,
        5_000,
        10_001,
    )


async def test_charged_usage_past_the_budget_wins_over_a_hook_failure() -> None:
    class ChargeThenRaise(Charging):
        async def before_model_call(self, request: CompactionRequest) -> None:
            await super().before_model_call(request)
            raise RuntimeError(_HOOK_SECRET)

    rt = await runtime(_script())
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_tokens=5_000),
    )
    hook = ChargeThenRaise(
        Usage(input_tokens=10_000, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    )
    result = await AgentRunner(rt, compaction=hook).run(task)
    assert result.failure is RunFailure.BUDGET_EXHAUSTED


@pytest.mark.parametrize(
    ("counted", "expected"),
    [
        (frozenset({TokenCount.OUTPUT}), 120),
        (frozenset({TokenCount.INPUT, TokenCount.OUTPUT}), 120),
        (frozenset({TokenCount.INPUT}), 300),
    ],
)
async def test_a_charged_call_may_not_produce_more_output_than_the_budget_has_left(
    counted: frozenset[TokenCount], expected: int
) -> None:
    hook = Charging(Usage.zero(), output=300)
    rt = await runtime(_script())
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("go"),),
        budget=RunBudget(max_tokens=120, counted_tokens=counted),
    )
    await AgentRunner(rt, compaction=hook).run(task)
    assert hook.allowed == [expected]
