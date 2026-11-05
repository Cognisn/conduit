"""conduit's agent errors keep neither cause nor context, and carry no content (D0040).

Cancellation always propagates unchanged, and a run cancelled or abandoned part-way is
recorded once in the sink.
"""

from __future__ import annotations

import asyncio

import pytest

from conduit.agent import (
    AgentRunner,
    AgentTask,
    BudgetExhausted,
    RunBudget,
    RunFailed,
    RunRefused,
)
from conduit.agent.events import RunFinishedEvent
from conduit.core.types import Message
from conduit.providers.stub import StubToolCall, StubTurn, echo, scripted
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, Sink, runtime, task_tools

SECRET = "zz secret 8812 zz"


def _clean(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), *map(str, error.args)):
        assert SECRET not in text
        assert "zz" not in text


async def test_a_refused_task_class_keeps_no_chain() -> None:
    rt = await runtime(ScriptLog(echo))
    with pytest.raises(RunRefused) as caught:
        AgentRunner(rt).start(AgentTask(task_class="triage", messages=(Message.user(SECRET),)))
    _clean(caught.value)


async def test_a_budget_failure_raised_for_the_result_keeps_no_chain() -> None:
    rt = await runtime(ScriptLog(lambda r, i: StubTurn(tool_calls=(StubToolCall(name="lookup"),))))
    tools = {"chat": task_tools([SampleTool("lookup")], Executor(answers={"lookup": SECRET}))}
    result = await AgentRunner(rt, tools=tools).run(
        AgentTask(
            task_class="chat",
            messages=(Message.user(SECRET),),
            budget=RunBudget(max_model_calls=2),
        )
    )
    with pytest.raises(BudgetExhausted) as caught:
        result.raise_for_failure()
    assert isinstance(caught.value, RunFailed)
    _clean(caught.value)


async def test_a_slow_tool_timeout_becomes_a_clean_budget_failure() -> None:
    rt = await runtime(ScriptLog(lambda r, i: StubTurn(tool_calls=(StubToolCall(name="lookup"),))))
    tools = {"chat": task_tools([SampleTool("lookup")], Executor(delay_s=5))}
    result = await AgentRunner(rt, tools=tools).run(
        AgentTask(
            task_class="chat",
            messages=(Message.user("go"),),
            budget=RunBudget(max_wall_time_s=0.1),
        )
    )
    with pytest.raises(BudgetExhausted) as caught:
        result.raise_for_failure()
    _clean(caught.value)


async def test_cancelling_a_run_propagates_and_is_recorded_once() -> None:
    sink = Sink()
    started = asyncio.Event()

    class Slow(Executor):
        async def __call__(self, tool, arguments, context):  # type: ignore[no-untyped-def]
            started.set()
            await asyncio.sleep(30)
            raise AssertionError("not reached")

    rt = await runtime(ScriptLog(scripted([StubTurn(tool_calls=(StubToolCall(name="lookup"),))])))
    runner = AgentRunner(
        rt, tools={"chat": task_tools([SampleTool("lookup")], Slow())}, events=sink
    )
    task = asyncio.create_task(
        runner.run(AgentTask(task_class="chat", messages=(Message.user("go"),)))
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finished = sink.of(RunFinishedEvent)
    assert [(e.outcome, e.error_type) for e in finished] == [("failed", "cancelled")]


async def test_a_stream_abandoned_part_way_is_recorded_once() -> None:
    sink = Sink()
    rt = await runtime(ScriptLog(echo))
    run = AgentRunner(rt, events=sink).start(
        AgentTask(task_class="chat", messages=(Message.user("a long reply to abandon"),))
    )
    stream = run.events()
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    finished = sink.of(RunFinishedEvent)
    assert [(e.outcome, e.error_type) for e in finished] == [("failed", "abandoned")]
    with pytest.raises(RuntimeError, match="not finished"):
        _ = run.result


async def test_closing_after_the_finished_event_is_not_an_abandonment() -> None:
    sink = Sink()
    rt = await runtime(ScriptLog(echo))
    run = AgentRunner(rt, events=sink).start(
        AgentTask(task_class="chat", messages=(Message.user("hi"),))
    )
    stream = run.events()
    async for event in stream:
        if isinstance(event, RunFinishedEvent):
            break
    await stream.aclose()
    assert [e.outcome for e in sink.of(RunFinishedEvent)] == ["completed"]
    assert run.result.text == "hi"


async def test_an_unexpected_error_propagates_and_is_recorded() -> None:
    sink = Sink()

    def broken(request: object, index: int) -> StubTurn:
        raise KeyError("zz")

    rt = await runtime(ScriptLog(broken))
    with pytest.raises(KeyError):
        await AgentRunner(rt, events=sink).run(
            AgentTask(task_class="chat", messages=(Message.user("go"),))
        )
    assert [(e.outcome, e.error_type) for e in sink.of(RunFinishedEvent)] == [
        ("failed", "KeyError")
    ]
