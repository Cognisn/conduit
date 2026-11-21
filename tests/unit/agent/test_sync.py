"""The runner from synchronous code, through the sync facade on a conduit loop."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from conduit.agent import AgentRun, AgentRunner, AgentTask, RunStatus
from conduit.agent.events import RunFinishedEvent, TextDeltaEvent
from conduit.core.errors import FacadeError
from conduit.core.facade import LoopThread, sync
from conduit.core.types import Message
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, runtime, task_tools


@pytest.fixture
def loop() -> Iterator[LoopThread]:
    owned = LoopThread(name="conduit-agent-sync-loop")
    yield owned
    owned.close(join_timeout=5)


def _script() -> ScriptLog:
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="lookup"),)),
            lambda results: StubTurn(text="all done here"),
        )
    )


def _runner(loop: LoopThread) -> AgentRunner:
    # The runtime is started on the facade's loop, as the facade requires.
    rt = loop.run(runtime(_script()))
    return AgentRunner(rt, tools={"chat": task_tools([SampleTool("lookup")], Executor())})


def _task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("go"),))


def test_a_run_completes_from_sync_code(loop: LoopThread) -> None:
    runner = sync(_runner(loop), loop=loop)
    result = runner.run(_task())
    assert result.status is RunStatus.COMPLETED
    assert result.text == "all done here"
    assert result.tool_calls == 1


def test_a_run_streams_to_sync_code_as_a_plain_iterator(loop: LoopThread) -> None:
    runner = _runner(loop)
    run = runner.start(_task())
    assert isinstance(run, AgentRun)
    events = list(sync(run, loop=loop).events())
    assert any(isinstance(e, TextDeltaEvent) for e in events)
    assert isinstance(events[-1], RunFinishedEvent)
    assert run.result.text == "all done here"


def test_the_sync_facade_refuses_its_own_loop_thread(loop: LoopThread) -> None:
    runner = sync(_runner(loop), loop=loop)

    async def inside() -> None:
        runner.run(_task())

    with pytest.raises(FacadeError):
        loop.run(inside())
