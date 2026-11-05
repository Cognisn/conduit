"""Approvals: a tool the product marks goes through the ApprovalHandler before it runs.

A refusal is a tool result the model sees, never an exception.
"""

from __future__ import annotations

import pytest

from conduit.agent import AgentRunner, AgentTask, RunStatus
from conduit.agent.events import ApprovalDecidedEvent, ApprovalRequestedEvent
from conduit.core.events import ToolCallEvent
from conduit.core.types import Message, ToolResultBlock
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.unit.agent.helpers import (
    Approvals,
    Executor,
    SampleTool,
    ScriptLog,
    Sink,
    result_text,
    runtime,
    task_tools,
)


def _script() -> ScriptLog:
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="delete_host", input={"host": "h-1"}),)),
            lambda results: StubTurn(text="error" if results[0].is_error else "deleted"),
        )
    )


def _task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("delete h-1"),))


async def _run(approvals: Approvals | None, *, marked: object = True):  # type: ignore[no-untyped-def]
    log = _script()
    executor = Executor()
    sink = Sink()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("delete_host", requires_approval=marked)], executor)}
    run = AgentRunner(rt, tools=tools, approvals=approvals, events=sink).start(_task())
    events = [e async for e in run]
    return run.result, events, executor, log, sink


async def test_an_approved_call_runs() -> None:
    approvals = Approvals(approve=True)
    result, events, executor, _, sink = await _run(approvals)
    assert result.text == "deleted"
    assert len(executor.calls) == 1
    assert len(approvals.requests) == 1
    request = approvals.requests[0]
    assert request.tool == "delete_host"
    assert request.arguments == {"host": "h-1"}
    assert request.run_id == result.run_id
    assert request.task_class == "chat"
    requested = [e for e in events if isinstance(e, ApprovalRequestedEvent)]
    decided = [e for e in events if isinstance(e, ApprovalDecidedEvent)]
    assert [e.outcome for e in decided] == ["approved"]
    assert requested[0].request_id == decided[0].request_id == request.request_id
    assert sink.of(ApprovalDecidedEvent) == decided


async def test_a_refused_call_never_runs_and_the_model_is_told() -> None:
    approvals = Approvals(approve=False)
    result, events, executor, log, _ = await _run(approvals)
    assert result.status is RunStatus.COMPLETED
    assert result.text == "error"
    assert executor.calls == []
    block = log.requests[1].messages[2].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.is_error
    assert result_text(block) == "The tool call was not approved, so it was not run."
    decided = [e for e in events if isinstance(e, ApprovalDecidedEvent)]
    assert [e.outcome for e in decided] == ["refused"]
    tool = next(e for e in events if isinstance(e, ToolCallEvent))
    assert (tool.outcome, tool.error_code) == ("failed", "not_approved")


async def test_without_a_handler_a_marked_tool_is_refused() -> None:
    result, events, executor, _, _ = await _run(None)
    assert result.text == "error"
    assert executor.calls == []
    decided = [e for e in events if isinstance(e, ApprovalDecidedEvent)]
    assert [e.outcome for e in decided] == ["unavailable"]


async def test_a_failing_handler_is_a_refusal_not_an_exception() -> None:
    approvals = Approvals(fail_with=RuntimeError("zz handler detail 3301 zz"))
    result, events, executor, log, _ = await _run(approvals)
    assert result.status is RunStatus.COMPLETED
    assert executor.calls == []
    decided = [e for e in events if isinstance(e, ApprovalDecidedEvent)]
    assert [e.outcome for e in decided] == ["failed"]
    assert "zz handler detail" not in log.requests[1].model_dump_json()


async def test_an_unmarked_tool_needs_no_approval() -> None:
    approvals = Approvals(approve=False)
    result, events, _, _, _ = await _run(approvals, marked=False)
    assert result.text == "deleted"
    assert approvals.requests == []
    assert not [e for e in events if isinstance(e, ApprovalRequestedEvent)]


@pytest.mark.parametrize("marked", [1, "yes", None, object()])
async def test_any_mark_other_than_false_needs_approval(marked: object) -> None:
    approvals = Approvals(approve=False)
    _, _, executor, _, _ = await _run(approvals, marked=marked)
    assert executor.calls == []
    assert len(approvals.requests) == 1


async def test_a_decision_that_is_not_strictly_approved_is_a_refusal() -> None:
    class Odd:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def decide(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            return "yes"  # not an ApprovalDecision

    log = _script()
    executor = Executor()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("delete_host", requires_approval=True)], executor)}
    result = await AgentRunner(rt, tools=tools, approvals=Odd()).run(_task())
    assert executor.calls == []
    assert result.text == "error"


async def test_a_product_predicate_can_mark_tools() -> None:
    approvals = Approvals(approve=False)
    log = _script()
    executor = Executor()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("delete_host")], executor)}
    runner = AgentRunner(
        rt,
        tools=tools,
        approvals=approvals,
        requires_approval=lambda tool: tool.name.startswith("delete_"),
    )
    await runner.run(_task())
    assert executor.calls == []
    assert len(approvals.requests) == 1


async def test_a_decision_with_a_truthy_non_boolean_approval_is_a_refusal() -> None:
    from conduit.core.backends import ApprovalDecision

    class Truthy:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def decide(self, request: object) -> ApprovalDecision:
            self.requests.append(request)
            return ApprovalDecision.model_construct(approved=1, reason="", decided_by=None)

    log = _script()
    executor = Executor()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("delete_host", requires_approval=True)], executor)}
    result = await AgentRunner(rt, tools=tools, approvals=Truthy()).run(_task())
    assert executor.calls == []
    assert result.text == "error"
