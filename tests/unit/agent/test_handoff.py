"""Hand-off: only to a declared task class the run allows, through its resolved assignment."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.agent import (
    HAND_OFF_TOOL,
    AgentRunner,
    AgentTask,
    BudgetKind,
    RunBudget,
    RunRefused,
    RunStatus,
)
from conduit.agent.events import HandOffEvent, RunFinishedEvent
from conduit.core.events import ToolCallEvent
from conduit.core.types import Message, ToolResultBlock, Usage
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.unit.agent.helpers import (
    Executor,
    SampleTool,
    ScriptLog,
    Sink,
    runtime,
    task_tools,
)


def _asks_for(arguments: dict[str, Any]) -> ScriptLog:
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name=HAND_OFF_TOOL, input=arguments),)),
            lambda results: StubTurn(text="stayed"),
        )
    )


def _chat(hand_offs: frozenset[str] = frozenset({"review"})) -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("look"),), hand_offs=hand_offs)


def _specialist() -> ScriptLog:
    return ScriptLog(lambda request, index: StubTurn(text="reviewed"))


async def test_a_hand_off_continues_on_the_targets_assignment() -> None:
    primary, specialist = _asks_for({"task_class": "review"}), _specialist()
    sink = Sink()
    rt = await runtime(primary, specialist)
    run = AgentRunner(rt, events=sink).start(_chat())
    events = [e async for e in run]
    result = run.result
    assert result.status is RunStatus.COMPLETED
    assert result.text == "reviewed"
    assert result.task_class == "review"
    assert result.hand_offs == ("review",)
    assert len(primary.requests) == 1
    assert len(specialist.requests) == 1
    request = specialist.requests[0]
    target = rt.assignments.for_task("review")
    assert (request.model, request.task_class, request.max_output_tokens) == (
        target.model,
        "review",
        target.max_output_tokens,
    )
    assert request.model == "stub-large"
    # The conversation carries over, with the hand-off's own tool result.
    result_block = request.messages[2].content[0]
    assert isinstance(result_block, ToolResultBlock) and not result_block.is_error
    handed = [e for e in events if isinstance(e, HandOffEvent)]
    assert len(handed) == 1
    event = handed[0]
    assert (event.outcome, event.from_task_class, event.to_task_class) == ("ok", "chat", "review")
    assert (event.provider, event.model) == ("specialist", "stub-large")
    assert sink.of(HandOffEvent) == handed


async def test_every_request_after_a_hand_off_stays_on_the_target() -> None:
    primary = _asks_for({"task_class": "review"})
    specialist = ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="inspect"),)),
            lambda results: StubTurn(text="done"),
        )
    )
    rt = await runtime(primary, specialist)
    review_tools = task_tools(
        [SampleTool("inspect", task_classes=frozenset({"review"}))], Executor(), "review"
    )
    chat_tools = task_tools([SampleTool("lookup")], Executor())
    result = await AgentRunner(rt, tools={"chat": chat_tools, "review": review_tools}).run(_chat())
    assert result.text == "done"
    assert {r.model for r in specialist.requests} == {"stub-large"}
    assert {r.task_class for r in specialist.requests} == {"review"}
    # The target's own tools are offered after the hand-off; the source's are not.
    names = {t.name for t in specialist.requests[0].tools}
    assert "inspect" in names
    assert "lookup" not in names
    assert {t.name for t in primary.requests[0].tools} == {"lookup", HAND_OFF_TOOL}


async def test_a_target_the_run_does_not_allow_is_refused_as_a_result() -> None:
    primary, specialist = _asks_for({"task_class": "triage"}), _specialist()
    rt = await runtime(primary, specialist, triage=True)  # triage is assigned, not allowed
    run = AgentRunner(rt).start(_chat())
    events = [e async for e in run]
    assert run.result.task_class == "chat"
    assert run.result.text == "stayed"
    assert run.result.hand_offs == ()
    assert specialist.requests == []
    assert len(primary.requests) == 2
    assert {r.task_class for r in primary.requests} == {"chat"}
    block = primary.requests[1].messages[2].content[0]
    assert isinstance(block, ToolResultBlock) and block.is_error
    handed = [e for e in events if isinstance(e, HandOffEvent)]
    assert [(e.outcome, e.to_task_class) for e in handed] == [("refused", None)]
    tool = next(e for e in events if isinstance(e, ToolCallEvent))
    assert (tool.tool, tool.outcome, tool.error_code) == (
        HAND_OFF_TOOL,
        "failed",
        "hand_off_refused",
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"task_class": 7},
        {"task_class": "chat"},
        {"task_class": "nonexistent"},
        {"task_class": "review", "extra": True},
        {"task_class": "review zz injected zz"},
    ],
)
async def test_a_malformed_or_impossible_hand_off_is_refused(arguments: dict[str, Any]) -> None:
    primary, specialist = _asks_for(arguments), _specialist()
    rt = await runtime(primary, specialist)
    result = await AgentRunner(rt).run(_chat())
    assert result.task_class == "chat"
    assert result.hand_offs == ()
    assert specialist.requests == []


async def test_a_hand_off_target_without_an_assignment_is_refused_at_the_start() -> None:
    primary = _asks_for({"task_class": "triage"})
    rt = await runtime(primary)  # triage declared nowhere
    with pytest.raises(RunRefused, match="triage"):
        AgentRunner(rt).start(_chat(frozenset({"triage"})))
    assert primary.requests == []


async def test_the_hand_off_tool_is_offered_only_when_the_run_allows_one() -> None:
    primary = _asks_for({"task_class": "review"})
    specialist = _specialist()
    rt = await runtime(primary, specialist)
    run = AgentRunner(rt).start(_chat(frozenset()))
    events = [e async for e in run]
    assert primary.requests[0].tools == ()
    assert run.result.task_class == "chat"
    assert specialist.requests == []
    # Without targets, a hand_off call is an unknown tool, not a refused hand-off.
    assert not [e for e in events if isinstance(e, HandOffEvent)]
    tool = next(e for e in events if isinstance(e, ToolCallEvent))
    assert (tool.tool, tool.error_code) == ("_unknown", "unknown_tool")


async def test_the_hand_off_tool_lists_only_the_allowed_targets() -> None:
    primary = _asks_for({"task_class": "review"})
    rt = await runtime(primary, _specialist(), triage=True)
    await AgentRunner(rt).run(_chat(frozenset({"review", "triage", "chat"})))
    spec = {t.name: t for t in primary.requests[0].tools}[HAND_OFF_TOOL]
    targets = spec.input_schema["properties"]["task_class"]["enum"]
    assert targets == ["review", "triage"]  # never the current task class


async def test_a_product_tool_named_like_the_hand_off_tool_is_refused() -> None:
    rt = await runtime(_asks_for({}))
    clash = task_tools([SampleTool(HAND_OFF_TOOL)], Executor())
    with pytest.raises(RunRefused, match=HAND_OFF_TOOL):
        AgentRunner(rt, tools={"chat": clash}).start(_chat())


async def test_only_the_first_hand_off_in_a_turn_is_taken() -> None:
    primary = ScriptLog(
        answer_tool_results(
            StubTurn(
                tool_calls=(
                    StubToolCall(name=HAND_OFF_TOOL, input={"task_class": "review"}),
                    StubToolCall(name=HAND_OFF_TOOL, input={"task_class": "triage"}),
                )
            ),
            lambda results: StubTurn(text="stayed"),
        )
    )
    rt = await runtime(primary, _specialist(), triage=True)
    result = await AgentRunner(rt).run(_chat(frozenset({"review", "triage"})))
    assert result.hand_offs == ("review",)
    results = [b for b in result.messages[2].content if isinstance(b, ToolResultBlock)]
    assert [r.is_error for r in results] == [False, True]


def _hands_off_then_loops() -> tuple[ScriptLog, ScriptLog]:
    """The primary hands off at once; the specialist asks for a tool on every turn."""
    primary = ScriptLog(
        lambda request, index: StubTurn(
            tool_calls=(StubToolCall(name=HAND_OFF_TOOL, input={"task_class": "review"}),),
            usage=_USAGE,
        )
    )
    specialist = ScriptLog(
        lambda request, index: StubTurn(tool_calls=(StubToolCall(name="inspect"),), usage=_USAGE)
    )
    return primary, specialist


_USAGE = Usage(input_tokens=30, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0)


async def _budgeted(budget: RunBudget) -> tuple[Any, list[Any], ScriptLog, Executor]:
    primary, specialist = _hands_off_then_loops()
    rt = await runtime(primary, specialist)
    executor = Executor()
    review_tools = task_tools(
        [SampleTool("inspect", task_classes=frozenset({"review"}))], executor, "review"
    )
    run = AgentRunner(rt, tools={"review": review_tools}).start(
        AgentTask(
            task_class="chat",
            messages=(Message.user("look"),),
            hand_offs=frozenset({"review"}),
            budget=budget,
        )
    )
    events = [e async for e in run]
    return run.result, events, specialist, executor


async def test_a_hand_off_keeps_the_runs_model_call_budget() -> None:
    result, events, specialist, _ = await _budgeted(RunBudget(max_model_calls=2))
    assert result.hand_offs == ("review",)
    assert result.budget is BudgetKind.MODEL_CALLS
    assert len(specialist.requests) == 1  # call 2; call 3 is refused
    assert (result.model_calls, result.tool_calls) == (2, 2)
    finished = events[-1]
    assert isinstance(finished, RunFinishedEvent)
    assert (finished.model_calls, finished.tool_calls, finished.hand_offs) == (2, 2, 1)
    assert finished.budget is BudgetKind.MODEL_CALLS


async def test_a_hand_off_keeps_the_runs_tool_call_budget() -> None:
    result, _, specialist, executor = await _budgeted(RunBudget(max_tool_calls=2))
    assert result.budget is BudgetKind.TOOL_CALLS
    assert len(executor.calls) == 1  # the hand-off was the first tool call
    assert result.tool_calls == 2
    assert len(specialist.requests) == 2


async def test_a_hand_off_keeps_the_runs_token_budget() -> None:
    result, _, specialist, _ = await _budgeted(RunBudget(max_tokens=100))
    assert result.budget is BudgetKind.TOKENS
    assert len(specialist.requests) == 2  # 40, 80, then 120 passes 100
    assert result.usage.input_tokens == 90
