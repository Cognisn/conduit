"""Output contracts in a run: one retry carrying the validation errors, then a typed failure."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from conduit.agent import (
    AgentRunner,
    AgentTask,
    OutputContract,
    OutputContractFailed,
    RunFailed,
    RunFailure,
    RunStatus,
)
from conduit.agent.events import OutputInvalidEvent, RunFinishedEvent
from conduit.core.types import Message, StopReason, TextBlock
from conduit.providers.stub import StubToolCall, StubTurn, scripted
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, runtime, task_tools


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    score: int


CONTRACT = OutputContract(model=Verdict, name="verdict")
GOOD = '{"label": "benign", "score": 2}'


def _contract_task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("judge"),), contract=CONTRACT)


async def test_a_valid_answer_is_returned_as_the_model() -> None:
    rt = await runtime(ScriptLog(scripted([StubTurn(text=GOOD)])))
    result = await AgentRunner(rt).run(_contract_task())
    assert result.status is RunStatus.COMPLETED
    assert result.output == Verdict(label="benign", score=2)
    result.raise_for_failure()  # does nothing on success


async def test_one_invalid_answer_is_retried_with_its_errors() -> None:
    log = ScriptLog(scripted([StubTurn(text='{"label": "benign"}'), StubTurn(text=GOOD)]))
    rt = await runtime(log)
    run = AgentRunner(rt).start(_contract_task())
    events = [e async for e in run]
    assert run.result.output == Verdict(label="benign", score=2)
    assert len(log.requests) == 2
    retry = log.requests[1].messages[-1]
    assert retry.role == "user"
    assert isinstance(retry.content[0], TextBlock)
    assert "score: missing" in retry.content[0].text
    invalid = [e for e in events if isinstance(e, OutputInvalidEvent)]
    assert [(e.attempt, e.error_count, e.retrying) for e in invalid] == [(1, 1, True)]


async def test_a_second_invalid_answer_is_a_typed_failure() -> None:
    log = ScriptLog(
        scripted([StubTurn(text="nope"), StubTurn(text="still nope"), StubTurn(text=GOOD)])
    )
    rt = await runtime(log)
    run = AgentRunner(rt).start(_contract_task())
    events = [e async for e in run]
    result = run.result
    assert result.status is RunStatus.FAILED
    assert result.failure is RunFailure.OUTPUT_INVALID
    assert result.output is None
    assert len(log.requests) == 2  # exactly one retry, never a third attempt
    invalid = [e for e in events if isinstance(e, OutputInvalidEvent)]
    assert [(e.attempt, e.retrying) for e in invalid] == [(1, True), (2, False)]
    finished = events[-1]
    assert isinstance(finished, RunFinishedEvent)
    assert finished.failure is RunFailure.OUTPUT_INVALID
    with pytest.raises(OutputContractFailed) as caught:
        result.raise_for_failure()
    assert isinstance(caught.value, RunFailed)
    assert caught.value.failure is RunFailure.OUTPUT_INVALID


async def test_tool_turns_between_attempts_do_not_reset_the_retry_limit() -> None:
    log = ScriptLog(
        scripted(
            [
                StubTurn(text="bad"),
                StubTurn(tool_calls=(StubToolCall(name="lookup"),)),
                StubTurn(text="bad again"),
                StubTurn(text=GOOD),
            ]
        )
    )
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    result = await AgentRunner(rt, tools=tools).run(_contract_task())
    assert result.failure is RunFailure.OUTPUT_INVALID
    assert len(log.requests) == 3


async def test_an_answer_cut_off_at_the_output_limit_is_invalid() -> None:
    log = ScriptLog(
        scripted([StubTurn(text=GOOD, stop_reason=StopReason.MAX_TOKENS), StubTurn(text=GOOD)])
    )
    rt = await runtime(log)
    result = await AgentRunner(rt).run(_contract_task())
    assert result.output == Verdict(label="benign", score=2)
    assert len(log.requests) == 2
    retry = log.requests[1].messages[-1].content[0]
    assert isinstance(retry, TextBlock)
    assert "cut off" in retry.text


async def test_the_output_filter_runs_before_the_contract() -> None:
    rt = await runtime(ScriptLog(scripted([StubTurn(text='{"label": "sk-zz-1", "score": 1}')])))

    async def mask(text: str) -> str:
        return text.replace("sk-zz-1", "[masked]")

    result = await AgentRunner(rt, output_filter=mask).run(_contract_task())
    assert result.output == Verdict(label="[masked]", score=1)


async def test_the_retry_message_never_echoes_the_invalid_value() -> None:
    log = ScriptLog(
        scripted(
            [
                StubTurn(text='{"label": "a", "score": "zz leaked value 6612 zz"}'),
                StubTurn(text=GOOD),
            ]
        )
    )
    rt = await runtime(log)
    await AgentRunner(rt).run(_contract_task())
    retry = log.requests[1].messages[-1].content[0]
    assert isinstance(retry, TextBlock)
    assert "zz leaked value" not in retry.text
    assert "score: int_parsing" in retry.text


async def test_the_contract_schema_is_requested_when_the_contract_asks_and_the_model_can() -> None:
    log = ScriptLog(scripted([StubTurn(text=GOOD)]))
    rt = await runtime(log)
    contract = OutputContract(model=Verdict, name="verdict", structured=True)
    await AgentRunner(rt).run(
        AgentTask(task_class="chat", messages=(Message.user("judge"),), contract=contract)
    )
    assert log.requests[0].output_schema == Verdict.model_json_schema()


async def test_without_structured_the_schema_is_not_requested() -> None:
    log = ScriptLog(scripted([StubTurn(text=GOOD)]))
    rt = await runtime(log)
    await AgentRunner(rt).run(_contract_task())
    assert log.requests[0].output_schema is None
