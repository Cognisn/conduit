"""The evaluation harness: cases, checks, the stub and live harnesses, and the report."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from conduit.agent import AgentTask, OutputContract, RunBudget, RunFailure
from conduit.core.backends import ApprovalDecision, ApprovalRequest
from conduit.core.events import CapturePolicy
from conduit.core.runtime import Runtime
from conduit.core.tasks import TaskClass
from conduit.core.types import Message
from conduit.providers.stub import StubToolCall, StubTurn
from conduit.testing._stub import start_stub_runtime
from conduit.testing.evaluation import (
    EvalCase,
    EvalReport,
    EvalTool,
    LiveHarness,
    Observation,
    StubHarness,
    completed,
    executed,
    failed,
    handed_off,
    never_executed,
    no_text_in_events,
    output_field,
)

CANARY = "zq-canary-7341-not-a-secret"


def _ask(text: str = "Check the host.", task_class: str = "chat", **kw: Any) -> AgentTask:
    return AgentTask(task_class=task_class, messages=(Message.user(text),), **kw)


def _answer(text: str = "done") -> tuple[StubTurn, ...]:
    return (StubTurn(text=text),)


async def _one(case: EvalCase, **harness: Any) -> Any:
    report = await StubHarness(**harness).run([case])
    [outcome] = report.outcomes
    return outcome


# --- Cases and the report ---------------------------------------------------------------


async def test_a_passing_case_passes_and_scores() -> None:
    case = EvalCase("answers", _ask(), checks=(completed(),), script=_answer())
    report = await StubHarness().run([case])
    assert [(o.case, o.passed, o.failures) for o in report.outcomes] == [("answers", True, ())]
    assert (report.passed, report.failed, report.skipped, report.ran) == (1, 0, 0, 1)
    assert report.score == 1.0
    assert report.outcomes[0].result is not None
    assert report.outcomes[0].result.text == "done"


async def test_a_failing_check_fails_the_case_and_the_score() -> None:
    good = EvalCase("good", _ask(), checks=(completed(),), script=_answer())
    bad = EvalCase("bad", _ask(), checks=(completed(), handed_off()), script=_answer())
    report = await StubHarness().run([good, bad])
    assert [o.passed for o in report.outcomes] == [True, False]
    assert report.outcomes[1].failures == ("expected a hand-off, got none",)
    assert report.score == 0.5
    assert report.failures() == {"bad": ("expected a hand-off, got none",)}


async def test_every_failing_check_is_reported_not_just_the_first() -> None:
    case = EvalCase(
        "two_wrong",
        _ask(),
        checks=(handed_off(), failed()),
        script=_answer(),
    )
    outcome = await _one(case)
    assert len(outcome.failures) == 2


async def test_a_check_that_raises_fails_the_case_naming_only_its_type() -> None:
    def broken(observation: Observation) -> str | None:
        raise KeyError(CANARY)

    case = EvalCase("raises", _ask(), checks=(broken,), script=_answer())
    outcome = await _one(case)
    assert not outcome.passed
    assert outcome.failures == ("a check raised KeyError",)
    assert CANARY not in repr(outcome.failures)


async def test_a_run_that_raises_fails_the_case_naming_only_its_type() -> None:
    # A hand-off target with no assignment is refused before the run starts.
    case = EvalCase(
        "refused",
        _ask(hand_offs=frozenset({"missing"})),
        checks=(completed(),),
        script=_answer(),
    )
    report = await StubHarness(task_classes=[TaskClass(name="chat")]).run([case])
    [outcome] = report.outcomes
    assert not outcome.passed
    assert outcome.failures == ("the run raised RunRefused",)
    assert outcome.result is None


async def test_cancellation_is_never_turned_into_a_failure() -> None:
    def script(request: Any, index: int) -> StubTurn:
        raise asyncio.CancelledError

    case = EvalCase("cancelled", _ask(), checks=(completed(),), script=script)
    with pytest.raises(asyncio.CancelledError):
        await StubHarness().run([case])


async def test_a_case_without_a_script_cannot_run_on_the_stub() -> None:
    case = EvalCase("unscripted", _ask(), checks=(completed(),))
    with pytest.raises(ValueError, match="unscripted has no stub script"):
        await StubHarness().run([case])


async def test_case_names_must_be_unique_in_a_run() -> None:
    case = EvalCase("same", _ask(), checks=(completed(),), script=_answer())
    with pytest.raises(ValueError, match="same"):
        await StubHarness().run([case, case])


def test_a_case_needs_a_name_and_a_check() -> None:
    with pytest.raises(ValueError, match="name"):
        EvalCase("", _ask(), checks=(completed(),))
    with pytest.raises(ValueError, match="check"):
        EvalCase("unchecked", _ask(), checks=())


def test_an_empty_report_scores_zero() -> None:
    assert EvalReport(()).score == 0.0


async def test_stub_checks_apply_only_on_the_stub() -> None:
    case = EvalCase(
        "stub_only",
        _ask(),
        checks=(completed(),),
        stub_checks=(handed_off(),),
        script=_answer(),
    )
    outcome = await _one(case)
    assert outcome.failures == ("expected a hand-off, got none",)


async def test_the_summary_names_cases_and_outcomes_only() -> None:
    case = EvalCase("leaky", _ask(CANARY), checks=(handed_off(),), script=_answer(CANARY))
    report = await StubHarness().run([case])
    summary = report.summary()
    assert "leaky" in summary and "failed" in summary and "0/1" in summary
    assert CANARY not in summary


# --- Checks -----------------------------------------------------------------------------


async def test_completed_and_failed() -> None:
    ok = await _one(EvalCase("ok", _ask(), checks=(completed(),), script=_answer()))
    assert ok.passed
    over = EvalCase(
        "over_budget",
        _ask(budget=RunBudget(max_model_calls=1)),
        checks=(completed(),),
        script=(StubTurn(tool_calls=(StubToolCall(name="lookup"),)), StubTurn(text="x")),
        tools=(EvalTool("lookup"),),
    )
    outcome = await _one(over)
    assert outcome.failures == ("expected completion, got failed (budget_exhausted)",)
    as_failure = EvalCase(
        "fails_as_expected",
        _ask(budget=RunBudget(max_model_calls=1)),
        checks=(failed(RunFailure.BUDGET_EXHAUSTED),),
        script=(StubTurn(tool_calls=(StubToolCall(name="lookup"),)), StubTurn(text="x")),
        tools=(EvalTool("lookup"),),
    )
    assert (await _one(as_failure)).passed
    wrong_failure = EvalCase(
        "fails_otherwise",
        _ask(budget=RunBudget(max_model_calls=1)),
        checks=(failed(RunFailure.MODEL_REFUSED),),
        script=(StubTurn(tool_calls=(StubToolCall(name="lookup"),)), StubTurn(text="x")),
        tools=(EvalTool("lookup"),),
    )
    assert (await _one(wrong_failure)).failures == (
        "expected a failure for model_refused, got budget_exhausted",
    )
    assert (await _one(EvalCase("no", _ask(), checks=(failed(),), script=_answer()))).failures == (
        "expected a failure, got completed",
    )


def _hand_off_case(name: str, *checks: Any) -> EvalCase:
    return EvalCase(
        name,
        _ask(hand_offs=frozenset({"review"})),
        checks=checks,
        script=(
            StubTurn(tool_calls=(StubToolCall(name="hand_off", input={"task_class": "review"}),)),
            StubTurn(text="reviewed"),
        ),
    )


async def test_handed_off() -> None:
    assert (await _one(_hand_off_case("handed", handed_off("review"), completed()))).passed
    assert (await _one(_hand_off_case("any", handed_off()))).passed
    wrong = await _one(_hand_off_case("elsewhere", handed_off("triage")))
    assert wrong.failures == ("expected a hand-off to triage, got review",)


class Verdict(BaseModel):
    verdict: str
    confidence: int


async def test_output_field() -> None:
    contract = OutputContract(model=Verdict, name="verdict")
    case = EvalCase(
        "verdict",
        _ask(contract=contract),
        checks=(output_field("verdict", "benign"), output_field("confidence", 3)),
        script=_answer('{"verdict": "benign", "confidence": 2}'),
    )
    outcome = await _one(case)
    assert outcome.failures == ("output field confidence did not hold the expected value",)
    no_output = await _one(
        EvalCase("plain", _ask(), checks=(output_field("verdict", "x"),), script=_answer())
    )
    assert no_output.failures == ("the run has no output",)


def _tool_case(name: str, *checks: Any, approvals: bool = False) -> EvalCase:
    return EvalCase(
        name,
        _ask(),
        checks=checks,
        script=(
            StubTurn(tool_calls=(StubToolCall(name="disable_rule", input={"rule": "r-1"}),)),
            StubTurn(text="done"),
        ),
        tools=(EvalTool("disable_rule", requires_approval=approvals),),
    )


async def test_never_executed_and_executed() -> None:
    ran = await _one(_tool_case("runs", never_executed("disable_rule"), executed("lookup")))
    assert ran.failures == ("tool disable_rule was executed", "tool lookup was not executed")
    assert (await _one(_tool_case("ok", executed("disable_rule")))).passed


class _Refuse:
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=False)


async def test_an_approval_handler_is_honoured_by_the_harness() -> None:
    case = _tool_case("refused", never_executed("disable_rule"), completed(), approvals=True)
    assert (await _one(case, approvals=_Refuse())).passed
    unanswered = await _one(case)  # no handler: a marked tool never runs
    assert unanswered.passed


async def test_no_text_in_events() -> None:
    script = (
        StubTurn(tool_calls=(StubToolCall(name="read_note", input={"id": "n-1"}),)),
        StubTurn(text=f"The note says {CANARY}."),
    )
    tools = (EvalTool("read_note", response=f"note: {CANARY}"),)
    quiet = EvalCase(
        "quiet", _ask(), checks=(no_text_in_events(CANARY),), script=script, tools=tools
    )
    assert (await _one(quiet)).passed  # the default capture policy keeps content out
    loud = await _one(quiet, capture=CapturePolicy(prompts=True, tool_content=True))
    assert not loud.passed
    [failure] = loud.failures
    assert CANARY not in failure
    assert failure == f"a {len(CANARY)}-character text reached an event"


async def test_observation_holds_the_stream_the_sink_and_the_executed_tools() -> None:
    seen: list[Observation] = []

    def keep(observation: Observation) -> str | None:
        seen.append(observation)
        return None

    await _one(_tool_case("observed", keep))
    [observation] = seen
    assert observation.executed == ("disable_rule",)
    stream = {event.type for event in observation.events}
    assert {"run_started", "tool_call", "run_finished"} <= stream
    delivered = {event.type for event in observation.delivered}
    assert {"model_call_finished", "run_finished"} <= delivered


# --- The live harness, on a runtime the product started ---------------------------------


async def _runtime(task_classes: list[str], text: str = "live answer") -> Runtime:
    turns = iter([StubTurn(text=text)] * 10)
    return await start_stub_runtime(
        lambda request, index: next(turns), [TaskClass(name=n) for n in task_classes]
    )


async def test_the_live_harness_runs_live_cases_on_the_given_runtime() -> None:
    runtime = await _runtime(["chat"])
    try:
        live = EvalCase(
            "live",
            _ask(),
            checks=(completed(),),
            stub_checks=(handed_off(),),  # never applied live
            script=_answer("scripted, never used live"),
            live=True,
        )
        report = await LiveHarness(runtime).run([live])
    finally:
        await runtime.aclose()
    [outcome] = report.outcomes
    assert outcome.passed, outcome.failures
    assert outcome.result is not None and outcome.result.text == "live answer"


async def test_a_case_not_marked_live_is_skipped_with_a_stated_reason() -> None:
    runtime = await _runtime(["chat"])
    try:
        stub_only = EvalCase("stub_only", _ask(), checks=(completed(),), script=_answer())
        report = await LiveHarness(runtime).run([stub_only])
    finally:
        await runtime.aclose()
    [outcome] = report.outcomes
    assert not outcome.passed
    assert outcome.skipped == "not marked live: its checks hold only for the scripted stub"
    assert (report.ran, report.skipped, report.score) == (0, 1, 0.0)


async def test_a_live_case_without_an_assignment_is_skipped_with_a_stated_reason() -> None:
    runtime = await _runtime(["chat"])
    try:
        case = EvalCase(
            "unassigned",
            _ask(task_class="triage"),
            checks=(completed(),),
            live=True,
        )
        handing = EvalCase(
            "unassigned_target",
            _ask(hand_offs=frozenset({"review"})),
            checks=(completed(),),
            live=True,
        )
        report = await LiveHarness(runtime).run([case, handing])
    finally:
        await runtime.aclose()
    assert [o.skipped for o in report.outcomes] == [
        "task class triage has no assignment in the runtime",
        "task class review has no assignment in the runtime",
    ]


async def test_the_score_counts_only_the_cases_that_ran() -> None:
    runtime = await _runtime(["chat"])
    try:
        live = EvalCase("live", _ask(), checks=(completed(),), live=True)
        stub_only = EvalCase("stub_only", _ask(), checks=(completed(),), script=_answer())
        report = await LiveHarness(runtime).run([live, stub_only])
    finally:
        await runtime.aclose()
    assert (report.passed, report.failed, report.skipped, report.ran) == (1, 0, 1, 1)
    assert report.score == 1.0
    assert report.failures() == {}


@pytest.mark.parametrize(
    ("returned", "reason"),
    [
        (False, "a check returned bool"),
        (0, "a check returned int"),
        ("", "a check returned an empty reason"),
        ([], "a check returned list"),
    ],
)
async def test_only_none_is_a_pass(returned: object, reason: str) -> None:
    case = EvalCase("odd", _ask(), checks=(lambda o: returned,), script=_answer())  # type: ignore[arg-type,return-value]
    outcome = await _one(case)
    assert not outcome.passed
    assert outcome.failures == (reason,)


async def test_an_output_field_failure_never_repeats_the_expected_value() -> None:
    contract = OutputContract(model=Verdict, name="verdict")
    case = EvalCase(
        "secret_value",
        _ask(contract=contract),
        checks=(output_field("verdict", CANARY),),
        script=_answer('{"verdict": "benign", "confidence": 2}'),
    )
    outcome = await _one(case)
    assert not outcome.passed
    assert CANARY not in repr(outcome.failures)
    assert outcome.failures == ("output field verdict did not hold the expected value",)
