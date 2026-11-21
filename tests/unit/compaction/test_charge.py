"""Summary calls charged to the run's budget (D0085).

The runner's ``CompactionRequest`` carries a charge; ``Summarise`` counts its call through
it before making it, and charges the call's usage once it is made. A summary call the
run's budget refuses is never made, and the run ends as for any over-budget call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from conduit.agent import AgentRunner, AgentTask, BudgetKind, RunBudget, RunFailure
from conduit.agent.events import BudgetExhaustedEvent
from conduit.compaction import CompactionEvent, Compactor, Summarise
from conduit.core.events import ModelCallFinishedEvent
from conduit.core.tools import TaskTools, ToolCallContext, ToolCatalogue, ToolOutcome
from conduit.core.types import Message, ModelRequest, Usage
from conduit.providers.stub import StubToolCall, StubTurn
from tests.unit.compaction.helpers import (
    Request,
    ScriptLog,
    Sink,
    conversation,
    filler,
    request,
    runtime,
)

SUMMARY_USAGE = Usage(input_tokens=900, output_tokens=60, cache_read_tokens=0, cache_write_tokens=0)


def _summariser() -> ScriptLog:
    return ScriptLog(
        lambda request, index: StubTurn(text="SUMMARY of it all.", usage=SUMMARY_USAGE)
    )


@dataclass
class Charge:
    """A charge that allows ``allow`` calls, then refuses, and records what it was given."""

    allow: int = 1
    cap: int | None = None
    accept: bool = True
    begun: list[int] = field(default_factory=list)
    recorded: list[Usage] = field(default_factory=list)

    def begin(self, max_output_tokens: int) -> int | None:
        self.begun.append(max_output_tokens)
        if len(self.begun) > self.allow:
            return None
        return max_output_tokens if self.cap is None else min(self.cap, max_output_tokens)

    def record(self, usage: Usage) -> bool:
        self.recorded.append(usage)
        return self.accept


def charged(shown: Request, charge: object) -> Request:
    return replace(shown, charge=charge)


async def _setup() -> tuple[Summarise, ScriptLog, Any]:
    summariser = _summariser()
    rt = await runtime(summariser=summariser)
    strategy = Summarise(
        rt, task_class="summary", threshold=0.1, keep_turns=1, max_summary_tokens=64
    )
    return strategy, summariser, rt.assignments["chat"]


async def test_a_summary_call_is_counted_and_charged() -> None:
    strategy, summariser, chat = await _setup()
    charge = Charge(cap=30)
    sink = Sink()
    compacted = await Compactor([strategy], events=sink).before_model_call(
        charged(request(chat, conversation(4)), charge)
    )
    assert compacted is not None
    assert charge.begun == [strategy.summary_allowance]
    [sent] = summariser.requests
    assert sent.max_output_tokens == 30  # the charge's cap, not the strategy's own
    assert charge.recorded == [SUMMARY_USAGE]


async def test_a_summary_call_the_budget_refuses_is_never_made() -> None:
    strategy, summariser, chat = await _setup()
    charge = Charge(allow=0)
    sink = Sink()
    compacted = await Compactor([strategy], events=sink).before_model_call(
        charged(request(chat, conversation(4)), charge)
    )
    assert compacted is None
    assert summariser.requests == []
    assert charge.recorded == []
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "budget_exhausted")


async def test_a_summary_past_the_budget_is_not_applied() -> None:
    strategy, summariser, chat = await _setup()
    charge = Charge(accept=False)
    sink = Sink()
    compacted = await Compactor([strategy], events=sink).before_model_call(
        charged(request(chat, conversation(4)), charge)
    )
    assert compacted is None
    assert len(summariser.requests) == 1
    assert charge.recorded == [SUMMARY_USAGE]
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "budget_exhausted")


async def test_without_a_charge_the_summary_call_goes_ahead() -> None:
    strategy, summariser, chat = await _setup()
    compacted = await Compactor([strategy]).before_model_call(request(chat, conversation(4)))
    assert compacted is not None
    [sent] = summariser.requests
    assert sent.max_output_tokens == strategy.summary_allowance


async def test_a_charge_that_is_not_a_charge_is_ignored() -> None:
    strategy, summariser, chat = await _setup()
    shown = charged(request(chat, conversation(4)), "not a charge")
    compacted = await Compactor([strategy]).before_model_call(shown)
    assert compacted is not None
    assert len(summariser.requests) == 1


# Through the runner -------------------------------------------------------------------


class Lookup:
    name = "lookup"
    description = "Look up a synthetic release item."
    task_classes = frozenset({"chat"})
    requires_approval = False
    input_schema: ClassVar[dict[str, Any]] = {"type": "object"}


async def _execute(tool: Any, arguments: dict[str, Any], context: ToolCallContext) -> ToolOutcome:
    return ToolOutcome(ok=True, content=f"{arguments.get('q')}: " + filler(1_200, "row"))


def _tools() -> dict[str, TaskTools]:
    return {"chat": TaskTools(ToolCatalogue.from_tools([Lookup()]).registry_for("chat"), _execute)}


def _primary(calls: int) -> ScriptLog:
    def play(request: ModelRequest, index: int) -> StubTurn:
        if index < calls:
            return StubTurn(tool_calls=(StubToolCall(name="lookup", input={"q": f"item {index}"}),))
        return StubTurn(text="All items are checked.")

    return ScriptLog(play)


def _task(budget: RunBudget) -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("Check."),), budget=budget)


async def test_summary_calls_count_against_the_run_s_budget() -> None:
    primary, summariser = _primary(6), _summariser()
    calls = Sink()
    rt = await runtime(primary, summariser, events=calls)
    strategy = Summarise(rt, task_class="summary", threshold=0.3, keep_turns=1)
    runner = AgentRunner(rt, tools=_tools(), compaction=Compactor([strategy]))
    result = await runner.run(_task(RunBudget(max_model_calls=50)))
    assert result.failure is None
    assert summariser.requests, "no summary was asked for"
    assert result.model_calls == len(primary.requests) + len(summariser.requests)
    # The run's usage is every call's, the summaries' included.
    finished = calls.of(ModelCallFinishedEvent)
    assert len(finished) == result.model_calls
    total = Usage.zero()
    for event in finished:
        assert event.usage is not None
        total = total + event.usage
    assert result.usage == total


async def test_a_summary_call_past_the_model_call_budget_ends_the_run() -> None:
    # A first run finds the call a summary is first asked before; a second, on a fresh
    # runtime (the stub counts calls per runtime), allows exactly that many calls.
    primary, summariser = _primary(6), _summariser()
    rt = await runtime(primary, summariser)
    strategy = Summarise(rt, task_class="summary", threshold=0.3, keep_turns=1)
    probe = await AgentRunner(rt, tools=_tools(), compaction=Compactor([strategy])).run(
        _task(RunBudget(max_model_calls=50))
    )
    assert probe.failure is None
    limit = _first_summary_call(primary.requests)

    primary, summariser = _primary(6), _summariser()
    rt = await runtime(primary, summariser)
    strategy = Summarise(rt, task_class="summary", threshold=0.3, keep_turns=1)
    sink = Sink()
    runner = AgentRunner(rt, tools=_tools(), compaction=Compactor([strategy]), events=sink)
    result = await runner.run(_task(RunBudget(max_model_calls=limit)))
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.MODEL_CALLS
    assert summariser.requests == []
    assert len(primary.requests) == limit - 1
    [exhausted] = sink.of(BudgetExhaustedEvent)
    assert (exhausted.budget, exhausted.limit, exhausted.used) == (
        BudgetKind.MODEL_CALLS,
        limit,
        limit,
    )


async def test_summary_usage_past_the_token_budget_ends_the_run() -> None:
    primary = _primary(6)
    summariser = ScriptLog(
        lambda request, index: StubTurn(
            text="SUMMARY.",
            usage=Usage(
                input_tokens=10_000_000, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
            ),
        )
    )
    rt = await runtime(primary, summariser)
    strategy = Summarise(rt, task_class="summary", threshold=0.3, keep_turns=1)
    sink = Sink()
    runner = AgentRunner(rt, tools=_tools(), compaction=Compactor([strategy]), events=sink)
    result = await runner.run(_task(RunBudget(max_model_calls=50, max_tokens=5_000_000)))
    assert result.failure is RunFailure.BUDGET_EXHAUSTED
    assert result.budget is BudgetKind.TOKENS
    assert len(summariser.requests) == 1
    # No call follows the over-budget summary.
    assert not any(_starts_with_summary(r) for r in primary.requests)
    [exhausted] = sink.of(BudgetExhaustedEvent)
    assert exhausted.budget is BudgetKind.TOKENS
    assert exhausted.used > 5_000_000


def _first_summary_call(primary: list[ModelRequest]) -> int:
    """The run's call count at which the first summary call is asked for: the calls
    before the first compacted call, plus that call, counted before its summary."""
    compacted = next(i for i, r in enumerate(primary) if _starts_with_summary(r))
    return compacted + 1


def _starts_with_summary(sent: ModelRequest) -> bool:
    first = sent.messages[0].content[0]
    return "SUMMARY" in getattr(first, "text", "")
