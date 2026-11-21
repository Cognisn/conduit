"""The budget meter: model calls, tool calls, tokens (chosen counts) and wall time."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conduit.agent.budgets import BudgetKind, BudgetMeter, RunBudget, TokenCount
from conduit.core.types import Usage


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _usage(i: int = 0, o: int = 0, r: int = 0, w: int = 0) -> Usage:
    return Usage(input_tokens=i, output_tokens=o, cache_read_tokens=r, cache_write_tokens=w)


def test_model_calls_are_refused_once_the_budget_is_spent() -> None:
    meter = BudgetMeter(RunBudget(max_model_calls=2), Clock())
    assert meter.begin_model_call() is None
    assert meter.begin_model_call() is None
    exhausted = meter.begin_model_call()
    assert exhausted is not None
    assert (exhausted.kind, exhausted.limit, exhausted.used) == (BudgetKind.MODEL_CALLS, 2, 2)
    assert meter.model_calls == 2


def test_tool_calls_are_refused_once_the_budget_is_spent() -> None:
    meter = BudgetMeter(RunBudget(max_tool_calls=1), Clock())
    assert meter.begin_tool_call() is None
    exhausted = meter.begin_tool_call()
    assert exhausted is not None
    assert (exhausted.kind, exhausted.limit, exhausted.used) == (BudgetKind.TOOL_CALLS, 1, 1)
    assert meter.tool_calls == 1


def test_a_zero_tool_budget_refuses_the_first_tool_call() -> None:
    meter = BudgetMeter(RunBudget(max_tool_calls=0), Clock())
    assert meter.begin_tool_call() is not None


def test_every_count_is_charged_by_default() -> None:
    meter = BudgetMeter(RunBudget(max_tokens=10), Clock())
    assert meter.record_usage(_usage(1, 2, 3, 4)) is None  # exactly 10: not over
    assert meter.tokens_used == 10
    exhausted = meter.begin_model_call()
    assert exhausted is not None
    assert (exhausted.kind, exhausted.limit, exhausted.used) == (BudgetKind.TOKENS, 10, 10)


def test_usage_over_the_token_budget_is_exhausted_at_once() -> None:
    meter = BudgetMeter(RunBudget(max_tokens=10), Clock())
    exhausted = meter.record_usage(_usage(5, 6))
    assert exhausted is not None
    assert (exhausted.kind, exhausted.used) == (BudgetKind.TOKENS, 11)


@pytest.mark.parametrize(
    ("counted", "used"),
    [
        ({TokenCount.INPUT}, 1),
        ({TokenCount.OUTPUT}, 2),
        ({TokenCount.CACHE_READ}, 3),
        ({TokenCount.CACHE_WRITE}, 4),
        ({TokenCount.INPUT, TokenCount.CACHE_WRITE}, 5),
        ({TokenCount.OUTPUT, TokenCount.CACHE_READ, TokenCount.CACHE_WRITE}, 9),
    ],
)
def test_the_product_chooses_which_counts_are_charged(counted: set[TokenCount], used: int) -> None:
    meter = BudgetMeter(RunBudget(max_tokens=100, counted_tokens=frozenset(counted)), Clock())
    meter.record_usage(_usage(1, 2, 3, 4))
    assert meter.tokens_used == used


def test_the_total_usage_keeps_all_four_counts_whichever_are_charged() -> None:
    meter = BudgetMeter(RunBudget(counted_tokens=frozenset({TokenCount.OUTPUT})), Clock())
    meter.record_usage(_usage(1, 2, 3, 4))
    meter.record_usage(_usage(10, 20, 30, 40))
    assert meter.usage == _usage(11, 22, 33, 44)


def test_no_token_budget_means_no_token_refusal() -> None:
    meter = BudgetMeter(RunBudget(), Clock())
    assert meter.record_usage(_usage(10**9)) is None
    assert meter.begin_model_call() is None


def test_the_wall_time_budget_runs_from_the_start() -> None:
    clock = Clock()
    meter = BudgetMeter(RunBudget(max_wall_time_s=30), clock)
    clock.now += 10
    assert meter.remaining_s() == pytest.approx(20)
    assert meter.wall_time_exhausted() is None
    clock.now += 20
    exhausted = meter.wall_time_exhausted()
    assert exhausted is not None
    assert exhausted.kind is BudgetKind.WALL_TIME
    assert exhausted.limit == 30
    assert exhausted.used == pytest.approx(30)


def test_a_model_call_after_the_deadline_is_refused_as_wall_time() -> None:
    clock = Clock()
    meter = BudgetMeter(RunBudget(max_wall_time_s=5), clock)
    clock.now += 6
    exhausted = meter.begin_model_call()
    assert exhausted is not None
    assert exhausted.kind is BudgetKind.WALL_TIME
    assert meter.model_calls == 0


def test_without_a_wall_time_budget_there_is_no_deadline() -> None:
    meter = BudgetMeter(RunBudget(), Clock())
    assert meter.remaining_s() is None
    assert meter.wall_time_exhausted() is None


def test_an_empty_choice_of_counts_is_refused() -> None:
    with pytest.raises(ValidationError):
        RunBudget(counted_tokens=frozenset())


@pytest.mark.parametrize(
    "fields",
    [
        {"max_model_calls": 0},
        {"max_tool_calls": -1},
        {"max_tokens": 0},
        {"max_wall_time_s": 0},
        {"unknown": 1},
    ],
)
def test_a_budget_out_of_range_is_refused(fields: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RunBudget.model_validate(fields)


def test_a_budget_is_buildable_from_a_mapping() -> None:
    budget = RunBudget.model_validate(
        {"max_model_calls": 4, "max_tokens": 900, "counted_tokens": ["output", "input"]}
    )
    assert budget.counted_tokens == frozenset({TokenCount.OUTPUT, TokenCount.INPUT})
