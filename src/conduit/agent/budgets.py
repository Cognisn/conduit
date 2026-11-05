"""Run budgets: model calls, tool calls, tokens and wall time.

A ``RunBudget`` is the product's limit for one run, including any hand-offs within it.
``BudgetMeter`` keeps one run's counts and says when a limit is reached:

- **Model calls:** a call is refused once ``max_model_calls`` have been made.
- **Tool calls:** a call is refused once ``max_tool_calls`` have been made. Every tool
  call the model asks for counts, including ones refused (unknown, not approved).
- **Tokens:** usage keeps all four counts apart; the product chooses which of them are
  charged to ``max_tokens`` (``counted_tokens``, every count by default). Usage beyond
  the budget ends the run at once, and no further call is made once it is spent.
- **Wall time:** measured on the runner's clock from the start of the run. Each model
  call, approval and tool call is bounded by the time left, and refused when none is.

A run that reaches a limit stops cleanly: it never concludes on a partial answer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from conduit.core.settings import SettingsModel
from conduit.core.types import Usage


class BudgetKind(StrEnum):
    MODEL_CALLS = "model_calls"
    TOOL_CALLS = "tool_calls"
    TOKENS = "tokens"
    WALL_TIME = "wall_time"


class TokenCount(StrEnum):
    """One of the four usage counts, by name."""

    INPUT = "input"  # uncached input
    OUTPUT = "output"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"


_ALL_COUNTS = frozenset(TokenCount)


class RunBudget(SettingsModel):
    """One run's limits. Buildable in code or from a mapping."""

    max_model_calls: PositiveInt = 20
    max_tool_calls: NonNegativeInt = 50
    max_tokens: PositiveInt | None = None
    counted_tokens: frozenset[TokenCount] = Field(default=_ALL_COUNTS, min_length=1)
    max_wall_time_s: PositiveFloat | None = None


@dataclass(frozen=True, slots=True)
class Exhausted:
    """A limit reached: which, the limit and what was used."""

    kind: BudgetKind
    limit: float
    used: float


def charged(usage: Usage, counts: frozenset[TokenCount]) -> int:
    """The tokens of ``usage`` charged to a budget counting ``counts``."""
    values = {
        TokenCount.INPUT: usage.input_tokens,
        TokenCount.OUTPUT: usage.output_tokens,
        TokenCount.CACHE_READ: usage.cache_read_tokens,
        TokenCount.CACHE_WRITE: usage.cache_write_tokens,
    }
    return sum(values[count] for count in counts)


class BudgetMeter:
    """One run's counts against its budget. Not shared between runs."""

    def __init__(self, budget: RunBudget, clock: Callable[[], float]) -> None:
        self.budget = budget
        self._clock = clock
        self._started = clock()
        self.model_calls = 0
        self.tool_calls = 0
        self.usage = Usage.zero()

    @property
    def tokens_used(self) -> int:
        return charged(self.usage, self.budget.counted_tokens)

    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def remaining_s(self) -> float | None:
        """Seconds left before the wall-time limit, or ``None`` when there is none."""
        if self.budget.max_wall_time_s is None:
            return None
        return self.budget.max_wall_time_s - self.elapsed_s()

    def wall_time_exhausted(self) -> Exhausted | None:
        limit = self.budget.max_wall_time_s
        if limit is None:
            return None
        elapsed = self.elapsed_s()
        if elapsed >= limit:
            return Exhausted(BudgetKind.WALL_TIME, limit, elapsed)
        return None

    def timed_out(self) -> Exhausted:
        """The wall-time limit, reached part-way through a bounded wait."""
        limit = self.budget.max_wall_time_s or 0.0
        return Exhausted(BudgetKind.WALL_TIME, limit, max(self.elapsed_s(), limit))

    def begin_model_call(self) -> Exhausted | None:
        """Count a model call about to be made, or say which limit refuses it."""
        if self.model_calls >= self.budget.max_model_calls:
            return Exhausted(BudgetKind.MODEL_CALLS, self.budget.max_model_calls, self.model_calls)
        limit = self.budget.max_tokens
        if limit is not None and self.tokens_used >= limit:
            return Exhausted(BudgetKind.TOKENS, limit, self.tokens_used)
        wall = self.wall_time_exhausted()
        if wall is not None:
            return wall
        self.model_calls += 1
        return None

    def record_usage(self, usage: Usage) -> Exhausted | None:
        """Add a call's usage; the token limit if it is now passed."""
        self.usage = self.usage + usage
        limit = self.budget.max_tokens
        if limit is not None and self.tokens_used > limit:
            return Exhausted(BudgetKind.TOKENS, limit, self.tokens_used)
        return None

    def begin_tool_call(self) -> Exhausted | None:
        """Count a tool call about to be made, or say which limit refuses it."""
        if self.tool_calls >= self.budget.max_tool_calls:
            return Exhausted(BudgetKind.TOOL_CALLS, self.budget.max_tool_calls, self.tool_calls)
        wall = self.wall_time_exhausted()
        if wall is not None:
            return wall
        self.tool_calls += 1
        return None
