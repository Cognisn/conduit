"""A run's result, its typed failures, and the agent layer's errors.

A run ends ``COMPLETED`` with its answer, or ``FAILED`` with a ``RunFailure`` saying why.
Failing is not an exception: ``RunResult.raise_for_failure`` raises one for a product
that prefers it. Every error here is raised fresh, outside any handler, so none keeps
the exception it replaced (no ``__cause__``, no ``__context__``), and its message names
identifiers and outcomes only, never content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from conduit.agent.budgets import BudgetKind
from conduit.core.errors import ConduitError, ProviderFailure
from conduit.core.types import Message, StopReason, Usage


class RunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class RunFailure(StrEnum):
    BUDGET_EXHAUSTED = "budget_exhausted"  # see RunResult.budget for which
    OUTPUT_INVALID = "output_invalid"  # the answer failed its contract twice
    MODEL_FAILED = "model_failed"  # the provider failed; see RunResult.provider_failure
    MODEL_REFUSED = "model_refused"  # the model declined (stop reason refusal)
    TOOL_FAILED = "tool_failed"  # the product's tool executor raised


class AgentError(ConduitError):
    """Base for the agent layer's errors."""


class RunRefused(AgentError):
    """A run cannot start (an unassigned task class, an empty conversation, a hand-off
    target without an assignment). Raised before anything runs."""


class RunIncomplete(AgentError, RuntimeError):
    """A run's result was asked for before the run finished, or a run was streamed twice."""


class CompactionRefused(AgentError):
    """The compaction hook returned something the runner cannot send to a model."""


class SystemPromptRefused(AgentError):
    """The system hook returned something other than system blocks."""


class RunFailed(AgentError):
    """A run ended without its answer. Raised only by ``RunResult.raise_for_failure``."""

    def __init__(self, failure: RunFailure, run_id: str) -> None:
        super().__init__(f"run {run_id} failed: {failure.value}")
        self.failure = failure
        self.run_id = run_id


class BudgetExhausted(RunFailed):
    def __init__(self, budget: BudgetKind, run_id: str) -> None:
        super().__init__(RunFailure.BUDGET_EXHAUSTED, run_id)
        self.args = (f"run {run_id} failed: the {budget.value} budget was exhausted",)
        self.budget = budget


class OutputContractFailed(RunFailed):
    def __init__(self, run_id: str) -> None:
        super().__init__(RunFailure.OUTPUT_INVALID, run_id)


@dataclass(frozen=True, slots=True)
class RunResult:
    """How a run ended. ``text`` and ``output`` are set only when it completed.

    ``task_class`` is the class the run ended on (after any hand-offs, which are listed
    in order in ``hand_offs``). ``messages`` is the conversation as it stood at the end.
    ``usage`` is the run's total, all four counts.
    """

    run_id: str
    status: RunStatus
    task_class: str
    text: str
    output: BaseModel | None
    failure: RunFailure | None
    budget: BudgetKind | None
    provider_failure: ProviderFailure | None
    stop_reason: StopReason | None
    usage: Usage
    model_calls: int
    tool_calls: int
    hand_offs: tuple[str, ...]
    messages: tuple[Message, ...]

    def raise_for_failure(self) -> None:
        """Raise the run's failure as a ``RunFailed``; nothing if it completed."""
        if self.failure is None:
            return
        if self.failure is RunFailure.BUDGET_EXHAUSTED and self.budget is not None:
            raise BudgetExhausted(self.budget, self.run_id)
        if self.failure is RunFailure.OUTPUT_INVALID:
            raise OutputContractFailed(self.run_id)
        raise RunFailed(self.failure, self.run_id)
