"""Compaction strategies: what replaces the older part of a conversation.

A strategy has a ``name`` (an identifier, recorded on compaction points and events), a
``threshold`` (the share of the assigned model's context window at which it triggers,
above 0 and at most 1), and ``keep_turns`` (how many of the last assistant turns stay as
they are). Given the messages being replaced, it returns the summary text. The
``Compactor`` decides when to compact and where to cut; the strategy only writes the
text, so a product can supply its own strategy with the same three attributes and one
``summarise`` method.

Three are supplied, mechanism only (C1): the texts they write are defaults a product
replaces with its own.

- ``KeepLastTurns`` drops the older turns and leaves a fixed notice in their place.
- ``DropToolResults`` keeps the older turns' words, as text, with every tool result left
  out (tool results are usually the bulk of a conversation).
- ``Summarise`` asks a model to summarise the older turns, through a task class the
  product declares and assigns. Its capability floor is checked when the strategy is
  built, from the runtime's resolved assignment, so a summary model below it is refused
  at start-up and never part-way through a task (C18). The conversation must also fit
  the summary model's context window, or the compaction fails before any call.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from conduit.compaction.conversation import estimate_tokens, render_messages
from conduit.compaction.errors import CompactionFailed
from conduit.core.errors import (
    CapabilityRefused,
    ConfigurationError,
    ProviderError,
    ProviderFailure,
)
from conduit.core.identifiers import IDENTIFIER_PATTERN
from conduit.core.runtime import TaskBinding
from conduit.core.tasks import CapabilityFloor, ResolvedAssignment
from conduit.core.types import Message, ModelResponse

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_NOTICE",
    "CompactionStrategy",
    "DropToolResults",
    "KeepLastTurns",
    "StrategyContext",
    "Summarise",
    "SummaryBindings",
    "check_strategy",
]

DEFAULT_NOTICE = (
    "Earlier messages in this conversation were removed to keep it within the model's "
    "context window. Only the most recent turns follow."
)
DROPPED_HEADER = "The earlier conversation, with its tool results left out:\n\n"
SUMMARY_HEADER = "A summary of the earlier conversation:\n\n"
DEFAULT_INSTRUCTIONS = (
    "Summarise the conversation below so that it can continue without it. Keep, in full: "
    "decisions and their reasons, open questions and unresolved problems, the current "
    "state of any work (names, paths, identifiers, figures), and the user's stated "
    "preferences and constraints. Keep only the findings of tool results and the outcome "
    "of finished tasks. Leave out pleasantries and anything later superseded. Write the "
    "summary only, with no preamble."
)


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """What a strategy knows about the call being compacted: the run, and the model the
    conversation goes to (its catalogue entry gives the context window)."""

    run_id: str | None
    assignment: ResolvedAssignment


@runtime_checkable
class CompactionStrategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    @property
    def keep_turns(self) -> int: ...

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        """The text that replaces ``replaced``. Raise ``CompactionFailed`` to leave the
        conversation as it is."""
        ...


def check_strategy(name: str, threshold: float, keep_turns: int) -> None:
    """Refuse a strategy's settings unless they are usable: an identifier name of at most
    128 characters, a threshold above 0 and at most 1, and at least one kept turn."""
    if not isinstance(name, str) or len(name) > 128 or not re.fullmatch(IDENTIFIER_PATTERN, name):
        raise ConfigurationError("a strategy name must be an identifier (no spaces)")
    if not (math.isfinite(threshold) and 0 < threshold <= 1):
        raise ConfigurationError("a compaction threshold must be above 0 and at most 1")
    if keep_turns < 1:
        raise ConfigurationError("keep_turns must be at least 1")


class KeepLastTurns:
    """Drop the older turns and leave ``notice`` in their place."""

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        keep_turns: int = 4,
        notice: str = DEFAULT_NOTICE,
        name: str = "keep_last_turns",
    ) -> None:
        check_strategy(name, threshold, keep_turns)
        self.name = name
        self.threshold = threshold
        self.keep_turns = keep_turns
        self.notice = notice

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        return self.notice


class DropToolResults:
    """Keep the older turns' words as text, with every tool result left out."""

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        keep_turns: int = 4,
        header: str = DROPPED_HEADER,
        max_message_chars: int = 2_000,
        name: str = "drop_tool_results",
    ) -> None:
        check_strategy(name, threshold, keep_turns)
        self.name = name
        self.threshold = threshold
        self.keep_turns = keep_turns
        self.header = header
        self.max_message_chars = max_message_chars

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        rendered = render_messages(
            replaced, tool_result_chars=None, max_message_chars=self.max_message_chars
        )
        return self.header + rendered


class SummaryBindings(Protocol):
    """Where the summariser finds its task class: a started ``Runtime``."""

    @property
    def assignments(self) -> Mapping[str, ResolvedAssignment]: ...

    def for_task(self, task_class: str) -> TaskBinding: ...


class Summarise:
    """Summarise the older turns with the model assigned to ``task_class``.

    ``floor`` is the least the summary model must offer; it is checked here, once. The
    request carries the run's id, so the provider registry's model-call events tie the
    summary call to the run. Its tokens are metered by the registry like any call, but
    are not charged to the run's own budget.
    """

    def __init__(
        self,
        bindings: SummaryBindings,
        *,
        task_class: str,
        floor: CapabilityFloor | None = None,
        threshold: float = 0.8,
        keep_turns: int = 4,
        instructions: str = DEFAULT_INSTRUCTIONS,
        header: str = SUMMARY_HEADER,
        tool_result_chars: int = 500,
        max_message_chars: int = 3_000,
        max_summary_tokens: int | None = None,
        name: str = "summarise",
    ) -> None:
        check_strategy(name, threshold, keep_turns)
        assignment = bindings.assignments.get(task_class)
        if assignment is None:
            raise ConfigurationError(f"summary task class {task_class!r} has no assignment")
        shortfalls = (floor or CapabilityFloor()).shortfalls(assignment.entry)
        if shortfalls:
            raise CapabilityRefused(
                f"{task_class}: model {assignment.model!r} is below the summary floor: "
                + "; ".join(shortfalls)
            )
        if max_summary_tokens is not None and max_summary_tokens < 1:
            raise ConfigurationError("max_summary_tokens must be positive")
        self._bindings = bindings
        self.task_class = task_class
        self.name = name
        self.threshold = threshold
        self.keep_turns = keep_turns
        self.instructions = instructions
        self.header = header
        self.tool_result_chars = tool_result_chars
        self.max_message_chars = max_message_chars
        self.max_summary_tokens = max_summary_tokens

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        binding = self._bindings.for_task(self.task_class)
        assignment = binding.assignment
        output = assignment.max_output_tokens
        if self.max_summary_tokens is not None:
            output = min(output, self.max_summary_tokens)
        rendered = render_messages(
            replaced,
            tool_result_chars=self.tool_result_chars,
            max_message_chars=self.max_message_chars,
        )
        prompt = Message.user(f"{self.instructions}\n\nCONVERSATION:\n\n{rendered}")
        if estimate_tokens(messages=(prompt,)) + output > assignment.entry.context_window:
            raise CompactionFailed("summary_too_large")
        request = binding.request(
            messages=(prompt,), max_output_tokens=output, run_id=context.run_id
        )
        response: ModelResponse | None = None
        failure: ProviderFailure | None = None
        try:
            response = await binding.provider.complete(request)
        except ProviderError as error:
            failure = error.failure
        if response is None:  # raised after the handler, so nothing is chained (D0040)
            raise CompactionFailed("summary_failed", failure=failure)
        text = response.text.strip()
        if not text:
            raise CompactionFailed("summary_empty")
        return self.header + text
