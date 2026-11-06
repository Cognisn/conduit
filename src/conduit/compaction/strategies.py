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

  The older turns are sent as they are, as messages, with tool calls and results in
  their own blocks (each result trimmed) and the instructions added to the last user
  turn. The provider registry therefore holds the call's tool content to the capture
  policy's tool-content setting, as for any call (C19). Because the request carries tool
  blocks it also carries the definitions of the tools those turns used (the run's own,
  or a minimal one, an object with any properties, for a tool the run no longer offers,
  after a hand-off for example), so the default floor asks for a model with tools.
  Requests have no tool choice, so a summary model may still answer with a tool call
  rather than text; that is reported as ``summary_empty`` and the compaction is tried
  again on a later call.

A strategy may also declare ``summary_allowance``: about how many tokens its summary may
take. The ``Compactor`` counts it, with the fixed prefix and the kept tail, before
compacting, so it never asks for a summary that could not bring the prompt under the
threshold. ``Summarise`` declares its output budget.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from conduit.compaction.conversation import (
    TOOL_RESULT_OMITTED,
    estimate_tokens,
    render_messages,
    trim_text,
)
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
from conduit.core.types import (
    CacheBreakpoint,
    ContentBlock,
    ImageBlock,
    Message,
    ModelResponse,
    ProviderStateBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_NOTICE",
    "CompactionStrategy",
    "DropToolResults",
    "KeepLastTurns",
    "StrategyContext",
    "Summarise",
    "SummaryBindings",
    "SummaryCharge",
    "check_strategy",
    "summary_messages",
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


@runtime_checkable
class SummaryCharge(Protocol):
    """The run's budget, for a strategy's own model calls: the agent runner's
    ``ModelCallCharge``, read structurally (D0085).

    ``begin`` counts a call about to be made and returns the output tokens it may
    produce, or ``None`` when the run's budget refuses it; ``record`` charges the call's
    usage and returns ``False`` when that takes the run past its budget. Either refusal
    ends the run once the hook returns.
    """

    def begin(self, max_output_tokens: int) -> int | None: ...

    def record(self, usage: Usage) -> bool: ...


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """What a strategy knows about the call being compacted: the run, the model the
    conversation goes to (its catalogue entry gives the context window), the tools the
    conversation's tool calls were made with, and the run's ``charge`` for any model call
    the strategy makes (``None`` outside a run, or with a hook caller that offers none)."""

    run_id: str | None
    assignment: ResolvedAssignment
    tools: tuple[ToolSpec, ...] = ()
    charge: SummaryCharge | None = None


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


def _result_text(block: ToolResultBlock, limit: int | None) -> str:
    if limit is None:
        return TOOL_RESULT_OMITTED
    parts = [c.text if isinstance(c, TextBlock) else "[image]" for c in block.content]
    return trim_text(" ".join(parts), limit) or "[no content]"


def _prepared_block(
    block: ContentBlock, tool_result_chars: int | None, max_message_chars: int
) -> ContentBlock | None:
    if isinstance(block, TextBlock):
        return TextBlock(text=trim_text(block.text, max_message_chars))
    if isinstance(block, ToolResultBlock):
        return ToolResultBlock(
            tool_use_id=block.tool_use_id,
            content=(TextBlock(text=_result_text(block, tool_result_chars)),),
            is_error=block.is_error,
        )
    if isinstance(block, ImageBlock):
        return TextBlock(text="[image]")
    if isinstance(block, CacheBreakpoint | ProviderStateBlock):
        return None  # the summary model is not the one these were made for
    return block  # a tool use, kept whole so its result still answers it


def summary_messages(
    replaced: Sequence[Message],
    instructions: str,
    *,
    tool_result_chars: int | None = 500,
    max_message_chars: int = 3_000,
) -> tuple[Message, ...]:
    """``replaced`` as a summary request sends it: tool calls and results in their own
    blocks, each result trimmed to ``tool_result_chars`` (or left out when ``None``),
    text trimmed to ``max_message_chars``, images named, cache markers and provider state
    left out, and ``instructions`` added to the last user turn."""
    messages: list[Message] = []
    for message in replaced:
        blocks: list[ContentBlock] = []
        for block in message.content:
            prepared = _prepared_block(block, tool_result_chars, max_message_chars)
            if prepared is not None:
                blocks.append(prepared)
        content = tuple(blocks) or (TextBlock(text="[no text]"),)
        messages.append(Message(role=message.role, content=content))
    if messages and messages[-1].role == "user":
        last = messages[-1]
        messages[-1] = Message(role="user", content=(*last.content, TextBlock(text=instructions)))
    else:
        messages.append(Message.user(instructions))
    return tuple(messages)


def _used_tools(messages: Sequence[Message], known: Sequence[ToolSpec]) -> tuple[ToolSpec, ...]:
    """A definition for every tool ``messages`` call, in order of first use: the run's own,
    or a minimal one for a tool it does not offer."""
    specs = {spec.name: spec for spec in known}
    used: dict[str, ToolSpec] = {}
    for message in messages:
        for block in message.content:
            if isinstance(block, ToolUseBlock) and block.name not in used:
                used[block.name] = specs.get(block.name) or ToolSpec(
                    name=block.name,
                    description="A tool used earlier in the conversation.",
                    input_schema={"type": "object"},
                )
    return tuple(used.values())


class Summarise:
    """Summarise the older turns with the model assigned to ``task_class``.

    ``floor`` is the least the summary model must offer (a model with tools unless the
    product says otherwise); it is checked here, once. The request carries the run's id,
    so the provider registry's model-call events tie the summary call to the run. Its
    tokens are metered by the registry like any call, and charged to the run's own
    budget through the context's ``charge`` (D0085): the call is counted before it is
    made, its output is held to what the run's token budget has left, and its usage is
    charged once it is made. A call the budget refuses is never made, and a summary
    whose usage takes the run past its budget is not used; both fail the compaction as
    ``budget_exhausted``, and the runner ends the run as for any over-budget call.
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
        tool_result_chars: int | None = 500,
        max_message_chars: int = 3_000,
        max_summary_tokens: int | None = None,
        name: str = "summarise",
    ) -> None:
        check_strategy(name, threshold, keep_turns)
        assignment = bindings.assignments.get(task_class)
        if assignment is None:
            raise ConfigurationError(f"summary task class {task_class!r} has no assignment")
        self.floor = floor if floor is not None else CapabilityFloor(tools=True)
        shortfalls = self.floor.shortfalls(assignment.entry)
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
        self.summary_allowance = self._output(assignment)

    def _output(self, assignment: ResolvedAssignment) -> int:
        output = assignment.max_output_tokens
        if self.max_summary_tokens is not None:
            output = min(output, self.max_summary_tokens)
        return output

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        binding = self._bindings.for_task(self.task_class)
        assignment = binding.assignment
        output = self._output(assignment)
        messages = summary_messages(
            replaced,
            self.instructions,
            tool_result_chars=self.tool_result_chars,
            max_message_chars=self.max_message_chars,
        )
        has_tools = any(
            isinstance(b, ToolUseBlock | ToolResultBlock) for m in messages for b in m.content
        )
        tools = _used_tools(messages, context.tools) if has_tools else ()
        if estimate_tokens(tools=tools, messages=messages) + output > (
            assignment.entry.context_window
        ):
            raise CompactionFailed("summary_too_large")
        charge = context.charge
        if charge is not None:
            allowed = charge.begin(output)
            if allowed is None:
                raise CompactionFailed("budget_exhausted")
            output = max(1, min(output, allowed))
        request = binding.request(
            messages=messages, tools=tools, max_output_tokens=output, run_id=context.run_id
        )
        response: ModelResponse | None = None
        failure: ProviderFailure | None = None
        try:
            response = await binding.provider.complete(request)
        except ProviderError as error:
            failure = error.failure
        if response is None:  # raised after the handler, so nothing is chained (D0040)
            raise CompactionFailed("summary_failed", failure=failure)
        if charge is not None and not charge.record(response.usage):
            raise CompactionFailed("budget_exhausted")
        text = response.text.strip()
        if not text:
            raise CompactionFailed("summary_empty")
        return self.header + text
