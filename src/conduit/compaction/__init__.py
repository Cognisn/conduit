"""Transcript compaction strategies that keep the cached prefix stable."""

# ``Compactor`` is the agent runner's compaction hook (``AgentRunner(compaction=...)``):
# before every model call it compacts the conversation once the prompt reaches a
# strategy's share of the assigned model's context window, or when a compaction is
# forced. Strategies drop old tool results, keep only the last turns, or summarise
# through a product-declared task class. The system prompt, tools and loaded skills are
# never touched, so the cached prefix stays byte-identical, and the summary follows it.
# Compaction points go to the transcript store, and compaction events to the event sink
# under the capture policy. The layer uses the core alone.

from conduit.compaction.compactor import (
    DEFAULT_MAX_FORCED,
    CompactionContext,
    Compactor,
    TokenEstimator,
)
from conduit.compaction.conversation import (
    DEFAULT_MAX_MESSAGE_CHARS,
    TOOL_RESULT_OMITTED,
    estimate_tokens,
    find_cut,
    render_messages,
)
from conduit.compaction.errors import REASONS, CompactionError, CompactionFailed
from conduit.compaction.events import CompactionEvent, CompactionOutcome, CompactionTrigger
from conduit.compaction.strategies import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_NOTICE,
    CompactionStrategy,
    DropToolResults,
    KeepLastTurns,
    StrategyContext,
    Summarise,
    SummaryBindings,
    check_strategy,
    summary_messages,
)

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_MAX_FORCED",
    "DEFAULT_MAX_MESSAGE_CHARS",
    "DEFAULT_NOTICE",
    "REASONS",
    "TOOL_RESULT_OMITTED",
    "CompactionContext",
    "CompactionError",
    "CompactionEvent",
    "CompactionFailed",
    "CompactionOutcome",
    "CompactionStrategy",
    "CompactionTrigger",
    "Compactor",
    "DropToolResults",
    "KeepLastTurns",
    "StrategyContext",
    "Summarise",
    "SummaryBindings",
    "TokenEstimator",
    "check_strategy",
    "estimate_tokens",
    "find_cut",
    "render_messages",
    "summary_messages",
]
