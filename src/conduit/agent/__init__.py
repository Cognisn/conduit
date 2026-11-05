"""Agent runner with budgets, hand-off, output contracts, untrusted framing and approvals."""

# ``AgentRunner`` drives an ``AgentTask`` on a started runtime's task binding through model
# calls and tool calls, as an async stream of typed run events (``conduit.agent.events``),
# to a ``RunResult``. The product supplies the policy: its tools and executor, approval
# handler, output filter, transcript store, event sink and capture policy, and the hooks.
# The layer uses the core alone.

from conduit.agent.budgets import BudgetKind, BudgetMeter, Exhausted, RunBudget, TokenCount
from conduit.agent.contracts import ContractCheck, OutputContract, parse_output
from conduit.agent.hooks import (
    CompactionHook,
    CompactionRequest,
    OutputFilter,
    SystemHook,
    ToolWrapper,
    tool_requires_approval,
    tool_results_prefamed,
)
from conduit.agent.results import (
    AgentError,
    BudgetExhausted,
    CompactionRefused,
    OutputContractFailed,
    RunFailed,
    RunFailure,
    RunIncomplete,
    RunRefused,
    RunResult,
    RunStatus,
    SystemPromptRefused,
)
from conduit.agent.runner import (
    HAND_OFF_TOOL,
    MAX_OUTPUT_ATTEMPTS,
    UNKNOWN_TOOL,
    AgentRun,
    AgentRunner,
    AgentTask,
    Bindings,
)
from conduit.agent.untrusted import (
    FRAMING_NOTE,
    UntrustedBlock,
    UntrustedText,
    frame_untrusted,
    sanitise_untrusted,
    untrusted_text_block,
)

__all__ = [
    "FRAMING_NOTE",
    "HAND_OFF_TOOL",
    "MAX_OUTPUT_ATTEMPTS",
    "UNKNOWN_TOOL",
    "AgentError",
    "AgentRun",
    "AgentRunner",
    "AgentTask",
    "Bindings",
    "BudgetExhausted",
    "BudgetKind",
    "BudgetMeter",
    "CompactionHook",
    "CompactionRefused",
    "CompactionRequest",
    "ContractCheck",
    "Exhausted",
    "OutputContract",
    "OutputContractFailed",
    "OutputFilter",
    "RunBudget",
    "RunFailed",
    "RunFailure",
    "RunIncomplete",
    "RunRefused",
    "RunResult",
    "RunStatus",
    "SystemHook",
    "SystemPromptRefused",
    "TokenCount",
    "ToolWrapper",
    "UntrustedBlock",
    "UntrustedText",
    "frame_untrusted",
    "parse_output",
    "sanitise_untrusted",
    "tool_requires_approval",
    "tool_results_prefamed",
    "untrusted_text_block",
]
