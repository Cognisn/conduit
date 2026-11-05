"""Hooks the product supplies to a runner: the output filter, approvals and compaction.

- ``OutputFilter``: an async callable given model text and returning it with anything the
  product must not let out (secrets, for example) masked. The runner applies it to the
  text of every assistant message before that message joins the conversation or the
  transcript, so the answer, the run-finished event and the stored transcript are all
  filtered. Text deltas cannot be filtered reliably (a secret may span deltas), so with a
  filter set they carry no text at all, whatever the capture policy. A pattern filter
  cannot find every secret; it is one control among several.
- ``requires_approval``: decides whether a tool needs approval. The default reads the
  tool's own ``requires_approval`` attribute, and treats any value other than ``False``
  (or its absence) as needing approval, so a mistyped mark fails safe.
- ``CompactionHook``: called before every model call with the conversation about to be
  sent; it may return a replacement conversation (a summary followed by recent messages)
  or ``None`` to send it unchanged. The compaction layer implements it; the runner only
  calls it, checks what comes back, and records a ``ConversationCompactedEvent``. Writing
  a ``CompactionPoint`` to the transcript is the hook's own business: the runner does not
  append the replacement.
- ``SystemHook`` (per run, ``AgentTask.system_hook``): given the run's base system blocks
  (``AgentTask.system``, or ``system_for`` after a hand-off) before every model call, it
  returns the system blocks to send. The skills layer's ``SkillSession.system`` fits it
  as it is, so loaded skills join the cached prefix as soon as they are loaded. It runs
  before the compaction hook, which sees the system the call will send.
- ``ToolWrapper`` (per run, ``AgentTask.tool_wrapper``): given the run's executor (which
  calls the active task class's ``TaskTools``), it returns the executor the runner calls
  instead. The skills layer's ``with_use_skill`` fits it as it is: it answers ``use_skill``
  itself and hands every other tool on. The wrapped tool must still be in the task's
  registry: the runner resolves every tool there first, so the wrapper never sees an
  unknown tool, and approvals still apply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from conduit.core.tasks import ResolvedAssignment
from conduit.core.tools import ToolExecutor
from conduit.core.types import Message, SystemBlock, ToolSpec, Usage


class OutputFilter(Protocol):
    async def __call__(self, text: str) -> str: ...


class SystemHook(Protocol):
    async def __call__(self, base: tuple[SystemBlock, ...]) -> Sequence[SystemBlock]: ...


#: Wraps a run's tool executor (``with_use_skill``, for example).
ToolWrapper = Callable[[ToolExecutor], ToolExecutor]


def tool_requires_approval(tool: Any) -> bool:
    """The default approval mark: a tool's ``requires_approval``, failing safe."""
    return getattr(tool, "requires_approval", False) is not False


def tool_results_prefamed(tool: Any) -> bool:
    """The default pre-framed mark: a tool's ``untrusted_framed`` is exactly ``True``.

    Such a tool's executor frames its results as untrusted itself (the MCP tool bridge
    does), so the runner sends them as they are. Anything else, including a missing or
    mistyped mark, is framed by the runner: framing twice is harmless, never framing is
    not. The declaration is the tool's, never read from the content.
    """
    return getattr(tool, "untrusted_framed", False) is True


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    """What the compaction hook is shown before a model call.

    ``messages`` is the conversation about to be sent, ``system`` and ``tools`` the rest
    of the prompt, and ``assignment`` the model it goes to (its catalogue entry gives the
    context window). ``usage`` is the run's total so far; ``last_usage`` the previous
    call's (its input counts are the best estimate of the prompt's size), or ``None``
    before the first call. ``model_calls`` is how many calls the run has made so far.
    """

    run_id: str
    task_class: str
    session_id: str | None
    assignment: ResolvedAssignment
    system: tuple[SystemBlock, ...]
    tools: tuple[ToolSpec, ...]
    messages: tuple[Message, ...]
    usage: Usage
    last_usage: Usage | None
    model_calls: int


@runtime_checkable
class CompactionHook(Protocol):
    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        """A replacement conversation, or ``None`` to keep it. It must be non-empty and
        end with a user message; the runner refuses anything else with
        ``CompactionRefused``."""
        ...
