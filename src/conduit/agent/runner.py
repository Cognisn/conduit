"""The agent runner: one task driven through model calls and tool calls to an answer.

``AgentRunner`` drives an ``AgentTask`` on its task class's ``TaskBinding``: every model
call goes to that binding's provider, built by ``TaskBinding.request`` from the resolved
assignment, so the assignment's model is the only one a run can use, and the provider
refuses anything else (``ModelNotAssigned``) before any client call. Tools are reached
only through the task class's ``TaskTools``; a tool outside it is refused as a tool
result and never run.

What the runner enforces, whatever the model says:

- **Budgets** (``RunBudget``): model calls, tool calls, tokens (the counts the product
  chose) and wall time. A run that reaches one stops with ``BUDGET_EXHAUSTED`` and a
  ``BudgetExhaustedEvent``; it never concludes on a partial answer.
- **Hand-off**: only to a task class the task allows (``AgentTask.hand_offs``) and the
  runtime has resolved an assignment for, through the model's ``hand_off`` tool. The run
  continues on that class's binding and tools; nothing else can change its model.
- **Output contracts**: an answer failing its contract is retried once with the errors,
  then the run fails with ``OUTPUT_INVALID``.
- **Untrusted framing**: every tool result reaches the model framed as untrusted, by
  the runner or, for a tool declared to (``preframed``), by its own executor.
- **Approvals**: a tool the product marks goes to its ``ApprovalHandler`` first; a
  refusal (or no handler, or a failing one) is a tool result the model sees.
- **Output filter**: the product's filter masks the text of every assistant message.
- **Fail safe**: a failing provider, a refusing model, a raising tool executor or a
  raising compaction hook ends the run with a typed failure; the conversation is left
  well formed (every tool use has a result), so the transcript stays loadable.
- **Hooks' own calls**: a model call the compaction hook makes itself (a summary) is
  counted and charged to the run's budget through the request's ``charge`` (D0085).

A run is an async stream of typed run events (``conduit.agent.events``); the result is
``AgentRun.result`` once the stream ends, or ``await runner.run(task)``. With a
``TranscriptStore`` and a session id, every message the run adds (its input first) is
appended as a ``MessageEntry``; the ``history`` the product passes is not appended again.
Provider state blocks stay in the assistant messages, so the provider gets them back.

A runner keeps no per-run state, so one runner serves concurrent runs. From sync code
use the facade: ``sync(runner, loop=loop).run(task)`` or
``sync(runner.start(task), loop=loop).events()``, with the runtime started on that loop.

Carried across from VOR's ``vor.ai.runner`` and generalised (C4): VOR's notebook, prompt
layers, secret filter and hand-off to a person become the transcript store, the
product's messages, the output filter hook and typed failures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from conduit.agent.budgets import BudgetKind, BudgetMeter, Exhausted, RunBudget, TokenCount
from conduit.agent.contracts import CUT_OFF, ContractCheck, OutputContract, parse_output
from conduit.agent.events import (
    ApprovalDecidedEvent,
    ApprovalOutcome,
    ApprovalRequestedEvent,
    BudgetExhaustedEvent,
    ConversationCompactedEvent,
    HandOffEvent,
    OutputInvalidEvent,
    RunEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextDeltaEvent,
    ToolCallStartedEvent,
)
from conduit.agent.hooks import (
    CompactionHook,
    CompactionRequest,
    OutputFilter,
    SystemHook,
    ToolWrapper,
    tool_requires_approval,
    tool_results_preframed,
)
from conduit.agent.results import (
    CompactionRefused,
    RunFailure,
    RunIncomplete,
    RunRefused,
    RunResult,
    RunStatus,
    SystemPromptRefused,
)
from conduit.core.backends import ApprovalDecision, ApprovalHandler, ApprovalRequest
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure, UnknownTool
from conduit.core.events import (
    CapturedContent,
    CapturePolicy,
    ContentKind,
    EventSink,
    FanOutSink,
    ToolCallEvent,
)
from conduit.core.identifiers import IDENTIFIER_PATTERN, new_id
from conduit.core.runtime import TaskBinding
from conduit.core.streaming import StreamEvent, TextDelta, collect
from conduit.core.tasks import ResolvedAssignment
from conduit.core.tools import TaskTools, ToolCallContext, ToolExecutor, ToolOutcome
from conduit.core.transcript import MessageEntry, TranscriptStore
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.core.untrusted import (
    DEFAULT_MAX_LENGTH,
    UntrustedText,
    frame_untrusted,
    untrusted_text_block,
)

logger = logging.getLogger("conduit.agent")

#: The tool through which the model hands a task to another task class.
HAND_OFF_TOOL = "hand_off"
#: The tool name events record for a tool the model asked for that does not exist.
UNKNOWN_TOOL = "_unknown"
#: The first invalid answer is retried once; the second ends the run.
MAX_OUTPUT_ATTEMPTS = 2
CANCELLED = "cancelled"
ABANDONED = "abandoned"

DEFAULT_HAND_OFF_DESCRIPTION = (
    "Hand this task to another task class, which continues the conversation from here. "
    "Use it only when that class is better suited to the task."
)
_UNKNOWN_TOOL_TEXT = "No tool with that name is available to this task."
_NOT_APPROVED_TEXT = "The tool call was not approved, so it was not run."
_BUDGET_TEXT = "The tool call was not run: the run's budget was exhausted."
_STOPPED_TEXT = "The tool call was not run: the run stopped."
_TOOL_FAILED_TEXT = "The tool failed."
_HAND_OFF_REFUSED_TEXT = "That hand-off is not available to this task."
_EMPTY_REPLY = "(empty reply)"
_IDENTIFIER = re.compile(IDENTIFIER_PATTERN)
_NOTHING = CapturePolicy()


class Bindings(Protocol):
    """Where a runner gets task bindings: a started ``Runtime``."""

    @property
    def assignments(self) -> Mapping[str, ResolvedAssignment]: ...

    def for_task(self, task_class: str) -> TaskBinding: ...


@dataclass(frozen=True, slots=True)
class AgentTask:
    """One run's work.

    ``history`` is the conversation so far (already stored by the product, never appended
    again); ``messages`` are the new messages this run starts with, and ``untrusted`` is
    external text, framed and added to the last of them (or as a user message of its own).
    ``system`` is the system prompt, and ``system_for`` replaces it for a task class the
    run hands off to. ``hand_offs`` are the task classes the model may hand the task to.

    ``system_hook`` rebuilds the system blocks from that base before every model call
    (``SkillSession.system``), and ``tool_wrapper`` wraps the run's tool executor
    (``with_use_skill``); see ``conduit.agent.hooks``. Both are per run, because what
    they wrap (a skill session) belongs to one conversation.
    """

    task_class: str
    messages: tuple[Message, ...] = ()
    history: tuple[Message, ...] = ()
    untrusted: tuple[UntrustedText, ...] = ()
    system: tuple[SystemBlock, ...] = ()
    system_for: Mapping[str, tuple[SystemBlock, ...]] = field(default_factory=dict)
    budget: RunBudget = field(default_factory=RunBudget)
    contract: OutputContract | None = None
    hand_offs: frozenset[str] = frozenset()
    session_id: str | None = None
    correlation_id: str | None = None
    subject_type: str | None = None
    subject_id: str | None = None
    actor_id: str | None = None
    cache_tools: bool = False
    system_hook: SystemHook | None = None
    tool_wrapper: ToolWrapper | None = None


@dataclass
class _Run:
    """One run's mutable state. Never shared between runs."""

    run_id: str
    task: AgentTask
    correlation_id: str
    meter: BudgetMeter
    binding: TaskBinding
    tools: TaskTools | None
    messages: list[Message]
    pending_input: tuple[Message, ...]
    began: float
    hand_offs: list[str] = field(default_factory=list)
    last_usage: Usage | None = None
    result: RunResult | None = None
    # A limit a hook's own model call reached through the run's charge (D0085).
    charged_over: Exhausted | None = None
    # Tool uses the conversation holds without results yet, and the turn answering them.
    open_uses: tuple[ToolUseBlock, ...] = ()
    open_turn: _Turn | None = None

    @property
    def task_class(self) -> str:
        return self.binding.assignment.task_class.name

    def common(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "task_class": self.task_class,
        }


class _Charge:
    """A run's ``ModelCallCharge``: a hook's own model calls, counted and charged to the
    run's budget like the runner's. A limit reached is kept on the run, which ends on it
    once the hook returns. Usage that spends the token budget exactly reaches it too, so
    the run's own call is not made with nothing left. Valid only during the
    ``before_model_call`` it was given to."""

    __slots__ = ("_run",)

    def __init__(self, run: _Run) -> None:
        self._run = run

    def __repr__(self) -> str:
        return f"_Charge({self._run.run_id!r})"

    def begin(self, max_output_tokens: int) -> int | None:
        run = self._run
        # Once a limit is reached the meter refuses every later call too.
        exhausted = run.meter.begin_model_call()
        if exhausted is not None:
            run.charged_over = exhausted
            return None
        allowed = max(1, max_output_tokens)
        budget = run.meter.budget
        if budget.max_tokens is not None and TokenCount.OUTPUT in budget.counted_tokens:
            allowed = max(1, min(allowed, budget.max_tokens - run.meter.tokens_used))
        return allowed

    def record(self, usage: Usage) -> bool:
        run = self._run
        over = run.meter.record_usage(usage)
        limit = run.meter.budget.max_tokens
        if over is None and limit is not None and run.meter.tokens_used >= limit:
            over = Exhausted(BudgetKind.TOKENS, limit, run.meter.tokens_used)
        if over is not None and run.charged_over is None:
            run.charged_over = over
        return over is None and run.charged_over is None


@dataclass
class _Compaction:
    """What the compaction step came to: the event to emit, or a failed hook."""

    event: ConversationCompactedEvent | None = None
    failed: bool = False


@dataclass
class _Call:
    """What a model call came to."""

    response: ModelResponse | None = None
    failure: ProviderFailure | None = None
    timed_out: bool = False


@dataclass
class _Turn:
    """What a turn of tool calls came to."""

    results: list[ToolResultBlock] = field(default_factory=list)
    exhausted: Exhausted | None = None
    tool_failed: bool = False
    hand_off: str | None = None

    @property
    def stopped(self) -> bool:
        return self.exhausted is not None or self.tool_failed


class AgentRun:
    """A run, started: an async stream of run events, then its result."""

    def __init__(self, runner: AgentRunner, run: _Run) -> None:
        self._runner = runner
        self._run = run
        self._streamed = False

    @property
    def run_id(self) -> str:
        return self._run.run_id

    def events(self) -> AsyncGenerator[RunEvent, None]:
        """The run's events, ending with ``RunFinishedEvent``. A run streams only once."""
        if self._streamed:
            raise RunIncomplete("a run streams only once")
        self._streamed = True
        return self._runner._drive(self._run)

    def __aiter__(self) -> AsyncIterator[RunEvent]:
        return self.events()

    @property
    def result(self) -> RunResult:
        """The result, once the stream has ended; ``RunIncomplete`` before."""
        if self._run.result is None:
            raise RunIncomplete("the run has not finished")
        return self._run.result

    async def wait(self) -> RunResult:
        """Run to the end, discarding the events, and return the result."""
        async with aclosing(self.events()) as events:
            async for _event in events:
                pass
        return self.result


def _identifier_or(value: str | None, fallback: str) -> str:
    return value if value is not None and _IDENTIFIER.fullmatch(value) else fallback


def _arguments_text(arguments: Mapping[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, default=str)


async def _replay(events: Sequence[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


class AgentRunner:
    """Runs agent tasks on a runtime's bindings. Holds no per-run state.

    ``tools`` gives each task class its ``TaskTools``; a class without an entry has no
    tools. ``events`` receives every run event but the text deltas, and ``capture`` is
    the content-capture policy for the stream and the sink. Both default to the
    runtime's (``Runtime.events`` and ``Runtime.capture``), so run events reach the
    product's sink under the deployment's policy; an explicit argument overrides either.
    Events delivered through the runtime's sink are also held to the runtime's policy.
    ``requires_approval`` marks the tools that need ``approvals``, and ``preframed`` the
    tools whose executor frames its results as untrusted itself (``tool_results_preframed``).
    ``clock`` measures the wall-time budget.
    """

    def __init__(
        self,
        bindings: Bindings,
        *,
        tools: Mapping[str, TaskTools] | None = None,
        approvals: ApprovalHandler | None = None,
        requires_approval: Callable[[Any], bool] = tool_requires_approval,
        preframed: Callable[[Any], bool] = tool_results_preframed,
        output_filter: OutputFilter | None = None,
        transcripts: TranscriptStore | None = None,
        events: EventSink | None = None,
        capture: CapturePolicy | None = None,
        compaction: CompactionHook | None = None,
        max_untrusted_length: int = DEFAULT_MAX_LENGTH,
        hand_off_description: str = DEFAULT_HAND_OFF_DESCRIPTION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for task_class, task_tools in (tools or {}).items():
            if task_tools.task_class != task_class:
                raise ConfigurationError(
                    f"the tools given for {task_class!r} are registered for task class "
                    f"{task_tools.task_class!r}"
                )
        if max_untrusted_length < 1:
            raise ConfigurationError("max_untrusted_length must be positive")
        self._bindings = bindings
        self._tools = dict(tools or {})
        self._approvals = approvals
        self._requires_approval = requires_approval
        self._preframed = preframed
        self._filter = output_filter
        self._transcripts = transcripts
        # Unless the product gives its own, run events go to the runtime's sink under the
        # runtime's capture policy (Q0027); bindings without them fall back to none.
        if capture is None:
            runtime_capture = getattr(bindings, "capture", None)
            capture = runtime_capture if isinstance(runtime_capture, CapturePolicy) else None
        if events is None:
            runtime_events = getattr(bindings, "events", None)
            events = runtime_events if isinstance(runtime_events, EventSink) else None
        self._capture = capture if capture is not None else CapturePolicy()
        self._sink = FanOutSink([events] if events is not None else [], policy=self._capture)
        self._compaction = compaction
        self._max_untrusted = max_untrusted_length
        self._hand_off_description = hand_off_description
        self._clock = clock

    # Starting ----------------------------------------------------------------------

    def start(self, task: AgentTask) -> AgentRun:
        """A run of ``task``, ready to stream. Raises ``RunRefused`` before anything runs."""
        assignments = self._bindings.assignments
        if task.task_class not in assignments:
            raise RunRefused(f"task class {task.task_class!r} has no assignment")
        for target in sorted(task.hand_offs):
            if target not in assignments:
                raise RunRefused(f"hand-off target {target!r} has no assignment")
        if task.correlation_id is not None and not _IDENTIFIER.fullmatch(task.correlation_id):
            raise RunRefused("a correlation id must be an identifier (no spaces)")
        if task.hand_offs:
            for task_class in sorted({task.task_class, *task.hand_offs}):
                task_tools = self._tools.get(task_class)
                if task_tools is not None and any(
                    spec.name == HAND_OFF_TOOL for spec in task_tools.specs()
                ):
                    raise RunRefused(
                        f"task class {task_class!r} has a tool named {HAND_OFF_TOOL!r}, "
                        "which the hand-off tool needs"
                    )
        pending = self._input(task)
        conversation = (*task.history, *pending)
        if not conversation:
            raise RunRefused("the task has no messages")
        if conversation[-1].role != "user":
            raise RunRefused("the conversation must end with a user message")
        run_id = new_id("run")
        run = _Run(
            run_id=run_id,
            task=task,
            correlation_id=task.correlation_id or run_id,
            meter=BudgetMeter(task.budget, self._clock),
            binding=self._bindings.for_task(task.task_class),
            tools=self._tools.get(task.task_class),
            messages=list(conversation),
            pending_input=pending,
            began=time.perf_counter(),
        )
        return AgentRun(self, run)

    async def run(self, task: AgentTask) -> RunResult:
        """Run ``task`` to the end and return its result."""
        return await self.start(task).wait()

    def _input(self, task: AgentTask) -> tuple[Message, ...]:
        messages = list(task.messages)
        framed = tuple(
            untrusted_text_block(part, max_length=self._max_untrusted) for part in task.untrusted
        )
        if framed:
            if messages and messages[-1].role == "user":
                last = messages[-1]
                messages[-1] = Message(role="user", content=(*last.content, *framed))
            else:
                messages.append(Message(role="user", content=framed))
        return tuple(messages)

    # Driving -----------------------------------------------------------------------

    async def _drive(self, run: _Run) -> AsyncGenerator[RunEvent, None]:
        # A run that ends any way but its own finish is recorded once in the sink.
        try:
            async with aclosing(self._steps(run)) as steps:
                async for event in steps:
                    yield event
        except GeneratorExit:
            if run.result is None:
                await self._close_open_uses(run)
                await self._record_end(run, ABANDONED)
            raise
        except asyncio.CancelledError:
            if run.result is None:
                await self._close_open_uses(run)
                await self._record_end(run, CANCELLED)
            raise
        except Exception as error:
            if run.result is None:
                await self._close_open_uses(run)
                await self._record_end(run, type(error).__name__)
            raise

    async def _close_open_uses(self, run: _Run) -> None:
        """Answer every tool use still open when a run ends part-way, so the conversation
        and the stored transcript stay well formed. Results already in are kept; the rest
        are stopped. A failure here is logged by class name and never replaces the end
        the run is already on (cancellation included), nor chains to it."""
        if not run.open_uses:
            return
        done = {r.tool_use_id: r for r in (run.open_turn.results if run.open_turn else [])}
        results = tuple(
            done.get(use.id) or self._error_result(use, _STOPPED_TEXT) for use in run.open_uses
        )
        run.open_uses = ()
        run.open_turn = None
        failed: str | None = None
        try:
            await self._add(run, Message(role="user", content=results))
        except Exception as error:
            failed = type(error).__name__
        if failed is not None:
            logger.warning("closing the open tool uses of run %s failed: %s", run.run_id, failed)

    async def _steps(self, run: _Run) -> AsyncGenerator[RunEvent, None]:
        task = run.task
        if run.pending_input:
            await self._append(run, run.pending_input)
        yield await self._emit(
            RunStartedEvent(
                **run.common(),
                provider=_identifier_or(run.binding.provider.name, "provider"),
                kind=run.binding.assignment.kind,
                model=run.binding.assignment.model,
                message_count=len(run.messages),
                tool_count=len(self._specs(run)),
                hand_off_count=len(self._targets(run)),
                contract=task.contract.name if task.contract is not None else None,
            )
        )
        invalid = 0
        while True:
            exhausted = run.meter.begin_model_call()
            if exhausted is not None:
                async for event in self._exhaust(run, exhausted):
                    yield event
                return
            # The hooks run within the wall-time budget, and the deadline is checked again
            # once they are done, so a slow hook cannot carry a call past it.
            system: tuple[SystemBlock, ...] | None = None
            compaction = _Compaction()
            expired = False
            try:
                async with asyncio.timeout(self._left(run)) as scope:
                    system = await self._system(run)
                    compaction = await self._compact(run, system)
            except TimeoutError:
                if not scope.expired():
                    raise
                expired = True
            # A hook that swallowed the deadline's cancellation is out of time all the same.
            expired = expired or scope.expired()
            # A hook's own call over the budget ends the run as any over-budget call
            # does, whatever the hook did next (D0085).
            if run.charged_over is not None:
                async for event in self._exhaust(run, run.charged_over):
                    yield event
                return
            if expired or system is None:
                async for event in self._exhaust(run, run.meter.timed_out()):
                    yield event
                return
            if compaction.event is not None:
                yield await self._emit(compaction.event)
            late = run.meter.wall_time_exhausted()
            if late is not None:
                async for event in self._exhaust(run, late):
                    yield event
                return
            if compaction.failed:
                yield await self._finish(run, RunFailure.COMPACTION_FAILED)
                return
            call = _Call()
            request = self._request(run, system)
            async with aclosing(self._call_model(run, request, call)) as deltas:
                async for delta in deltas:
                    yield delta
            if call.timed_out:
                async for event in self._exhaust(run, run.meter.timed_out()):
                    yield event
                return
            if call.response is None:
                yield await self._finish(
                    run, RunFailure.MODEL_FAILED, provider_failure=call.failure
                )
                return
            response = call.response
            message, text, filtered = await self._assistant(response)
            run.last_usage = response.usage
            uses = response.tool_uses
            await self._add(
                run, message, model=run.binding.assignment.model_id, usage=response.usage
            )
            run.open_uses = uses
            over = run.meter.record_usage(response.usage)
            if over is not None or response.stop_reason is StopReason.REFUSAL:
                if uses:
                    await self._add(run, self._unrun(uses, _BUDGET_TEXT if over else _STOPPED_TEXT))
                    run.open_uses = ()
                if over is not None:
                    async for event in self._exhaust(run, over):
                        yield event
                else:
                    yield await self._finish(run, RunFailure.MODEL_REFUSED)
                return
            if uses:
                turn = _Turn()
                run.open_turn = turn
                async with aclosing(self._tool_turn(run, uses, turn)) as tool_events:
                    async for event in tool_events:
                        yield event
                await self._add(run, Message(role="user", content=tuple(turn.results)))
                run.open_uses = ()
                run.open_turn = None
                if turn.exhausted is not None:
                    async for event in self._exhaust(run, turn.exhausted):
                        yield event
                    return
                if turn.tool_failed:
                    yield await self._finish(run, RunFailure.TOOL_FAILED)
                    return
                if turn.hand_off is not None:
                    self._switch(run, turn.hand_off)
                continue
            contract = task.contract
            if contract is None:
                yield await self._finish(
                    run, None, text=text, stop_reason=response.stop_reason, filtered=filtered
                )
                return
            check = (
                ContractCheck(None, (CUT_OFF,))
                if response.stop_reason is StopReason.MAX_TOKENS
                else parse_output(contract, text)
            )
            if check.value is not None:
                yield await self._finish(
                    run,
                    None,
                    text=text,
                    output=check.value,
                    stop_reason=response.stop_reason,
                    filtered=filtered,
                )
                return
            invalid += 1
            retrying = invalid < MAX_OUTPUT_ATTEMPTS
            yield await self._emit(
                OutputInvalidEvent(
                    **run.common(),
                    contract=contract.name,
                    attempt=invalid,
                    error_count=len(check.errors),
                    retrying=retrying,
                )
            )
            if not retrying:
                yield await self._finish(run, RunFailure.OUTPUT_INVALID)
                return
            await self._add(run, Message.user(contract.retry_message(check.errors)))

    # Model calls -------------------------------------------------------------------

    def _targets(self, run: _Run) -> list[str]:
        return sorted(t for t in run.task.hand_offs if t != run.task_class)

    def _specs(self, run: _Run) -> tuple[ToolSpec, ...]:
        specs = list(run.tools.specs()) if run.tools is not None else []
        targets = self._targets(run)
        if targets:
            specs.append(
                ToolSpec(
                    name=HAND_OFF_TOOL,
                    description=self._hand_off_description,
                    input_schema={
                        "type": "object",
                        "properties": {"task_class": {"type": "string", "enum": targets}},
                        "required": ["task_class"],
                        "additionalProperties": False,
                    },
                )
            )
        return tuple(specs)

    async def _system(self, run: _Run) -> tuple[SystemBlock, ...]:
        """The system blocks for the next call: the base, through the run's system hook."""
        task = run.task
        base = task.system_for.get(run.task_class, task.system)
        if task.system_hook is None:
            return base
        returned: Any = await task.system_hook(base)
        usable = isinstance(returned, Sequence) and not isinstance(returned, str | bytes)
        blocks = tuple(returned) if usable else ()
        if not usable or not all(isinstance(b, TextBlock | CacheBreakpoint) for b in blocks):
            raise SystemPromptRefused("the system hook must return a sequence of system blocks")
        return blocks

    def _request(self, run: _Run, system: tuple[SystemBlock, ...]) -> ModelRequest:
        task = run.task
        fields: dict[str, Any] = {
            "run_id": run.run_id,
            "correlation_id": run.correlation_id,
            "cache_tools": task.cache_tools,
        }
        remaining = run.meter.remaining_s()
        if remaining is not None:
            fields["timeout_s"] = min(max(remaining, 0.001), 3_600)
        budget = task.budget
        if budget.max_tokens is not None and TokenCount.OUTPUT in budget.counted_tokens:
            # A call may not produce more output than the token budget has left.
            left = budget.max_tokens - run.meter.tokens_used
            fields["max_output_tokens"] = max(
                1, min(run.binding.assignment.max_output_tokens, left)
            )
        contract = task.contract
        if (
            contract is not None
            and contract.structured
            and run.binding.assignment.entry.capabilities.structured_output
        ):
            fields["output_schema"] = contract.schema()
        return run.binding.request(
            messages=run.messages,
            system=system,
            tools=self._specs(run),
            **fields,
        )

    async def _call_model(
        self, run: _Run, request: ModelRequest, call: _Call
    ) -> AsyncGenerator[TextDeltaEvent, None]:
        index = run.meter.model_calls
        collected: list[StreamEvent] = []
        failure: ProviderFailure | None = None
        async with aclosing(run.binding.provider.stream(request)) as stream:
            while True:
                expired = False
                try:
                    async with asyncio.timeout(self._left(run)) as scope:
                        event = await anext(stream)
                except StopAsyncIteration:
                    break
                except ProviderError as error:
                    failure = error.failure
                    break
                except TimeoutError:
                    if not scope.expired():
                        raise
                    expired = True
                if expired:
                    call.timed_out = True
                    return
                collected.append(event)
                if isinstance(event, TextDelta):
                    policy = _NOTHING if self._filter is not None else self._capture
                    yield TextDeltaEvent(
                        **run.common(),
                        model_call=index,
                        text=policy.capture(ContentKind.PROMPT, event.text),
                    )
        if failure is not None:
            call.failure = failure
            return
        try:
            call.response = await collect(_replay(collected), model=request.model)
        except ProviderError as error:
            failure = error.failure
        call.failure = failure

    def _left(self, run: _Run) -> float | None:
        remaining = run.meter.remaining_s()
        return None if remaining is None else max(remaining, 0.0)

    async def _assistant(self, response: ModelResponse) -> tuple[Message, str, bool]:
        """The response as the conversation keeps it, filtered; its text; whether the
        filter changed anything."""
        blocks: list[Any] = []
        texts: list[str] = []
        filtered = False
        for block in response.content:
            if isinstance(block, TextBlock) and self._filter is not None:
                masked = await self._filter(block.text)
                filtered = filtered or masked != block.text
                block = TextBlock(text=masked)
            if isinstance(block, TextBlock):
                texts.append(block.text)
            blocks.append(block)
        if not blocks:
            blocks.append(TextBlock(text=_EMPTY_REPLY))
        return Message(role="assistant", content=tuple(blocks)), "".join(texts), filtered

    # Tool calls --------------------------------------------------------------------

    async def _tool_turn(
        self, run: _Run, uses: tuple[ToolUseBlock, ...], turn: _Turn
    ) -> AsyncGenerator[RunEvent, None]:
        for use in uses:
            if turn.stopped:
                text = _BUDGET_TEXT if turn.exhausted is not None else _STOPPED_TEXT
                turn.results.append(self._error_result(use, text))
                continue
            exhausted = run.meter.begin_tool_call()
            if exhausted is not None:
                turn.exhausted = exhausted
                turn.results.append(self._error_result(use, _BUDGET_TEXT))
                continue
            async with aclosing(self._tool_call(run, use, turn)) as events:
                async for event in events:
                    yield event

    def _error_result(self, use: ToolUseBlock, text: str) -> ToolResultBlock:
        return ToolResultBlock(tool_use_id=use.id, content=(TextBlock(text=text),), is_error=True)

    def _unrun(self, uses: Sequence[ToolUseBlock], text: str) -> Message:
        return Message(role="user", content=tuple(self._error_result(u, text) for u in uses))

    def _finished(
        self,
        run: _Run,
        call_id: str,
        tool: str,
        arguments: CapturedContent,
        began: float,
        *,
        ok: bool,
        error_code: str | None,
        result: str,
    ) -> ToolCallEvent:
        return ToolCallEvent(
            **run.common(),
            call_id=call_id,
            tool=tool,
            outcome="ok" if ok else "failed",
            error_code=None if error_code is None else _identifier_or(error_code, "error"),
            arguments=arguments,
            result=self._capture.capture(ContentKind.TOOL, result),
            duration_ms=(time.perf_counter() - began) * 1000,
        )

    async def _tool_call(
        self, run: _Run, use: ToolUseBlock, turn: _Turn
    ) -> AsyncGenerator[RunEvent, None]:
        began = time.perf_counter()
        call_id = new_id("tool")
        arguments = self._capture.capture(ContentKind.TOOL, _arguments_text(use.input))
        if use.name == HAND_OFF_TOOL and self._targets(run):
            async for event in self._hand_off(run, use, turn, call_id, arguments, began):
                yield event
            return
        tool: Any = None
        if run.tools is not None:
            try:
                tool = run.tools.resolve(use.name)
            except UnknownTool:
                tool = None
        name = tool.name if tool is not None else UNKNOWN_TOOL
        yield await self._emit(
            ToolCallStartedEvent(**run.common(), call_id=call_id, tool=name, arguments=arguments)
        )
        if tool is None or run.tools is None:
            turn.results.append(self._error_result(use, _UNKNOWN_TOOL_TEXT))
            yield await self._emit(
                self._finished(
                    run,
                    call_id,
                    name,
                    arguments,
                    began,
                    ok=False,
                    error_code="unknown_tool",
                    result=_UNKNOWN_TOOL_TEXT,
                )
            )
            return
        if self._requires_approval(tool):
            outcome: ApprovalOutcome | None = None
            async with aclosing(self._approve(run, use, call_id, name, turn)) as approval:
                async for event in approval:
                    if isinstance(event, ApprovalDecidedEvent):
                        outcome = event.outcome
                    yield event
            if turn.exhausted is not None:
                turn.results.append(self._error_result(use, _BUDGET_TEXT))
                yield await self._emit(
                    self._finished(
                        run,
                        call_id,
                        name,
                        arguments,
                        began,
                        ok=False,
                        error_code="timeout",
                        result=_BUDGET_TEXT,
                    )
                )
                return
            if outcome != "approved":
                turn.results.append(self._error_result(use, _NOT_APPROVED_TEXT))
                yield await self._emit(
                    self._finished(
                        run,
                        call_id,
                        name,
                        arguments,
                        began,
                        ok=False,
                        error_code="not_approved",
                        result=_NOT_APPROVED_TEXT,
                    )
                )
                return
        context = ToolCallContext(
            task_class=run.task_class,
            run_id=run.run_id,
            call_id=call_id,
            correlation_id=run.correlation_id,
            subject_type=run.task.subject_type,
            subject_id=run.task.subject_id,
            actor_id=run.task.actor_id,
        )
        failed: str | None = None
        expired = False
        result: Any = None
        try:
            async with asyncio.timeout(self._left(run)) as scope:
                result = await self._executor(run, run.tools)(tool, dict(use.input), context)
        except TimeoutError:
            if scope.expired():
                expired = True
            else:
                failed = "TimeoutError"
        except Exception as error:
            failed = type(error).__name__
        if expired:
            turn.exhausted = run.meter.timed_out()
            turn.results.append(self._error_result(use, _BUDGET_TEXT))
            yield await self._emit(
                self._finished(
                    run,
                    call_id,
                    name,
                    arguments,
                    began,
                    ok=False,
                    error_code="timeout",
                    result=_BUDGET_TEXT,
                )
            )
            return
        if failed is not None or not isinstance(result, ToolOutcome):
            # An executor that raised, or returned something other than a ToolOutcome.
            why = failed if failed is not None else type(result).__name__
            logger.warning("tool %s failed in run %s: %s", name, run.run_id, why)
            turn.tool_failed = True
            turn.results.append(self._error_result(use, _TOOL_FAILED_TEXT))
            yield await self._emit(
                self._finished(
                    run,
                    call_id,
                    name,
                    arguments,
                    began,
                    ok=False,
                    error_code="tool_failed",
                    result=_TOOL_FAILED_TEXT,
                )
            )
            return
        # A tool whose executor frames its own results (the MCP bridge) is declared so
        # by the tool or the product, never by its content; everything else is framed.
        sent = (
            result.content
            if self._preframed(tool) is True
            else frame_untrusted(
                result.content,
                source=tool.name,
                record=result.record_id,
                max_length=self._max_untrusted,
            ).text
        )
        turn.results.append(
            ToolResultBlock(
                tool_use_id=use.id,
                content=(TextBlock(text=sent),),
                is_error=not result.ok,
            )
        )
        yield await self._emit(
            self._finished(
                run,
                call_id,
                name,
                arguments,
                began,
                ok=result.ok,
                error_code=result.error_code,
                result=result.content,
            )
        )

    def _executor(self, run: _Run, tools: TaskTools) -> ToolExecutor:
        """The executor a tool call goes to: the task's tools, through the run's wrapper."""
        executor: ToolExecutor = tools.call
        wrapper = run.task.tool_wrapper
        return wrapper(executor) if wrapper is not None else executor

    async def _approve(
        self, run: _Run, use: ToolUseBlock, call_id: str, name: str, turn: _Turn
    ) -> AsyncGenerator[RunEvent, None]:
        request_id = new_id("apr")
        yield await self._emit(
            ApprovalRequestedEvent(
                **run.common(), request_id=request_id, call_id=call_id, tool=name
            )
        )
        outcome: ApprovalOutcome
        if self._approvals is None:
            outcome = "unavailable"
        else:
            request = ApprovalRequest(
                request_id=request_id,
                run_id=run.run_id,
                task_class=run.task_class,
                tool=name,
                arguments=dict(use.input),
            )
            decision: Any = None
            failed: str | None = None
            expired = False
            try:
                async with asyncio.timeout(self._left(run)) as scope:
                    decision = await self._approvals.decide(request)
            except TimeoutError:
                if scope.expired():
                    expired = True
                else:
                    failed = "TimeoutError"
            except Exception as error:
                failed = type(error).__name__
            if expired:
                turn.exhausted = run.meter.timed_out()
                outcome = "failed"
            elif failed is not None:
                logger.warning("approval for %s failed in run %s: %s", name, run.run_id, failed)
                outcome = "failed"
            elif isinstance(decision, ApprovalDecision) and decision.approved is True:
                outcome = "approved"
            else:
                outcome = "refused"
        yield await self._emit(
            ApprovalDecidedEvent(
                **run.common(),
                request_id=request_id,
                call_id=call_id,
                tool=name,
                outcome=outcome,
            )
        )

    # Hand-off ----------------------------------------------------------------------

    async def _hand_off(
        self,
        run: _Run,
        use: ToolUseBlock,
        turn: _Turn,
        call_id: str,
        arguments: CapturedContent,
        began: float,
    ) -> AsyncGenerator[RunEvent, None]:
        yield await self._emit(
            ToolCallStartedEvent(
                **run.common(), call_id=call_id, tool=HAND_OFF_TOOL, arguments=arguments
            )
        )
        target = use.input.get("task_class") if set(use.input) == {"task_class"} else None
        assignments = self._bindings.assignments
        allowed = (
            isinstance(target, str)
            and turn.hand_off is None
            # Every target was checked against the assignments when the run started.
            and target in self._targets(run)
        )
        if not allowed or not isinstance(target, str):
            turn.results.append(self._error_result(use, _HAND_OFF_REFUSED_TEXT))
            yield await self._emit(
                HandOffEvent(
                    **run.common(),
                    call_id=call_id,
                    outcome="refused",
                    from_task_class=run.task_class,
                )
            )
            yield await self._emit(
                self._finished(
                    run,
                    call_id,
                    HAND_OFF_TOOL,
                    arguments,
                    began,
                    ok=False,
                    error_code="hand_off_refused",
                    result=_HAND_OFF_REFUSED_TEXT,
                )
            )
            return
        assignment = assignments[target]
        turn.hand_off = target
        text = f"Handed off to task class {target}."
        turn.results.append(ToolResultBlock(tool_use_id=use.id, content=(TextBlock(text=text),)))
        yield await self._emit(
            HandOffEvent(
                **run.common(),
                call_id=call_id,
                outcome="ok",
                from_task_class=run.task_class,
                to_task_class=target,
                provider=_identifier_or(assignment.provider, "provider"),
                model=assignment.model,
            )
        )
        yield await self._emit(
            self._finished(
                run,
                call_id,
                HAND_OFF_TOOL,
                arguments,
                began,
                ok=True,
                error_code=None,
                result=text,
            )
        )

    def _switch(self, run: _Run, target: str) -> None:
        """Continue the run on ``target``'s binding: its resolved assignment and tools."""
        run.binding = self._bindings.for_task(target)
        run.tools = self._tools.get(target)
        run.hand_offs.append(target)

    # Compaction --------------------------------------------------------------------

    async def _compact(self, run: _Run, system: tuple[SystemBlock, ...]) -> _Compaction:
        """Offer the conversation to the compaction hook. A hook that raises fails the
        compaction step: the run ends with ``COMPACTION_FAILED``, a fixed reason, and the
        error is logged by class name only and never kept (D0040). Cancellation
        propagates unchanged."""
        if self._compaction is None:
            return _Compaction()
        before = tuple(run.messages)
        replacement: Any = None
        failed: str | None = None
        try:
            replacement = await self._compaction.before_model_call(
                CompactionRequest(
                    run_id=run.run_id,
                    task_class=run.task_class,
                    session_id=run.task.session_id,
                    assignment=run.binding.assignment,
                    system=system,
                    tools=self._specs(run),
                    messages=before,
                    usage=run.meter.usage,
                    last_usage=run.last_usage,
                    model_calls=run.meter.model_calls - 1,
                    charge=_Charge(run),
                )
            )
        except Exception as error:
            failed = type(error).__name__
        if failed is not None:
            logger.warning("the compaction hook failed in run %s: %s", run.run_id, failed)
            return _Compaction(failed=True)
        if replacement is None:
            return _Compaction()
        usable = isinstance(replacement, Sequence) and not isinstance(replacement, str | bytes)
        messages = tuple(replacement) if usable else ()
        if (
            not messages
            or not all(isinstance(m, Message) for m in messages)
            or messages[-1].role != "user"
        ):
            raise CompactionRefused(
                "the compaction hook must return a non-empty conversation ending with a "
                "user message"
            )
        run.messages = list(messages)
        return _Compaction(
            event=ConversationCompactedEvent(
                **run.common(), messages_before=len(before), messages_after=len(messages)
            )
        )

    # Recording ---------------------------------------------------------------------

    async def _emit[E: RunEvent](self, event: E) -> E:
        await self._sink.emit(event)
        return event

    async def _add(
        self, run: _Run, message: Message, *, model: str | None = None, usage: Usage | None = None
    ) -> None:
        run.messages.append(message)
        await self._append(run, (message,), model=model, usage=usage)

    async def _append(
        self,
        run: _Run,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        session = run.task.session_id
        if self._transcripts is None or session is None:
            return
        await self._transcripts.append(
            session, [MessageEntry(message=m, model=model, usage=usage) for m in messages]
        )

    async def _exhaust(self, run: _Run, exhausted: Exhausted) -> AsyncGenerator[RunEvent, None]:
        yield await self._emit(
            BudgetExhaustedEvent(
                **run.common(),
                budget=exhausted.kind,
                limit=exhausted.limit,
                used=exhausted.used,
            )
        )
        yield await self._finish(run, RunFailure.BUDGET_EXHAUSTED, budget=exhausted)

    def _ended(self, run: _Run, **fields: Any) -> RunFinishedEvent:
        return RunFinishedEvent(
            **run.common(),
            usage=run.meter.usage,
            model_calls=run.meter.model_calls,
            tool_calls=run.meter.tool_calls,
            hand_offs=len(run.hand_offs),
            duration_ms=(time.perf_counter() - run.began) * 1000,
            **fields,
        )

    async def _finish(
        self,
        run: _Run,
        failure: RunFailure | None,
        *,
        budget: Exhausted | None = None,
        provider_failure: ProviderFailure | None = None,
        text: str = "",
        output: BaseModel | None = None,
        stop_reason: StopReason | None = None,
        filtered: bool = False,
    ) -> RunFinishedEvent:
        completed = failure is None
        run.result = RunResult(
            run_id=run.run_id,
            status=RunStatus.COMPLETED if completed else RunStatus.FAILED,
            task_class=run.task_class,
            text=text if completed else "",
            output=output if completed else None,
            failure=failure,
            budget=budget.kind if budget is not None else None,
            provider_failure=provider_failure,
            stop_reason=stop_reason,
            usage=run.meter.usage,
            model_calls=run.meter.model_calls,
            tool_calls=run.meter.tool_calls,
            hand_offs=tuple(run.hand_offs),
            messages=tuple(run.messages),
        )
        return await self._emit(
            self._ended(
                run,
                outcome="completed" if completed else "failed",
                failure=failure,
                budget=budget.kind if budget is not None else None,
                provider_failure=provider_failure,
                stop_reason=stop_reason,
                output=self._capture.capture(ContentKind.PROMPT, text) if completed else None,
                filtered=filtered,
            )
        )

    async def _record_end(self, run: _Run, error_type: str) -> None:
        await self._sink.emit(
            self._ended(run, outcome="failed", error_type=_identifier_or(error_type, "error"))
        )


__all__ = [
    "ABANDONED",
    "CANCELLED",
    "DEFAULT_HAND_OFF_DESCRIPTION",
    "HAND_OFF_TOOL",
    "MAX_OUTPUT_ATTEMPTS",
    "UNKNOWN_TOOL",
    "AgentRun",
    "AgentRunner",
    "AgentTask",
    "Bindings",
]
