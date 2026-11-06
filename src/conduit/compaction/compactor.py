"""The compactor: the agent runner's compaction hook.

``Compactor.before_model_call`` has the shape of ``conduit.agent.CompactionHook``, so a
product passes a compactor to ``AgentRunner(compaction=...)`` as it is. Before every
model call it estimates the prompt's size (the larger of its own estimate and the
previous call's reported input and output) and compares it with the assigned model's
context window. The most aggressive strategy whose threshold is reached (the highest
threshold at or below the prompt's share of the window) is applied; below every
threshold nothing changes. ``force`` makes the next call of a session compact whatever
its size, which is how a terminal interface offers a compaction command.

**The cached prefix stays stable (C19).** The hook returns messages only: the system
prompt, the tools and the loaded skills (which the skills layer places in the system
blocks, before their cache breakpoint) are never touched, so they stay byte-identical
before the breakpoint and the provider keeps reading them from its cache. The summary
is the first message, after that prefix. By default it ends with a cache breakpoint of
its own (``cache_summary``), so from the call after a compaction the summary is read
from the cache too. The kept tail is passed through as the same message objects.

**Transcripts.** With a transcript store and a session, the compactor loads the
session's transcript, checks that its active conversation is exactly the conversation
it was shown (the runner appends every message it adds, so they agree when the runner
and the compactor share a store), and appends a ``CompactionPoint`` replacing through
the last message before the kept tail, with the tokens before and after. A reload then
gives the compacted conversation (``Transcript.active_messages``), summary first, byte
for byte as it was sent.

**Events and errors.** Each compaction, skip or failure emits a ``CompactionEvent``
through a fan-out sink under the capture policy (taken the way the agent runner takes
them). A failure raises ``CompactionFailed`` with a fixed reason, after the event, and
the conversation is left as it was; no error keeps the exception it was raised for
(D0040), and cancellation always propagates unchanged.

One compactor may serve many runs at once: it keeps no per-run state, only the forced
compactions waiting for their sessions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, NoReturn, Protocol

from conduit.compaction.conversation import estimate_tokens, find_cut
from conduit.compaction.errors import CompactionFailed
from conduit.compaction.events import CompactionEvent, CompactionTrigger
from conduit.compaction.strategies import CompactionStrategy, StrategyContext, check_strategy
from conduit.core.errors import ConfigurationError
from conduit.core.events import CapturePolicy, ContentKind, EventSink, FanOutSink
from conduit.core.tasks import ResolvedAssignment
from conduit.core.transcript import (
    CompactionPoint,
    MessageEntry,
    Transcript,
    TranscriptStore,
)
from conduit.core.types import (
    CacheBreakpoint,
    ContentBlock,
    Message,
    SystemBlock,
    TextBlock,
    ToolSpec,
    Usage,
)

__all__ = ["CompactionContext", "Compactor", "TokenEstimator"]

logger = logging.getLogger("conduit.compaction")


class CompactionContext(Protocol):
    """What the hook reads from the runner's ``CompactionRequest``."""

    @property
    def run_id(self) -> str: ...

    @property
    def task_class(self) -> str: ...

    @property
    def session_id(self) -> str | None: ...

    @property
    def assignment(self) -> ResolvedAssignment: ...

    @property
    def system(self) -> tuple[SystemBlock, ...]: ...

    @property
    def tools(self) -> tuple[ToolSpec, ...]: ...

    @property
    def messages(self) -> tuple[Message, ...]: ...

    @property
    def last_usage(self) -> Usage | None: ...


class TokenEstimator(Protocol):
    def __call__(
        self,
        *,
        system: Sequence[SystemBlock] = (),
        tools: Sequence[ToolSpec] = (),
        messages: Sequence[Message] = (),
    ) -> int: ...


class Compactor:
    """Compacts conversations before model calls, by threshold or on request.

    ``strategies`` are tried by threshold; their names must be unique. ``transcripts``
    receives a ``CompactionPoint`` for each compaction of a run with a session.
    ``events`` receives a ``CompactionEvent`` for each attempt, under ``capture`` (nothing
    captured by default: pass the deployment's ``ConduitSettings.capture``). ``estimate``
    replaces the token estimate, and ``cache_summary`` closes the summary with a cache
    breakpoint.
    """

    def __init__(
        self,
        strategies: Sequence[CompactionStrategy],
        *,
        transcripts: TranscriptStore | None = None,
        events: EventSink | None = None,
        capture: CapturePolicy | None = None,
        estimate: TokenEstimator = estimate_tokens,
        cache_summary: bool = True,
    ) -> None:
        if not strategies:
            raise ConfigurationError("a compactor needs at least one strategy")
        names: set[str] = set()
        for strategy in strategies:
            check_strategy(strategy.name, strategy.threshold, strategy.keep_turns)
            if strategy.name in names:
                raise ConfigurationError(f"two strategies are named {strategy.name!r}")
            names.add(strategy.name)
        # Stable: strategies with equal thresholds keep the order they were given in.
        self._strategies = tuple(sorted(strategies, key=lambda s: s.threshold))
        self._names = frozenset(names)
        self._transcripts = transcripts
        self._capture = capture if capture is not None else CapturePolicy()
        self._sink = FanOutSink([events] if events is not None else [], policy=self._capture)
        self._estimate = estimate
        self._cache_summary = cache_summary
        self._forced: dict[str | None, str | None] = {}

    @property
    def strategies(self) -> tuple[CompactionStrategy, ...]:
        """The strategies, from the lowest threshold to the highest."""
        return self._strategies

    def force(self, session_id: str | None = None, *, strategy: str | None = None) -> None:
        """Compact the next model call of ``session_id`` (runs without a session when
        ``None``) whatever its size, with ``strategy`` or the most aggressive one."""
        if strategy is not None and strategy not in self._names:
            raise ConfigurationError(f"no strategy named {strategy!r}")
        self._forced[session_id] = strategy

    # The hook ---------------------------------------------------------------------------

    def _tokens(self, request: CompactionContext) -> int:
        estimated = self._estimate(
            system=request.system, tools=request.tools, messages=request.messages
        )
        usage = request.last_usage
        reported = usage.total_input_tokens + usage.output_tokens if usage is not None else 0
        return max(estimated, reported)

    def _by_threshold(self, tokens: int, window: int) -> CompactionStrategy | None:
        chosen: CompactionStrategy | None = None
        for strategy in self._strategies:
            if tokens / window >= strategy.threshold:
                chosen = strategy
        return chosen

    def _named(self, name: str | None) -> CompactionStrategy:
        if name is None:
            return self._strategies[-1]
        return next(s for s in self._strategies if s.name == name)

    async def before_model_call(self, request: CompactionContext) -> tuple[Message, ...] | None:
        """The compacted conversation, or ``None`` to send it unchanged."""
        messages = tuple(request.messages)
        window = request.assignment.entry.context_window
        tokens = self._tokens(request)
        trigger: CompactionTrigger
        if request.session_id in self._forced:
            strategy = self._named(self._forced.pop(request.session_id))
            trigger = "forced"
        else:
            found = self._by_threshold(tokens, window)
            if found is None:
                return None
            strategy, trigger = found, "threshold"
        event: dict[str, Any] = {
            "run_id": request.run_id,
            "task_class": request.task_class,
            "strategy": strategy.name,
            "trigger": trigger,
            "context_window": window,
            "tokens_before": tokens,
            "messages_before": len(messages),
        }
        cut = find_cut(messages, strategy.keep_turns)
        if cut is None:
            if trigger == "forced":
                await self._sink.emit(
                    CompactionEvent(**event, outcome="skipped", reason="nothing_to_compact")
                )
            return None

        # Where the compaction point goes, checked before any summary is asked for.
        store, session = self._transcripts, request.session_id
        sequence: list[int] | None = None
        previous = 0
        if store is not None and session is not None:
            loaded: Transcript | None = None
            error_class: str | None = None
            try:
                loaded = await store.load(session)
            except Exception as error:
                error_class = type(error).__name__
            if loaded is None:
                await self._fail(event, CompactionFailed("store_failed", error_class=error_class))
            sequence = _sequence_numbers(loaded, messages)
            if sequence is None:
                await self._fail(event, CompactionFailed("transcript_mismatch"))
            point = loaded.latest_compaction()
            previous = point.replaces_through if point is not None else 0

        text = await self._summary(event, strategy, messages[:cut], request)
        summary_content: tuple[ContentBlock, ...] = (TextBlock(text=text),)
        if self._cache_summary:
            summary_content = (*summary_content, CacheBreakpoint())
        summary = Message(role="user", content=summary_content)
        replacement = (summary, *messages[cut:])
        after = self._estimate(system=request.system, tools=request.tools, messages=replacement)

        through: int | None = None
        replaced = cut
        if store is not None and session is not None and sequence is not None:
            through = sequence[cut - 1]
            replaced = sum(1 for s in sequence[:cut] if s > previous)
            failed_class: str | None = None
            stored = False
            try:
                await store.append(
                    session,
                    [
                        CompactionPoint(
                            summary=summary,
                            replaces_through=through,
                            strategy=strategy.name,
                            tokens_before=tokens,
                            tokens_after=after,
                        )
                    ],
                )
                stored = True
            except Exception as error:
                failed_class = type(error).__name__
            if not stored:
                await self._fail(event, CompactionFailed("store_failed", error_class=failed_class))

        await self._sink.emit(
            CompactionEvent(
                **event,
                outcome="compacted",
                tokens_after=after,
                messages_after=len(replacement),
                entries_replaced=replaced,
                replaces_through=through,
                summary=self._capture.capture(ContentKind.PROMPT, text),
            )
        )
        logger.info(
            "compacted run %s with %s: %d to %d tokens",
            request.run_id,
            strategy.name,
            tokens,
            after,
        )
        return replacement

    async def _summary(
        self,
        event: dict[str, Any],
        strategy: CompactionStrategy,
        replaced: tuple[Message, ...],
        request: CompactionContext,
    ) -> str:
        context = StrategyContext(run_id=request.run_id, assignment=request.assignment)
        failure: CompactionFailed | None = None
        text: str | None = None
        try:
            text = await strategy.summarise(replaced, context)
        except CompactionFailed as error:
            # A fresh error: a product strategy's may carry a chain (D0040).
            failure = CompactionFailed(
                error.reason, failure=error.failure, error_class=error.error_class
            )
        except Exception as error:
            failure = CompactionFailed("strategy_failed", error_class=type(error).__name__)
        if failure is not None:
            await self._fail(event, failure)
        if not isinstance(text, str):
            await self._fail(event, CompactionFailed("strategy_failed", error_class="TypeError"))
        if not text.strip():  # an empty text block is refused by providers
            await self._fail(event, CompactionFailed("summary_empty"))
        return text

    async def _fail(self, event: dict[str, Any], failure: CompactionFailed) -> NoReturn:
        """Record the failure and raise it. Never called inside an exception handler."""
        await self._sink.emit(CompactionEvent(**event, outcome="failed", reason=failure.reason))
        logger.warning("compaction failed: %s", failure.reason)
        raise failure


def _sequence_numbers(transcript: Transcript, messages: tuple[Message, ...]) -> list[int] | None:
    """The sequence number behind each of ``messages`` in ``transcript``'s active
    conversation (an earlier summary stands at its point's ``replaces_through``), or
    ``None`` when the active conversation is not exactly ``messages``."""
    point = transcript.latest_compaction()
    active: list[tuple[Message, int]] = []
    since = 0
    if point is not None:
        active.append((point.summary, point.replaces_through))
        since = point.replaces_through
    active.extend(
        (e.message, e.seq or 0)
        for e in transcript.entries
        if isinstance(e, MessageEntry) and (e.seq or 0) > since
    )
    if tuple(m for m, _ in active) != messages:
        return None
    return [seq for _, seq in active]
