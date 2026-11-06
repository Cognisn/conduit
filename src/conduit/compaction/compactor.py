"""The compactor: the agent runner's compaction hook.

``Compactor.before_model_call`` has the shape of ``conduit.agent.CompactionHook``, so a
product passes a compactor to ``AgentRunner(compaction=...)`` as it is. Before every
model call it estimates the prompt's size (the larger of its own estimate and the
previous call's reported input and output) and compares it with the assigned model's
context window. The most aggressive strategy whose threshold is reached (the highest
threshold at or below the prompt's share of the window) is tried; below every threshold
nothing changes. ``force`` makes the next call of a session compact whatever its size,
which is how a terminal interface offers a compaction command.

**While the prompt fits, a compaction must bring it under its threshold.** "Fits" means
the prompt and the assignment's output budget together are within the context window.
While they are, before anything is asked of a strategy, the fixed prefix (system and
tools), the kept tail and the strategy's ``summary_allowance`` are estimated together:
if they already reach the threshold, no compaction could get under it, so none is tried
(``skipped``, ``prefix_too_large``). A compaction whose result is still at or over the
threshold is not applied either (``skipped``, ``not_below_threshold``). So a conversation
is never compacted again and again on every call, and no summary is paid for that could
not help.

**Once the prompt no longer fits, a compaction is needed**, so it is applied whenever its
result fits, even above the threshold. If the fixed prefix and the kept tail alone
already leave no room for the output budget, it fails before any summary is asked for;
the summary allowance is not counted here, since a shorter summary may still fit. A
result that does not fit fails too (``result_too_large``, raised, since the call could
not succeed). A forced compaction skips all of these checks: it is
applied whenever there is something to replace.

The estimate counts about four characters a token. It undercounts some text (Chinese,
Japanese and Korean, for example, take more tokens a character), so a prompt it judges
to fit may still be refused by the provider for its length; pass ``estimate`` with the
model's own count where that matters.

**The cached prefix stays stable (C19).** The hook returns messages only: the system
prompt, the tools and the loaded skills (which the skills layer places in the system
blocks, before their cache breakpoint) are never touched, so they stay byte-identical
before the breakpoint and the provider keeps reading them from its cache. The summary
is the first message, after that prefix. By default it ends with a cache breakpoint of
its own (``cache_summary``), so from the call after a compaction the summary is read
from the cache too. The kept tail is passed through as the same message objects. Some
providers limit the breakpoints in one request (Anthropic allows four): with tools,
system and skills already using three or more, the summary's breakpoint can push one
out, so a product near that limit passes ``cache_summary=False``.

**Transcripts.** With a transcript store and a session, the compactor loads the
session's transcript, checks that its active conversation is exactly the conversation
it was shown (the runner appends every message it adds, so they agree when the runner
and the compactor share a store), and appends a ``CompactionPoint`` replacing through
the last message before the kept tail, with the tokens before and after. A reload then
gives the compacted conversation (``Transcript.active_messages``), summary first, byte
for byte as it was sent.

**Events and failures.** Each compaction, skip or failure emits a ``CompactionEvent``
through a fan-out sink under the capture policy (taken the way the agent runner takes
them). When a compaction fails (``CompactionFailed``, with a fixed reason), the
conversation is left as it was: while the prompt and the model's output budget still
fit the context window, the hook returns ``None`` and the call goes ahead uncompacted
(the next call tries again); once they no longer fit, the hook raises the
``CompactionFailed``, after the event, since the call could not succeed. No error keeps
the exception it was raised for (D0040), and cancellation always propagates unchanged.

**Forced compactions** wait, per session, until one succeeds or finds nothing to
replace. A call claims its session's request when it starts, so concurrent calls do not
run it twice; a failed or cancelled attempt puts it back for the next call, unless the
session was forced again meanwhile, and a force made during an attempt is kept for the
next call whatever that attempt's outcome. At most ``max_forced`` sessions wait at once:
forcing another drops the one waiting longest.

One compactor may serve many runs at once: it keeps no per-run state, only the forced
compactions waiting for their sessions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol

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

__all__ = ["DEFAULT_MAX_FORCED", "CompactionContext", "Compactor", "TokenEstimator"]

logger = logging.getLogger("conduit.compaction")

DEFAULT_MAX_FORCED = 1_024


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


def _allowance(strategy: CompactionStrategy) -> int:
    value = getattr(strategy, "summary_allowance", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class Compactor:
    """Compacts conversations before model calls, by threshold or on request.

    ``strategies`` are tried by threshold; their names must be unique. ``transcripts``
    receives a ``CompactionPoint`` for each compaction of a run with a session.
    ``events`` receives a ``CompactionEvent`` for each attempt, under ``capture`` (nothing
    captured by default: pass the deployment's ``ConduitSettings.capture``). ``estimate``
    replaces the token estimate, ``cache_summary`` closes the summary with a cache
    breakpoint, and ``max_forced`` bounds the forced compactions waiting.
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
        max_forced: int = DEFAULT_MAX_FORCED,
    ) -> None:
        if not strategies:
            raise ConfigurationError("a compactor needs at least one strategy")
        if max_forced < 1:
            raise ConfigurationError("max_forced must be at least 1")
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
        self._max_forced = max_forced
        self._forced: dict[str | None, str | None] = {}

    @property
    def strategies(self) -> tuple[CompactionStrategy, ...]:
        """The strategies, from the lowest threshold to the highest."""
        return self._strategies

    def force(self, session_id: str | None = None, *, strategy: str | None = None) -> None:
        """Compact the next model call of ``session_id`` (runs without a session when
        ``None``) whatever its size, with ``strategy`` or the most aggressive one. It
        waits until a compaction succeeds or finds nothing to replace."""
        if strategy is not None and strategy not in self._names:
            raise ConfigurationError(f"no strategy named {strategy!r}")
        self._queue(session_id, strategy)

    def _queue(self, session_id: str | None, strategy: str | None) -> None:
        if session_id not in self._forced and len(self._forced) >= self._max_forced:
            del self._forced[next(iter(self._forced))]  # the one waiting longest
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
        session = request.session_id
        if session not in self._forced:
            result, _ = await self._attempt(request, None, claimed=False)
            return result
        # Claimed now, so concurrent calls of the session do not run it too; put back
        # unless it was used, and unless a newer force has arrived meanwhile.
        name = self._forced.pop(session)
        used = False
        try:
            result, used = await self._attempt(request, name, claimed=True)
        finally:
            if not used and session not in self._forced:
                self._queue(session, name)
        return result

    async def _attempt(
        self, request: CompactionContext, forced: str | None, *, claimed: bool
    ) -> tuple[tuple[Message, ...] | None, bool]:
        """The compacted conversation or ``None``, and whether a forced request was used
        (it compacted, or found nothing to replace)."""
        messages = tuple(request.messages)
        assignment = request.assignment
        window = assignment.entry.context_window
        budget = assignment.max_output_tokens
        tokens = self._tokens(request)
        fits = tokens + budget <= window
        trigger: CompactionTrigger
        if claimed:
            strategy = self._named(forced)
            trigger = "forced"
        else:
            found = self._by_threshold(tokens, window)
            if found is None:
                return None, False
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
                await self._skip(event, "nothing_to_compact")
                return None, True
            return None, False
        tail = messages[cut:]
        if trigger == "threshold":
            least = self._estimate(system=request.system, tools=request.tools, messages=tail)
            if fits:
                if (least + _allowance(strategy)) / window >= strategy.threshold:
                    await self._skip(event, "prefix_too_large")
                    return None, False
            # Past the window the allowance is not counted: it is what a summary might
            # take, and a shorter one may still fit. The result is checked once built.
            elif least + budget > window:
                return await self._failed(event, CompactionFailed("result_too_large"), fits), False

        session = request.session_id
        # Where the compaction point goes, checked before any summary is asked for.
        store = self._transcripts
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
                failed = CompactionFailed("store_failed", error_class=error_class)
                return await self._failed(event, failed, fits), False
            sequence = _sequence_numbers(loaded, messages)
            if sequence is None:
                return await self._failed(
                    event, CompactionFailed("transcript_mismatch"), fits
                ), False
            point = loaded.latest_compaction()
            previous = point.replaces_through if point is not None else 0

        text, failure = await self._summary(strategy, messages[:cut], request)
        if failure is not None or text is None:
            return await self._failed(
                event, failure or CompactionFailed("strategy_failed"), fits
            ), False
        summary_content: tuple[ContentBlock, ...] = (TextBlock(text=text),)
        if self._cache_summary:
            summary_content = (*summary_content, CacheBreakpoint())
        summary = Message(role="user", content=summary_content)
        replacement = (summary, *tail)
        after = self._estimate(system=request.system, tools=request.tools, messages=replacement)
        if trigger == "threshold":
            if fits and after / window >= strategy.threshold:
                await self._skip(event, "not_below_threshold", tokens_after=after)
                return None, False
            if not fits and after + budget > window:
                return await self._failed(event, CompactionFailed("result_too_large"), fits), False

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
                failed = CompactionFailed("store_failed", error_class=failed_class)
                return await self._failed(event, failed, fits), False

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
        return replacement, True

    async def _summary(
        self,
        strategy: CompactionStrategy,
        replaced: tuple[Message, ...],
        request: CompactionContext,
    ) -> tuple[str | None, CompactionFailed | None]:
        """The strategy's text, or the failure to report. Never raises but cancellation."""
        context = StrategyContext(
            run_id=request.run_id, assignment=request.assignment, tools=tuple(request.tools)
        )
        try:
            text: object = await strategy.summarise(replaced, context)
        except CompactionFailed as error:
            # A fresh error: a product strategy's may carry a chain or free text (D0040).
            return None, CompactionFailed(
                error.reason, failure=error.failure, error_class=error.error_class
            )
        except Exception as error:
            return None, CompactionFailed("strategy_failed", error_class=type(error).__name__)
        if not isinstance(text, str):
            return None, CompactionFailed("strategy_failed", error_class="TypeError")
        if not text.strip():  # an empty text block is refused by providers
            return None, CompactionFailed("summary_empty")
        return text, None

    async def _skip(self, event: dict[str, Any], reason: str, **fields: Any) -> None:
        await self._sink.emit(CompactionEvent(**event, outcome="skipped", reason=reason, **fields))
        logger.debug("compaction skipped: %s", reason)

    async def _failed(
        self, event: dict[str, Any], failure: CompactionFailed, fits: bool
    ) -> tuple[Message, ...] | None:
        """Record the failure; leave the conversation as it was while it still fits the
        window, and raise otherwise. Never called inside an exception handler."""
        await self._sink.emit(CompactionEvent(**event, outcome="failed", reason=failure.reason))
        logger.warning("compaction failed: %s", failure.reason)
        if not fits:
            raise failure
        return None


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
