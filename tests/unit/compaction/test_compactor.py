"""The Compactor as the runner's compaction hook, driven directly with a request."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from conduit.compaction import (
    CompactionEvent,
    CompactionFailed,
    Compactor,
    DropToolResults,
    KeepLastTurns,
    StrategyContext,
    Summarise,
    estimate_tokens,
)
from conduit.core.errors import ConfigurationError
from conduit.core.events import CapturePolicy
from conduit.core.tasks import ResolvedAssignment
from conduit.core.transcript import CompactionPoint, MessageEntry, Transcript, TranscriptEntry
from conduit.core.types import CacheBreakpoint, Message, TextBlock, Usage
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.compaction.helpers import (
    SMALL_WINDOW,
    Request,
    Sink,
    chat,
    conversation,
    request,
    runtime,
    summary_script,
)


async def _chat_assignment() -> ResolvedAssignment:
    return (await runtime()).assignments["chat"]


def _over(fraction: float, messages: Sequence[Message]) -> bool:
    return estimate_tokens(messages=messages) >= fraction * SMALL_WINDOW


# Construction ------------------------------------------------------------------------


def test_a_compactor_needs_a_strategy() -> None:
    with pytest.raises(ConfigurationError):
        Compactor([])


def test_strategy_names_must_be_unique() -> None:
    with pytest.raises(ConfigurationError):
        Compactor([KeepLastTurns(threshold=0.5), KeepLastTurns(threshold=0.9)])


def test_forcing_an_unknown_strategy_is_refused() -> None:
    compactor = Compactor([KeepLastTurns()])
    with pytest.raises(ConfigurationError):
        compactor.force(strategy="absent")


# Thresholds ----------------------------------------------------------------------------


async def test_below_the_threshold_nothing_changes() -> None:
    assignment = await _chat_assignment()
    messages = chat(3, chars=200)
    assert not _over(0.9, messages)
    compactor = Compactor([KeepLastTurns(threshold=0.9, keep_turns=1)])
    assert await compactor.before_model_call(request(assignment, messages)) is None


async def test_at_the_threshold_the_conversation_is_compacted() -> None:
    assignment = await _chat_assignment()
    messages = chat(4, chars=400)
    tokens = estimate_tokens(messages=messages)
    exact = tokens / SMALL_WINDOW  # the threshold is reached exactly
    compactor = Compactor([KeepLastTurns(threshold=exact, keep_turns=1)])
    replaced = await compactor.before_model_call(request(assignment, messages))
    assert replaced is not None
    just_above = Compactor([KeepLastTurns(threshold=(tokens + 1) / SMALL_WINDOW, keep_turns=1)])
    assert await just_above.before_model_call(request(assignment, messages)) is None


async def test_the_threshold_is_against_the_assigned_models_window() -> None:
    rt = await runtime()
    messages = chat(4, chars=400)
    share = estimate_tokens(messages=messages) / SMALL_WINDOW
    compactor = Compactor([KeepLastTurns(threshold=share, keep_turns=1)])
    # The summary model's window is four times larger, so the same conversation is under.
    assert await compactor.before_model_call(request(rt.assignments["summary"], messages)) is None
    assert await compactor.before_model_call(request(rt.assignments["chat"], messages))


async def test_reported_usage_counts_when_it_is_larger_than_the_estimate() -> None:
    assignment = await _chat_assignment()
    messages = chat(3, chars=200)
    compactor = Compactor([KeepLastTurns(threshold=0.5, keep_turns=1)])
    assert await compactor.before_model_call(request(assignment, messages)) is None
    reported = Usage(
        input_tokens=1_000, output_tokens=100, cache_read_tokens=900, cache_write_tokens=0
    )
    assert await compactor.before_model_call(request(assignment, messages, last_usage=reported))


async def test_the_most_aggressive_triggered_strategy_is_chosen() -> None:
    assignment = await _chat_assignment()
    messages = conversation(6)
    share = estimate_tokens(messages=messages) / SMALL_WINDOW
    sink = Sink()
    drop = DropToolResults(threshold=share / 2, keep_turns=1)
    keep = KeepLastTurns(threshold=share, keep_turns=1)
    later = KeepLastTurns(threshold=min(1.0, share * 1.5), keep_turns=1, name="later")
    compactor = Compactor([keep, later, drop], events=sink)
    await compactor.before_model_call(request(assignment, messages))
    assert [e.strategy for e in sink.of(CompactionEvent)] == ["keep_last_turns"]
    sink.events.clear()
    smaller = messages[:7]
    await compactor.before_model_call(request(assignment, smaller))
    assert [e.strategy for e in sink.of(CompactionEvent)] == ["drop_tool_results"]


# The replacement -------------------------------------------------------------------------


async def test_the_replacement_is_the_summary_then_the_tail_unchanged() -> None:
    assignment = await _chat_assignment()
    messages = conversation(5)
    compactor = Compactor([KeepLastTurns(threshold=0.01, keep_turns=2, notice="NOTICE")])
    replaced = await compactor.before_model_call(request(assignment, messages))
    assert replaced is not None
    summary, *tail = replaced
    assert summary == Message(role="user", content=(TextBlock(text="NOTICE"), CacheBreakpoint()))
    assert tuple(tail) == messages[-4:]
    assert all(a is b for a, b in zip(tail, messages[-4:], strict=True))
    assert replaced[-1].role == "user"


async def test_the_summary_can_be_left_out_of_the_cached_prefix() -> None:
    assignment = await _chat_assignment()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1, notice="N")], cache_summary=False
    )
    replaced = await compactor.before_model_call(request(assignment, conversation(3)))
    assert replaced is not None
    assert replaced[0] == Message.user("N")


async def test_nothing_is_compacted_when_too_little_would_go() -> None:
    assignment = await _chat_assignment()
    compactor = Compactor([KeepLastTurns(threshold=0.01, keep_turns=3)])
    assert await compactor.before_model_call(request(assignment, conversation(3))) is None


async def test_system_and_tools_are_never_touched() -> None:
    # The hook returns messages only; this checks it reads but never rewrites the prefix.
    assignment = await _chat_assignment()
    system = (TextBlock(text="You help example.com."), CacheBreakpoint())
    shown = request(assignment, conversation(4), system=system)
    compactor = Compactor([KeepLastTurns(threshold=0.01, keep_turns=1)])
    replaced = await compactor.before_model_call(shown)
    assert replaced is not None
    assert shown.system == system
    assert all(
        not isinstance(b, TextBlock) or b.text != "You help example.com."
        for m in replaced
        for b in m.content
    )


# Forcing ---------------------------------------------------------------------------------


async def test_a_forced_compaction_runs_below_the_threshold_once() -> None:
    assignment = await _chat_assignment()
    sink = Sink()
    compactor = Compactor([KeepLastTurns(threshold=1.0, keep_turns=1)], events=sink)
    messages = chat(3, chars=100)
    compactor.force()
    assert await compactor.before_model_call(request(assignment, messages)) is not None
    assert await compactor.before_model_call(request(assignment, messages)) is None
    [event] = sink.of(CompactionEvent)
    assert (event.trigger, event.outcome) == ("forced", "compacted")


async def test_forcing_is_per_session() -> None:
    assignment = await _chat_assignment()
    compactor = Compactor([KeepLastTurns(threshold=1.0, keep_turns=1)])
    compactor.force("s-2")
    messages = chat(3, chars=100)
    assert (
        await compactor.before_model_call(request(assignment, messages, session_id="s-1")) is None
    )
    assert await compactor.before_model_call(request(assignment, messages, session_id="s-2"))


async def test_forcing_chooses_the_named_strategy_or_the_most_aggressive() -> None:
    assignment = await _chat_assignment()
    sink = Sink()
    drop = DropToolResults(threshold=0.5, keep_turns=1)
    keep = KeepLastTurns(threshold=0.9, keep_turns=1)
    compactor = Compactor([drop, keep], events=sink)
    compactor.force()
    await compactor.before_model_call(request(assignment, conversation(3)))
    compactor.force(strategy="drop_tool_results")
    await compactor.before_model_call(request(assignment, conversation(3)))
    assert [e.strategy for e in sink.of(CompactionEvent)] == [
        "keep_last_turns",
        "drop_tool_results",
    ]


async def test_a_forced_compaction_with_nothing_to_cut_is_recorded_as_skipped() -> None:
    assignment = await _chat_assignment()
    sink = Sink()
    compactor = Compactor([KeepLastTurns(keep_turns=1)], events=sink)
    compactor.force()
    assert await compactor.before_model_call(request(assignment, (Message.user("hi"),))) is None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("skipped", "nothing_to_compact")


# Events ----------------------------------------------------------------------------------


async def test_the_event_carries_strategy_tokens_and_entries_replaced() -> None:
    assignment = await _chat_assignment()
    sink = Sink()
    messages = conversation(5)
    compactor = Compactor([KeepLastTurns(threshold=0.01, keep_turns=2, notice="N")], events=sink)
    replaced = await compactor.before_model_call(request(assignment, messages))
    assert replaced is not None
    [event] = sink.of(CompactionEvent)
    assert event.strategy == "keep_last_turns"
    assert event.trigger == "threshold"
    assert event.outcome == "compacted"
    assert event.run_id == "run_test"
    assert event.task_class == "chat"
    assert event.context_window == SMALL_WINDOW
    assert event.tokens_before == estimate_tokens(messages=messages)
    assert event.tokens_after == estimate_tokens(messages=replaced)
    assert event.tokens_after < event.tokens_before
    assert event.messages_before == len(messages)
    assert event.messages_after == len(replaced)
    assert event.entries_replaced == len(messages) - 4
    assert event.replaces_through is None  # no transcript store


async def test_the_summary_text_is_captured_only_when_prompts_are() -> None:
    assignment = await _chat_assignment()
    for policy, expected in ((None, None), (CapturePolicy(prompts=True), "SECRET-NOTICE")):
        sink = Sink()
        compactor = Compactor(
            [KeepLastTurns(threshold=0.01, keep_turns=1, notice="SECRET-NOTICE")],
            events=sink,
            capture=policy,
        )
        await compactor.before_model_call(request(assignment, conversation(3)))
        [event] = sink.of(CompactionEvent)
        assert event.summary is not None
        assert event.summary.size == len("SECRET-NOTICE")
        assert event.summary.text == expected
        assert "SECRET-NOTICE" not in event.model_dump_json(exclude={"summary"})


async def test_tool_content_capture_alone_does_not_capture_the_summary() -> None:
    assignment = await _chat_assignment()
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1, notice="SECRET-NOTICE")],
        events=sink,
        capture=CapturePolicy(tool_content=True),
    )
    await compactor.before_model_call(request(assignment, conversation(3)))
    [event] = sink.of(CompactionEvent)
    assert event.summary is not None and event.summary.text is None


# Transcripts -----------------------------------------------------------------------------


async def _stored(store: MemoryTranscriptStore, session: str, messages: Sequence[Message]) -> None:
    await store.append(session, [MessageEntry(message=m) for m in messages])


async def test_a_compaction_point_is_written_with_tokens_before_and_after() -> None:
    assignment = await _chat_assignment()
    store = MemoryTranscriptStore()
    messages = conversation(5)
    await _stored(store, "s-1", messages)
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=2, notice="N")], transcripts=store, events=sink
    )
    replaced = await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert replaced is not None
    transcript = await store.load("s-1")
    point = transcript.latest_compaction()
    assert point is not None
    assert point.strategy == "keep_last_turns"
    assert point.replaces_through == len(messages) - 4  # the sequence number of the last gone
    assert point.summary == replaced[0]
    [event] = sink.of(CompactionEvent)
    assert (point.tokens_before, point.tokens_after) == (event.tokens_before, event.tokens_after)
    assert event.replaces_through == point.replaces_through
    assert transcript.active_messages() == tuple(replaced)


async def test_a_second_compaction_folds_in_the_first() -> None:
    assignment = await _chat_assignment()
    store = MemoryTranscriptStore()
    messages = conversation(4)
    await _stored(store, "s-1", messages)
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1, notice="N")], transcripts=store, events=sink
    )
    first = await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert first is not None
    more = conversation(7)[len(messages) :]  # three more tool turns
    await _stored(store, "s-1", more)
    shown = (*first, *more)
    second = await compactor.before_model_call(request(assignment, shown, session_id="s-1"))
    assert second is not None
    transcript = await store.load("s-1")
    points = [e for e in transcript.entries if isinstance(e, CompactionPoint)]
    assert len(points) == 2
    assert points[0].replaces_through < points[1].replaces_through
    assert transcript.active_messages() == tuple(second)
    entries = sink.of(CompactionEvent)
    # The first summary is not counted again: only the newly replaced messages are.
    assert entries[1].entries_replaced == len(shown) - 2 - 1


async def test_the_point_replaces_through_the_last_message_before_the_tail() -> None:
    assignment = await _chat_assignment()
    store = MemoryTranscriptStore()
    messages = chat(4, chars=100)
    await _stored(store, "s-1", messages)
    compactor = Compactor([KeepLastTurns(threshold=0.01, keep_turns=1)], transcripts=store)
    replaced = await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert replaced is not None
    transcript = await store.load("s-1")
    point = transcript.latest_compaction()
    assert point is not None
    first_kept = next(
        e for e in transcript.entries if isinstance(e, MessageEntry) and e.message == replaced[1]
    )
    assert point.replaces_through == (first_kept.seq or 0) - 1


async def test_a_transcript_that_does_not_match_is_refused() -> None:
    assignment = await _chat_assignment()
    store = MemoryTranscriptStore()
    messages = conversation(4)
    await _stored(store, "s-1", messages[:-1])  # the last message never stored
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1)], transcripts=store, events=sink
    )
    with pytest.raises(CompactionFailed) as raised:
        await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert raised.value.reason == "transcript_mismatch"
    assert (await store.load("s-1")).latest_compaction() is None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "transcript_mismatch")


async def test_without_a_session_nothing_is_written() -> None:
    assignment = await _chat_assignment()
    store = MemoryTranscriptStore()
    compactor = Compactor([KeepLastTurns(threshold=0.01, keep_turns=1)], transcripts=store)
    assert await compactor.before_model_call(request(assignment, conversation(3)))
    assert await store.list_sessions() == []


# Failures (D0040) ------------------------------------------------------------------------


class FailingStore(MemoryTranscriptStore):
    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on
        self.armed = False

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        if self.armed and self.fail_on == "append":
            raise OSError("disk full at /srv/example.com/secret-path")
        return await super().append(session_id, entries)

    async def load(self, session_id: str) -> Transcript:
        if self.armed and self.fail_on == "load":
            raise OSError("cannot read /srv/example.com/secret-path")
        return await super().load(session_id)


def _clean(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), *map(str, error.args)):
        assert "example.com" not in text and "secret" not in text and "SUMMARY" not in text


@pytest.mark.parametrize("fail_on", ["append", "load"])
async def test_a_failing_store_is_reported_by_class_only(fail_on: str) -> None:
    assignment = await _chat_assignment()
    store = FailingStore(fail_on)
    messages = conversation(4)
    await _stored(store, "s-1", messages)
    store.armed = True
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1)], transcripts=store, events=sink
    )
    with pytest.raises(CompactionFailed) as raised:
        await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert raised.value.reason == "store_failed"
    assert raised.value.error_class == "OSError"
    _clean(raised.value)
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "store_failed")


async def test_a_failing_product_strategy_is_wrapped_without_its_text() -> None:
    class Broken:
        name = "broken"
        threshold = 0.01
        keep_turns = 1

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            raise ValueError("SUMMARY with example.com secret")

    assignment = await _chat_assignment()
    with pytest.raises(CompactionFailed) as raised:
        await Compactor([Broken()]).before_model_call(request(assignment, conversation(3)))
    assert (raised.value.reason, raised.value.error_class) == ("strategy_failed", "ValueError")
    _clean(raised.value)


async def test_a_failed_summary_leaves_no_point_and_is_reported() -> None:
    from conduit.core.errors import ProviderError, ProviderFailure
    from conduit.providers.stub import scripted

    rt = await runtime(summariser=scripted([ProviderError(ProviderFailure.TIMEOUT)]))
    store = MemoryTranscriptStore()
    messages = conversation(4)
    await _stored(store, "s-1", messages)
    sink = Sink()
    compactor = Compactor(
        [Summarise(rt, task_class="summary", threshold=0.01, keep_turns=1)],
        transcripts=store,
        events=sink,
    )
    with pytest.raises(CompactionFailed) as raised:
        await compactor.before_model_call(
            request(rt.assignments["chat"], messages, session_id="s-1")
        )
    _clean(raised.value)
    assert raised.value.failure is ProviderFailure.TIMEOUT
    assert (await store.load("s-1")).latest_compaction() is None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason, event.strategy) == (
        "failed",
        "summary_failed",
        "summarise",
    )


async def test_cancellation_propagates_unchanged() -> None:
    class Slow:
        name = "slow"
        threshold = 0.01
        keep_turns = 1

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            await asyncio.sleep(10)
            return "never"

    assignment = await _chat_assignment()
    sink = Sink()
    compactor = Compactor([Slow()], events=sink)
    task = asyncio.create_task(compactor.before_model_call(request(assignment, conversation(3))))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sink.of(CompactionEvent) == []


async def test_a_summarised_compaction_end_to_end() -> None:
    log = summary_script("SUMMARY: items 0 to 2 checked.")
    rt = await runtime(summariser=log)
    sink = Sink()
    compactor = Compactor(
        [Summarise(rt, task_class="summary", threshold=0.01, keep_turns=1)], events=sink
    )
    replaced = await compactor.before_model_call(request(rt.assignments["chat"], conversation(4)))
    assert replaced is not None
    first = replaced[0].content[0]
    assert isinstance(first, TextBlock)
    assert first.text.endswith("SUMMARY: items 0 to 2 checked.")
    assert len(log.requests) == 1
    assert sink.of(CompactionEvent)[0].strategy == "summarise"


def test_request_helper_matches_the_runner_shape() -> None:
    from dataclasses import fields

    from conduit.agent import CompactionRequest

    assert [f.name for f in fields(Request)] == [f.name for f in fields(CompactionRequest)]


@pytest.mark.parametrize(
    ("returned", "reason"),
    [
        ("", "summary_empty"),
        ("  \n", "summary_empty"),
        (None, "strategy_failed"),
        (42, "strategy_failed"),
    ],
)
async def test_a_strategy_must_return_some_text(returned: object, reason: str) -> None:
    class Odd:
        name = "odd"
        threshold = 0.01
        keep_turns = 1

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> object:
            return returned

    assignment = await _chat_assignment()
    sink = Sink()
    compactor = Compactor([Odd()], events=sink)  # type: ignore[list-item]
    with pytest.raises(CompactionFailed) as raised:
        await compactor.before_model_call(request(assignment, conversation(3)))
    assert raised.value.reason == reason
    _clean(raised.value)
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", reason)


async def test_a_strategys_own_chained_failure_reaches_the_caller_unchained() -> None:
    class Chained:
        name = "chained"
        threshold = 0.01
        keep_turns = 1

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            try:
                raise KeyError("SUMMARY example.com secret")
            except KeyError as error:
                raise CompactionFailed("summary_failed", error_class="KeyError") from error

    assignment = await _chat_assignment()
    with pytest.raises(CompactionFailed) as raised:
        await Compactor([Chained()]).before_model_call(request(assignment, conversation(3)))
    assert (raised.value.reason, raised.value.error_class) == ("summary_failed", "KeyError")
    _clean(raised.value)


async def test_the_event_leaves_the_compactor_already_within_the_policy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The fan-out sink strips disallowed content and warns when it has to; the compactor
    # must never give it the summary text to strip.
    assignment = await _chat_assignment()
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1, notice="SECRET-NOTICE")], events=sink
    )
    with caplog.at_level("WARNING", logger="conduit.events"):
        await compactor.before_model_call(request(assignment, conversation(3)))
    assert [r for r in caplog.records if r.name == "conduit.events"] == []
    [event] = sink.of(CompactionEvent)
    assert event.summary is not None and event.summary.text is None
