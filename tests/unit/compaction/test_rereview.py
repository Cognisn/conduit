"""The re-review's findings: compaction past the window, the guard boundaries, forced
compactions under concurrency, and the tools a summary request carries."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from conduit.compaction import (
    CompactionEvent,
    CompactionFailed,
    Compactor,
    KeepLastTurns,
    StrategyContext,
    Summarise,
)
from conduit.core.tasks import ResolvedAssignment
from conduit.core.transcript import MessageEntry
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ProviderStateBlock,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from conduit.providers.stub import StubToolCall, StubTurn
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.compaction.helpers import (
    SMALL_WINDOW,
    ScriptLog,
    Sink,
    conversation,
    filler,
    request,
    runtime,
    summary_script,
)

LOOKUP = ToolSpec(name="lookup", description="Look up an item.", input_schema={"type": "object"})


async def _chat() -> ResolvedAssignment:
    return (await runtime()).assignments["chat"]


class Sizes:
    """An estimator with fixed answers: ``shown`` for the conversation as shown, ``tail``
    for the kept tail alone, and ``after`` for anything else (the compacted result)."""

    def __init__(self, shown: Sequence[Message], *, total: int, tail: int, after: int) -> None:
        self.shown = tuple(shown)
        self.total, self.tail, self.after = total, tail, after

    def __call__(
        self,
        *,
        system: Sequence[SystemBlock] = (),
        tools: Sequence[ToolSpec] = (),
        messages: Sequence[Message] = (),
    ) -> int:
        given = tuple(messages)
        if given == self.shown:
            return self.total
        if given == self.shown[-2:]:
            return self.tail
        return self.after


# N-B1: past the window, a compaction is applied whenever its result fits --------------


async def test_past_the_window_a_result_over_the_threshold_is_applied() -> None:
    assignment = await _chat()
    sink = Sink()
    compactor = Compactor([KeepLastTurns(threshold=0.5, keep_turns=1, notice="N")], events=sink)
    system = (TextBlock(text=filler(8_200, "sys")),)
    shown = request(assignment, conversation(8, result_chars=1_200), system=system)
    replaced = await compactor.before_model_call(shown)
    assert replaced is not None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.trigger) == ("compacted", "threshold")
    budget = assignment.max_output_tokens
    assert event.tokens_before + budget > SMALL_WINDOW
    assert event.tokens_after is not None
    assert event.tokens_after / SMALL_WINDOW >= 0.5  # still over the threshold, but fits
    assert event.tokens_after + budget <= SMALL_WINDOW


async def test_past_the_window_with_no_room_for_the_tail_it_fails_before_any_summary() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    sink = Sink()
    strategy = Summarise(rt, task_class="summary", threshold=0.5, keep_turns=1)
    system = (TextBlock(text=filler(15_200, "sys")),)  # about 3,800 tokens: no room left
    shown = request(rt.assignments["chat"], conversation(4), system=system)
    with pytest.raises(CompactionFailed) as raised:
        await Compactor([strategy], events=sink).before_model_call(shown)
    assert raised.value.reason == "result_too_large"
    assert log.requests == []
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "result_too_large")


async def test_past_the_window_a_result_that_still_does_not_fit_fails() -> None:
    assignment = await _chat()
    store = MemoryTranscriptStore()
    messages = conversation(14, result_chars=1_200)
    await store.append("s-1", [MessageEntry(message=m) for m in messages])
    sink = Sink()
    wordy = KeepLastTurns(threshold=0.5, keep_turns=1, notice=filler(16_000, "note"))
    compactor = Compactor([wordy], transcripts=store, events=sink)
    with pytest.raises(CompactionFailed) as raised:
        await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert raised.value.reason == "result_too_large"
    assert (await store.load("s-1")).latest_compaction() is None


@pytest.mark.parametrize(("extra", "fails"), [(0, False), (1, True)])
async def test_past_the_window_the_tail_must_leave_room_for_the_output(
    extra: int, fails: bool
) -> None:
    assignment = await _chat()
    budget = assignment.max_output_tokens
    messages = conversation(4)
    sizes = Sizes(messages, total=SMALL_WINDOW, tail=SMALL_WINDOW - budget + extra, after=10)
    compactor = Compactor([KeepLastTurns(threshold=0.5, keep_turns=1)], estimate=sizes)
    call = compactor.before_model_call(request(assignment, messages))
    if fails:
        with pytest.raises(CompactionFailed):
            await call
    else:
        assert await call is not None


@pytest.mark.parametrize(("extra", "fails"), [(0, False), (1, True)])
async def test_past_the_window_the_result_must_leave_room_for_the_output(
    extra: int, fails: bool
) -> None:
    assignment = await _chat()
    budget = assignment.max_output_tokens
    messages = conversation(4)
    sizes = Sizes(messages, total=SMALL_WINDOW, tail=10, after=SMALL_WINDOW - budget + extra)
    compactor = Compactor([KeepLastTurns(threshold=0.5, keep_turns=1)], estimate=sizes)
    call = compactor.before_model_call(request(assignment, messages))
    if fails:
        with pytest.raises(CompactionFailed):
            await call
    else:
        assert await call is not None


# M7: the guard boundaries -----------------------------------------------------------


@pytest.mark.parametrize(("tail", "skipped"), [(2_000, True), (1_999, False)])
async def test_the_prefix_check_counts_reaching_the_threshold(tail: int, skipped: bool) -> None:
    assignment = await _chat()
    messages = conversation(4)
    sizes = Sizes(messages, total=3_000, tail=tail, after=100)  # fits; threshold 2,000
    sink = Sink()
    compactor = Compactor([KeepLastTurns(threshold=0.5, keep_turns=1)], estimate=sizes, events=sink)
    result = await compactor.before_model_call(request(assignment, messages))
    assert (result is None) is skipped
    expected = ("skipped", "prefix_too_large") if skipped else ("compacted", None)
    assert [(e.outcome, e.reason) for e in sink.of(CompactionEvent)] == [expected]


async def test_the_summary_size_check_counts_the_tool_definitions() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    replaced = conversation(1, result_chars=200)  # a question, a tool call, its result
    wide = ToolSpec(name="lookup", description=filler(2_400, "doc"), input_schema={})
    strategy = Summarise(rt, task_class="tiny", instructions="S.", max_summary_tokens=16)
    narrow = StrategyContext(run_id="r", assignment=rt.assignments["chat"], tools=(LOOKUP,))
    assert await strategy.summarise(replaced, narrow)
    broad = StrategyContext(run_id="r", assignment=rt.assignments["chat"], tools=(wide,))
    with pytest.raises(CompactionFailed) as raised:
        await strategy.summarise(replaced, broad)
    assert raised.value.reason == "summary_too_large"
    assert len(log.requests) == 1


async def test_cache_markers_and_provider_state_stay_out_of_the_summary_request() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    state = ProviderStateBlock(provider_kind="stub", state_type="thinking", payload=b"opaque")
    replaced = (
        Message(role="user", content=(TextBlock(text="earlier summary"), CacheBreakpoint())),
        Message(role="assistant", content=(state, TextBlock(text="noted"))),
        Message.user("go on"),
    )
    strategy = Summarise(rt, task_class="summary")
    await strategy.summarise(
        replaced, StrategyContext(run_id="r", assignment=rt.assignments["chat"])
    )
    [sent] = log.requests
    blocks = [b for m in sent.messages for b in m.content]
    assert not any(isinstance(b, CacheBreakpoint | ProviderStateBlock) for b in blocks)
    assert [m.role for m in sent.messages] == ["user", "assistant", "user"]


# M8: one forced compaction runs once ------------------------------------------------


class Slow:
    name = "slow"
    threshold = 1.0
    keep_turns = 1

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.release = asyncio.Event()

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        self.calls += 1
        await self.release.wait()
        if self.fail:
            raise CompactionFailed("summary_failed")
        return "SUMMARY"


async def test_one_force_compacts_once_across_concurrent_calls() -> None:
    assignment = await _chat()
    slow = Slow()
    compactor = Compactor([slow])
    compactor.force()
    calls = [
        asyncio.create_task(compactor.before_model_call(request(assignment, conversation(3))))
        for _ in range(3)
    ]
    await asyncio.sleep(0.05)
    slow.release.set()
    results = await asyncio.gather(*calls)
    assert sum(r is not None for r in results) == 1
    assert slow.calls == 1


async def test_a_force_made_during_a_forced_compaction_survives_its_success() -> None:
    assignment = await _chat()
    slow = Slow()
    compactor = Compactor([slow])
    compactor.force("s-1")
    first = asyncio.create_task(
        compactor.before_model_call(request(assignment, conversation(3), session_id="s-1"))
    )
    await asyncio.sleep(0.05)
    compactor.force("s-1")  # asked again while the first is still running
    slow.release.set()
    assert await first is not None
    again = await compactor.before_model_call(
        request(assignment, conversation(3), session_id="s-1")
    )
    assert again is not None
    assert slow.calls == 2


async def test_a_failed_forced_compaction_does_not_replace_a_newer_force() -> None:
    assignment = await _chat()
    failing = Slow(fail=True)
    other = KeepLastTurns(threshold=0.9, keep_turns=1, name="other")
    sink = Sink()
    compactor = Compactor([failing, other], events=sink)
    compactor.force(strategy="slow")
    first = asyncio.create_task(compactor.before_model_call(request(assignment, conversation(3))))
    await asyncio.sleep(0.05)
    compactor.force(strategy="other")
    failing.release.set()
    assert await first is None
    assert await compactor.before_model_call(request(assignment, conversation(3)))
    assert [(e.strategy, e.outcome) for e in sink.of(CompactionEvent)] == [
        ("slow", "failed"),
        ("other", "compacted"),
    ]


async def test_a_cancelled_forced_compaction_is_kept_for_the_next_call() -> None:
    assignment = await _chat()
    slow = Slow()
    compactor = Compactor([slow])
    compactor.force()
    task = asyncio.create_task(compactor.before_model_call(request(assignment, conversation(3))))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    slow.release.set()
    assert await compactor.before_model_call(request(assignment, conversation(3)))


# M9: the tools a summary request carries ---------------------------------------------


async def test_the_summary_request_carries_only_the_tools_the_turns_used() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    other = ToolSpec(name="other", description="Unused here.", input_schema={"type": "object"})
    legacy_use = ToolUseBlock(id="toolu_legacy", name="legacy", input={"q": "x"})
    legacy_result = ToolResultBlock(tool_use_id="toolu_legacy", content=(TextBlock(text="r"),))
    replaced = (
        *conversation(1)[:3],
        Message(role="assistant", content=(legacy_use,)),
        Message(role="user", content=(legacy_result,)),
    )
    context = StrategyContext(run_id="r", assignment=rt.assignments["chat"], tools=(other, LOOKUP))
    await Summarise(rt, task_class="summary").summarise(replaced, context)
    [sent] = log.requests
    assert [t.name for t in sent.tools] == ["lookup", "legacy"]
    assert sent.tools[0] == LOOKUP
    assert sent.tools[1].input_schema == {"type": "object"}


async def test_a_summary_model_answering_with_a_tool_call_gives_summary_empty() -> None:
    answer = ScriptLog(lambda r, i: StubTurn(tool_calls=(StubToolCall(name="lookup"),)))
    rt = await runtime(summariser=answer)
    context = StrategyContext(run_id="r", assignment=rt.assignments["chat"], tools=(LOOKUP,))
    with pytest.raises(CompactionFailed) as raised:
        await Summarise(rt, task_class="summary").summarise(conversation(2)[:3], context)
    assert raised.value.reason == "summary_empty"


async def test_a_forced_compaction_ignores_the_prefix_check_while_the_prompt_fits() -> None:
    assignment = await _chat()
    sink = Sink()
    compactor = Compactor([KeepLastTurns(threshold=0.5, keep_turns=1, notice="N")], events=sink)
    system = (TextBlock(text=filler(8_200, "sys")),)  # over the threshold on its own
    shown = request(assignment, conversation(3), system=system)
    compactor.force()
    assert await compactor.before_model_call(shown) is not None
    [event] = sink.of(CompactionEvent)
    assert event.tokens_before + assignment.max_output_tokens <= SMALL_WINDOW  # it fits
    assert (event.outcome, event.trigger) == ("compacted", "forced")


# M11: past the window, the allowance does not refuse a summary that would fit ----------


async def test_past_the_window_a_short_summary_is_not_refused_for_its_allowance() -> None:
    class Brief:
        name = "brief"
        threshold = 0.5
        keep_turns = 1
        summary_allowance = 512  # what the summary might take, not what it does

        def __init__(self) -> None:
            self.calls = 0

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            self.calls += 1
            return "SUMMARY"

    assignment = await _chat()
    budget = assignment.max_output_tokens
    messages = conversation(4)
    tail = SMALL_WINDOW - budget - 100  # room for a short summary, not for 512 tokens
    sizes = Sizes(messages, total=SMALL_WINDOW, tail=tail, after=tail + 20)
    brief = Brief()
    sink = Sink()
    compactor = Compactor([brief], estimate=sizes, events=sink)
    assert await compactor.before_model_call(request(assignment, messages)) is not None
    assert brief.calls == 1
    [event] = sink.of(CompactionEvent)
    assert event.outcome == "compacted"
    assert (event.tokens_after or 0) + budget <= SMALL_WINDOW
