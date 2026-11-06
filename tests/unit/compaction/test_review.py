"""The independent review's findings on the compaction layer, each pinned by a test.

B1: no compaction is applied unless it brings the prompt under its threshold, and none is
attempted when the fixed prefix and the kept tail already reach it. B2: the summary call
sends tool content in tool blocks, so the registry's capture policy strips it. B3: a
product's own ``CompactionFailed`` cannot carry free text. M1 to M6: the summary size
boundary, product strategies validated, failures that leave a fitting conversation as it
was, forced compactions kept until they succeed, and a bounded queue of them.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pytest

from conduit.agent import AgentRunner, AgentTask
from conduit.compaction import (
    CompactionEvent,
    CompactionFailed,
    Compactor,
    KeepLastTurns,
    StrategyContext,
    Summarise,
    estimate_tokens,
)
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.events import CapturePolicy, ModelCallStartedEvent
from conduit.core.tasks import ResolvedAssignment
from conduit.core.transcript import MessageEntry
from conduit.core.types import (
    Message,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from conduit.providers.stub import scripted
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.compaction.helpers import (
    SMALL_WINDOW,
    ScriptLog,
    Sink,
    conversation,
    filler,
    oversized,
    request,
    runtime,
    summary_script,
)
from tests.unit.compaction.test_runner import _script, _skills, _tools

LOOKUP = ToolSpec(name="lookup", description="Look up an item.", input_schema={"type": "object"})


async def _chat() -> ResolvedAssignment:
    return (await runtime()).assignments["chat"]


def _events(sink: Sink, outcome: str) -> list[CompactionEvent]:
    return [e for e in sink.of(CompactionEvent) if e.outcome == outcome]


# B1: compaction must bring the prompt under its threshold -------------------------------


async def test_no_compaction_is_tried_when_prefix_and_tail_reach_the_threshold() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    sink = Sink()
    strategy = Summarise(rt, task_class="summary", threshold=0.5, keep_turns=1)
    compactor = Compactor([strategy], events=sink)
    system = (TextBlock(text=filler(8_200, "sys")),)  # about 2,050 tokens: over 0.5 alone
    shown = request(rt.assignments["chat"], conversation(4), system=system)
    assert await compactor.before_model_call(shown) is None
    assert log.requests == []
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason, event.trigger) == (
        "skipped",
        "prefix_too_large",
        "threshold",
    )


async def test_the_summary_allowance_counts_in_the_check_before_any_call() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    messages = conversation(8, result_chars=1_200)  # about 2,920 tokens; tail about 360
    sink = Sink()
    # 0.5 of 4,000 is 2,000: the tail and 512 summary tokens fit, the tail and 1,700 do not.
    roomy = Summarise(rt, task_class="summary", threshold=0.5, keep_turns=1)
    assert strategy_allowance(roomy) == 512
    assert await Compactor([roomy], events=sink).before_model_call(
        request(rt.assignments["chat"], messages)
    )
    assert len(log.requests) == 1

    class Wordy:
        name = "wordy"
        threshold = 0.5
        keep_turns = 1
        summary_allowance = 1_700

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            raise AssertionError("never asked: the allowance rules it out first")

    sink = Sink()
    assert (
        await Compactor([Wordy()], events=sink).before_model_call(
            request(rt.assignments["chat"], messages)
        )
        is None
    )
    assert [(e.outcome, e.reason) for e in sink.of(CompactionEvent)] == [
        ("skipped", "prefix_too_large")
    ]


def strategy_allowance(strategy: Summarise) -> int:
    return strategy.summary_allowance


async def test_a_result_still_over_the_threshold_is_not_applied() -> None:
    assignment = await _chat()
    store = MemoryTranscriptStore()
    messages = conversation(8, result_chars=1_200)
    await store.append("s-1", [MessageEntry(message=m) for m in messages])
    sink = Sink()
    long_notice = KeepLastTurns(threshold=0.5, keep_turns=1, notice=filler(8_000, "note"))
    compactor = Compactor([long_notice], transcripts=store, events=sink)
    shown = request(assignment, messages, session_id="s-1")
    assert await compactor.before_model_call(shown) is None
    assert (await store.load("s-1")).latest_compaction() is None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("skipped", "not_below_threshold")
    assert event.tokens_after is not None and event.tokens_after / SMALL_WINDOW >= 0.5


async def test_a_result_just_under_the_threshold_is_applied() -> None:
    assignment = await _chat()
    messages = conversation(8, result_chars=1_200)
    probe = KeepLastTurns(threshold=0.5, keep_turns=1, notice="x")
    tail = await Compactor([probe]).before_model_call(request(assignment, messages))
    assert tail is not None
    after = estimate_tokens(messages=tail)
    # A threshold just above the result's share applies it; exactly at the share does not.
    above = KeepLastTurns(threshold=(after + 1) / SMALL_WINDOW, keep_turns=1, notice="x")
    assert await Compactor([above]).before_model_call(request(assignment, messages))
    at = KeepLastTurns(threshold=after / SMALL_WINDOW, keep_turns=1, notice="x")
    sink = Sink()
    assert (
        await Compactor([at], events=sink).before_model_call(request(assignment, messages)) is None
    )
    assert [e.reason for e in sink.of(CompactionEvent)] == ["not_below_threshold"]


async def test_a_forced_compaction_is_applied_even_above_the_threshold() -> None:
    assignment = await _chat()
    sink = Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.5, keep_turns=1, notice=filler(8_000, "note"))], events=sink
    )
    compactor.force()
    system = (TextBlock(text=filler(8_200, "sys")),)
    shown = request(assignment, conversation(8, result_chars=1_200), system=system)
    assert await compactor.before_model_call(shown) is not None
    assert [(e.outcome, e.trigger) for e in sink.of(CompactionEvent)] == [("compacted", "forced")]


@pytest.mark.parametrize("use_summary", [False, True])
async def test_a_run_whose_prefix_is_over_the_threshold_compacts_only_past_the_window(
    use_summary: bool,
) -> None:
    # The prefix alone is over the threshold, so while the prompt fits the window no
    # compaction is tried; once it no longer fits, compacting is needed and is applied
    # whenever its result fits. Every call the model receives therefore fits.
    log = _script(12)
    summariser = summary_script()
    rt = await runtime(log, summariser)
    budget = rt.assignments["chat"].max_output_tokens
    store, sink = MemoryTranscriptStore(), Sink()
    strategy = (
        Summarise(rt, task_class="summary", threshold=0.5, keep_turns=1)
        if use_summary
        else KeepLastTurns(threshold=0.5, keep_turns=1, notice="NOTICE")
    )
    compactor = Compactor([strategy], transcripts=store, events=sink)
    runner = AgentRunner(rt, tools=_tools(), transcripts=store, compaction=compactor)
    session = await _skills()
    task = AgentTask(
        task_class="chat",
        messages=(Message.user("Check."),),
        system=(TextBlock(text="Base. " + filler(8_200, "sys")),),
        system_hook=session.system,
        cache_tools=True,
        session_id="s-1",
    )
    result = await runner.run(task)
    assert result.text == "All items are checked."
    for sent in log.requests:
        size = estimate_tokens(system=sent.system, tools=sent.tools, messages=sent.messages)
        assert size + budget <= SMALL_WINDOW
    compacted = _events(sink, "compacted")
    skipped = _events(sink, "skipped")
    assert compacted and skipped
    assert all(e.tokens_before + budget > SMALL_WINDOW for e in compacted)
    assert all((e.tokens_after or 0) + budget <= SMALL_WINDOW for e in compacted)
    assert all(e.tokens_before + budget <= SMALL_WINDOW for e in skipped)
    assert {e.reason for e in skipped} == {"prefix_too_large"}
    assert len(summariser.requests) == (len(compacted) if use_summary else 0)
    assert _events(sink, "failed") == []


async def test_in_a_run_each_compaction_lands_below_its_threshold_and_holds() -> None:
    log = _script(16)
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=0.5, keep_turns=1, notice="NOTICE")],
        transcripts=store,
        events=sink,
    )
    runner = AgentRunner(rt, tools=_tools(), transcripts=store, compaction=compactor)
    await runner.run(
        AgentTask(
            task_class="chat",
            messages=(Message.user("Check."),),
            system=(TextBlock(text="Base."),),
            system_hook=(await _skills()).system,
            cache_tools=True,
            session_id="s-1",
        )
    )
    compacted = _events(sink, "compacted")
    assert 1 <= len(compacted) < len(log.requests) // 2
    assert all((e.tokens_after or 0) / SMALL_WINDOW < 0.5 for e in compacted)
    first_of = [i for i, r in enumerate(log.requests) if _starts_with_notice(r.messages)]
    # A compacted call is never followed by another compaction on the very next call.
    fresh = [i for i in first_of if len(log.requests[i].messages) <= 3]
    assert all(b - a > 1 for a, b in itertools.pairwise(fresh))


def _starts_with_notice(messages: Sequence[Message]) -> bool:
    first = messages[0].content[0]
    return isinstance(first, TextBlock) and first.text.startswith("NOTICE")


# B2: tool content in the summary call stays under the capture policy -------------------


@pytest.mark.parametrize("tool_content", [False, True])
async def test_the_summary_call_carries_tool_results_only_under_the_tool_policy(
    tool_content: bool,
) -> None:
    events = Sink()
    policy = CapturePolicy(prompts=True, tool_content=tool_content)
    rt = await runtime(summariser=summary_script(), events=events, capture=policy)
    strategy = Summarise(
        rt, task_class="summary", threshold=0.1, keep_turns=1, max_summary_tokens=64
    )
    compactor = Compactor([strategy])
    shown = request(rt.assignments["chat"], conversation(4), tools=(LOOKUP,))
    assert await compactor.before_model_call(shown)
    [started] = [e for e in events.of(ModelCallStartedEvent) if e.task_class == "summary"]
    text = started.request.text
    assert text is not None
    assert "Please check the example.com release items." in text
    assert ("RESULT-0" in text) is tool_content
    assert ('"q":"item 0"' in text) is tool_content  # a tool call's arguments


async def test_the_summary_request_keeps_tool_blocks_with_trimmed_results() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    strategy = Summarise(
        rt, task_class="summary", instructions="SUMMARISE NOW.", tool_result_chars=40
    )
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"], tools=(LOOKUP,))
    replaced = conversation(3, result_chars=2_000)[:5]
    await strategy.summarise(replaced, context)
    [sent] = log.requests
    assert sent.tools == (LOOKUP,)
    assert [m.role for m in sent.messages] == [m.role for m in replaced]
    uses = [b for m in sent.messages for b in m.content if isinstance(b, ToolUseBlock)]
    results = [b for m in sent.messages for b in m.content if isinstance(b, ToolResultBlock)]
    assert len(uses) == len(results) == 2
    for result in results:
        [block] = result.content
        assert isinstance(block, TextBlock) and len(block.text) <= 40
    last = sent.messages[-1].content[-1]
    assert isinstance(last, TextBlock) and last.text == "SUMMARISE NOW."


async def test_without_tool_blocks_the_summary_request_sends_no_tools() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    strategy = Summarise(rt, task_class="summary")
    context = StrategyContext(run_id="r", assignment=rt.assignments["chat"], tools=(LOOKUP,))
    await strategy.summarise((Message.user("hi"), Message.assistant("hello")), context)
    [sent] = log.requests
    assert sent.tools == ()
    assert sent.messages[-1].role == "user"


async def test_the_summary_task_class_needs_tools_by_default() -> None:
    from conduit.core.errors import CapabilityRefused
    from conduit.core.tasks import CapabilityFloor

    rt = await runtime()
    # Every test model offers tools; a floor asking for more shows the default is checked.
    assert Summarise(rt, task_class="summary").floor == CapabilityFloor(tools=True)
    with pytest.raises(CapabilityRefused):
        Summarise(rt, task_class="summary", floor=CapabilityFloor(tools=True, vision=True))


# B3: a product's own failure cannot carry text ------------------------------------------


def test_compaction_failed_keeps_only_documented_reasons_and_class_names() -> None:
    error = CompactionFailed("SECRET text from example.com", error_class="a class name with spaces")
    assert error.reason == "strategy_failed"
    assert error.error_class is None
    assert "SECRET" not in str(error) and "example.com" not in repr(error)
    kept = CompactionFailed("summary_failed", error_class="botocore.exceptions.ClientError")
    assert (kept.reason, kept.error_class) == ("summary_failed", "botocore.exceptions.ClientError")
    assert CompactionFailed("store_failed", error_class="x" * 129).error_class is None


async def test_a_products_free_text_failure_becomes_a_fixed_reason() -> None:
    class Leaky:
        name = "leaky"
        threshold = 0.1
        keep_turns = 1

        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            failure = CompactionFailed("summary_failed")
            failure.reason = "SECRET text from example.com"
            failure.error_class = "SECRET class text"
            raise failure

    assignment = await _chat()
    sink = Sink()
    with pytest.raises(CompactionFailed) as raised:
        await Compactor([Leaky()], events=sink).before_model_call(request(assignment, oversized()))
    assert (raised.value.reason, raised.value.error_class) == ("strategy_failed", None)
    assert "SECRET" not in str(raised.value) and "SECRET" not in repr(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "strategy_failed")


# M1: the summary size boundary -----------------------------------------------------------


async def test_a_summary_request_exactly_filling_the_window_is_allowed() -> None:
    log = summary_script("ok summary")
    rt = await runtime(summariser=log)
    context = StrategyContext(run_id="r", assignment=rt.assignments["chat"])
    replaced = (Message.user(filler(2_150)), Message.assistant("ok"))
    probe = Summarise(rt, task_class="tiny", instructions="S.", max_summary_tokens=1)
    await probe.summarise(replaced, context)
    sent = log.requests[-1]
    size = estimate_tokens(tools=sent.tools, messages=sent.messages)
    assert 536 <= size <= 599  # so the exact output budget is within the assignment's 64
    exact = Summarise(rt, task_class="tiny", instructions="S.", max_summary_tokens=600 - size)
    assert (await exact.summarise(replaced, context)).endswith("ok summary")
    over = Summarise(rt, task_class="tiny", instructions="S.", max_summary_tokens=601 - size)
    with pytest.raises(CompactionFailed) as raised:
        await over.summarise(replaced, context)
    assert raised.value.reason == "summary_too_large"


# M2: a product strategy is checked like a supplied one ----------------------------------


@pytest.mark.parametrize(
    ("name", "threshold", "keep_turns"),
    [("ok", 0.0, 1), ("ok", 1.5, 1), ("bad name", 0.5, 1), ("ok", 0.5, 0)],
)
def test_a_product_strategy_is_validated(name: str, threshold: float, keep_turns: int) -> None:
    class Product:
        async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
            return "s"

    strategy = Product()
    strategy.name = name  # type: ignore[attr-defined]
    strategy.threshold = threshold  # type: ignore[attr-defined]
    strategy.keep_turns = keep_turns  # type: ignore[attr-defined]
    with pytest.raises(ConfigurationError):
        Compactor([strategy])  # type: ignore[list-item]


# M3: a failure leaves a conversation that still fits as it was ---------------------------


class Flaky:
    name = "flaky"
    threshold = 0.1
    keep_turns = 1

    def __init__(self, failures: int) -> None:
        self.failures = failures

    async def summarise(self, replaced: Sequence[Message], context: StrategyContext) -> str:
        if self.failures:
            self.failures -= 1
            raise CompactionFailed("summary_failed", failure=ProviderFailure.UNAVAILABLE)
        return "SUMMARY"


async def test_a_failure_while_the_prompt_fits_leaves_the_conversation_unchanged() -> None:
    assignment = await _chat()
    sink = Sink()
    compactor = Compactor([Flaky(failures=1)], events=sink)
    messages = conversation(4)
    assert await compactor.before_model_call(request(assignment, messages)) is None
    [event] = sink.of(CompactionEvent)
    assert (event.outcome, event.reason) == ("failed", "summary_failed")
    # The next call tries again, and succeeds.
    assert await compactor.before_model_call(request(assignment, messages))


async def test_a_failed_summary_call_in_a_run_does_not_end_the_run() -> None:
    log = _script(6)
    summariser = ScriptLog(scripted([ProviderError(ProviderFailure.UNAVAILABLE)] * 20))
    rt = await runtime(log, summariser)
    sink = Sink()
    strategy = Summarise(
        rt, task_class="summary", threshold=0.3, keep_turns=1, max_summary_tokens=64
    )
    runner = AgentRunner(rt, tools=_tools(), compaction=Compactor([strategy], events=sink))
    result = await runner.run(
        AgentTask(task_class="chat", messages=(Message.user("Check."),), cache_tools=True)
    )
    assert result.text == "All items are checked."
    assert _events(sink, "failed")
    assert {e.reason for e in _events(sink, "failed")} == {"summary_failed"}


async def test_a_failure_when_the_prompt_no_longer_fits_raises() -> None:
    assignment = await _chat()
    sink = Sink()
    with pytest.raises(CompactionFailed):
        await Compactor([Flaky(failures=1)], events=sink).before_model_call(
            request(assignment, oversized())
        )
    assert [e.outcome for e in sink.of(CompactionEvent)] == ["failed"]


class _Exact:
    """An estimator giving ``tokens`` for the conversation shown and 10 for anything else."""

    def __init__(self, shown: Sequence[Message], tokens: int) -> None:
        self.shown = tuple(shown)
        self.tokens = tokens

    def __call__(
        self,
        *,
        system: Sequence[SystemBlock] = (),
        tools: Sequence[ToolSpec] = (),
        messages: Sequence[Message] = (),
    ) -> int:
        return self.tokens if tuple(messages) == self.shown else 10


@pytest.mark.parametrize(("extra", "raises"), [(0, False), (1, True)])
async def test_fitting_counts_the_output_budget(extra: int, raises: bool) -> None:
    assignment = await _chat()
    base = conversation(4)
    tokens = SMALL_WINDOW - assignment.max_output_tokens + extra
    compactor = Compactor([Flaky(failures=1)], estimate=_Exact(base, tokens))
    if raises:
        with pytest.raises(CompactionFailed):
            await compactor.before_model_call(request(assignment, base))
    else:
        assert await compactor.before_model_call(request(assignment, base)) is None


# M4: a forced compaction stays until it succeeds ---------------------------------------


async def test_a_failed_forced_compaction_is_tried_again() -> None:
    assignment = await _chat()
    sink = Sink()
    compactor = Compactor([Flaky(failures=1)], events=sink)
    small = conversation(2)  # under the threshold: only a forced compaction applies
    assert estimate_tokens(messages=small) < 0.1 * SMALL_WINDOW
    compactor.force("s-1")
    assert await compactor.before_model_call(request(assignment, small, session_id="s-1")) is None
    again = await compactor.before_model_call(request(assignment, small, session_id="s-1"))
    assert again is not None
    assert [(e.outcome, e.trigger) for e in sink.of(CompactionEvent)] == [
        ("failed", "forced"),
        ("compacted", "forced"),
    ]
    assert await compactor.before_model_call(request(assignment, small, session_id="s-1")) is None


async def test_a_forced_compaction_with_nothing_to_cut_is_consumed() -> None:
    assignment = await _chat()
    sink = Sink()
    compactor = Compactor([KeepLastTurns(keep_turns=1)], events=sink)
    compactor.force()
    one = (Message.user("hi"),)
    assert await compactor.before_model_call(request(assignment, one)) is None
    assert await compactor.before_model_call(request(assignment, one)) is None
    assert [e.reason for e in sink.of(CompactionEvent)] == ["nothing_to_compact"]


# M6: forced compactions waiting for sessions are bounded --------------------------------


async def test_forced_compactions_waiting_are_bounded_oldest_first() -> None:
    assignment = await _chat()
    compactor = Compactor([KeepLastTurns(threshold=1.0, keep_turns=1)], max_forced=2)
    for session in ("a", "b", "c"):
        compactor.force(session)
    compactor.force("b")  # forcing again does not add an entry
    messages = conversation(3)
    assert await compactor.before_model_call(request(assignment, messages, session_id="a")) is None
    assert await compactor.before_model_call(request(assignment, messages, session_id="b"))
    assert await compactor.before_model_call(request(assignment, messages, session_id="c"))


def test_the_forced_bound_must_be_positive() -> None:
    with pytest.raises(ConfigurationError):
        Compactor([KeepLastTurns()], max_forced=0)
