"""Strategies: where the cut falls, what each strategy puts in the summary, and refusals."""

from __future__ import annotations

import pytest

from conduit.compaction import (
    CompactionFailed,
    DropToolResults,
    KeepLastTurns,
    StrategyContext,
    Summarise,
    estimate_tokens,
    find_cut,
    render_messages,
)
from conduit.core.errors import (
    CapabilityRefused,
    ConfigurationError,
    ProviderError,
    ProviderFailure,
)
from conduit.core.tasks import CapabilityFloor
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    TextBlock,
    ToolSpec,
)
from conduit.providers.stub import StubTurn, scripted
from tests.unit.compaction.helpers import (
    ScriptLog,
    chat,
    conversation,
    filler,
    runtime,
    summary_script,
)

# The cut ---------------------------------------------------------------------------


def test_the_kept_tail_starts_at_an_assistant_message() -> None:
    messages = conversation(4)  # U, (A, U) x 4
    cut = find_cut(messages, keep_turns=2)
    assert cut == 5
    assert messages[cut].role == "assistant"
    assert messages[cut - 1].role == "user"


def test_the_tail_holds_exactly_the_last_turns() -> None:
    messages = chat(5)  # (U, A) x 5, U
    for keep in (1, 2, 3, 4):
        cut = find_cut(messages, keep_turns=keep)
        assert cut is not None
        tail = messages[cut:]
        assert sum(1 for m in tail if m.role == "assistant") == keep


def test_nothing_is_cut_when_fewer_than_two_messages_would_go() -> None:
    assert find_cut((Message.user("hi"),), keep_turns=1) is None
    assert find_cut(chat(1), keep_turns=1) is None  # U, A, U: only the first U would go
    assert find_cut(chat(2), keep_turns=2) is None
    assert find_cut(chat(2), keep_turns=1) == 3


def test_keep_turns_must_be_positive() -> None:
    with pytest.raises(ConfigurationError):
        find_cut(chat(3), keep_turns=0)


# Token estimates ---------------------------------------------------------------------


def test_the_estimate_counts_system_tools_and_messages() -> None:
    base = estimate_tokens(messages=(Message.user("x" * 400),))
    assert 100 <= base <= 140
    with_system = estimate_tokens(
        system=(TextBlock(text="y" * 400), CacheBreakpoint()), messages=(Message.user("x" * 400),)
    )
    assert with_system >= base + 100
    tool = ToolSpec(name="lookup", description="z" * 400, input_schema={"type": "object"})
    assert estimate_tokens(tools=(tool,), messages=(Message.user("x" * 400),)) >= base + 100


def test_the_estimate_grows_with_tool_results() -> None:
    assert estimate_tokens(messages=conversation(4)) > estimate_tokens(messages=conversation(2))


# Rendering ---------------------------------------------------------------------------


def test_rendering_omits_tool_results_when_asked() -> None:
    text = render_messages(conversation(2), tool_result_chars=None)
    assert "RESULT-0" not in text and "RESULT-1" not in text
    assert "[tool result omitted]" in text
    assert "[tool call: lookup]" in text
    assert "Please check the example.com release items." in text


def test_rendering_trims_tool_results_and_long_messages() -> None:
    text = render_messages(conversation(1, result_chars=5_000), tool_result_chars=50)
    assert "RESULT-0" in text
    assert len(text) < 1_000
    long = render_messages((Message.user(filler(9_000)),), max_message_chars=1_000)
    assert len(long) < 1_200
    assert long.endswith("[truncated]")


def test_rendering_names_images_and_skips_markers() -> None:
    image = ImageBlock(media_type="image/png", data=b"\x89PNG synthetic")
    message = Message(role="user", content=(TextBlock(text="see"), image, CacheBreakpoint()))
    text = render_messages((message,))
    assert "[image]" in text
    assert "PNG" not in text


# Strategy settings ---------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, float("nan")])
def test_a_threshold_outside_zero_to_one_is_refused(threshold: float) -> None:
    with pytest.raises(ConfigurationError):
        KeepLastTurns(threshold=threshold)
    with pytest.raises(ConfigurationError):
        DropToolResults(threshold=threshold)


def test_a_threshold_of_one_is_allowed() -> None:
    assert KeepLastTurns(threshold=1.0).threshold == 1.0


def test_keep_turns_below_one_is_refused() -> None:
    with pytest.raises(ConfigurationError):
        KeepLastTurns(keep_turns=0)


def test_a_strategy_name_must_be_an_identifier() -> None:
    with pytest.raises(ConfigurationError):
        KeepLastTurns(name="keep last turns")


# What each strategy writes -------------------------------------------------------------


async def test_keep_last_turns_writes_its_notice() -> None:
    rt = await runtime()
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    strategy = KeepLastTurns(notice="Earlier turns were removed.")
    assert await strategy.summarise(conversation(3)[:3], context) == "Earlier turns were removed."
    assert strategy.name == "keep_last_turns"


async def test_drop_tool_results_keeps_the_words_and_drops_the_results() -> None:
    rt = await runtime()
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    strategy = DropToolResults()
    text = await strategy.summarise(conversation(3)[:5], context)
    assert strategy.name == "drop_tool_results"
    assert "Please check the example.com release items." in text
    assert "Looking up item 1." in text
    assert "RESULT-" not in text


async def test_summarise_calls_the_summary_task_class() -> None:
    log = summary_script("SUMMARY: three items were checked.")
    rt = await runtime(summariser=log)
    strategy = Summarise(rt, task_class="summary", instructions="Summarise this, please.")
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    text = await strategy.summarise(conversation(3)[:5], context)
    assert text.endswith("SUMMARY: three items were checked.")
    assert strategy.name == "summarise"
    [sent] = log.requests
    assert sent.model == "stub-summary"
    assert sent.task_class == "summary"
    assert sent.run_id == "run_x"
    assert sent.tools == ()
    prompt = sent.messages[0].content[0]
    assert isinstance(prompt, TextBlock)
    assert prompt.text.startswith("Summarise this, please.")
    assert "Looking up item 1." in prompt.text
    assert sent.max_output_tokens <= 512


# The summary task class's floor --------------------------------------------------------


async def test_the_summary_task_class_must_be_assigned() -> None:
    rt = await runtime()
    with pytest.raises(ConfigurationError, match="no assignment"):
        Summarise(rt, task_class="absent")


async def test_the_summary_model_must_meet_the_floor_at_construction() -> None:
    rt = await runtime()
    floor = CapabilityFloor(min_context_window=8_000)
    assert Summarise(rt, task_class="summary", floor=floor).task_class == "summary"
    with pytest.raises(CapabilityRefused, match="tiny"):
        Summarise(rt, task_class="tiny", floor=floor)


async def test_the_floor_is_checked_once_and_never_part_way() -> None:
    rt = await runtime()
    floor = CapabilityFloor(min_output_limit=1_024)
    with pytest.raises(CapabilityRefused):
        Summarise(rt, task_class="tiny", floor=floor)
    with pytest.raises(CapabilityRefused):
        Summarise(rt, task_class="tiny", floor=CapabilityFloor(kind="embedding"))


async def test_a_conversation_too_large_for_the_summary_model_is_refused() -> None:
    log = summary_script()
    rt = await runtime(summariser=log)
    strategy = Summarise(rt, task_class="tiny")
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    with pytest.raises(CompactionFailed) as raised:
        await strategy.summarise(chat(6, chars=400)[:-1], context)
    assert raised.value.reason == "summary_too_large"
    assert log.requests == []


async def test_a_conversation_that_just_fits_is_summarised() -> None:
    log = summary_script("ok summary")
    rt = await runtime(summariser=log)
    strategy = Summarise(rt, task_class="tiny", instructions="S.", max_summary_tokens=16)
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    # 600-token window, 16 output tokens: a prompt of about 2,000 characters fits.
    replaced = (Message.user(filler(1_800)), Message.assistant("ok"))
    assert (await strategy.summarise(replaced, context)).endswith("ok summary")
    assert log.requests[0].max_output_tokens == 16


async def test_a_failed_summary_call_keeps_no_provider_error() -> None:
    script = ScriptLog(scripted([ProviderError(ProviderFailure.UNAVAILABLE)]))
    rt = await runtime(summariser=script)
    strategy = Summarise(rt, task_class="summary")
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    with pytest.raises(CompactionFailed) as raised:
        await strategy.summarise(conversation(2)[:3], context)
    error = raised.value
    assert error.reason == "summary_failed"
    assert error.failure is ProviderFailure.UNAVAILABLE
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "example.com" not in str(error) and "example.com" not in repr(error)
    assert all("example.com" not in str(arg) for arg in error.args)


async def test_an_empty_summary_is_refused() -> None:
    rt = await runtime(summariser=ScriptLog(lambda r, i: StubTurn(text="   ")))
    strategy = Summarise(rt, task_class="summary")
    context = StrategyContext(run_id="run_x", assignment=rt.assignments["chat"])
    with pytest.raises(CompactionFailed) as raised:
        await strategy.summarise(conversation(2)[:3], context)
    assert raised.value.reason == "summary_empty"


def test_a_strategy_name_fits_a_compaction_point() -> None:
    assert KeepLastTurns(name="n" * 128).name == "n" * 128
    with pytest.raises(ConfigurationError):
        KeepLastTurns(name="n" * 129)
