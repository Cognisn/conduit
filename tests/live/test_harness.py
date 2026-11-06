"""The live checks proven offline against the stub provider. Runs in the default suite.

Each check in ``tests/live/harness.py`` passes on a well-behaved stub and fails on one
scripted to break the property it checks, so a live pass means what it says. Nothing
here reaches a network.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import EmbeddingResponse
from conduit.core.runtime import TaskBinding
from conduit.core.streaming import StreamEvent, StreamStopped, TextDelta, ToolCallStarted
from conduit.core.types import ModelRequest, ProviderStateBlock, StopReason, Usage
from conduit.embeddings import EmbeddingBinding
from conduit.providers.stub import (
    StubFactory,
    StubToolCall,
    StubTurn,
    answer_tool_results,
    scripted,
)
from tests.live.harness import (
    EMBEDDING_TEXTS,
    LOOKUP_TOOL,
    ask,
    assert_usage,
    check_caching,
    check_chat,
    check_embeddings,
    check_output_limit,
    check_structured_output,
    check_tools,
    check_vision,
    started,
    started_embeddings,
    stream_shape,
)
from tests.live.report import REPORT, Verdict
from tests.live.targets import Target

STUB_TEXT = "stub-said-this-and-it-must-never-be-reported"


def _target(script: str = "echo") -> Target:
    return Target(
        "stub", {"kind": "stub", "not_for_production": True, "script": script}, "stub-echo"
    )


def _factory(**scripts: object) -> list[StubFactory]:
    return [StubFactory(scripts=scripts)]  # type: ignore[arg-type]


def _tool_turn(state: tuple[ProviderStateBlock, ...] = ()) -> StubTurn:
    return StubTurn(tool_calls=(StubToolCall(LOOKUP_TOOL.name, {"code": "R-1234"}),), state=state)


def _no_cache_usage() -> Usage:
    return Usage(input_tokens=100, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)


@pytest.fixture(autouse=True)
def _report_left_as_found() -> Iterator[None]:
    """These offline checks record to the run's report; take their lines back out."""
    kept = len(REPORT.lines)
    yield
    del REPORT.lines[kept:]


def _reported_text() -> str:
    return " ".join(f"{line.subject} {line.check} {line.detail}" for line in REPORT.lines)


async def test_chat_passes_on_a_well_behaved_stream() -> None:
    before = len(REPORT.lines)
    async with started(_target(), factories=_factory()) as binding:
        usage = await check_chat(binding, "stub chat")
    assert usage.output_tokens > 0
    assert REPORT.lines[before].verdict is Verdict.MATCH
    assert "ready" not in _reported_text()  # the stub echoes the prompt: never reported


async def test_chat_fails_when_the_turn_does_not_end() -> None:
    script = scripted([StubTurn(text=STUB_TEXT, stop_reason=StopReason.MAX_TOKENS)])
    async with started(_target("cut"), factories=_factory(cut=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_chat(binding, "stub chat")
    assert "max_tokens" in str(failed.value)
    assert STUB_TEXT not in str(failed.value)


async def test_chat_fails_without_text() -> None:
    # Output tokens but no text (a model that only reasoned): the text check must catch it.
    usage = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    script = scripted([StubTurn(text="", usage=usage, stop_reason=StopReason.END_TURN)])
    async with started(_target("mute"), factories=_factory(mute=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_chat(binding, "stub chat")
    assert "no text" in str(failed.value)


async def test_tool_loop_passes_and_counts_the_state_sent_back() -> None:
    state = (ProviderStateBlock(provider_kind="stub", state_type="thinking", payload=b"s"),)
    script = answer_tool_results(_tool_turn(state), lambda results: StubTurn(text=STUB_TEXT))
    async with started(_target("tools"), factories=_factory(tools=script)) as binding:
        blocks = await check_tools(binding, "stub tools")
    assert blocks == 1
    assert STUB_TEXT not in _reported_text()


async def test_tool_loop_fails_when_the_model_does_not_call_the_tool() -> None:
    async with started(_target(), factories=_factory()) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_tools(binding, "stub tools")
    assert "ToolCallStarted" in str(failed.value)


async def test_tool_loop_fails_when_no_answer_follows_the_result() -> None:
    script = answer_tool_results(_tool_turn(), lambda results: _tool_turn())
    async with started(_target("again"), factories=_factory(again=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_tools(binding, "stub tools")
    assert "after the result" in str(failed.value)


async def test_tool_loop_fails_on_another_tool() -> None:
    other = StubTurn(tool_calls=(StubToolCall("other_tool", {"code": "R-1234"}),))
    script = answer_tool_results(other, lambda results: StubTurn(text="done"))
    async with started(_target("other"), factories=_factory(other=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_tools(binding, "stub tools")
    assert "another tool" in str(failed.value)


async def test_caching_passes_on_a_write_then_a_read() -> None:
    before = len(REPORT.lines)
    async with started(_target(), factories=_factory()) as binding:
        first, second = await check_caching(binding, "stub cache", explicit=True, strict=True)
    assert first.cache_write_tokens > 0
    assert second.cache_read_tokens > 0
    assert REPORT.lines[before].verdict is Verdict.MATCH


async def test_caching_fails_an_explicit_cache_with_no_write() -> None:
    turn = StubTurn(text="ok", usage=_no_cache_usage())
    script = scripted([turn, turn])
    async with started(_target("flat"), factories=_factory(flat=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_caching(binding, "stub cache", explicit=True, strict=True)
    assert "cache write" in str(failed.value)


async def test_caching_fails_a_strict_implicit_cache_with_no_read() -> None:
    turn = StubTurn(text="ok", usage=_no_cache_usage())
    script = scripted([turn] * 4)
    before = len(REPORT.lines)
    async with started(_target("flat"), factories=_factory(flat=script)) as binding:
        with pytest.raises(AssertionError):
            await check_caching(binding, "stub cache", explicit=False, strict=True, attempts=1)
    assert REPORT.lines[before].verdict is Verdict.MISMATCH


async def test_caching_reports_an_unobserved_best_effort_cache() -> None:
    turn = StubTurn(text="ok", usage=_no_cache_usage())
    script = scripted([turn] * 4)
    before = len(REPORT.lines)
    async with started(_target("flat"), factories=_factory(flat=script)) as binding:
        await check_caching(binding, "stub cache", explicit=False, strict=False, attempts=1)
    assert REPORT.lines[before].verdict is Verdict.UNOBSERVED


async def test_usage_with_no_output_fails() -> None:
    usage = Usage(input_tokens=10, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    script = scripted([StubTurn(text="ok", usage=usage)])
    async with started(_target("empty"), factories=_factory(empty=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_chat(binding, "stub chat")
    assert "output tokens" in str(failed.value)


async def test_capability_checks_report_acceptance_and_shape() -> None:
    async with started(_target(), factories=_factory()) as binding:
        limit_ok, _ = await check_output_limit(binding, 4096)
        vision_ok, _ = await check_vision(binding)
        shaped, why = await check_structured_output(binding)
    assert limit_ok
    assert vision_ok
    assert not shaped  # the echo stub answers with the prompt, not the schema's JSON
    assert why == "the answer was not the schema's JSON"


async def test_structured_output_passes_on_the_schema_json() -> None:
    script = scripted([StubTurn(text='{"answer": "ok"}')])
    async with started(_target("json"), factories=_factory(json=script)) as binding:
        shaped, _ = await check_structured_output(binding)
    assert shaped


async def test_embeddings_pass_at_the_model_length_and_a_shorter_one() -> None:
    target = Target("stub", {"kind": "stub", "not_for_production": True}, "stub-embed")
    async with started_embeddings(target) as binding:
        await check_embeddings(binding, "stub embed", 8)
    async with started_embeddings(target, dimensions=4) as binding:
        await check_embeddings(binding, "stub embed", 4)


async def test_embeddings_fail_at_the_wrong_length() -> None:
    target = Target("stub", {"kind": "stub", "not_for_production": True}, "stub-embed")
    async with started_embeddings(target) as binding:
        with pytest.raises(AssertionError):
            await check_embeddings(binding, "stub embed", 9)


# Shapes a registry-built provider never produces, through a fake binding ------------


class _FakeProvider:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


class _FakeBinding:
    """Just what the checks use of a TaskBinding: ``request`` and ``provider.stream``."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self.provider = _FakeProvider(events)

    def request(self, **fields: Any) -> ModelRequest:
        return ModelRequest(model="fake", max_output_tokens=16, **fields)


def _fake(events: list[StreamEvent]) -> TaskBinding:
    return cast(TaskBinding, _FakeBinding(events))


def _stop(reason: StopReason = StopReason.END_TURN) -> StreamStopped:
    return StreamStopped(
        stop_reason=reason,
        usage=Usage(input_tokens=5, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0),
    )


async def test_a_stream_without_a_final_event_fails() -> None:
    binding = _fake([TextDelta(text="x")])
    with pytest.raises(AssertionError) as failed:
        await stream_shape(binding, binding.request(messages=ask("x")))
    assert "exactly one StreamStopped" in str(failed.value)


async def test_a_stream_with_two_final_events_fails() -> None:
    binding = _fake([TextDelta(text="x"), _stop(), _stop()])
    with pytest.raises(AssertionError) as failed:
        await stream_shape(binding, binding.request(messages=ask("x")))
    assert "exactly one StreamStopped" in str(failed.value)


async def test_events_after_the_final_event_fail() -> None:
    binding = _fake([_stop(), TextDelta(text="x")])
    with pytest.raises(AssertionError) as failed:
        await stream_shape(binding, binding.request(messages=ask("x")))
    assert "events followed StreamStopped" in str(failed.value)


async def test_an_unfinished_tool_call_is_named_before_the_protocol_failure() -> None:
    binding = _fake([ToolCallStarted(id="c1", name=LOOKUP_TOOL.name), _stop(StopReason.TOOL_USE)])
    with pytest.raises(AssertionError) as failed:
        await check_tools(binding, "fake tools")
    assert "a started tool call was not finished" in str(failed.value)


async def test_a_broken_stream_with_balanced_calls_still_fails_as_protocol() -> None:
    binding = _fake([TextDelta(text="x")])  # no final event: collect refuses it
    with pytest.raises(ProviderError) as failed:
        await check_tools(binding, "fake tools")
    assert failed.value.failure is ProviderFailure.PROTOCOL
    assert failed.value.__cause__ is None
    assert failed.value.__context__ is None


def test_usage_with_no_input_fails() -> None:
    usage = Usage(input_tokens=0, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    with pytest.raises(AssertionError) as failed:
        assert_usage(usage)
    assert "no input tokens" in str(failed.value)


def test_usage_with_a_negative_count_fails() -> None:
    usage = Usage.model_construct(
        input_tokens=-1, output_tokens=5, cache_read_tokens=10, cache_write_tokens=0
    )
    with pytest.raises(AssertionError) as failed:
        assert_usage(usage)
    assert "negative" in str(failed.value)


async def test_a_tool_call_without_a_string_code_fails() -> None:
    turn = StubTurn(tool_calls=(StubToolCall(LOOKUP_TOOL.name, {"code": 1234}),))
    script = answer_tool_results(turn, lambda results: StubTurn(text="done"))
    async with started(_target("num"), factories=_factory(num=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_tools(binding, "stub tools")
    assert "no string code" in str(failed.value)


async def test_a_silent_answer_after_the_tool_result_fails() -> None:
    usage = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    silent = StubTurn(text="", usage=usage, stop_reason=StopReason.END_TURN)
    script = answer_tool_results(_tool_turn(), lambda results: silent)
    async with started(_target("silent"), factories=_factory(silent=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_tools(binding, "stub tools")
    assert "no answer followed the tool result" in str(failed.value)


async def test_tool_calls_ending_the_turn_fail() -> None:
    ended = StubTurn(
        tool_calls=(StubToolCall(LOOKUP_TOOL.name, {"code": "R-1234"}),),
        stop_reason=StopReason.END_TURN,
    )
    script = answer_tool_results(ended, lambda results: StubTurn(text="done"))
    async with started(_target("ended"), factories=_factory(ended=script)) as binding:
        with pytest.raises(AssertionError) as failed:
            await check_tools(binding, "stub tools")
    assert "expected tool_use" in str(failed.value)


class _FakeEmbedder:
    def __init__(self, response: EmbeddingResponse) -> None:
        self._response = response

    async def embed(self, texts: Any, **fields: Any) -> EmbeddingResponse:
        return self._response


def _vectors(value: float) -> tuple[tuple[float, ...], ...]:
    return tuple((value, 0.5) for _ in EMBEDDING_TEXTS)


async def test_a_non_finite_embedding_fails() -> None:
    response = EmbeddingResponse(model="fake", vectors=_vectors(math.nan), usage=None)
    with pytest.raises(AssertionError) as failed:
        await check_embeddings(cast(EmbeddingBinding, _FakeEmbedder(response)), "fake", 2)
    assert "non-finite" in str(failed.value)


async def test_embedding_usage_beyond_input_fails() -> None:
    usage = Usage(input_tokens=3, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0)
    response = EmbeddingResponse(model="fake", vectors=_vectors(0.25), usage=usage)
    with pytest.raises(AssertionError) as failed:
        await check_embeddings(cast(EmbeddingBinding, _FakeEmbedder(response)), "fake", 2)
    assert "more than input tokens" in str(failed.value)


async def test_fewer_vectors_than_texts_fail() -> None:
    response = EmbeddingResponse(model="fake", vectors=_vectors(0.25)[:-1], usage=None)
    with pytest.raises(AssertionError) as failed:
        await check_embeddings(cast(EmbeddingBinding, _FakeEmbedder(response)), "fake", 2)
    assert "vectors for" in str(failed.value)


async def test_a_well_formed_fake_embedding_passes() -> None:
    usage = Usage(input_tokens=3, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    response = EmbeddingResponse(model="fake", vectors=_vectors(0.25), usage=usage)
    await check_embeddings(cast(EmbeddingBinding, _FakeEmbedder(response)), "fake", 2)
