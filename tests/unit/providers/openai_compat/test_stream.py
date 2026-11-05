"""Stream chunks to conduit events, and usage with cached prompt tokens split out."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import StopReason, Usage
from conduit.providers.openai_compat.mapping import StreamMapper, usage_from_openai


def _chunk(delta: dict[str, Any] | None = None, finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-example",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        "usage": None,
    }


def _usage_chunk(prompt: int = 10, completion: int = 5, cached: int | None = 0) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"id": "chatcmpl-example", "choices": [], "usage": usage}


def _tool_delta(
    index: int, *, id: str | None = None, name: str | None = None, arguments: str | None = None
) -> dict[str, Any]:
    call: dict[str, Any] = {"index": index}
    function: dict[str, Any] = {}
    if id is not None:
        call["id"] = id
        call["type"] = "function"
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    call["function"] = function
    return {"tool_calls": [call]}


def _run(*chunks: dict[str, Any]) -> list[StreamEvent]:
    mapper = StreamMapper()
    events: list[StreamEvent] = []
    for chunk in chunks:
        events.extend(mapper.feed(chunk))
    events.extend(mapper.finish())
    return events


class TestUsage:
    """Prompt tokens include cached tokens; each count is checked on its own."""

    BODY: ClassVar[dict[str, Any]] = {
        "prompt_tokens": 2048,
        "completion_tokens": 120,
        "total_tokens": 2168,
        "prompt_tokens_details": {"cached_tokens": 1920, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 64},
    }

    def test_input_tokens_are_prompt_tokens_minus_cached(self) -> None:
        assert usage_from_openai(self.BODY).input_tokens == 128

    def test_cache_read_tokens_are_the_cached_count(self) -> None:
        assert usage_from_openai(self.BODY).cache_read_tokens == 1920

    def test_cache_write_tokens_are_always_zero(self) -> None:
        assert usage_from_openai(self.BODY).cache_write_tokens == 0

    def test_output_tokens_are_completion_tokens_reasoning_included(self) -> None:
        assert usage_from_openai(self.BODY).output_tokens == 120

    def test_total_input_is_the_prompt(self) -> None:
        assert usage_from_openai(self.BODY).total_input_tokens == 2048

    def test_missing_details_mean_nothing_was_cached(self) -> None:
        usage = usage_from_openai({"prompt_tokens": 40, "completion_tokens": 2})
        assert usage == Usage(
            input_tokens=40, output_tokens=2, cache_read_tokens=0, cache_write_tokens=0
        )
        usage = usage_from_openai(
            {"prompt_tokens": 40, "completion_tokens": 2, "prompt_tokens_details": None}
        )
        assert usage.input_tokens == 40
        usage = usage_from_openai(
            {
                "prompt_tokens": 40,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": None},
            }
        )
        assert usage.cache_read_tokens == 0

    @pytest.mark.parametrize(
        "body",
        [
            {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 11},
            },
            {"prompt_tokens": 10},
            {"completion_tokens": 1},
            {"prompt_tokens": "10", "completion_tokens": 1},
            {"prompt_tokens": -1, "completion_tokens": 1},
            {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": -1},
            },
            {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": "cached"},
            {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": [5]},
            {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": 5},
        ],
    )
    def test_impossible_usage_is_a_protocol_error(self, body: dict[str, Any]) -> None:
        with pytest.raises(ProviderError) as caught:
            usage_from_openai(body)
        assert caught.value.failure is ProviderFailure.PROTOCOL


class TestText:
    def test_text_deltas_then_one_stop_with_the_final_usage(self) -> None:
        events = _run(
            _chunk({"role": "assistant", "content": ""}),
            _chunk({"content": "Hel"}),
            _chunk({"content": "lo"}),
            _chunk({}, finish="stop"),
            _usage_chunk(prompt=100, completion=2, cached=64),
        )
        assert events == [
            TextDelta(text="Hel"),
            TextDelta(text="lo"),
            StreamStopped(
                stop_reason=StopReason.END_TURN,
                usage=Usage(
                    input_tokens=36, output_tokens=2, cache_read_tokens=64, cache_write_tokens=0
                ),
            ),
        ]
        assert sum(isinstance(e, StreamStopped) for e in events) == 1

    def test_other_choices_are_ignored(self) -> None:
        stray = {"choices": [{"index": 1, "delta": {"content": "other"}, "finish_reason": None}]}
        events = _run(_chunk({"content": "mine"}), stray, _chunk(finish="stop"), _usage_chunk())
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["mine"]

    def test_a_refusal_is_surfaced_as_text_with_a_refusal_stop(self) -> None:
        events = _run(
            _chunk({"refusal": "I can't help with that."}), _chunk(finish="stop"), _usage_chunk()
        )
        assert events[0] == TextDelta(text="I can't help with that.")
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is StopReason.REFUSAL


class TestStopReasons:
    @pytest.mark.parametrize(
        ("finish", "reason"),
        [
            ("stop", StopReason.END_TURN),
            ("length", StopReason.MAX_TOKENS),
            ("tool_calls", StopReason.TOOL_USE),
            ("function_call", StopReason.TOOL_USE),
            ("content_filter", StopReason.REFUSAL),
            ("something_new", StopReason.OTHER),
        ],
    )
    def test_finish_reasons_map(self, finish: str, reason: StopReason) -> None:
        stop = _run(_chunk({"content": "x"}), _chunk(finish=finish), _usage_chunk())[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is reason


class TestToolCalls:
    def test_arguments_are_assembled_from_deltas(self) -> None:
        events = _run(
            _chunk({"role": "assistant", "content": None}),
            _chunk(_tool_delta(0, id="call_a", name="get_weather", arguments="")),
            _chunk(_tool_delta(0, arguments='{"ci')),
            _chunk(_tool_delta(0, arguments='ty": "Syd')),
            _chunk(_tool_delta(0, arguments='ney"}')),
            _chunk(finish="tool_calls"),
            _usage_chunk(),
        )
        assert events[:2] == [
            ToolCallStarted(id="call_a", name="get_weather"),
            ToolCallFinished(id="call_a", name="get_weather", input={"city": "Sydney"}),
        ]
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is StopReason.TOOL_USE

    def test_parallel_calls_are_kept_apart_by_index(self) -> None:
        events = _run(
            _chunk(_tool_delta(0, id="call_a", name="get_weather", arguments='{"city":')),
            _chunk(_tool_delta(1, id="call_b", name="get_weather", arguments='{"city":')),
            _chunk(_tool_delta(0, arguments=' "Sydney"}')),
            _chunk(_tool_delta(1, arguments=' "Hobart"}')),
            _chunk(finish="tool_calls"),
            _usage_chunk(),
        )
        finished = [e for e in events if isinstance(e, ToolCallFinished)]
        assert [(e.id, e.input) for e in finished] == [
            ("call_a", {"city": "Sydney"}),
            ("call_b", {"city": "Hobart"}),
        ]
        started = [e for e in events if isinstance(e, ToolCallStarted)]
        assert [e.id for e in started] == ["call_a", "call_b"]

    def test_a_whole_call_in_one_chunk(self) -> None:
        events = _run(
            _chunk(_tool_delta(0, id="call_a", name="lookup", arguments='{"q": 1}')),
            _chunk(finish="tool_calls"),
            _usage_chunk(),
        )
        assert ToolCallFinished(id="call_a", name="lookup", input={"q": 1}) in events

    def test_empty_arguments_are_an_empty_input(self) -> None:
        events = _run(
            _chunk(_tool_delta(0, id="call_a", name="now")),
            _chunk(finish="tool_calls"),
            _usage_chunk(),
        )
        assert ToolCallFinished(id="call_a", name="now", input={}) in events

    def test_text_before_a_call_is_kept(self) -> None:
        events = _run(
            _chunk({"content": "Let me check."}),
            _chunk(_tool_delta(0, id="call_a", name="now", arguments="{}")),
            _chunk(finish="tool_calls"),
            _usage_chunk(),
        )
        assert events[0] == TextDelta(text="Let me check.")
        assert isinstance(events[1], ToolCallStarted)

    @pytest.mark.parametrize(
        "chunks",
        [
            # arguments that are not JSON
            [_chunk(_tool_delta(0, id="call_a", name="now", arguments="{not json"))],
            # arguments that are JSON but not an object
            [_chunk(_tool_delta(0, id="call_a", name="now", arguments="[1, 2]"))],
            # a call that never names itself
            [_chunk(_tool_delta(0, id="call_a", arguments="{}"))],
            # a call with no id
            [_chunk(_tool_delta(0, name="now", arguments="{}"))],
            # a name conduit cannot carry
            [_chunk(_tool_delta(0, id="call_a", name="bad name!", arguments="{}"))],
            # a delta that changes the call's id
            [
                _chunk(_tool_delta(0, id="call_a", name="now", arguments="{")),
                _chunk(_tool_delta(0, id="call_b", arguments="}")),
            ],
            # a delta with no index
            [_chunk({"tool_calls": [{"id": "call_a", "function": {"name": "now"}}]})],
        ],
    )
    def test_broken_tool_calls_are_protocol_errors(self, chunks: list[dict[str, Any]]) -> None:
        with pytest.raises(ProviderError) as caught:
            _run(*chunks, _chunk(finish="tool_calls"), _usage_chunk())
        assert caught.value.failure is ProviderFailure.PROTOCOL


class TestBrokenStreams:
    def test_no_usage_is_a_protocol_error(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run(_chunk({"content": "hi"}), _chunk(finish="stop"))
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_no_finish_reason_is_a_protocol_error(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run(_chunk({"content": "hi"}), _usage_chunk())
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_the_last_usage_wins_and_is_never_summed(self) -> None:
        events = _run(
            _chunk({"content": "hi"}),
            _chunk(finish="stop"),
            _usage_chunk(prompt=5, completion=1),
            _usage_chunk(prompt=50, completion=3),
        )
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage.input_tokens == 50
        assert stop.usage.output_tokens == 3
