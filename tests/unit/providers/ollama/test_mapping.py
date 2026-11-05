"""Mapping conduit requests to /api/chat bodies, and NDJSON stream lines to events."""

from __future__ import annotations

import base64
import itertools
import json
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.ollama.mapping import StreamMapper, build_chat_body, failure_for_status
from tests.unit.providers.ollama.chains import assert_nothing_attached

MODEL = "llama3.1:8b"
PNG = b"\x89PNG\r\n\x1a\nsynthetic"
TOOL = ToolSpec(
    name="lookup_host",
    description="Look a host up in the inventory.",
    input_schema={"type": "object", "properties": {"host": {"type": "string"}}},
)


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model=MODEL,
        messages=messages or (Message.user("Hello"),),
        max_output_tokens=256,
        **fields,
    )


class TestRequestBody:
    def test_a_plain_request(self) -> None:
        body = build_chat_body(_request(), num_ctx=None)
        assert body == {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "options": {"num_predict": 256},
        }

    def test_parameters_and_context_length(self) -> None:
        body = build_chat_body(_request(temperature=0.2, stop_sequences=("END",)), num_ctx=131072)
        assert body["options"] == {
            "num_predict": 256,
            "temperature": 0.2,
            "stop": ["END"],
            "num_ctx": 131072,
        }

    def test_system_blocks_join_and_cache_breakpoints_are_ignored(self) -> None:
        request = _request(
            system=(TextBlock(text="First."), CacheBreakpoint(), TextBlock(text="Second.")),
        )
        body = build_chat_body(request, num_ctx=None)
        assert body["messages"][0] == {"role": "system", "content": "First.\n\nSecond."}
        assert "cache" not in json.dumps(body)

    def test_a_breakpoint_inside_a_message_is_ignored(self) -> None:
        message = Message(
            role="user", content=(TextBlock(text="A"), CacheBreakpoint(), TextBlock(text="B"))
        )
        body = build_chat_body(_request(message), num_ctx=None)
        assert body["messages"] == [{"role": "user", "content": "A\n\nB"}]

    def test_images_are_sent_as_base64(self) -> None:
        message = Message(
            role="user",
            content=(TextBlock(text="What is this?"), ImageBlock(media_type="image/png", data=PNG)),
        )
        body = build_chat_body(_request(message), num_ctx=None)
        assert body["messages"] == [
            {
                "role": "user",
                "content": "What is this?",
                "images": [base64.b64encode(PNG).decode("ascii")],
            }
        ]

    def test_tools_and_a_tool_round_trip(self) -> None:
        call = ToolUseBlock(id="call_1", name="lookup_host", input={"host": "a.example.com"})
        result = ToolResultBlock(
            tool_use_id="call_1", content=(TextBlock(text="host a is up"),), is_error=False
        )
        request = _request(
            Message.user("Is host a up?"),
            Message(role="assistant", content=(TextBlock(text="Checking."), call)),
            Message(role="user", content=(result, TextBlock(text="Thanks."))),
            tools=(TOOL,),
        )
        body = build_chat_body(request, num_ctx=None)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "lookup_host",
                    "description": "Look a host up in the inventory.",
                    "parameters": TOOL.input_schema,
                },
            }
        ]
        assert body["messages"] == [
            {"role": "user", "content": "Is host a up?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {"function": {"name": "lookup_host", "arguments": {"host": "a.example.com"}}}
                ],
            },
            {"role": "tool", "tool_name": "lookup_host", "content": "host a is up"},
            {"role": "user", "content": "Thanks."},
        ]

    def test_an_error_result_is_marked(self) -> None:
        call = ToolUseBlock(id="call_1", name="lookup_host", input={})
        result = ToolResultBlock(
            tool_use_id="call_1", content=(TextBlock(text="no such host"),), is_error=True
        )
        request = _request(
            Message.user("Go"),
            Message(role="assistant", content=(call,)),
            Message(role="user", content=(result,)),
        )
        body = build_chat_body(request, num_ctx=None)
        assert body["messages"][-1] == {
            "role": "tool",
            "tool_name": "lookup_host",
            "content": "Error: no such host",
        }

    def test_a_result_for_an_unknown_call_is_rejected_before_any_call(self) -> None:
        result = ToolResultBlock(tool_use_id="call_9", content=(TextBlock(text="x"),))
        request = _request(Message(role="user", content=(result,)))
        with pytest.raises(ProviderError) as caught:
            build_chat_body(request, num_ctx=None)
        assert caught.value.failure is ProviderFailure.REJECTED

    def test_structured_output_sends_the_schema_as_format(self) -> None:
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        body = build_chat_body(_request(output_schema=schema), num_ctx=None)
        assert body["format"] == schema


def _line(**fields: Any) -> str:
    return json.dumps({"model": MODEL, "created_at": "2026-11-04T00:00:00Z", **fields})


def _chunk(content: str = "", **message: Any) -> str:
    return _line(message={"role": "assistant", "content": content, **message}, done=False)


def _done(reason: str = "stop", **counts: int) -> str:
    return _line(
        message={"role": "assistant", "content": ""},
        done=True,
        done_reason=reason,
        **counts,
    )


def _run(lines: list[str]) -> list[StreamEvent]:
    ids = (f"toolu_test_{n}" for n in itertools.count())
    mapper = StreamMapper(new_call_id=lambda: next(ids))
    events: list[StreamEvent] = []
    for line in lines:
        events += mapper.feed(line)
    return events + mapper.finish()


class TestStream:
    def test_text_then_one_stop_with_usage(self) -> None:
        events = _run([_chunk("Hel"), _chunk("lo."), _done(prompt_eval_count=26, eval_count=5)])
        assert events == [
            TextDelta(text="Hel"),
            TextDelta(text="lo."),
            StreamStopped(
                stop_reason=StopReason.END_TURN,
                usage=Usage(
                    input_tokens=26, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0
                ),
            ),
        ]

    def test_blank_lines_and_thinking_are_skipped(self) -> None:
        events = _run(["", _chunk("", thinking="hmm"), _chunk("Hi"), "  ", _done()])
        assert events[:-1] == [TextDelta(text="Hi")]

    def test_prompt_eval_count_is_input(self) -> None:
        stop = _run([_done(prompt_eval_count=40)])[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage == Usage(
            input_tokens=40, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_eval_count_is_output(self) -> None:
        stop = _run([_done(eval_count=9)])[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage == Usage(
            input_tokens=0, output_tokens=9, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_cached_prompt_tokens_are_split_out_of_input(self) -> None:
        stop = _run([_done(prompt_eval_count=40, prompt_eval_cached_count=30)])[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage == Usage(
            input_tokens=10, output_tokens=0, cache_read_tokens=30, cache_write_tokens=0
        )

    @pytest.mark.parametrize("key", ["prompt_eval_count", "prompt_eval_cached_count", "eval_count"])
    def test_a_negative_count_is_a_protocol_failure_without_the_value(self, key: str) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_done("stop", **{key: -7031})])
        assert caught.value.failure is ProviderFailure.PROTOCOL
        assert "-7031" not in str(caught.value)
        assert_nothing_attached(caught.value, "-7031")

    def test_a_done_line_without_counts_reports_zero_usage(self) -> None:
        stop = _run([_done()])[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage == Usage.zero()

    def test_a_cached_count_above_the_prompt_count_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_done(prompt_eval_count=10, prompt_eval_cached_count=30)])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("stop", StopReason.END_TURN),
            ("length", StopReason.MAX_TOKENS),
            ("load", StopReason.OTHER),
            ("unload", StopReason.OTHER),
        ],
    )
    def test_done_reasons(self, reason: str, expected: StopReason) -> None:
        stop = _run([_done(reason)])[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is expected

    def test_tool_calls_start_and_finish_and_stop_for_tool_use(self) -> None:
        calls = [
            {"function": {"name": "lookup_host", "arguments": {"host": "a.example.com"}}},
            {"function": {"name": "lookup_host", "arguments": '{"host": "b.example.com"}'}},
        ]
        events = _run([_chunk("Checking."), _chunk(tool_calls=calls), _done()])
        assert events[:-1] == [
            TextDelta(text="Checking."),
            ToolCallStarted(id="toolu_test_0", name="lookup_host"),
            ToolCallFinished(
                id="toolu_test_0", name="lookup_host", input={"host": "a.example.com"}
            ),
            ToolCallStarted(id="toolu_test_1", name="lookup_host"),
            ToolCallFinished(
                id="toolu_test_1", name="lookup_host", input={"host": "b.example.com"}
            ),
        ]
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is StopReason.TOOL_USE

    def test_a_call_id_from_the_server_is_kept(self) -> None:
        calls = [{"id": "call_abc", "function": {"name": "lookup_host", "arguments": {}}}]
        events = _run([_chunk(tool_calls=calls), _done()])
        assert events[0] == ToolCallStarted(id="call_abc", name="lookup_host")

    @pytest.mark.parametrize(
        "call",
        [
            {"function": {"name": "lookup_host", "arguments": "[1, 2]"}},
            {"function": {"name": "lookup_host", "arguments": "{not json"}},
            {"function": {"name": "not a tool name", "arguments": {}}},
            {"function": {"arguments": {}}},
            {"name": "lookup_host"},
        ],
    )
    def test_a_malformed_tool_call_is_a_protocol_failure(self, call: dict[str, Any]) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_chunk(tool_calls=[call]), _done()])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    @pytest.mark.parametrize(
        "lines",
        [
            [],
            ["{not json"],
            ['["a list"]'],
            [_chunk("Hi")],
        ],
    )
    def test_a_broken_stream_is_a_protocol_failure(self, lines: list[str]) -> None:
        with pytest.raises(ProviderError) as caught:
            _run(lines)
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_a_line_after_done_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_done(), _chunk("late")])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_an_error_line_is_unavailable_and_carries_no_provider_text(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_chunk("Hi"), json.dumps({"error": "model runner crashed: secret detail"})])
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        assert "secret detail" not in str(caught.value)
        assert_nothing_attached(caught.value, "secret detail")


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (400, ProviderFailure.REJECTED),
        (401, ProviderFailure.REJECTED),
        (403, ProviderFailure.REJECTED),
        (404, ProviderFailure.REJECTED),
        (408, ProviderFailure.TIMEOUT),
        (429, ProviderFailure.UNAVAILABLE),
        (500, ProviderFailure.UNAVAILABLE),
        (503, ProviderFailure.UNAVAILABLE),
        (504, ProviderFailure.TIMEOUT),
    ],
)
def test_http_status_failures(status: int, failure: ProviderFailure) -> None:
    assert failure_for_status(status) is failure
