"""Mapping conduit requests to generate_content parameters, and stream chunks to events.

Chunks are the JSON form of the SDK's ``GenerateContentResponse`` (snake_case field
names, bytes as base64), which is what the provider hands the mapper.
"""

from __future__ import annotations

import base64
import itertools
import json
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    ProviderStateReported,
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
    ProviderStateBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.gemini.mapping import (
    GENERATED_PREFIX,
    SKIP_SIGNATURE,
    StreamMapper,
    build_request,
)
from tests.unit.providers.gemini.chains import assert_nothing_attached

MODEL = "gemini-3.8-flash"
PNG = b"\x89PNG\r\n\x1a\nsynthetic"
SCHEMA = {"type": "object", "properties": {"host": {"type": "string"}}, "required": ["host"]}
TOOL = ToolSpec(name="lookup_host", description="Look a host up.", input_schema=SCHEMA)


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model=MODEL, messages=messages or (Message.user("Hello"),), max_output_tokens=256, **fields
    )


def _signed(call_id: str, signature: bytes) -> ProviderStateBlock:
    """The state block the mapper reports for ``call_id``'s thought signature."""
    payload = {"call_id": call_id, "signature": base64.b64encode(signature).decode("ascii")}
    return ProviderStateBlock(
        provider_kind="gemini",
        state_type="thought_signature",
        payload=json.dumps(payload).encode("utf-8"),
    )


class TestRequest:
    def test_a_plain_request(self) -> None:
        assert build_request(_request()) == {
            "model": MODEL,
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "config": {"max_output_tokens": 256},
        }

    def test_parameters(self) -> None:
        config = build_request(_request(temperature=0.3, stop_sequences=("END",)))["config"]
        assert config == {"max_output_tokens": 256, "temperature": 0.3, "stop_sequences": ["END"]}

    def test_system_text_joins_and_cache_breakpoints_are_ignored(self) -> None:
        request = _request(
            Message(role="user", content=(TextBlock(text="A"), CacheBreakpoint())),
            system=(TextBlock(text="One."), CacheBreakpoint(), TextBlock(text="Two.")),
            cache_tools=True,
            tools=(TOOL,),
        )
        params = build_request(request)
        assert params["config"]["system_instruction"] == "One.\n\nTwo."
        assert params["contents"] == [{"role": "user", "parts": [{"text": "A"}]}]
        assert "cache" not in repr(params)

    def test_a_message_of_only_a_breakpoint_is_left_out(self) -> None:
        request = _request(Message.user("A"), Message(role="user", content=(CacheBreakpoint(),)))
        assert build_request(request)["contents"] == [{"role": "user", "parts": [{"text": "A"}]}]

    def test_images_are_inline_data(self) -> None:
        message = Message(
            role="user", content=(ImageBlock(media_type="image/png", data=PNG), TextBlock(text="?"))
        )
        assert build_request(_request(message))["contents"] == [
            {
                "role": "user",
                "parts": [{"inline_data": {"mime_type": "image/png", "data": PNG}}, {"text": "?"}],
            }
        ]

    def test_tools_are_declared_with_their_json_schema(self) -> None:
        config = build_request(_request(tools=(TOOL,)))["config"]
        assert config["tools"] == [
            {
                "function_declarations": [
                    {
                        "name": "lookup_host",
                        "description": "Look a host up.",
                        "parameters_json_schema": SCHEMA,
                    }
                ]
            }
        ]

    def test_structured_output(self) -> None:
        config = build_request(_request(output_schema=SCHEMA))["config"]
        assert config["response_mime_type"] == "application/json"
        assert config["response_json_schema"] == SCHEMA

    def test_a_tool_round_trip_with_a_server_id_and_its_signature_from_state(self) -> None:
        call = ToolUseBlock(id="fc_1", name="lookup_host", input={"host": "a.example.com"})
        result = ToolResultBlock(tool_use_id="fc_1", content=(TextBlock(text="up"),))
        answer = (TextBlock(text="Checking."), _signed("fc_1", b"sig-1"), call)
        request = _request(
            Message.user("Is a up?"),
            Message(role="assistant", content=answer),
            Message(role="user", content=(result,)),
        )
        contents = build_request(request)["contents"]
        assert contents[1:] == [
            {
                "role": "model",
                "parts": [
                    {"text": "Checking."},
                    {
                        "function_call": {
                            "id": "fc_1",
                            "name": "lookup_host",
                            "args": {"host": "a.example.com"},
                        },
                        "thought_signature": b"sig-1",
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "fc_1",
                            "name": "lookup_host",
                            "response": {"output": "up"},
                        }
                    }
                ],
            },
        ]

    def test_a_generated_id_is_never_sent(self) -> None:
        call_id = f"{GENERATED_PREFIX}abc"
        call = ToolUseBlock(id=call_id, name="lookup_host", input={})
        result = ToolResultBlock(tool_use_id=call_id, content=(TextBlock(text="x"),))
        request = _request(
            Message.user("Go"),
            Message(role="assistant", content=(_signed(call_id, b"sig"), call)),
            Message(role="user", content=(result,)),
        )
        contents = build_request(request)["contents"]
        assert contents[1]["parts"][0]["thought_signature"] == b"sig"
        assert contents[1]["parts"][0]["function_call"] == {"name": "lookup_host", "args": {}}
        assert contents[2]["parts"][0]["function_response"] == {
            "name": "lookup_host",
            "response": {"output": "x"},
        }

    def test_an_unknown_signature_falls_back_to_the_skip_value_on_the_first_call_only(
        self,
    ) -> None:
        first = ToolUseBlock(id="toolu_a", name="lookup_host", input={})
        second = ToolUseBlock(id="toolu_b", name="lookup_host", input={})
        request = _request(Message.user("Go"), Message(role="assistant", content=(first, second)))
        parts = build_request(request)["contents"][1]["parts"]
        assert parts[0]["thought_signature"] == SKIP_SIGNATURE
        assert "thought_signature" not in parts[1]

    def test_a_signature_on_a_later_call_is_kept(self) -> None:
        first = ToolUseBlock(id="toolu_a", name="lookup_host", input={})
        second = ToolUseBlock(id="toolu_b", name="lookup_host", input={})
        answer = (first, _signed("toolu_b", b"sig-b"), second)
        request = _request(Message.user("Go"), Message(role="assistant", content=answer))
        parts = build_request(request)["contents"][1]["parts"]
        assert parts[0]["thought_signature"] == SKIP_SIGNATURE
        assert parts[1]["thought_signature"] == b"sig-b"

    def test_a_signature_applies_only_within_its_own_message(self) -> None:
        call = ToolUseBlock(id="fc_1", name="lookup_host", input={})
        result = ToolResultBlock(tool_use_id="fc_1", content=())
        request = _request(
            Message.user("Go"),
            Message(role="assistant", content=(_signed("fc_2", b"sig-2"), TextBlock(text="x"))),
            Message.user("Again"),
            Message(role="assistant", content=(call,)),
            Message(role="user", content=(result,)),
        )
        contents = build_request(request)["contents"]
        assert contents[3]["parts"][0]["thought_signature"] == SKIP_SIGNATURE
        assert contents[1]["parts"] == [{"text": "x"}]

    def test_the_state_block_itself_is_never_a_part(self) -> None:
        call = ToolUseBlock(id="fc_1", name="lookup_host", input={})
        request = _request(
            Message.user("Go"),
            Message(role="assistant", content=(_signed("fc_1", b"zz-sig-zz"), call)),
        )
        parts = build_request(request)["contents"][1]["parts"]
        assert parts == [
            {
                "function_call": {"id": "fc_1", "name": "lookup_host", "args": {}},
                "thought_signature": b"zz-sig-zz",
            }
        ]

    @pytest.mark.parametrize(
        "payload",
        [
            b"not json zz-state-marker-zz",
            b"\xff zz-state-marker-zz",
            b'["zz-state-marker-zz"]',
            b'{"call_id": "fc_1"}',
            b'{"call_id": 1, "signature": "c2ln"}',
            b'{"call_id": "fc_1", "signature": "%%zz-state-marker-zz%%"}',
            # Decodes once the stray characters are dropped: only strict base64 is taken.
            b'{"call_id": "fc_1", "signature": "c2ln!!"}',
            b'{"call_id": "fc_1", "signature": "c2ln", "extra": "zz-state-marker-zz"}',
        ],
    )
    def test_own_state_that_is_not_a_signature_is_refused_cleanly(self, payload: bytes) -> None:
        state = ProviderStateBlock(
            provider_kind="gemini", state_type="thought_signature", payload=payload
        )
        call = ToolUseBlock(id="fc_1", name="lookup_host", input={})
        request = _request(Message.user("Go"), Message(role="assistant", content=(state, call)))
        with pytest.raises(ProviderError) as caught:
            build_request(request)
        assert caught.value.failure is ProviderFailure.REJECTED
        assert_nothing_attached(caught.value, "zz-state-marker-zz")

    def test_own_state_of_an_unknown_type_is_refused(self) -> None:
        # A well-formed signature payload under another state type is still refused.
        signed = _signed("fc_1", b"sig")
        state = ProviderStateBlock(
            provider_kind="gemini", state_type="future", payload=signed.payload
        )
        call = ToolUseBlock(id="fc_1", name="lookup_host", input={})
        request = _request(Message.user("Go"), Message(role="assistant", content=(state, call)))
        with pytest.raises(ProviderError) as caught:
            build_request(request)
        assert caught.value.failure is ProviderFailure.REJECTED

    def test_an_error_result_and_images_in_a_result(self) -> None:
        call = ToolUseBlock(id="toolu_a", name="lookup_host", input={})
        result = ToolResultBlock(
            tool_use_id="toolu_a",
            content=(TextBlock(text="no such host"), ImageBlock(media_type="image/png", data=PNG)),
            is_error=True,
        )
        request = _request(
            Message.user("Go"),
            Message(role="assistant", content=(call,)),
            Message(role="user", content=(result,)),
        )
        parts = build_request(request)["contents"][2]["parts"]
        assert parts == [
            {
                "function_response": {
                    "id": "toolu_a",
                    "name": "lookup_host",
                    "response": {"error": "no such host"},
                }
            },
            {"inline_data": {"mime_type": "image/png", "data": PNG}},
        ]

    def test_a_result_for_an_unknown_call_is_rejected_before_any_call(self) -> None:
        result = ToolResultBlock(tool_use_id="toolu_x", content=())
        with pytest.raises(ProviderError) as caught:
            build_request(_request(Message(role="user", content=(result,))))
        assert caught.value.failure is ProviderFailure.REJECTED


def _chunk(
    *parts: dict[str, Any], finish: str | None = None, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": list(parts)}, "index": 0}
    if finish is not None:
        candidate["finish_reason"] = finish
    chunk: dict[str, Any] = {"candidates": [candidate], "model_version": MODEL}
    if usage is not None:
        chunk["usage_metadata"] = usage
    return chunk


def _mapper() -> StreamMapper:
    ids = (f"{GENERATED_PREFIX}{n}" for n in itertools.count())
    return StreamMapper(new_call_id=lambda: next(ids))


def _run(chunks: list[dict[str, Any]], mapper: StreamMapper | None = None) -> list[StreamEvent]:
    mapper = mapper or _mapper()
    events: list[StreamEvent] = []
    for chunk in chunks:
        events += mapper.feed(chunk)
    return events + mapper.finish()


def _usage(chunks: list[dict[str, Any]]) -> Usage:
    stop = _run(chunks)[-1]
    assert isinstance(stop, StreamStopped)
    return stop.usage


def _final(**usage: int) -> list[dict[str, Any]]:
    return [_chunk({"text": "x"}, finish="STOP", usage=usage)]


class TestStream:
    def test_text_then_one_stop_with_the_final_usage(self) -> None:
        events = _run(
            [
                _chunk({"text": "Hel"}, usage={"prompt_token_count": 9}),
                _chunk(
                    {"text": "lo."},
                    finish="STOP",
                    usage={"prompt_token_count": 9, "candidates_token_count": 3},
                ),
            ]
        )
        assert events == [
            TextDelta(text="Hel"),
            TextDelta(text="lo."),
            StreamStopped(
                stop_reason=StopReason.END_TURN,
                usage=Usage(
                    input_tokens=9, output_tokens=3, cache_read_tokens=0, cache_write_tokens=0
                ),
            ),
        ]

    def test_thought_parts_are_not_streamed(self) -> None:
        events = _run(
            [_chunk({"text": "thinking", "thought": True}, {"text": "Hi"}, finish="STOP", usage={})]
        )
        assert events[:-1] == [TextDelta(text="Hi")]

    def test_prompt_tokens_are_input(self) -> None:
        assert _usage(_final(prompt_token_count=40)) == Usage(
            input_tokens=40, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_candidate_tokens_are_output(self) -> None:
        assert _usage(_final(candidates_token_count=7)) == Usage(
            input_tokens=0, output_tokens=7, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_thinking_tokens_are_output(self) -> None:
        assert _usage(_final(thoughts_token_count=11)) == Usage(
            input_tokens=0, output_tokens=11, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_cached_tokens_are_split_out_of_the_prompt(self) -> None:
        assert _usage(_final(prompt_token_count=40, cached_content_token_count=32)) == Usage(
            input_tokens=8, output_tokens=0, cache_read_tokens=32, cache_write_tokens=0
        )

    def test_tool_use_prompt_tokens_are_input(self) -> None:
        assert _usage(_final(tool_use_prompt_token_count=5)) == Usage(
            input_tokens=5, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_a_cached_count_above_the_prompt_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _usage(_final(prompt_token_count=4, cached_content_token_count=5))
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_usage_comes_from_the_last_chunk_and_is_never_summed(self) -> None:
        usage = _usage(
            [
                _chunk({"text": "a"}, usage={"prompt_token_count": 9, "candidates_token_count": 1}),
                _chunk(
                    {"text": "b"},
                    finish="STOP",
                    usage={"prompt_token_count": 9, "candidates_token_count": 2},
                ),
            ]
        )
        assert usage.input_tokens == 9
        assert usage.output_tokens == 2

    def test_a_final_chunk_without_usage_is_a_protocol_failure(self) -> None:
        # Review finding 1: an earlier chunk's partial usage must never stand in
        # for the call's usage (C19); the last chunk fed must carry it.
        with pytest.raises(ProviderError) as caught:
            _run(
                [
                    _chunk({"text": "a"}, usage={"prompt_token_count": 100}),
                    _chunk({"text": "b"}, finish="STOP"),
                ]
            )
        assert caught.value.failure is ProviderFailure.PROTOCOL

    @pytest.mark.parametrize(
        "key",
        [
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "tool_use_prompt_token_count",
        ],
    )
    def test_a_negative_count_is_a_protocol_failure_without_the_value(self, key: str) -> None:
        with pytest.raises(ProviderError) as caught:
            _usage(_final(**{key: -7031}))
        assert caught.value.failure is ProviderFailure.PROTOCOL
        assert "-7031" not in str(caught.value)
        assert_nothing_attached(caught.value, "-7031")

    def test_a_stream_without_usage_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_chunk({"text": "a"}, finish="STOP")])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_a_stream_without_a_finish_reason_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_chunk({"text": "a"}, usage={"prompt_token_count": 1})])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    @pytest.mark.parametrize(
        ("finish", "expected"),
        [
            ("STOP", StopReason.END_TURN),
            ("MAX_TOKENS", StopReason.MAX_TOKENS),
            ("SAFETY", StopReason.REFUSAL),
            ("RECITATION", StopReason.REFUSAL),
            ("BLOCKLIST", StopReason.REFUSAL),
            ("PROHIBITED_CONTENT", StopReason.REFUSAL),
            ("SPII", StopReason.REFUSAL),
            ("MALFORMED_FUNCTION_CALL", StopReason.OTHER),
            ("OTHER", StopReason.OTHER),
        ],
    )
    def test_finish_reasons(self, finish: str, expected: StopReason) -> None:
        stop = _run([_chunk(finish=finish, usage={})])[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is expected

    def test_a_blocked_prompt_is_a_refusal(self) -> None:
        chunk = {
            "prompt_feedback": {"block_reason": "SAFETY"},
            "usage_metadata": {"prompt_token_count": 12},
        }
        events = _run([chunk])
        assert events == [
            StreamStopped(
                stop_reason=StopReason.REFUSAL,
                usage=Usage(
                    input_tokens=12, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
                ),
            )
        ]

    def test_function_calls_and_their_signatures(self) -> None:
        signature = base64.b64encode(b"sig-1").decode("ascii")
        mapper = _mapper()
        events = _run(
            [
                _chunk(
                    {"text": "Checking."},
                    {
                        "function_call": {
                            "id": "fc_1",
                            "name": "lookup_host",
                            "args": {"host": "a.example.com"},
                        },
                        "thought_signature": signature,
                    },
                    {"function_call": {"name": "lookup_host", "args": {"host": "b.example.com"}}},
                ),
                _chunk(finish="STOP", usage={"prompt_token_count": 3}),
            ],
            mapper,
        )
        assert events[:-1] == [
            TextDelta(text="Checking."),
            ProviderStateReported(state=_signed("fc_1", b"sig-1")),
            ToolCallStarted(id="fc_1", name="lookup_host"),
            ToolCallFinished(id="fc_1", name="lookup_host", input={"host": "a.example.com"}),
            ToolCallStarted(id=f"{GENERATED_PREFIX}0", name="lookup_host"),
            ToolCallFinished(
                id=f"{GENERATED_PREFIX}0", name="lookup_host", input={"host": "b.example.com"}
            ),
        ]
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is StopReason.TOOL_USE

    def test_a_signature_on_a_generated_id_is_reported_under_that_id(self) -> None:
        signature = base64.b64encode(b"sig-g").decode("ascii")
        events = _run(
            [
                _chunk(
                    {"function_call": {"name": "lookup_host"}, "thought_signature": signature},
                    finish="STOP",
                    usage={},
                )
            ]
        )
        assert events[0] == ProviderStateReported(state=_signed(f"{GENERATED_PREFIX}0", b"sig-g"))

    def test_a_call_without_args_has_empty_input(self) -> None:
        events = _run([_chunk({"function_call": {"name": "lookup_host"}}, finish="STOP", usage={})])
        assert events[1] == ToolCallFinished(
            id=f"{GENERATED_PREFIX}0", name="lookup_host", input={}
        )

    @pytest.mark.parametrize(
        "part",
        [
            {"function_call": {"name": "not a name", "args": {}}},
            {"function_call": {"args": {}}},
            {"function_call": {"name": "lookup_host", "args": [1]}},
            {"function_call": {"name": "lookup_host"}, "thought_signature": "%%%"},
        ],
    )
    def test_a_malformed_call_is_a_protocol_failure(self, part: dict[str, Any]) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_chunk(part, finish="STOP", usage={})])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_a_chunk_after_the_finish_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _run([_chunk(finish="STOP", usage={}), _chunk({"text": "late"})])
        assert caught.value.failure is ProviderFailure.PROTOCOL

    def test_a_trailing_usage_only_chunk_after_the_finish_is_accepted(self) -> None:
        usage = _usage(
            [
                _chunk({"text": "a"}, finish="STOP", usage={"prompt_token_count": 1}),
                {"usage_metadata": {"prompt_token_count": 1, "candidates_token_count": 4}},
            ]
        )
        assert usage.output_tokens == 4
