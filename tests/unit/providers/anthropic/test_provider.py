"""The Anthropic adapter's streaming: text, tool use, usage, stop reasons and errors."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    UsageReported,
)
from conduit.core.types import Message, ModelRequest, StopReason, ToolUseBlock, Usage
from conduit.providers.anthropic import AnthropicProvider
from conduit.providers.anthropic.errors import failure_for
from conduit.providers.anthropic.stream import translate

from . import fakes
from .fakes import (
    message_end,
    message_start,
    scripted_client,
    text_block,
    thinking_block,
    tool_block,
    usage,
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _request(**fields: Any) -> ModelRequest:
    return ModelRequest(
        model="claude-sonnet-5-5",
        messages=(Message.user("hello"),),
        max_output_tokens=256,
        **fields,
    )


def _provider(client: Any, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(client, name="claude", **kwargs)


async def _events(provider: AnthropicProvider, request: ModelRequest | None = None) -> list[Any]:
    return [event async for event in provider.stream(request or _request())]


def _simple(*body: dict[str, Any], start: dict[str, Any] | None = None, **end: Any) -> list[Any]:
    events: list[Any] = [message_start(start or usage(input_tokens=10, output_tokens=1))]
    events += body
    events += message_end(
        end.get("stop_reason", "end_turn"), end.get("delta", usage(10, output_tokens=5))
    )
    return events


def _stopped(events: list[StreamEvent]) -> StreamStopped:
    stops = [e for e in events if isinstance(e, StreamStopped)]
    assert len(stops) == 1
    assert events[-1] is stops[0]
    return stops[0]


class TestText:
    async def test_text_deltas_stream_in_order(self) -> None:
        client = scripted_client(_simple(*text_block(0, "Hel", "lo", "!")))
        events = await _events(_provider(client))
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo", "!"]

    async def test_complete_collects_the_stream(self) -> None:
        client = scripted_client(_simple(*text_block(0, "Hi ", "there")))
        response = await _provider(client).complete(_request())
        assert response.text == "Hi there"
        assert response.stop_reason is StopReason.END_TURN
        assert response.model == "claude-sonnet-5-5"

    async def test_sdk_event_models_are_read_through_model_dump(self) -> None:
        stream = fakes.FakeStream(_simple(*text_block(0, "ok")), wrap=True)
        client = fakes.FakeClient(lambda params: stream)
        response = await _provider(client).complete(_request())
        assert response.text == "ok"

    async def test_thinking_and_unknown_blocks_are_not_streamed(self) -> None:
        body = [
            *thinking_block(0),
            *text_block(1, "answer"),
            {"type": "ping"},
            {"type": "some_future_event"},
        ]
        events = await _events(_provider(scripted_client(_simple(*body))))
        kinds = {type(e) for e in events}
        assert kinds == {UsageReported, TextDelta, StreamStopped}
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["answer"]

    async def test_the_request_goes_to_the_sdk_as_mapped(self) -> None:
        client = scripted_client(_simple(*text_block(0, "x")))
        await _events(_provider(client))
        (params,) = client.messages.calls
        assert params["model"] == "claude-sonnet-5-5"
        assert params["stream"] is True
        assert params["max_tokens"] == 256

    async def test_the_sdk_stream_is_closed(self) -> None:
        client = scripted_client(_simple(*text_block(0, "x")))
        await _events(_provider(client))
        assert client.messages.streams[0].closed

    async def test_an_abandoned_stream_is_closed(self) -> None:
        client = scripted_client(_simple(*text_block(0, "a", "b", "c")))
        stream = _provider(client).stream(_request())
        async for event in stream:
            if isinstance(event, TextDelta):
                break
        await stream.aclose()  # type: ignore[attr-defined]
        assert client.messages.streams[0].closed

    async def test_aclose_closes_the_client(self) -> None:
        client = scripted_client()
        await _provider(client).aclose()
        assert client.closed


class TestToolUse:
    async def test_tool_input_is_assembled_from_json_deltas(self) -> None:
        body = [
            *text_block(0, "Let me look."),
            *tool_block(1, "toolu_01", "lookup", '{"query": "ex', 'ample", "limit"', ": 3}"),
        ]
        client = scripted_client(_simple(*body, stop_reason="tool_use"))
        events = await _events(_provider(client))
        started = [e for e in events if isinstance(e, ToolCallStarted)]
        finished = [e for e in events if isinstance(e, ToolCallFinished)]
        assert started == [ToolCallStarted(id="toolu_01", name="lookup")]
        assert finished == [
            ToolCallFinished(id="toolu_01", name="lookup", input={"query": "example", "limit": 3})
        ]
        assert _stopped(events).stop_reason is StopReason.TOOL_USE

    async def test_a_tool_with_no_input_deltas_has_an_empty_input(self) -> None:
        client = scripted_client(_simple(*tool_block(0, "toolu_01", "now"), stop_reason="tool_use"))
        response = await _provider(client).complete(_request())
        assert response.tool_uses == (ToolUseBlock(id="toolu_01", name="now", input={}),)

    async def test_parallel_tool_calls_keep_their_order(self) -> None:
        body = [
            *tool_block(0, "toolu_a", "one", "{}"),
            *tool_block(1, "toolu_b", "two", '{"n": 1}'),
        ]
        client = scripted_client(_simple(*body, stop_reason="tool_use"))
        response = await _provider(client).complete(_request())
        assert [t.id for t in response.tool_uses] == ["toolu_a", "toolu_b"]
        assert response.tool_uses[1].input == {"n": 1}

    async def test_invalid_tool_json_is_a_protocol_failure(self) -> None:
        client = scripted_client(
            _simple(*tool_block(0, "toolu_01", "x", '{"a": '), stop_reason="tool_use")
        )
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.PROTOCOL

    async def test_tool_json_that_is_not_an_object_is_a_protocol_failure(self) -> None:
        client = scripted_client(
            _simple(*tool_block(0, "toolu_01", "x", "[1, 2]"), stop_reason="tool_use")
        )
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.PROTOCOL

    async def test_the_translator_itself_refuses_non_object_input(self) -> None:
        events = _simple(*tool_block(0, "toolu_01", "x", '"text"'), stop_reason="tool_use")

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        with pytest.raises(ProviderError) as caught:
            _ = [event async for event in translate(source())]
        assert caught.value.failure is ProviderFailure.PROTOCOL

    async def test_a_tool_call_cut_off_before_any_input_is_dropped(self) -> None:
        body = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_01", "name": "x", "input": {}},
            },
            {"type": "content_block_stop", "index": 0},
        ]
        client = scripted_client(_simple(*body, stop_reason="max_tokens"))
        events = await _events(_provider(client))
        assert not [e for e in events if isinstance(e, ToolCallStarted | ToolCallFinished)]
        assert _stopped(events).stop_reason is StopReason.MAX_TOKENS

    async def test_a_tool_call_followed_by_another_block_survives_max_tokens(self) -> None:
        body = [*tool_block(0, "toolu_01", "x", '{"a": 1}'), *text_block(1, "and then")]
        client = scripted_client(_simple(*body, stop_reason="max_tokens"))
        response = await _provider(client).complete(_request())
        assert response.tool_uses == (ToolUseBlock(id="toolu_01", name="x", input={"a": 1}),)

    async def test_invalid_tool_json_followed_by_another_block_is_a_protocol_failure(
        self,
    ) -> None:
        body = [*tool_block(0, "toolu_01", "x", '{"a": '), *text_block(1, "more")]
        client = scripted_client(_simple(*body, stop_reason="max_tokens"))
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.PROTOCOL

    async def test_interleaved_tool_blocks_are_assembled_by_index(self) -> None:
        def start(index: int, call_id: str, name: str) -> dict[str, Any]:
            return {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
            }

        def piece(index: int, text: str) -> dict[str, Any]:
            return {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": text},
            }

        body = [
            start(0, "a", "x"),
            start(1, "b", "y"),
            piece(1, '{"q":'),
            piece(0, '{"p":1}'),
            piece(1, "2}"),
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_stop", "index": 0},
        ]
        client = scripted_client(_simple(*body, stop_reason="tool_use"))
        response = await _provider(client).complete(_request())
        assert {t.id: t.input for t in response.tool_uses} == {"a": {"p": 1}, "b": {"q": 2}}

    async def test_a_tool_call_cut_off_by_max_tokens_is_dropped(self) -> None:
        body = [*text_block(0, "partial"), *tool_block(1, "toolu_01", "x", '{"a": "unfinis')]
        client = scripted_client(_simple(*body, stop_reason="max_tokens"))
        response = await _provider(client).complete(_request())
        assert response.tool_uses == ()
        assert response.text == "partial"
        assert response.stop_reason is StopReason.MAX_TOKENS


class TestUsage:
    """Each count on its own, from the final message events, never double-counted."""

    async def _usage(self, start: dict[str, Any], delta: dict[str, Any]) -> Usage:
        client = scripted_client(_simple(*text_block(0, "x"), start=start, delta=delta))
        return _stopped(await _events(_provider(client))).usage

    async def test_input_tokens_alone(self) -> None:
        got = await self._usage(usage(input_tokens=42), {"output_tokens": 0})
        assert got == Usage(
            input_tokens=42, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
        )

    async def test_output_tokens_alone(self) -> None:
        got = await self._usage(usage(output_tokens=1), {"output_tokens": 17})
        assert got == Usage(
            input_tokens=0, output_tokens=17, cache_read_tokens=0, cache_write_tokens=0
        )

    async def test_cache_read_tokens_alone(self) -> None:
        got = await self._usage(usage(cache_read=900), {"output_tokens": 0})
        assert got == Usage(
            input_tokens=0, output_tokens=0, cache_read_tokens=900, cache_write_tokens=0
        )

    async def test_cache_write_tokens_alone(self) -> None:
        got = await self._usage(usage(cache_write=1200), {"output_tokens": 0})
        assert got == Usage(
            input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=1200
        )

    async def test_the_final_delta_replaces_counts_rather_than_adding(self) -> None:
        start = usage(input_tokens=30, output_tokens=2, cache_read=500, cache_write=40)
        delta = usage(input_tokens=30, output_tokens=88, cache_read=500, cache_write=40)
        got = await self._usage(start, delta)
        assert got == Usage(
            input_tokens=30, output_tokens=88, cache_read_tokens=500, cache_write_tokens=40
        )

    async def test_counts_missing_from_the_delta_keep_their_start_value(self) -> None:
        start = usage(input_tokens=30, output_tokens=2, cache_read=500, cache_write=40)
        delta = {"output_tokens": 9, "input_tokens": None, "cache_read_input_tokens": None}
        got = await self._usage(start, delta)
        assert got == Usage(
            input_tokens=30, output_tokens=9, cache_read_tokens=500, cache_write_tokens=40
        )

    async def test_null_cache_counts_are_zero(self) -> None:
        got = await self._usage(
            usage(input_tokens=5, cache_read=None, cache_write=None), {"output_tokens": 3}
        )
        assert got == Usage(
            input_tokens=5, output_tokens=3, cache_read_tokens=0, cache_write_tokens=0
        )

    async def test_progress_is_reported_after_message_start(self) -> None:
        client = scripted_client(_simple(*text_block(0, "x"), start=usage(7, 1, 3, 2)))
        events = await _events(_provider(client))
        assert isinstance(events[0], UsageReported)
        assert events[0].usage == Usage(
            input_tokens=7, output_tokens=1, cache_read_tokens=3, cache_write_tokens=2
        )

    async def test_complete_reports_the_final_usage_once(self) -> None:
        start = usage(input_tokens=30, output_tokens=2, cache_read=500, cache_write=40)
        delta = usage(input_tokens=30, output_tokens=88, cache_read=500, cache_write=40)
        client = scripted_client(_simple(*text_block(0, "x"), start=start, delta=delta))
        response = await _provider(client).complete(_request())
        assert response.usage.total_tokens == 30 + 88 + 500 + 40


class TestStopReasons:
    @pytest.mark.parametrize(
        ("wire", "expected"),
        [
            ("end_turn", StopReason.END_TURN),
            ("tool_use", StopReason.TOOL_USE),
            ("max_tokens", StopReason.MAX_TOKENS),
            ("model_context_window_exceeded", StopReason.MAX_TOKENS),
            ("stop_sequence", StopReason.STOP_SEQUENCE),
            ("refusal", StopReason.REFUSAL),
            ("pause_turn", StopReason.OTHER),
            ("something_new", StopReason.OTHER),
            (None, StopReason.OTHER),
        ],
    )
    async def test_mapping(self, wire: str | None, expected: StopReason) -> None:
        client = scripted_client(_simple(*text_block(0, "x"), stop_reason=wire))
        assert _stopped(await _events(_provider(client))).stop_reason is expected


class TestBrokenStreams:
    async def _failure(self, events: list[Any]) -> ProviderFailure:
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(scripted_client(events)))
        return caught.value.failure

    async def test_no_message_stop(self) -> None:
        events = _simple(*text_block(0, "x"))[:-1]
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_no_message_start(self) -> None:
        events = _simple(*text_block(0, "x"))[1:]
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_no_message_delta(self) -> None:
        events = [e for e in _simple(*text_block(0, "x")) if e["type"] != "message_delta"]
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_events_after_message_stop(self) -> None:
        events = [*_simple(*text_block(0, "x")), *text_block(1, "late")]
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_a_delta_for_an_unknown_block(self) -> None:
        events = _simple(
            {
                "type": "content_block_delta",
                "index": 3,
                "delta": {"type": "text_delta", "text": "?"},
            }
        )
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_a_tool_block_never_stopped(self) -> None:
        events = _simple(*tool_block(0, "toolu_01", "x", "{}")[:-1], stop_reason="tool_use")
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_a_tool_block_stopped_after_message_delta(self) -> None:
        events = [
            message_start(usage(5)),
            *tool_block(0, "toolu_01", "x", "{}")[:-1],
            *message_end("tool_use", {"output_tokens": 2})[:1],
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
        assert await self._failure(events) is ProviderFailure.PROTOCOL

    async def test_an_error_event_in_the_stream(self) -> None:
        events = [
            message_start(usage(5)),
            {"type": "error", "error": {"type": "overloaded_error", "message": "secret text"}},
        ]
        assert await self._failure(events) is ProviderFailure.UNAVAILABLE

    async def test_an_invalid_request_error_event_is_rejected(self) -> None:
        events = [
            message_start(usage(5)),
            {"type": "error", "error": {"type": "invalid_request_error", "message": "secret"}},
        ]
        assert await self._failure(events) is ProviderFailure.REJECTED


class TestErrors:
    @pytest.mark.parametrize(
        ("error", "failure"),
        [
            (fakes.APITimeoutError("timed out: secret"), ProviderFailure.TIMEOUT),
            (TimeoutError(), ProviderFailure.TIMEOUT),
            (fakes.APIConnectionError("secret"), ProviderFailure.UNAVAILABLE),
            (fakes.RateLimitError("secret", status_code=429), ProviderFailure.UNAVAILABLE),
            (fakes.InternalServerError("secret", status_code=500), ProviderFailure.UNAVAILABLE),
            (fakes.OverloadedError("secret", status_code=529), ProviderFailure.UNAVAILABLE),
            (fakes.APIStatusError("secret", status_code=503), ProviderFailure.UNAVAILABLE),
            (fakes.APIStatusError("secret", status_code=408), ProviderFailure.UNAVAILABLE),
            (fakes.BadRequestError("secret", status_code=400), ProviderFailure.REJECTED),
            (fakes.AuthenticationError("secret", status_code=401), ProviderFailure.REJECTED),
            (fakes.APIStatusError("secret", status_code=404), ProviderFailure.REJECTED),
            (fakes.APIError("secret"), ProviderFailure.REJECTED),
            (KeyError("bug"), ProviderFailure.PROTOCOL),
        ],
    )
    def test_classification(self, error: BaseException, failure: ProviderFailure) -> None:
        assert failure_for(error) is failure

    @pytest.mark.parametrize("when", ["create", "stream"])
    async def test_errors_become_provider_failures_without_provider_text(self, when: str) -> None:
        error = fakes.RateLimitError("rate limited for prompt: SECRET-MARKER", status_code=429)
        if when == "create":
            client = scripted_client(error)
        else:
            client = scripted_client([message_start(usage(5)), error])
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        fakes.assert_clean(caught.value, "SECRET-MARKER")

    async def test_a_non_sdk_error_mid_stream_is_a_clean_protocol_failure(self) -> None:
        client = scripted_client([message_start(usage(5)), KeyError("SECRET-MARKER")])
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.PROTOCOL
        fakes.assert_clean(caught.value, "SECRET-MARKER")

    async def test_invalid_tool_json_is_a_clean_protocol_failure(self) -> None:
        body = [*tool_block(0, "toolu_01", "x", '{"SECRET-MARKER": '), *text_block(1, "more")]
        client = scripted_client(_simple(*body, stop_reason="tool_use"))
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.PROTOCOL
        fakes.assert_clean(caught.value, "SECRET-MARKER")

    async def test_a_failing_close_does_not_replace_or_chain_the_failure(self) -> None:
        stream = fakes.FakeStream(
            [message_start(usage(5)), fakes.RateLimitError("SECRET-MARKER", status_code=429)],
            close_error=RuntimeError("SECRET-MARKER on close"),
        )
        client = fakes.FakeClient(lambda params: stream)
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client))
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        fakes.assert_clean(caught.value, "SECRET-MARKER")
        assert stream.closed

    async def test_a_failing_close_after_a_whole_stream_is_not_raised(self) -> None:
        stream = fakes.FakeStream(
            _simple(*text_block(0, "fine")), close_error=RuntimeError("SECRET-MARKER")
        )
        client = fakes.FakeClient(lambda params: stream)
        response = await _provider(client).complete(_request())
        assert response.text == "fine"

    async def test_error_event_text_is_not_carried(self) -> None:
        events = [
            message_start(usage(5)),
            {"type": "error", "error": {"type": "api_error", "message": "SECRET-MARKER"}},
        ]
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(scripted_client(events)))
        fakes.assert_clean(caught.value, "SECRET-MARKER")

    async def test_cancellation_is_not_swallowed(self) -> None:
        import asyncio

        client = scripted_client(asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await _events(_provider(client))


class TestStructuredOutput:
    async def test_schema_is_sent_where_the_model_supports_it(self) -> None:
        client = scripted_client(_simple(*text_block(0, '{"answer": "x"}')))
        provider = _provider(client, structured_output=lambda model: True)
        await _events(provider, _request(output_schema=SCHEMA))
        assert client.messages.calls[0]["output_config"] == {
            "format": {"type": "json_schema", "schema": SCHEMA}
        }

    async def test_schema_is_refused_where_the_model_does_not(self) -> None:
        client = scripted_client()
        provider = _provider(client, structured_output=lambda model: False)
        with pytest.raises(ProviderError) as caught:
            await _events(provider, _request(output_schema=SCHEMA))
        assert caught.value.failure is ProviderFailure.REJECTED
        assert client.messages.calls == []

    async def test_no_schema_needs_no_support(self) -> None:
        client = scripted_client(_simple(*text_block(0, "x")))
        provider = _provider(client, structured_output=lambda model: False)
        await _events(provider)
        assert "output_config" not in client.messages.calls[0]

    async def test_by_default_support_comes_from_the_default_catalogue(self) -> None:
        client = scripted_client(_simple(*text_block(0, "{}")))
        await _events(_provider(client), _request(output_schema=SCHEMA))
        assert "output_config" in client.messages.calls[0]

    async def test_a_model_the_default_catalogue_does_not_hold_is_refused(self) -> None:
        client = scripted_client()
        request = ModelRequest(
            model="claude-unknown-9",
            messages=(Message.user("x"),),
            max_output_tokens=10,
            output_schema=SCHEMA,
        )
        with pytest.raises(ProviderError) as caught:
            await _events(_provider(client), request)
        assert caught.value.failure is ProviderFailure.REJECTED
