"""Opaque provider state: a content block and a stream event that carry it, never as text.

A provider can return state that must be sent back unchanged on the next request
(Claude's thinking blocks with their signatures, Gemini's thought signatures). conduit
carries it as a ``ProviderStateBlock``: the provider kind that made it, its type, and
an opaque payload. Each adapter round-trips its own and ignores every other kind's.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from pydantic import TypeAdapter, ValidationError

from conduit.core.errors import ProviderError
from conduit.core.streaming import (
    ProviderStateReported,
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    collect,
)
from conduit.core.types import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderStateBlock,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from conduit.providers.stub import StubProvider, StubTurn, scripted

PAYLOAD = b"\x00\xffopaque synthetic state\x80\x81"


def _state(
    kind: str = "stub", state_type: str = "thinking", payload: bytes = PAYLOAD
) -> ProviderStateBlock:
    return ProviderStateBlock(provider_kind=kind, state_type=state_type, payload=payload)


async def _aiter(events: Sequence[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


class TestTheBlock:
    def test_it_round_trips_through_json_with_any_bytes(self) -> None:
        block = _state()
        adapter: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        again = adapter.validate_json(adapter.dump_json(block))
        assert again == block
        assert isinstance(again, ProviderStateBlock)
        assert again.payload == PAYLOAD

    def test_a_message_holding_it_round_trips(self) -> None:
        message = Message(role="assistant", content=(_state(), TextBlock(text="Done.")))
        assert Message.model_validate_json(message.model_dump_json()) == message

    def test_the_payload_is_never_rendered(self) -> None:
        block = _state(payload=b"zz-visible-marker-zz")
        assert "zz-visible-marker-zz" not in repr(block)
        assert "zz-visible-marker-zz" not in str(block)
        message = Message(role="assistant", content=(block,))
        assert "zz-visible-marker-zz" not in repr(message)

    def test_it_is_never_text(self) -> None:
        response = ModelResponse(
            content=(_state(), TextBlock(text="answer")),
            stop_reason=StopReason.END_TURN,
            usage=Usage.zero(),
            model="stub-echo",
        )
        assert response.text == "answer"
        assert response.provider_state == (_state(),)

    @pytest.mark.parametrize(
        "fields",
        [
            {"provider_kind": "Not A Kind", "state_type": "thinking", "payload": b""},
            {"provider_kind": "stub", "state_type": "has space", "payload": b""},
            {"provider_kind": "stub", "state_type": "", "payload": b""},
            {"provider_kind": "stub", "state_type": "thinking", "payload": "text"},
        ],
    )
    def test_kind_and_type_are_identifiers_and_the_payload_bytes(
        self, fields: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            ProviderStateBlock.model_validate(fields, strict=True)

    def test_a_user_message_cannot_carry_provider_state(self) -> None:
        with pytest.raises(ValidationError) as caught:
            Message(role="user", content=(_state(),))
        assert "opaque synthetic" not in str(caught.value)

    def test_a_message_gives_one_kind_its_own_state(self) -> None:
        mine, theirs = _state("stub"), _state("gemini", "thought_signature")
        message = Message(role="assistant", content=(mine, TextBlock(text="x"), theirs))
        assert message.provider_state("stub") == (mine,)
        assert message.provider_state("gemini") == (theirs,)
        assert message.provider_state("ollama") == ()


class TestTheStreamEvent:
    def test_it_round_trips_as_a_stream_event(self) -> None:
        adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)
        event = ProviderStateReported(state=_state())
        again = adapter.validate_json(adapter.dump_json(event))
        assert again == event

    async def test_collect_keeps_it_in_stream_order(self) -> None:
        first, second = _state(state_type="thinking"), _state(state_type="redacted")
        response = await collect(
            _aiter(
                [
                    ProviderStateReported(state=first),
                    TextDelta(text="Hel"),
                    TextDelta(text="lo"),
                    ProviderStateReported(state=second),
                    ToolCallStarted(id="toolu_1", name="lookup"),
                    ToolCallFinished(id="toolu_1", name="lookup", input={"q": "x"}),
                    StreamStopped(stop_reason=StopReason.TOOL_USE, usage=Usage.zero()),
                ]
            ),
            model="stub-echo",
        )
        assert response.content == (
            first,
            TextBlock(text="Hello"),
            second,
            ToolUseBlock(id="toolu_1", name="lookup", input={"q": "x"}),
        )
        assert response.text == "Hello"

    async def test_state_after_the_stop_breaks_the_stream(self) -> None:
        with pytest.raises(ProviderError):
            await collect(
                _aiter(
                    [
                        StreamStopped(stop_reason=StopReason.END_TURN, usage=Usage.zero()),
                        ProviderStateReported(state=_state()),
                    ]
                ),
                model="stub-echo",
            )


class TestTheStub:
    """The stub emits provider state and accepts its own back, ignoring other kinds'."""

    def _provider(self, turns: Sequence[StubTurn]) -> StubProvider:
        return StubProvider(scripted(turns), not_for_production=True)

    async def test_it_streams_a_turn_state_before_the_text(self) -> None:
        provider = self._provider([StubTurn(text="Hi", state=(_state(),))])
        request = ModelRequest(
            model="stub-echo", messages=(Message.user("hello"),), max_output_tokens=16
        )
        events = [e async for e in provider.stream(request)]
        reported = [e for e in events if isinstance(e, ProviderStateReported)]
        assert [e.state for e in reported] == [_state()]
        first_text = next(i for i, e in enumerate(events) if isinstance(e, TextDelta))
        assert events.index(reported[0]) < first_text
        response = await self._provider([StubTurn(text="Hi", state=(_state(),))]).complete(request)
        assert response.provider_state == (_state(),) and response.text == "Hi"

    async def test_it_keeps_its_own_state_and_ignores_others(self) -> None:
        provider = self._provider([StubTurn(text="one"), StubTurn(text="two")])
        mine, theirs = _state("stub"), _state("anthropic", "signature")
        history = (
            Message.user("first"),
            Message(role="assistant", content=(theirs, mine, TextBlock(text="one"))),
            Message.user("second"),
        )
        await provider.complete(
            ModelRequest(model="stub-echo", messages=history[:1], max_output_tokens=16)
        )
        await provider.complete(
            ModelRequest(model="stub-echo", messages=history, max_output_tokens=16)
        )
        assert provider.state_received == [(), (mine,)]
