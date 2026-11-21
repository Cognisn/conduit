"""Typed stream events, and a non-streaming call defined as the collected stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider, Provider
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    UsageReported,
    collect,
)
from conduit.core.types import (
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)


def _usage(input_tokens: int, output_tokens: int, read: int = 0, write: int = 0) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )


async def _aiter(events: Sequence[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


async def test_collect_assembles_text_tool_use_and_final_usage() -> None:
    events: list[StreamEvent] = [
        UsageReported(usage=_usage(100, 0, 40, 60)),
        TextDelta(text="Let me "),
        TextDelta(text="check."),
        ToolCallStarted(id="toolu_1", name="lookup"),
        UsageReported(usage=_usage(100, 7, 40, 60)),
        ToolCallFinished(id="toolu_1", name="lookup", input={"host": "example.net"}),
        StreamStopped(stop_reason=StopReason.TOOL_USE, usage=_usage(100, 12, 40, 60)),
    ]
    response = await collect(_aiter(events), model="stub-echo")
    assert response.content == (
        TextBlock(text="Let me check."),
        ToolUseBlock(id="toolu_1", name="lookup", input={"host": "example.net"}),
    )
    assert response.stop_reason is StopReason.TOOL_USE
    # Usage comes from the final stream event, never a sum of the progress reports.
    assert response.usage == _usage(100, 12, 40, 60)
    assert response.model == "stub-echo"


async def test_text_after_a_tool_call_is_a_new_block() -> None:
    events: list[StreamEvent] = [
        TextDelta(text="a"),
        ToolCallStarted(id="t", name="n"),
        ToolCallFinished(id="t", name="n", input={}),
        TextDelta(text="b"),
        StreamStopped(stop_reason=StopReason.END_TURN, usage=Usage.zero()),
    ]
    response = await collect(_aiter(events), model="m")
    assert [type(b).__name__ for b in response.content] == [
        "TextBlock",
        "ToolUseBlock",
        "TextBlock",
    ]


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([TextDelta(text="no stop")], id="no-stop"),
        pytest.param(
            [
                StreamStopped(stop_reason=StopReason.END_TURN, usage=Usage.zero()),
                TextDelta(text="late"),
            ],
            id="event-after-stop",
        ),
        pytest.param(
            [
                ToolCallFinished(id="t", name="n", input={}),
                StreamStopped(stop_reason=StopReason.TOOL_USE, usage=Usage.zero()),
            ],
            id="finished-without-start",
        ),
        pytest.param(
            [
                ToolCallStarted(id="t", name="n"),
                ToolCallFinished(id="t", name="other", input={}),
                StreamStopped(stop_reason=StopReason.TOOL_USE, usage=Usage.zero()),
            ],
            id="finished-with-another-name",
        ),
        pytest.param(
            [
                ToolCallStarted(id="t", name="n"),
                StreamStopped(stop_reason=StopReason.TOOL_USE, usage=Usage.zero()),
            ],
            id="started-never-finished",
        ),
    ],
)
async def test_a_malformed_stream_is_a_protocol_failure(events: list[StreamEvent]) -> None:
    with pytest.raises(ProviderError) as raised:
        await collect(_aiter(events), model="m")
    assert raised.value.failure is ProviderFailure.PROTOCOL


class _Fixed(BaseProvider):
    kind = "fixed"

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        return _aiter(self._events)


async def test_complete_is_the_collected_stream() -> None:
    events: list[StreamEvent] = [
        TextDelta(text="hi"),
        StreamStopped(stop_reason=StopReason.END_TURN, usage=_usage(3, 1, 0, 2)),
    ]
    provider = _Fixed(events)
    request = ModelRequest(model="m", messages=(Message.user("x"),), max_output_tokens=10)
    response = await provider.complete(request)
    assert response.text == "hi"
    assert response.usage == _usage(3, 1, 0, 2)
    assert response.model == "m"
    assert isinstance(provider, Provider)
    await provider.aclose()


def test_provider_errors_carry_only_the_failure_class() -> None:
    error = ProviderError(ProviderFailure.TIMEOUT)
    assert str(error) == "model provider call failed: timeout"
