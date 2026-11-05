"""The adapter against recorded (synthetic) Anthropic interactions, replayed offline."""

from __future__ import annotations

import pytest

from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.anthropic import AnthropicProvider
from conduit.testing.replay import Recording, Replay, load_recording

from . import recordings
from .fakes import replay_client

SYSTEM = (TextBlock(text=recordings.SYSTEM), CacheBreakpoint())


def _load(recording: Recording) -> Recording:
    return load_recording(recordings.path_for(recording))


@pytest.mark.parametrize("recording", recordings.all_recordings(), ids=lambda r: r.name)
def test_committed_recordings_are_current_and_scrubbed(recording: Recording) -> None:
    assert _load(recording) == recording


def _request(*messages: Message, tools: tuple[ToolSpec, ...] = ()) -> ModelRequest:
    return ModelRequest(
        model=recordings.MODEL,
        system=SYSTEM,
        messages=messages,
        tools=tools,
        cache_tools=bool(tools),
        max_output_tokens=1024,
    )


async def test_cached_conversation() -> None:
    replay = Replay(_load(recordings.cached_conversation()))
    provider = AnthropicProvider(replay_client(replay), name="claude")

    first = await provider.complete(_request(Message.user("Say hello.")))
    assert first.text == "Hello."
    assert first.usage == Usage(
        input_tokens=9, output_tokens=4, cache_read_tokens=0, cache_write_tokens=2048
    )

    second = await provider.complete(
        _request(
            Message.user("Say hello."), Message.assistant("Hello."), Message.user("And goodbye.")
        )
    )
    assert second.text == "Goodbye."
    assert second.usage == Usage(
        input_tokens=21, output_tokens=5, cache_read_tokens=2048, cache_write_tokens=0
    )
    replay.assert_exhausted()


async def test_tool_round_trip() -> None:
    replay = Replay(_load(recordings.tool_round_trip()))
    provider = AnthropicProvider(replay_client(replay), name="claude")
    tool = ToolSpec.model_validate(recordings.TOOL)
    ask = Message.user("Which network is host-a.example.com on?")

    first = await provider.complete(_request(ask, tools=(tool,)))
    assert first.stop_reason is StopReason.TOOL_USE
    assert first.text == "Checking."
    assert first.tool_uses == (
        ToolUseBlock(
            id="toolu_synthetic_01", name="lookup_host", input={"host": "host-a.example.com"}
        ),
    )
    assert first.usage.cache_write_tokens == 1100

    answer = Message(
        role="user",
        content=(
            ToolResultBlock(
                tool_use_id="toolu_synthetic_01",
                content=(TextBlock(text="host-a.example.com is 192.0.2.10"),),
            ),
        ),
    )
    second = await provider.complete(
        _request(ask, Message(role="assistant", content=first.content), answer, tools=(tool,))
    )
    assert second.text == "It is on 192.0.2.0/24."
    assert second.stop_reason is StopReason.END_TURN
    assert second.usage.cache_read_tokens == 1100
    replay.assert_exhausted()
