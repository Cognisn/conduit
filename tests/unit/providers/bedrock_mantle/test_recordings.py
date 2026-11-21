"""The mantle adapter against recorded (synthetic) interactions, replayed offline."""

from __future__ import annotations

import pytest

from conduit.core.catalogue import default_catalogue
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
from conduit.providers.bedrock_mantle import BedrockMantleProvider
from conduit.testing.replay import Recording, Replay, load_recording

from ..anthropic.fakes import replay_client
from . import recordings

SYSTEM = (TextBlock(text=recordings.SYSTEM), CacheBreakpoint())


def _load(recording: Recording) -> Recording:
    return load_recording(recordings.path_for(recording))


@pytest.mark.parametrize("recording", recordings.all_recordings(), ids=lambda r: r.name)
def test_committed_recordings_are_current_and_scrubbed(recording: Recording) -> None:
    assert _load(recording) == recording


def _provider(replay: Replay) -> BedrockMantleProvider:
    return BedrockMantleProvider(
        replay_client(replay), name="mantle_melbourne", catalogue=default_catalogue()
    )


def _request(model: str, *messages: Message, tools: tuple[ToolSpec, ...] = ()) -> ModelRequest:
    return ModelRequest(
        model=model,
        system=SYSTEM,
        messages=messages,
        tools=tools,
        cache_tools=bool(tools),
        max_output_tokens=1024,
    )


async def test_sonnet_5_in_melbourne_writes_then_reads_its_cache() -> None:
    replay = Replay(_load(recordings.melbourne_cached()))
    provider = _provider(replay)

    first = await provider.complete(_request(recordings.SONNET_5, Message.user("Say hello.")))
    assert first.text == "Hello."
    assert first.usage == Usage(
        input_tokens=11, output_tokens=4, cache_read_tokens=0, cache_write_tokens=1536
    )

    second = await provider.complete(
        _request(
            recordings.SONNET_5,
            Message.user("Say hello."),
            Message.assistant("Hello."),
            Message.user("And goodbye."),
        )
    )
    assert second.text == "Goodbye."
    assert second.usage == Usage(
        input_tokens=23, output_tokens=5, cache_read_tokens=1536, cache_write_tokens=0
    )
    replay.assert_exhausted()


async def test_haiku_4_5_tool_round_trip_sends_no_cache_control() -> None:
    replay = Replay(_load(recordings.haiku_tool_uncached()))
    provider = _provider(replay)
    tool = ToolSpec.model_validate(recordings.TOOL)
    ask = Message.user("Which network is host-b.example.org on?")

    first = await provider.complete(_request(recordings.HAIKU_4_5, ask, tools=(tool,)))
    assert first.stop_reason is StopReason.TOOL_USE
    assert first.text == "Checking."
    assert first.tool_uses == (
        ToolUseBlock(
            id="toolu_synthetic_02", name="lookup_host", input={"host": "host-b.example.org"}
        ),
    )
    assert first.usage == Usage(
        input_tokens=52, output_tokens=29, cache_read_tokens=0, cache_write_tokens=0
    )

    answer = Message(
        role="user",
        content=(
            ToolResultBlock(
                tool_use_id="toolu_synthetic_02",
                content=(TextBlock(text="host-b.example.org is 198.51.100.7"),),
            ),
        ),
    )
    second = await provider.complete(
        _request(
            recordings.HAIKU_4_5,
            ask,
            Message(role="assistant", content=first.content),
            answer,
            tools=(tool,),
        )
    )
    assert second.text == "It is on 198.51.100.0/24."
    assert second.stop_reason is StopReason.END_TURN
    replay.assert_exhausted()
