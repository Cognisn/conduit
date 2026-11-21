"""Amazon Nova's <thinking> spans, through the provider, against synthetic recordings."""

from __future__ import annotations

import json
from typing import Any

from conduit.core.streaming import (
    ProviderStateReported,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import ProviderStateBlock, StopReason
from conduit.testing.replay import load_recording
from tests.unit.providers.bedrock import recordings
from tests.unit.providers.bedrock.recordings import RecordedRuntime
from tests.unit.providers.bedrock.test_provider import built, plain


async def replay(build: Any, region: str, *, streaming: bool = True) -> tuple[Any, RecordedRuntime]:
    runtime = RecordedRuntime(load_recording(recordings.path(build())), region=region)
    provider, _ = await built(runtime, region=region, use_converse_stream=streaming)
    return provider, runtime


def thoughts(events: list[Any]) -> list[str]:
    """The text of each reasoning state block reported, in order."""
    out: list[str] = []
    for event in events:
        if isinstance(event, ProviderStateReported):
            assert event.state.provider_kind == "bedrock"
            assert event.state.state_type == "reasoning_content"
            out.append(json.loads(event.state.payload)["reasoningText"]["text"])
    return out


class TestNovaThinkingRecordings:
    async def test_a_streamed_reply_with_split_tags(self) -> None:
        provider, runtime = await replay(recordings.nova_thinking_split_stream, "us-east-1")
        events = [e async for e in provider.stream(plain("amazon.nova-lite-v1:0"))]
        assert thoughts(events) == [recordings.NOVA_THOUGHT]
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == recordings.NOVA_ANSWER
        assert isinstance(events[-1], StreamStopped)
        runtime.replay.assert_exhausted()

    async def test_the_assembled_response_holds_reasoning_apart_from_the_answer(self) -> None:
        provider, runtime = await replay(recordings.nova_thinking_split_stream, "us-east-1")
        response = await provider.complete(plain("amazon.nova-lite-v1:0"))
        assert response.text == recordings.NOVA_ANSWER
        states = [b for b in response.content if isinstance(b, ProviderStateBlock)]
        assert len(states) == 1
        assert "thinking" not in response.text
        runtime.replay.assert_exhausted()

    async def test_spans_either_side_of_a_streamed_tool_call(self) -> None:
        provider, runtime = await replay(recordings.nova_thinking_tool_stream, "us-east-1")
        events = [e async for e in provider.stream(plain("amazon.nova-lite-v1:0", tools=True))]
        assert thoughts(events) == ["I should look the host up.", "A second lookup helps."]
        assert [type(e).__name__ for e in events] == [
            "ProviderStateReported",
            "ToolCallStarted",
            "ToolCallFinished",
            "ProviderStateReported",
            "ToolCallStarted",
            "ToolCallFinished",
            "StreamStopped",
        ]
        assert events[2] == ToolCallFinished(
            id="tooluse_example0001", name="lookup_host", input={"host": "host-a.example.com"}
        )
        assert events[4] == ToolCallStarted(id="tooluse_example0002", name="lookup_host")
        assert not [e for e in events if isinstance(e, TextDelta)]
        runtime.replay.assert_exhausted()

    async def test_a_whole_converse_reply_with_several_spans_and_a_tool_call(self) -> None:
        provider, runtime = await replay(
            recordings.nova_thinking_converse, "us-east-1", streaming=False
        )
        events = [e async for e in provider.stream(plain("amazon.nova-pro-v1:0", tools=True))]
        assert thoughts(events) == ["First thought.", "Second thought.", "Third thought."]
        assert [e.text for e in events if isinstance(e, TextDelta)] == [
            "Checking now.\n",
            "One moment.",
        ]
        stopped = events[-1]
        assert isinstance(stopped, StreamStopped)
        assert stopped.stop_reason is StopReason.TOOL_USE
        runtime.replay.assert_exhausted()

    async def test_a_reply_with_no_thinking_is_plain_text(self) -> None:
        provider, runtime = await replay(recordings.nova_plain_stream, "us-east-1")
        events = [e async for e in provider.stream(plain("amazon.nova-lite-v1:0"))]
        assert thoughts(events) == []
        assert [e.text for e in events if isinstance(e, TextDelta)] == [
            "host-a.example.com ",
            "stopped answering at 09:00.",
        ]
        runtime.replay.assert_exhausted()

    async def test_a_reply_cut_off_inside_a_span_is_reasoning_not_answer_text(self) -> None:
        provider, runtime = await replay(recordings.nova_thinking_unclosed_stream, "us-east-1")
        events = [e async for e in provider.stream(plain("amazon.nova-lite-v1:0"))]
        assert thoughts(events) == ["The notes name one host and one time. I sho"]
        assert not [e for e in events if isinstance(e, TextDelta)]
        stopped = events[-1]
        assert isinstance(stopped, StreamStopped)
        assert stopped.stop_reason is StopReason.MAX_TOKENS
        runtime.replay.assert_exhausted()

    async def test_another_model_that_mentions_the_tag_is_untouched(self) -> None:
        provider, runtime = await replay(recordings.deepseek_tag_mention, "ap-southeast-2")
        events = [e async for e in provider.stream(plain("deepseek.v3.2"))]
        assert thoughts(events) == []
        texts = [e.text for e in events if isinstance(e, TextDelta)]
        # Not parsed and not held back: the first chunk ends mid-way through a tag.
        assert texts == [recordings.TAG_MENTION[:20], recordings.TAG_MENTION[20:]]
        assert "".join(texts) == recordings.TAG_MENTION
        runtime.replay.assert_exhausted()
