"""Recorded-response replay, for provider tests without a network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conduit.core.errors import ConfigurationError, ReplayMismatch
from conduit.core.streaming import StreamEvent, StreamStopped, TextDelta
from conduit.core.types import Message, ModelRequest, StopReason, Usage
from conduit.testing.replay import (
    Interaction,
    Recording,
    Replay,
    ReplayProvider,
    check_scrubbed,
    load_recording,
    save_recording,
)


def _recording() -> Recording:
    return Recording(
        name="converse-basic",
        provider="bedrock",
        interactions=(
            Interaction(
                request={"modelId": "anthropic.example-v1:0", "messages": [{"role": "user"}]},
                response=[{"contentBlockDelta": {"delta": {"text": "Hi"}}}, {"messageStop": {}}],
            ),
            Interaction(request={"op": "second"}, response={"ok": True}),
        ),
    )


class TestReplay:
    def test_plays_interactions_in_order_when_requests_match(self) -> None:
        replay = Replay(_recording())
        first = replay.next({"messages": [{"role": "user"}], "modelId": "anthropic.example-v1:0"})
        assert first[0]["contentBlockDelta"]["delta"]["text"] == "Hi"
        assert replay.next({"op": "second"}) == {"ok": True}
        replay.assert_exhausted()

    def test_a_different_request_is_a_mismatch(self) -> None:
        replay = Replay(_recording())
        with pytest.raises(ReplayMismatch, match="modelId"):
            replay.next({"modelId": "other", "messages": [{"role": "user"}]})

    def test_running_out_or_leaving_some_unplayed_is_a_mismatch(self) -> None:
        replay = Replay(_recording())
        with pytest.raises(ReplayMismatch):
            replay.assert_exhausted()
        replay.next({"modelId": "anthropic.example-v1:0", "messages": [{"role": "user"}]})
        replay.next({"op": "second"})
        with pytest.raises(ReplayMismatch, match="no more"):
            replay.next({"op": "third"})


class TestFiles:
    def test_save_and_load_round_trip_with_lf(self, tmp_path: Path) -> None:
        path = tmp_path / "converse-basic.json"
        save_recording(_recording(), path)
        assert b"\r\n" not in path.read_bytes()
        assert load_recording(path) == _recording()

    @pytest.mark.parametrize(
        "leak",
        [
            "AKIAABCDEFGHIJKLMNOP",
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "sk-proj-abcdefghijklmnopqrstuvwx",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlLXZhbHVl",
            "someone@cognisn.com",
            "8.8.8.8",
            "arn:aws:bedrock:ap-southeast-2:987654321098:inference-profile/x",
            "account 987654321098 in the response",
            '"OwnerId": "111122223333"',
            "2606:4700:4700::1111",
            "fe80::1ff:fe23:4567:890a",
            "[2a00:1450:4001:82a::200e]:443",
        ],
    )
    def test_an_unscrubbed_recording_is_refused(self, leak: str, tmp_path: Path) -> None:
        recording = Recording(
            name="leaky",
            provider="bedrock",
            interactions=(Interaction(request={"note": f"value {leak}"}, response={}),),
        )
        with pytest.raises(ConfigurationError):
            check_scrubbed(recording)
        with pytest.raises(ConfigurationError):
            save_recording(recording, tmp_path / "leaky.json")
        (tmp_path / "raw.json").write_text(
            recording.model_dump_json(), encoding="utf-8", newline="\n"
        )
        with pytest.raises(ConfigurationError):
            load_recording(tmp_path / "raw.json")

    @pytest.mark.parametrize(
        "allowed",
        [
            "analyst@example.com",
            "192.0.2.10",
            "198.51.100.1",
            "203.0.113.200",
            "127.0.0.1",
            "arn:aws:bedrock:ap-southeast-2:123456789012:inference-profile/x",
            "account 123456789012 in the response",
            "2001:db8::10",
            "2001:DB8:0:0:1::1",
            "::1",
            "the time was 12:30:45",
            "1234567890123 is thirteen digits; 12345678901 is eleven",
            "a version of 0.123456789012",
        ],
    )
    def test_synthetic_values_pass(self, allowed: str) -> None:
        check_scrubbed(
            Recording(
                name="ok",
                provider="bedrock",
                interactions=(Interaction(request={"note": allowed}, response={}),),
            )
        )


class TestReplayProvider:
    async def test_replays_recorded_stream_events(self) -> None:
        events: list[StreamEvent] = [
            TextDelta(text="Hello"),
            StreamStopped(stop_reason=StopReason.END_TURN, usage=Usage.zero()),
        ]
        request = ModelRequest(
            model="stub-echo", messages=(Message.user("x"),), max_output_tokens=5
        )
        recording = ReplayProvider.record(request, events, name="hello")
        assert json.loads(recording.model_dump_json())["interactions"][0]["response"][0] == {
            "type": "text_delta",
            "text": "Hello",
        }
        provider = ReplayProvider(recording)
        assert (await provider.complete(request)).text == "Hello"
        provider.replay.assert_exhausted()
