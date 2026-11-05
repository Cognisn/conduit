"""Synthetic recorded Anthropic interactions, written to tests/recordings/anthropic.

Every value here is invented: model output, identifiers and token counts. The requests
are the SDK parameters the adapter sends; the responses are stream events in the API's
wire format. Regenerate the files with ``python -m tests.unit.providers.anthropic.recordings``
from the repository root; ``test_recordings.py`` checks the committed files match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, save_recording

from .fakes import message_end, message_start, text_block, thinking_block, tool_block, usage

DIRECTORY = Path(__file__).resolve().parents[3] / "recordings" / "anthropic"
MODEL = "claude-sonnet-5-5"
SYSTEM = "You are a synthetic assistant for conduit's tests. Answer briefly."
TOOL = {
    "name": "lookup_host",
    "description": "Look up a host in the synthetic inventory.",
    "input_schema": {
        "type": "object",
        "properties": {"host": {"type": "string"}},
        "required": ["host"],
    },
}
EPHEMERAL = {"type": "ephemeral"}


def _params(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": 1024,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}],
        "messages": messages,
        "stream": True,
        "timeout": 120.0,
        **extra,
    }


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def cached_conversation() -> Recording:
    """Two calls sharing a cached system prompt: the first writes it, the second reads it."""
    first = _params([_user("Say hello.")])
    second = _params(
        [
            _user("Say hello."),
            {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
            _user("And goodbye."),
        ]
    )
    return Recording(
        name="anthropic-cached-conversation",
        provider="anthropic",
        interactions=(
            Interaction(
                request=first,
                response=[
                    message_start(usage(input_tokens=9, output_tokens=1, cache_write=2048)),
                    *text_block(0, "Hel", "lo."),
                    *message_end("end_turn", usage(9, 4, cache_read=0, cache_write=2048)),
                ],
            ),
            Interaction(
                request=second,
                response=[
                    message_start(usage(input_tokens=21, output_tokens=1, cache_read=2048)),
                    *text_block(0, "Good", "bye."),
                    *message_end("end_turn", usage(21, 5, cache_read=2048, cache_write=0)),
                ],
            ),
        ),
    )


def tool_round_trip() -> Recording:
    """A tool call assembled from JSON deltas, then an answer from the tool's result."""
    ask = _user("Which network is host-a.example.com on?")
    call = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "toolu_synthetic_01",
                "name": "lookup_host",
                "input": {"host": "host-a.example.com"},
            },
        ],
    }
    result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_synthetic_01",
                "content": [{"type": "text", "text": "host-a.example.com is 192.0.2.10"}],
                "is_error": False,
            }
        ],
    }
    tools = [{**TOOL, "cache_control": EPHEMERAL}]
    return Recording(
        name="anthropic-tool-round-trip",
        provider="anthropic",
        interactions=(
            Interaction(
                request=_params([ask], tools=tools),
                response=[
                    message_start(usage(input_tokens=40, output_tokens=1, cache_write=1100)),
                    *thinking_block(0),
                    *text_block(1, "Checking."),
                    *tool_block(
                        2, "toolu_synthetic_01", "lookup_host", '{"host": ', '"host-a.example.com"}'
                    ),
                    *message_end("tool_use", usage(40, 31, cache_read=0, cache_write=1100)),
                ],
            ),
            Interaction(
                request=_params([ask, call, result], tools=tools),
                response=[
                    message_start(usage(input_tokens=75, output_tokens=1, cache_read=1100)),
                    *text_block(0, "It is on 192.0.2.0/24."),
                    *message_end("end_turn", usage(75, 12, cache_read=1100, cache_write=0)),
                ],
            ),
        ),
    )


def all_recordings() -> list[Recording]:
    return [cached_conversation(), tool_round_trip()]


def path_for(recording: Recording) -> Path:
    return DIRECTORY / f"{recording.name}.json"


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for recording in all_recordings():
        save_recording(recording, path_for(recording))


if __name__ == "__main__":
    main()
