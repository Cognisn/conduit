"""Synthetic recorded bedrock-mantle interactions, written to tests/recordings/bedrock_mantle.

Every value here is invented: model output, identifiers and token counts. The requests
are the SDK parameters the adapter sends to the mantle client's ``messages.create``;
the responses are Messages API stream events in the wire format, which is the same on
bedrock-mantle as on Anthropic's own API. Regenerate the files with
``python -m tests.unit.providers.bedrock_mantle.recordings`` from the repository root;
``test_recordings.py`` checks the committed files match.

- ``mantle-sonnet-5-melbourne-cached``: Claude Sonnet 5 in-region in ap-southeast-4,
  two calls sharing a cached system prompt (written, then read).
- ``mantle-haiku-4-5-tool-uncached``: Claude Haiku 4.5, which the catalogue says does not
  cache on bedrock-mantle, so no ``cache_control`` is sent; a tool call with a thinking
  block that goes back unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, save_recording

from ..anthropic.fakes import (
    message_end,
    message_start,
    text_block,
    thinking_block,
    tool_block,
    usage,
)

DIRECTORY = Path(__file__).resolve().parents[3] / "recordings" / "bedrock_mantle"
PROVIDER = "bedrock_mantle"
SONNET_5 = "anthropic.claude-sonnet-5"
HAIKU_4_5 = "anthropic.claude-haiku-4-5"
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


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _params(model: str, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 1024,
        "messages": messages,
        "stream": True,
        "timeout": 120.0,
        **extra,
    }


def melbourne_cached() -> Recording:
    """Two calls sharing a cached system prompt: the first writes it, the second reads it."""
    system = [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}]
    first = _params(SONNET_5, [_user("Say hello.")], system=system)
    second = _params(
        SONNET_5,
        [
            _user("Say hello."),
            {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
            _user("And goodbye."),
        ],
        system=system,
    )
    return Recording(
        name="mantle-sonnet-5-melbourne-cached",
        provider=PROVIDER,
        interactions=(
            Interaction(
                request=first,
                response=[
                    message_start(
                        usage(input_tokens=11, output_tokens=1, cache_write=1536), model=SONNET_5
                    ),
                    *text_block(0, "Hel", "lo."),
                    *message_end("end_turn", usage(11, 4, cache_read=0, cache_write=1536)),
                ],
            ),
            Interaction(
                request=second,
                response=[
                    message_start(
                        usage(input_tokens=23, output_tokens=1, cache_read=1536), model=SONNET_5
                    ),
                    *text_block(0, "Good", "bye."),
                    *message_end("end_turn", usage(23, 5, cache_read=1536, cache_write=0)),
                ],
            ),
        ),
    )


def haiku_tool_uncached() -> Recording:
    """A tool call assembled from JSON deltas, then an answer from the tool's result.

    The request carries no ``cache_control`` although the conduit request asks for
    breakpoints: the catalogue says Haiku 4.5 does not cache on bedrock-mantle. The
    first response's thinking block goes back unchanged, before the call.
    """
    system = [{"type": "text", "text": SYSTEM}]
    ask = _user("Which network is host-b.example.org on?")
    call = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"},
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "toolu_synthetic_02",
                "name": "lookup_host",
                "input": {"host": "host-b.example.org"},
            },
        ],
    }
    result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_synthetic_02",
                "content": [{"type": "text", "text": "host-b.example.org is 198.51.100.7"}],
                "is_error": False,
            }
        ],
    }
    return Recording(
        name="mantle-haiku-4-5-tool-uncached",
        provider=PROVIDER,
        interactions=(
            Interaction(
                request=_params(HAIKU_4_5, [ask], tools=[TOOL], system=system),
                response=[
                    message_start(usage(input_tokens=52, output_tokens=1), model=HAIKU_4_5),
                    *thinking_block(0),
                    *text_block(1, "Checking."),
                    *tool_block(
                        2, "toolu_synthetic_02", "lookup_host", '{"host": ', '"host-b.example.org"}'
                    ),
                    *message_end("tool_use", usage(52, 29)),
                ],
            ),
            Interaction(
                request=_params(HAIKU_4_5, [ask, call, result], tools=[TOOL], system=system),
                response=[
                    message_start(usage(input_tokens=88, output_tokens=1), model=HAIKU_4_5),
                    *text_block(0, "It is on 198.51.100.0/24."),
                    *message_end("end_turn", usage(88, 13)),
                ],
            ),
        ),
    )


def all_recordings() -> list[Recording]:
    return [melbourne_cached(), haiku_tool_uncached()]


def path_for(recording: Recording) -> Path:
    return DIRECTORY / f"{recording.name}.json"


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for recording in all_recordings():
        save_recording(recording, path_for(recording))


if __name__ == "__main__":
    main()
