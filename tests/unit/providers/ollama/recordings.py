"""Synthetic Ollama recordings: /api/chat bodies and the NDJSON lines answering them.

The files in ``tests/recordings/ollama/`` are written by this module through
``conduit.testing.replay.save_recording``, which refuses anything unscrubbed. They are
synthetic: prompts, hosts and identifiers are invented, and the responses follow the
/api/chat streaming shape (one JSON object per line, the last with ``done: true`` and
the token counts). Regenerate them with

    python -m tests.unit.providers.ollama.recordings

A test checks that the committed files equal what this module builds.

Each recorded request is ``{"method", "path", "body"}`` as the adapter sent it; each
response is ``{"status", "lines"}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, save_recording

DIRECTORY = Path(__file__).resolve().parents[3] / "recordings" / "ollama"

MODEL = "llama3.1:8b"
CONTEXT = 131072
SYSTEM = "You answer questions about the example.com inventory."
QUESTION = "Is host-a.example.com answering on 192.0.2.10?"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_host",
        "description": "Look a host up in the inventory.",
        "parameters": {
            "type": "object",
            "properties": {"host": {"type": "string"}},
            "required": ["host"],
        },
    },
}
CALL = {"function": {"name": "lookup_host", "arguments": {"host": "host-a.example.com"}}}
RESULT = "host-a.example.com: up, last seen 09:00 on 192.0.2.10"


def _line(done: bool, content: str = "", **fields: Any) -> str:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if "tool_calls" in fields:
        message["tool_calls"] = fields.pop("tool_calls")
    return json.dumps(
        {
            "model": MODEL,
            "created_at": "2026-11-04T00:00:00.000000Z",
            "message": message,
            "done": done,
            **fields,
        },
        sort_keys=True,
    )


def _final(reason: str, prompt: int, output: int, cached: int | None = None) -> str:
    counts: dict[str, Any] = {"prompt_eval_count": prompt, "eval_count": output}
    if cached is not None:
        counts["prompt_eval_cached_count"] = cached
    return _line(
        True,
        done_reason=reason,
        total_duration=912000000,
        load_duration=21000000,
        prompt_eval_duration=310000000,
        eval_duration=540000000,
        **counts,
    )


def _body(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, *messages],
        "stream": True,
        "options": {"num_predict": 512, "num_ctx": CONTEXT},
        "tools": [TOOL],
        **extra,
    }


FIRST: list[dict[str, Any]] = [{"role": "user", "content": QUESTION}]
SECOND: list[dict[str, Any]] = [
    *FIRST,
    {"role": "assistant", "content": "Checking the inventory.", "tool_calls": [CALL]},
    {"role": "tool", "tool_name": "lookup_host", "content": RESULT},
]


def tool_round_trip() -> Recording:
    """A tool call, then the answer once the tool result is sent (prefix cached)."""
    return Recording(
        name="ollama-tool-round-trip",
        provider="ollama",
        interactions=(
            Interaction(
                request={"method": "POST", "path": "/api/chat", "body": _body(FIRST)},
                response={
                    "status": 200,
                    "lines": [
                        _line(False, "Checking "),
                        _line(False, "the inventory."),
                        _line(False, tool_calls=[CALL]),
                        _final("stop", 84, 31),
                    ],
                },
            ),
            Interaction(
                request={"method": "POST", "path": "/api/chat", "body": _body(SECOND)},
                response={
                    "status": 200,
                    "lines": [
                        _line(False, "Yes: host-a.example.com "),
                        _line(False, "is up."),
                        _final("stop", 131, 12, cached=84),
                    ],
                },
            ),
        ),
    )


def cut_short() -> Recording:
    """A plain answer stopped by the output limit."""
    body = _body([{"role": "user", "content": "Describe the example.com network."}])
    del body["tools"]
    body["options"]["num_predict"] = 8
    return Recording(
        name="ollama-cut-short",
        provider="ollama",
        interactions=(
            Interaction(
                request={"method": "POST", "path": "/api/chat", "body": body},
                response={
                    "status": 200,
                    "lines": [
                        _line(False, "The example.com network "),
                        _line(False, "has two"),
                        _final("length", 30, 8),
                    ],
                },
            ),
        ),
    )


RECORDINGS = {"tool-round-trip.json": tool_round_trip, "cut-short.json": cut_short}


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename, build in RECORDINGS.items():
        save_recording(build(), DIRECTORY / filename)


if __name__ == "__main__":
    main()
