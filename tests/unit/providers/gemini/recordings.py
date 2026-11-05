"""Synthetic Gemini recordings, and a fake google-genai client that replays them.

The files in ``tests/recordings/gemini/`` are written by this module through
``conduit.testing.replay.save_recording``, which refuses anything unscrubbed. They are
synthetic: prompts, hosts, identifiers and signatures are invented, and each chunk is
the JSON form of the SDK's ``GenerateContentResponse`` (snake_case, bytes as base64).
Regenerate them with

    python -m tests.unit.providers.gemini.recordings

A test checks that the committed files equal what this module builds.

Each recorded request is the ``generate_content_stream`` parameters the adapter
built (``model``, ``contents``, ``config``) in JSON form; each response is the list
of chunks, or ``{"raise": <status>}`` for an API error.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, Replay, save_recording

DIRECTORY = Path(__file__).resolve().parents[3] / "recordings" / "gemini"

MODEL = "gemini-3.8-flash"
SYSTEM = "You answer questions about the example.com inventory."
QUESTION = "Is host-a.example.com answering on 192.0.2.10?"
RESULT = "host-a.example.com: up, last seen 09:00 on 192.0.2.10"
SIGNATURE = base64.b64encode(b"synthetic-thought-signature-0001").decode("ascii")
SCHEMA = {
    "type": "object",
    "properties": {"host": {"type": "string"}},
    "required": ["host"],
}
TOOLS = [
    {
        "function_declarations": [
            {
                "name": "lookup_host",
                "description": "Look a host up in the inventory.",
                "parameters_json_schema": SCHEMA,
            }
        ]
    }
]
CALL = {"id": "fc_example0001", "name": "lookup_host", "args": {"host": "host-a.example.com"}}


def jsonable(value: Any) -> Any:
    """``value`` with bytes as base64 text, as recordings hold requests."""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value


def _chunk(
    *parts: dict[str, Any], finish: str | None = None, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": list(parts)}, "index": 0}
    if finish is not None:
        candidate["finish_reason"] = finish
    chunk: dict[str, Any] = {
        "candidates": [candidate],
        "model_version": MODEL,
        "response_id": "resp_example0001",
    }
    if usage is not None:
        chunk["usage_metadata"] = usage
    return chunk


def _request(contents: list[dict[str, Any]], **config: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "contents": contents,
        "config": {
            "max_output_tokens": 512,
            "system_instruction": SYSTEM,
            "tools": TOOLS,
            **config,
        },
    }


FIRST: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": QUESTION}]}]
SECOND: list[dict[str, Any]] = [
    *FIRST,
    {
        "role": "model",
        "parts": [
            {"text": "Checking the inventory."},
            {"function_call": CALL, "thought_signature": SIGNATURE},
        ],
    },
    {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "id": CALL["id"],
                    "name": "lookup_host",
                    "response": {"output": RESULT},
                }
            }
        ],
    },
]


def tool_round_trip() -> Recording:
    """A function call with a thought signature, then the answer (prefix cached)."""
    return Recording(
        name="gemini-tool-round-trip",
        provider="gemini",
        interactions=(
            Interaction(
                request=_request(FIRST),
                response=[
                    _chunk({"text": "Checking "}, usage={"prompt_token_count": 96}),
                    _chunk(
                        {"text": "the inventory."},
                        {"function_call": CALL, "thought_signature": SIGNATURE},
                        finish="STOP",
                        usage={
                            "prompt_token_count": 96,
                            "candidates_token_count": 24,
                            "thoughts_token_count": 40,
                            "total_token_count": 160,
                        },
                    ),
                ],
            ),
            Interaction(
                request=_request(SECOND),
                response=[
                    _chunk({"text": "Yes: host-a.example.com "}),
                    _chunk(
                        {"text": "is up."},
                        finish="STOP",
                        usage={
                            "prompt_token_count": 150,
                            "cached_content_token_count": 96,
                            "candidates_token_count": 9,
                            "total_token_count": 159,
                        },
                    ),
                ],
            ),
        ),
    )


def cut_short() -> Recording:
    """A plain answer stopped by the output limit."""
    request = _request([{"role": "user", "parts": [{"text": "Describe the example.com network."}]}])
    del request["config"]["tools"]
    request["config"]["max_output_tokens"] = 8
    return Recording(
        name="gemini-cut-short",
        provider="gemini",
        interactions=(
            Interaction(
                request=request,
                response=[
                    _chunk({"text": "The example.com network "}),
                    _chunk(
                        {"text": "has two"},
                        finish="MAX_TOKENS",
                        usage={"prompt_token_count": 30, "candidates_token_count": 8},
                    ),
                ],
            ),
        ),
    )


RECORDINGS: dict[str, Callable[[], Recording]] = {
    "tool-round-trip.json": tool_round_trip,
    "cut-short.json": cut_short,
}


class FakeModels:
    """``client.aio.models`` answering from a handler of the request parameters."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self.closed_streams = 0

    async def generate_content_stream(self, **params: Any) -> AsyncIterator[Any]:
        self.calls.append(params)
        chunks = self.handler(params)
        models = self

        async def stream() -> AsyncIterator[Any]:
            try:
                for chunk in chunks:
                    if isinstance(chunk, BaseException):
                        raise chunk
                    yield chunk
            finally:
                models.closed_streams += 1

        return stream()


class FakeAio:
    def __init__(self, models: FakeModels) -> None:
        self.models = models
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    """Stands in for ``google.genai.Client``: only ``aio`` is used by the adapter."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.aio = FakeAio(FakeModels(handler))


def replaying(replay: Replay) -> Callable[[dict[str, Any]], Any]:
    def handle(params: dict[str, Any]) -> Any:
        return replay.next(jsonable(params))

    return handle


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename, build in RECORDINGS.items():
        save_recording(build(), DIRECTORY / filename)


if __name__ == "__main__":
    main()
