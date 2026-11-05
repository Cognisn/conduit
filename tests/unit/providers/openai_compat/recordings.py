"""The adapter's recorded responses, built here and saved through ``save_recording``.

Every recording is synthetic: invented prompts, ``chatcmpl-example`` ids, no keys,
accounts, hosts or real prompts. Each request is the Chat Completions body the adapter
sends, written out literally so a change in the mapping fails replay. Regenerate with
``python -m tests.unit.providers.openai_compat.recordings`` from the repository root;
a test checks the committed files equal what this module builds.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, load_recording, save_recording

DIRECTORY = Path(__file__).resolve().parents[3] / "recordings" / "openai_compat"

PNG = b"\x89PNG\r\n\x1a\nsynthetic-image"
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
    "required": ["city", "celsius"],
    "additionalProperties": False,
}


def _chunk(
    delta: dict[str, Any] | None = None, finish: str | None = None, *, model: str
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-example",
        "object": "chat.completion.chunk",
        "created": 1793750400,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        "usage": None,
    }


def _usage(
    *, model: str, prompt: int, completion: int, cached: int | None, reasoning: int = 0
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "id": "chatcmpl-example",
        "object": "chat.completion.chunk",
        "created": 1793750400,
        "model": model,
        "choices": [],
        "usage": usage,
    }


def _body(model: str, messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
        **fields,
    }


SYSTEM = {"role": "system", "content": "You are a concise assistant."}


def openai_text_cached() -> Recording:
    """OpenAI: a text answer whose prompt was mostly read from the cache."""
    model = "gpt-5"
    return Recording(
        name="openai-text-cached",
        provider="openai_compat",
        interactions=(
            Interaction(
                request=_body(
                    model,
                    [SYSTEM, {"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
                ),
                response=[
                    _chunk({"role": "assistant", "content": ""}, model=model),
                    _chunk({"content": "Hello"}, model=model),
                    _chunk({"content": ", there."}, model=model),
                    _chunk({}, "stop", model=model),
                    _usage(model=model, prompt=2048, completion=24, cached=1920, reasoning=16),
                ],
            ),
        ),
    )


def openai_parallel_tool_calls() -> Recording:
    """OpenAI: two tool calls in one turn, their arguments streamed in pieces."""
    model = "gpt-5"
    call = {"type": "function"}
    return Recording(
        name="openai-parallel-tool-calls",
        provider="openai_compat",
        interactions=(
            Interaction(
                request=_body(
                    model,
                    [
                        SYSTEM,
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Weather in Sydney and Hobart?"}],
                        },
                    ],
                    tools=[WEATHER_TOOL],
                ),
                response=[
                    _chunk({"role": "assistant", "content": None}, model=model),
                    _chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_example_1",
                                    **call,
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        model=model,
                    ),
                    _chunk(
                        {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Syd'}}]},
                        model=model,
                    ),
                    _chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_example_2",
                                    **call,
                                    "function": {"name": "get_weather", "arguments": '{"ci'},
                                }
                            ]
                        },
                        model=model,
                    ),
                    _chunk(
                        {"tool_calls": [{"index": 0, "function": {"arguments": 'ney"}'}}]},
                        model=model,
                    ),
                    _chunk(
                        {"tool_calls": [{"index": 1, "function": {"arguments": 'ty": "Hobart"}'}}]},
                        model=model,
                    ),
                    _chunk({}, "tool_calls", model=model),
                    _usage(model=model, prompt=310, completion=42, cached=0),
                ],
            ),
        ),
    )


def openai_tool_results() -> Recording:
    """OpenAI: the turn after the calls, with a result, an error result and an image."""
    model = "gpt-5"
    return Recording(
        name="openai-tool-results",
        provider="openai_compat",
        interactions=(
            Interaction(
                request=_body(
                    model,
                    [
                        SYSTEM,
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Weather in Sydney and Hobart?"}],
                        },
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_example_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Sydney"}',
                                    },
                                },
                                {
                                    "id": "call_example_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Hobart"}',
                                    },
                                },
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_example_1",
                            "content": "24 C, sunny",
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_example_2",
                            "content": "Error: weather service unavailable",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Image from tool result call_example_1:"},
                                {"type": "image_url", "image_url": {"url": PNG_URL}},
                            ],
                        },
                    ],
                    tools=[WEATHER_TOOL],
                ),
                response=[
                    _chunk({"role": "assistant", "content": ""}, model=model),
                    _chunk({"content": "Sydney is 24 C and sunny; "}, model=model),
                    _chunk({"content": "Hobart is unavailable."}, model=model),
                    _chunk({}, "stop", model=model),
                    _usage(model=model, prompt=420, completion=18, cached=None),
                ],
            ),
        ),
    )


def xai_text_cached() -> Recording:
    """X.AI: a text answer, cached tokens reported as OpenAI does (xAI's own example)."""
    model = "grok-4.3"
    return Recording(
        name="xai-text-cached",
        provider="openai_compat",
        interactions=(
            Interaction(
                request=_body(
                    model,
                    [SYSTEM, {"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
                ),
                response=[
                    _chunk({"role": "assistant", "content": "Hello"}, model=model),
                    _chunk({"content": "."}, model=model),
                    _chunk({}, "stop", model=model),
                    _usage(model=model, prompt=125, completion=2, cached=98),
                ],
            ),
        ),
    )


def foundry_structured_output() -> Recording:
    """Microsoft Foundry: a deployment answering to a JSON schema, cut off by length."""
    model = "example-gpt-deployment"
    return Recording(
        name="foundry-structured-output",
        provider="openai_compat",
        interactions=(
            Interaction(
                request=_body(
                    model,
                    [
                        SYSTEM,
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Weather in Hobart as JSON."}],
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "output", "schema": ANSWER_SCHEMA, "strict": True},
                    },
                ),
                response=[
                    _chunk({"role": "assistant", "content": '{"city": "Hob'}, model=model),
                    _chunk({"content": 'art", "celsius": 11'}, model=model),
                    _chunk({"content": "}"}, model=model),
                    _chunk({}, "stop", model=model),
                    _usage(model=model, prompt=1100, completion=12, cached=1024),
                ],
            ),
        ),
    )


BUILDERS: dict[str, Callable[[], Recording]] = {
    builder().name: builder
    for builder in (
        openai_text_cached,
        openai_parallel_tool_calls,
        openai_tool_results,
        xai_text_cached,
        foundry_structured_output,
    )
}


def path(name: str) -> Path:
    return DIRECTORY / f"{name}.json"


def load(name: str) -> Recording:
    return load_recording(path(name))


def write_all() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        save_recording(builder(), path(name))


if __name__ == "__main__":
    write_all()
