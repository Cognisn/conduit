"""Requests: conduit's provider-neutral request as a Chat Completions body."""

from __future__ import annotations

import base64
from typing import Any, ClassVar

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from conduit.providers.openai_compat.mapping import build_request

PNG = b"\x89PNG\r\n\x1a\nsynthetic-image"
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model="gpt-5",
        messages=messages or (Message.user("hello"),),
        max_output_tokens=256,
        **fields,
    )


def _body(request: ModelRequest, *, structured_output: bool = False) -> dict[str, Any]:
    return build_request(request, structured_output=structured_output)


class TestBasics:
    def test_a_minimal_request_streams_with_usage_on_the_final_chunk(self) -> None:
        assert _body(_request()) == {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_completion_tokens": 256,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def test_system_blocks_become_one_system_message(self) -> None:
        body = _body(
            _request(system=(TextBlock(text="Be terse."), TextBlock(text=" Use metric units.")))
        )
        assert body["messages"][0] == {
            "role": "system",
            "content": "Be terse. Use metric units.",
        }

    def test_optional_parameters_are_sent_only_when_set(self) -> None:
        body = _body(_request(temperature=0.2, stop_sequences=("END",)))
        assert body["temperature"] == 0.2
        assert body["stop"] == ["END"]
        plain = _body(_request())
        assert "temperature" not in plain
        assert "stop" not in plain
        assert "tools" not in plain
        assert "response_format" not in plain

    def test_the_request_is_not_mutated_by_the_call_context(self) -> None:
        one = _body(_request(run_id="run-1", correlation_id="c-1", task_class="chat"))
        assert one == _body(_request())


class TestCacheBreakpointsAreIgnored:
    def test_breakpoints_anywhere_leave_the_body_unchanged(self) -> None:
        plain = _request(
            Message.user("first"),
            system=(TextBlock(text="System."),),
            tools=(WEATHER,),
        )
        marked = _request(
            Message(role="user", content=(TextBlock(text="first"), CacheBreakpoint())),
            system=(TextBlock(text="System."), CacheBreakpoint()),
            tools=(WEATHER,),
            cache_tools=True,
        )
        assert _body(marked) == _body(plain)


class TestContent:
    def test_images_are_data_urls(self) -> None:
        message = Message(
            role="user",
            content=(TextBlock(text="What is this?"), ImageBlock(media_type="image/png", data=PNG)),
        )
        assert _body(_request(message))["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": PNG_URL}},
                ],
            }
        ]

    def test_assistant_tool_use_becomes_tool_calls(self) -> None:
        assistant = Message(
            role="assistant",
            content=(
                TextBlock(text="Checking."),
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "Hobart"}),
            ),
        )
        assert _body(_request(Message.user("Weather?"), assistant))["messages"][1] == {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Hobart"}'},
                }
            ],
        }

    def test_assistant_with_only_tool_calls_has_null_content(self) -> None:
        assistant = Message(
            role="assistant",
            content=(ToolUseBlock(id="call_1", name="get_weather", input={}),),
        )
        message = _body(_request(Message.user("Weather?"), assistant))["messages"][1]
        assert message["content"] is None
        assert message["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_tool_results_are_tool_messages_before_the_rest_of_the_turn(self) -> None:
        turn = Message(
            role="user",
            content=(
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=(TextBlock(text="18 C"), TextBlock(text=", fine")),
                ),
                ToolResultBlock(
                    tool_use_id="call_2", content=(TextBlock(text="unknown city"),), is_error=True
                ),
                TextBlock(text="And tomorrow?"),
            ),
        )
        messages = _body(_request(turn))["messages"]
        assert messages == [
            {"role": "tool", "tool_call_id": "call_1", "content": "18 C, fine"},
            {"role": "tool", "tool_call_id": "call_2", "content": "Error: unknown city"},
            {"role": "user", "content": [{"type": "text", "text": "And tomorrow?"}]},
        ]

    def test_images_in_tool_results_follow_as_a_user_message(self) -> None:
        turn = Message(
            role="user",
            content=(
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=(TextBlock(text="chart"), ImageBlock(media_type="image/png", data=PNG)),
                ),
            ),
        )
        messages = _body(_request(turn))["messages"]
        assert messages == [
            {"role": "tool", "tool_call_id": "call_1", "content": "chart"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image from tool result call_1:"},
                    {"type": "image_url", "image_url": {"url": PNG_URL}},
                ],
            },
        ]

    def test_tools_are_functions(self) -> None:
        assert _body(_request(tools=(WEATHER,)))["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]


class TestStructuredOutput:
    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def test_a_schema_becomes_a_strict_json_schema_response_format(self) -> None:
        body = _body(_request(output_schema=self.SCHEMA), structured_output=True)
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": self.SCHEMA, "strict": True},
        }

    def test_a_schema_for_a_model_without_structured_output_is_rejected(self) -> None:
        with pytest.raises(ProviderError) as caught:
            _body(_request(output_schema=self.SCHEMA), structured_output=False)
        assert caught.value.failure is ProviderFailure.REJECTED
