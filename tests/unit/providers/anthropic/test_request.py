"""The Anthropic adapter's request mapping: messages, tools, images and cache control."""

from __future__ import annotations

import base64
from typing import Any, ClassVar

import pytest

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
from conduit.providers.anthropic.request import MAX_CACHE_BREAKPOINTS, build_params

EPHEMERAL = {"type": "ephemeral"}
PNG = b"\x89PNG\r\n\x1a\nsynthetic"


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model="claude-sonnet-5-5",
        messages=messages or (Message.user("hello"),),
        max_output_tokens=512,
        **fields,
    )


def _tool(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"the {name} tool",
        input_schema={"type": "object", "properties": {}},
    )


def _marked(params: dict[str, Any]) -> list[Any]:
    """Every block carrying cache_control, in prompt order (tools, system, messages)."""
    blocks: list[Any] = list(params.get("tools", []))
    blocks += list(params.get("system", []))
    for message in params["messages"]:
        blocks += list(message["content"])
    return [b for b in blocks if "cache_control" in b]


class TestBasicShape:
    def test_minimal_request(self) -> None:
        params = build_params(_request(), structured_output=False)
        assert params == {
            "model": "claude-sonnet-5-5",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "stream": True,
            "timeout": 120.0,
        }

    def test_system_tools_and_stop_sequences(self) -> None:
        request = _request(
            system=(TextBlock(text="be brief"),),
            tools=(_tool("lookup"),),
            stop_sequences=("END",),
            timeout_s=30,
        )
        params = build_params(request, structured_output=False)
        assert params["system"] == [{"type": "text", "text": "be brief"}]
        assert params["tools"] == [
            {
                "name": "lookup",
                "description": "the lookup tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        assert params["stop_sequences"] == ["END"]
        assert params["timeout"] == 30.0

    def test_temperature_is_never_sent(self) -> None:
        # API version 2026-10-21 rejects temperature, top_p and top_k on every model.
        params = build_params(_request(temperature=0.2), structured_output=False)
        assert "temperature" not in params

    def test_tool_use_and_tool_results(self) -> None:
        request = _request(
            Message.user("find it"),
            Message(
                role="assistant",
                content=(
                    TextBlock(text="looking"),
                    ToolUseBlock(id="toolu_01", name="lookup", input={"q": "x"}),
                ),
            ),
            Message(
                role="user",
                content=(
                    ToolResultBlock(
                        tool_use_id="toolu_01",
                        content=(
                            TextBlock(text="found"),
                            ImageBlock(media_type="image/png", data=PNG),
                        ),
                    ),
                    ToolResultBlock(
                        tool_use_id="toolu_02", content=(TextBlock(text="boom"),), is_error=True
                    ),
                ),
            ),
        )
        messages = build_params(request, structured_output=False)["messages"]
        assert messages[1] == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "looking"},
                {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {"q": "x"}},
            ],
        }
        encoded = base64.standard_b64encode(PNG).decode("ascii")
        assert messages[2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": [
                    {"type": "text", "text": "found"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
                    },
                ],
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_02",
                "content": [{"type": "text", "text": "boom"}],
                "is_error": True,
            },
        ]

    def test_image_in_a_user_message(self) -> None:
        request = _request(
            Message(
                role="user",
                content=(
                    ImageBlock(media_type="image/webp", data=b"RIFFxx"),
                    TextBlock(text="what?"),
                ),
            )
        )
        content = build_params(request, structured_output=False)["messages"][0]["content"]
        assert content[0] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/webp",
                "data": base64.standard_b64encode(b"RIFFxx").decode("ascii"),
            },
        }

    def test_the_request_is_not_mutated_between_builds(self) -> None:
        request = _request(system=(TextBlock(text="s"), CacheBreakpoint()))
        first = build_params(request, structured_output=False)
        second = build_params(request, structured_output=False)
        assert first == second


class TestStructuredOutput:
    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def test_schema_becomes_output_config_format(self) -> None:
        params = build_params(_request(output_schema=self.SCHEMA), structured_output=True)
        assert params["output_config"] == {"format": {"type": "json_schema", "schema": self.SCHEMA}}

    def test_no_schema_no_output_config(self) -> None:
        assert "output_config" not in build_params(_request(), structured_output=True)


class TestCacheControl:
    def test_system_breakpoint_marks_the_preceding_block(self) -> None:
        request = _request(
            system=(TextBlock(text="stable"), CacheBreakpoint(), TextBlock(text="volatile"))
        )
        params = build_params(request, structured_output=False)
        assert params["system"] == [
            {"type": "text", "text": "stable", "cache_control": EPHEMERAL},
            {"type": "text", "text": "volatile"},
        ]

    def test_cache_tools_marks_the_last_tool(self) -> None:
        request = _request(tools=(_tool("a"), _tool("b")), cache_tools=True)
        tools = build_params(request, structured_output=False)["tools"]
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == EPHEMERAL

    def test_tools_without_cache_tools_are_unmarked(self) -> None:
        request = _request(tools=(_tool("a"),))
        assert _marked(build_params(request, structured_output=False)) == []

    def test_message_breakpoint_marks_the_preceding_block_and_is_not_sent(self) -> None:
        request = _request(
            Message(
                role="user",
                content=(TextBlock(text="document"), CacheBreakpoint(), TextBlock(text="question")),
            )
        )
        content = build_params(request, structured_output=False)["messages"][0]["content"]
        assert content == [
            {"type": "text", "text": "document", "cache_control": EPHEMERAL},
            {"type": "text", "text": "question"},
        ]

    def test_breakpoint_opening_a_message_marks_the_previous_message(self) -> None:
        request = _request(
            Message.user("first"),
            Message.assistant("reply"),
            Message(role="user", content=(CacheBreakpoint(), TextBlock(text="next"))),
        )
        messages = build_params(request, structured_output=False)["messages"]
        assert messages[1]["content"] == [
            {"type": "text", "text": "reply", "cache_control": EPHEMERAL}
        ]
        assert messages[2]["content"] == [{"type": "text", "text": "next"}]

    def test_breakpoint_opening_the_messages_marks_the_system_prompt(self) -> None:
        request = _request(
            Message(role="user", content=(CacheBreakpoint(), TextBlock(text="hi"))),
            system=(TextBlock(text="sys"),),
        )
        params = build_params(request, structured_output=False)
        assert params["system"][0]["cache_control"] == EPHEMERAL

    def test_breakpoint_with_nothing_before_it_is_ignored(self) -> None:
        request = _request(system=(CacheBreakpoint(), TextBlock(text="sys")))
        assert _marked(build_params(request, structured_output=False)) == []

    def test_consecutive_breakpoints_count_once(self) -> None:
        request = _request(system=(TextBlock(text="sys"), CacheBreakpoint(), CacheBreakpoint()))
        params = build_params(request, structured_output=False)
        assert len(_marked(params)) == 1

    def test_tool_result_and_image_blocks_can_be_marked(self) -> None:
        request = _request(
            Message(
                role="user",
                content=(
                    ToolResultBlock(tool_use_id="toolu_01", content=(TextBlock(text="r"),)),
                    CacheBreakpoint(),
                    ImageBlock(media_type="image/png", data=PNG),
                    CacheBreakpoint(),
                ),
            )
        )
        content = build_params(request, structured_output=False)["messages"][0]["content"]
        assert content[0]["cache_control"] == EPHEMERAL
        assert content[1]["cache_control"] == EPHEMERAL

    def test_the_provider_limit_is_four(self) -> None:
        assert MAX_CACHE_BREAKPOINTS == 4

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    def test_up_to_the_limit_every_breakpoint_is_kept(self, count: int) -> None:
        blocks: list[Any] = []
        for i in range(count):
            blocks += [TextBlock(text=f"part {i}"), CacheBreakpoint()]
        request = _request(Message(role="user", content=tuple(blocks)))
        assert len(_marked(build_params(request, structured_output=False))) == count

    def test_repeated_breakpoints_do_not_use_up_the_limit(self) -> None:
        content: list[Any] = [TextBlock(text="a"), CacheBreakpoint(), CacheBreakpoint()]
        for name in ("b", "c", "d"):
            content += [TextBlock(text=name), CacheBreakpoint()]
        request = _request(Message(role="user", content=tuple(content)))
        marked = _marked(build_params(request, structured_output=False))
        assert [b["text"] for b in marked] == ["a", "b", "c", "d"]

    def test_beyond_the_limit_the_first_three_and_the_last_are_kept(self) -> None:
        request = _request(
            Message(
                role="user",
                content=(
                    TextBlock(text="m0"),
                    CacheBreakpoint(),
                    TextBlock(text="m1"),
                    CacheBreakpoint(),
                    TextBlock(text="m2"),
                ),
            ),
            Message.assistant("m3"),
            Message(
                role="user", content=(CacheBreakpoint(), TextBlock(text="m4"), CacheBreakpoint())
            ),
            system=(TextBlock(text="sys"), CacheBreakpoint()),
            tools=(_tool("a"),),
            cache_tools=True,
        )
        params = build_params(request, structured_output=False)
        marked = _marked(params)
        assert len(marked) == MAX_CACHE_BREAKPOINTS
        names = [b.get("name") or b.get("text") for b in marked]
        # tools, system, m0 are the first three; m4 (the last) is kept; m1 and m3 merge on.
        assert names == ["a", "sys", "m0", "m4"]
