"""Messages, content blocks, tool specifications, usage and the normalised response."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from conduit.core.types import (
    CacheBreakpoint,
    ContentBlock,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)


def _usage(**overrides: int) -> Usage:
    values = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 3,
        "cache_write_tokens": 2,
    }
    values.update(overrides)
    return Usage(**values)


class TestUsage:
    def test_all_four_counts_are_required(self) -> None:
        # Termograph lesson: a provider adapter must state cache tokens explicitly, so a
        # missing cache count is an error, never a silent zero.
        with pytest.raises(ValidationError):
            Usage(input_tokens=1, output_tokens=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Usage(input_tokens=1, output_tokens=1, cache_read_tokens=0)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "missing", ["input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"]
    )
    def test_each_count_is_required_on_its_own(self, missing: str) -> None:
        values = {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_tokens": 1,
            "cache_write_tokens": 1,
        }
        del values[missing]
        with pytest.raises(ValidationError):
            Usage.model_validate(values)

    def test_a_provider_wire_name_is_refused_rather_than_ignored(self) -> None:
        # Reading a wrong key once meant cache tokens were never billed. A mapping that
        # carries a provider's own names is refused, not dropped.
        with pytest.raises(ValidationError):
            Usage.model_validate(
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cacheReadInputTokens": 7,
                }
            )

    def test_counts_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            _usage(cache_write_tokens=-1)

    def test_addition_keeps_each_count_separate(self) -> None:
        total = _usage() + _usage(input_tokens=1, output_tokens=2, cache_read_tokens=4)
        assert total == Usage(
            input_tokens=11, output_tokens=7, cache_read_tokens=7, cache_write_tokens=4
        )

    def test_zero_and_totals(self) -> None:
        assert Usage.zero() == Usage(
            input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
        )
        usage = _usage()
        assert usage.total_input_tokens == 15  # uncached, cache read and cache write
        assert usage.total_tokens == 20


class TestContent:
    def test_every_block_kind_round_trips_through_json(self) -> None:
        message = Message(
            role="user",
            content=(
                TextBlock(text="Summarise the report for example.org."),
                ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\n\x00\x01"),
                CacheBreakpoint(),
                ToolResultBlock(
                    tool_use_id="toolu_1",
                    content=(TextBlock(text="42"),),
                    is_error=False,
                ),
            ),
        )
        reply = Message(
            role="assistant",
            content=(ToolUseBlock(id="toolu_1", name="lookup", input={"host": "example.com"}),),
        )
        for original in (message, reply):
            again = Message.model_validate_json(original.model_dump_json())
            assert again == original

    def test_blocks_are_discriminated_by_type(self) -> None:
        adapter: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
        block = adapter.validate_python({"type": "tool_use", "id": "t", "name": "n", "input": {}})
        assert isinstance(block, ToolUseBlock)
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "audio", "data": ""})

    def test_image_media_type_is_restricted(self) -> None:
        with pytest.raises(ValidationError):
            ImageBlock(media_type="application/pdf", data=b"x")  # type: ignore[arg-type]

    def test_user_and_assistant_helpers(self) -> None:
        assert Message.user("hello") == Message(role="user", content=(TextBlock(text="hello"),))
        assert Message.assistant("hi").role == "assistant"

    def test_messages_are_immutable(self) -> None:
        message = Message.user("hello")
        with pytest.raises(ValidationError):
            message.role = "assistant"


class TestToolSpec:
    def test_name_must_be_a_tool_name(self) -> None:
        ToolSpec(name="get_weather", description="d", input_schema={"type": "object"})
        with pytest.raises(ValidationError):
            ToolSpec(name="get weather", description="d", input_schema={"type": "object"})


class TestRequestAndResponse:
    def test_response_exposes_text_and_tool_uses(self) -> None:
        response = ModelResponse(
            content=(
                TextBlock(text="Looking "),
                ToolUseBlock(id="a", name="lookup", input={}),
                TextBlock(text="it up."),
            ),
            stop_reason=StopReason.TOOL_USE,
            usage=Usage.zero(),
            model="stub-echo",
        )
        assert response.text == "Looking it up."
        assert [t.id for t in response.tool_uses] == ["a"]

    def test_request_digest_is_stable_and_ignores_the_call_context(self) -> None:
        def build(run_id: str) -> ModelRequest:
            return ModelRequest(
                model="stub-echo",
                system=(TextBlock(text="You are terse."), CacheBreakpoint()),
                messages=(Message.user("hello"),),
                max_output_tokens=256,
                run_id=run_id,
            )

        assert build("run-1").digest() == build("run-2").digest()
        changed = build("run-1").model_copy(update={"max_output_tokens": 512})
        assert changed.digest() != build("run-1").digest()

    def test_request_needs_a_positive_output_budget(self) -> None:
        with pytest.raises(ValidationError):
            ModelRequest(model="m", messages=(Message.user("x"),), max_output_tokens=0)
