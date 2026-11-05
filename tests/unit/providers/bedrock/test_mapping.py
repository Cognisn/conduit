"""Mapping between conduit's types and the Converse wire format."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import StreamStopped, TextDelta, ToolCallFinished, ToolCallStarted
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    ProviderStateBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.bedrock.mapping import (
    CACHE_POINT,
    MAX_CACHE_POINTS,
    StreamMapper,
    build_converse_request,
    events_from_converse_response,
    stop_reason_from_bedrock,
    usage_from_bedrock,
)

MODEL = "anthropic.claude-sonnet-4-6"
TOOL = ToolSpec(
    name="lookup",
    description="Look a host up.",
    input_schema={"type": "object", "properties": {"host": {"type": "string"}}},
)


def request(**fields: Any) -> ModelRequest:
    fields.setdefault("messages", (Message.user("Hello"),))
    return ModelRequest(model=MODEL, max_output_tokens=256, **fields)


def cache_points(params: dict[str, Any]) -> int:
    blocks = list(params.get("system", []))
    blocks += params.get("toolConfig", {}).get("tools", [])
    for message in params["messages"]:
        blocks += message["content"]
    return sum(1 for b in blocks if "cachePoint" in b)


class TestRequest:
    def test_a_plain_request(self) -> None:
        assert build_converse_request(request(), prompt_caching=True) == {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            "inferenceConfig": {"maxTokens": 256},
        }

    def test_system_temperature_and_stop_sequences(self) -> None:
        params = build_converse_request(
            request(
                system=(TextBlock(text="Be brief."),),
                temperature=0.2,
                stop_sequences=("END",),
            ),
            prompt_caching=False,
        )
        assert params["system"] == [{"text": "Be brief."}]
        assert params["inferenceConfig"] == {
            "maxTokens": 256,
            "temperature": 0.2,
            "stopSequences": ["END"],
        }

    def test_images_tool_use_and_tool_results(self) -> None:
        messages = (
            Message(
                role="user",
                content=(
                    TextBlock(text="What is this?"),
                    ImageBlock(media_type="image/png", data=b"\x89PNG"),
                ),
            ),
            Message(
                role="assistant",
                content=(ToolUseBlock(id="tu-1", name="lookup", input={"host": "a"}),),
            ),
            Message(
                role="user",
                content=(
                    ToolResultBlock(tool_use_id="tu-1", content=(TextBlock(text="192.0.2.1"),)),
                    ToolResultBlock(
                        tool_use_id="tu-2",
                        content=(ImageBlock(media_type="image/jpeg", data=b"\xff\xd8"),),
                        is_error=True,
                    ),
                ),
            ),
        )
        params = build_converse_request(request(messages=messages), prompt_caching=False)
        assert params["messages"] == [
            {
                "role": "user",
                "content": [
                    {"text": "What is this?"},
                    {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "tu-1", "name": "lookup", "input": {"host": "a"}}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tu-1",
                            "content": [{"text": "192.0.2.1"}],
                            "status": "success",
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "tu-2",
                            "content": [
                                {"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8"}}}
                            ],
                            "status": "error",
                        }
                    },
                ],
            },
        ]

    def test_tools(self) -> None:
        params = build_converse_request(request(tools=(TOOL,)), prompt_caching=False)
        assert params["toolConfig"] == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "lookup",
                        "description": "Look a host up.",
                        "inputSchema": {"json": TOOL.input_schema},
                    }
                }
            ]
        }

    def test_structured_output_is_refused_until_it_is_mapped(self) -> None:
        with pytest.raises(ProviderError) as raised:
            build_converse_request(request(output_schema={"type": "object"}), prompt_caching=True)
        assert raised.value.failure is ProviderFailure.REJECTED


class TestCachePoints:
    def cached(self, *, prompt_caching: bool) -> dict[str, Any]:
        return build_converse_request(
            request(
                system=(TextBlock(text="Stable."), CacheBreakpoint()),
                tools=(TOOL,),
                cache_tools=True,
                messages=(
                    Message(role="user", content=(TextBlock(text="Doc."), CacheBreakpoint())),
                ),
            ),
            prompt_caching=prompt_caching,
        )

    def test_breakpoints_become_cache_points_in_system_tools_and_messages(self) -> None:
        params = self.cached(prompt_caching=True)
        assert params["system"] == [{"text": "Stable."}, CACHE_POINT]
        assert params["toolConfig"]["tools"][-1] == CACHE_POINT
        assert params["messages"][0]["content"] == [{"text": "Doc."}, CACHE_POINT]
        assert CACHE_POINT == {"cachePoint": {"type": "default"}}

    def test_breakpoints_are_dropped_where_the_model_does_not_cache(self) -> None:
        params = self.cached(prompt_caching=False)
        assert cache_points(params) == 0
        assert params["system"] == [{"text": "Stable."}]
        assert params["toolConfig"]["tools"] == [
            {
                "toolSpec": {
                    "name": "lookup",
                    "description": "Look a host up.",
                    "inputSchema": {"json": TOOL.input_schema},
                }
            }
        ]

    def test_only_the_last_four_breakpoints_are_kept(self) -> None:
        # Bedrock refuses more than four cache points. Earlier breakpoints are dropped:
        # the cache still reads the longest prefix ending at a later one.
        messages = tuple(
            Message(role="user", content=(TextBlock(text=f"part {i}"), CacheBreakpoint()))
            for i in range(5)
        )
        params = build_converse_request(
            request(
                system=(TextBlock(text="S."), CacheBreakpoint()),
                tools=(TOOL,),
                cache_tools=True,
                messages=messages,
            ),
            prompt_caching=True,
        )
        assert MAX_CACHE_POINTS == 4
        assert cache_points(params) == 4
        assert CACHE_POINT not in params["toolConfig"]["tools"]
        assert params["system"] == [{"text": "S."}]
        kept = [CACHE_POINT in m["content"] for m in params["messages"]]
        assert kept == [False, True, True, True, True]


class TestUsage:
    """Each Bedrock count on its own, so a misread key cannot hide behind another."""

    def test_input_tokens(self) -> None:
        assert usage_from_bedrock({"inputTokens": 11, "outputTokens": 0}) == Usage(
            input_tokens=11, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_output_tokens(self) -> None:
        assert usage_from_bedrock({"inputTokens": 0, "outputTokens": 7}) == Usage(
            input_tokens=0, output_tokens=7, cache_read_tokens=0, cache_write_tokens=0
        )

    def test_cache_read_input_tokens(self) -> None:
        assert usage_from_bedrock(
            {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 5}
        ) == Usage(input_tokens=0, output_tokens=0, cache_read_tokens=5, cache_write_tokens=0)

    def test_cache_write_input_tokens(self) -> None:
        assert usage_from_bedrock(
            {"inputTokens": 0, "outputTokens": 0, "cacheWriteInputTokens": 3}
        ) == Usage(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=3)

    def test_all_four_together_and_total_tokens_ignored(self) -> None:
        usage = usage_from_bedrock(
            {
                "inputTokens": 11,
                "outputTokens": 7,
                "cacheReadInputTokens": 5,
                "cacheWriteInputTokens": 3,
                "totalTokens": 26,
            }
        )
        assert usage == Usage(
            input_tokens=11, output_tokens=7, cache_read_tokens=5, cache_write_tokens=3
        )
        assert usage.total_tokens == 26

    @pytest.mark.parametrize(
        "raw",
        [
            {"outputTokens": 1},
            {"inputTokens": 1},
            {"inputTokens": -1, "outputTokens": 1},
            {"inputTokens": "1", "outputTokens": 1},
            {"inputTokens": 1, "outputTokens": 1, "cacheReadInputTokens": True},
        ],
    )
    def test_a_missing_or_malformed_count_breaks_the_protocol(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ProviderError) as raised:
            usage_from_bedrock(raw)
        assert raised.value.failure is ProviderFailure.PROTOCOL


class TestStopReasons:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("end_turn", StopReason.END_TURN),
            ("tool_use", StopReason.TOOL_USE),
            ("max_tokens", StopReason.MAX_TOKENS),
            ("model_context_window_exceeded", StopReason.MAX_TOKENS),
            ("stop_sequence", StopReason.STOP_SEQUENCE),
            ("guardrail_intervened", StopReason.REFUSAL),
            ("content_filtered", StopReason.REFUSAL),
            ("refusal", StopReason.REFUSAL),
            ("malformed_tool_use", StopReason.OTHER),
            ("something_new", StopReason.OTHER),
        ],
    )
    def test_mapping(self, raw: str, expected: StopReason) -> None:
        assert stop_reason_from_bedrock(raw) is expected


METADATA = {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 7}, "metrics": {}}}
USAGE = Usage(input_tokens=11, output_tokens=7, cache_read_tokens=0, cache_write_tokens=0)


def run(raw_events: list[dict[str, Any]]) -> list[Any]:
    mapper = StreamMapper()
    out: list[Any] = []
    for raw in raw_events:
        out.extend(mapper.feed(raw))
    out.extend(mapper.finish())
    return out


class TestStream:
    def test_text_then_one_stop_with_the_metadata_usage(self) -> None:
        events = run(
            [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
                {"contentBlockDelta": {"delta": {"text": ""}, "contentBlockIndex": 0}},
                {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
                METADATA,
            ]
        )
        assert events == [
            TextDelta(text="Hel"),
            TextDelta(text="lo"),
            StreamStopped(stop_reason=StopReason.END_TURN, usage=USAGE),
        ]

    def test_tool_input_is_assembled_from_its_deltas(self) -> None:
        events = run(
            [
                {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup"}},
                        "contentBlockIndex": 1,
                    }
                },
                {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": '{"ho'}},
                        "contentBlockIndex": 1,
                    }
                },
                {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": 'st": "a"}'}},
                        "contentBlockIndex": 1,
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 1}},
                {"messageStop": {"stopReason": "tool_use"}},
                METADATA,
            ]
        )
        assert events == [
            ToolCallStarted(id="tu-1", name="lookup"),
            ToolCallFinished(id="tu-1", name="lookup", input={"host": "a"}),
            StreamStopped(stop_reason=StopReason.TOOL_USE, usage=USAGE),
        ]

    def test_a_tool_call_with_no_input_deltas_has_empty_input(self) -> None:
        events = run(
            [
                {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup"}},
                        "contentBlockIndex": 0,
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
                METADATA,
            ]
        )
        assert events[1] == ToolCallFinished(id="tu-1", name="lookup", input={})

    def test_reasoning_deltas_are_not_surfaced(self) -> None:
        events = run(
            [
                {
                    "contentBlockDelta": {
                        "delta": {"reasoningContent": {"text": "thinking"}},
                        "contentBlockIndex": 0,
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
                METADATA,
            ]
        )
        assert events == [StreamStopped(stop_reason=StopReason.END_TURN, usage=USAGE)]

    @pytest.mark.parametrize(
        "raw_events",
        [
            pytest.param([{"messageStop": {"stopReason": "end_turn"}}], id="no metadata"),
            pytest.param([{"messageStop": {}}], id="a message stop without a reason"),
            pytest.param(
                [{"contentBlockDelta": {"delta": {"text": 5}, "contentBlockIndex": 0}}],
                id="text that is not a string",
            ),
            pytest.param([METADATA], id="no message stop"),
            pytest.param(
                [{"messageStop": {"stopReason": "end_turn"}}, METADATA, METADATA],
                id="a second metadata event",
            ),
            pytest.param(
                [
                    {
                        "contentBlockStart": {
                            "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup"}},
                            "contentBlockIndex": 0,
                        }
                    },
                    {"messageStop": {"stopReason": "tool_use"}},
                    METADATA,
                ],
                id="tool call never stopped",
            ),
            pytest.param(
                [
                    {
                        "contentBlockStart": {
                            "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup"}},
                            "contentBlockIndex": 0,
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "delta": {"toolUse": {"input": "[1, 2]"}},
                            "contentBlockIndex": 0,
                        }
                    },
                    {"contentBlockStop": {"contentBlockIndex": 0}},
                ],
                id="tool input not an object",
            ),
            pytest.param(
                [
                    {
                        "contentBlockStart": {
                            "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup"}},
                            "contentBlockIndex": 0,
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "delta": {"toolUse": {"input": "{not json"}},
                            "contentBlockIndex": 0,
                        }
                    },
                    {"contentBlockStop": {"contentBlockIndex": 0}},
                ],
                id="tool input not JSON",
            ),
        ],
    )
    def test_a_broken_stream_is_a_protocol_failure(self, raw_events: list[dict[str, Any]]) -> None:
        with pytest.raises(ProviderError) as raised:
            run(raw_events)
        assert raised.value.failure is ProviderFailure.PROTOCOL
        assert raised.value.__cause__ is None and raised.value.__context__ is None


class TestConverseResponse:
    def test_a_converse_response_becomes_the_same_events(self) -> None:
        events = events_from_converse_response(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "Looking."},
                            {
                                "toolUse": {
                                    "toolUseId": "tu-1",
                                    "name": "lookup",
                                    "input": {"host": "a"},
                                }
                            },
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 11, "outputTokens": 7, "cacheReadInputTokens": 5},
                "metrics": {"latencyMs": 10},
            }
        )
        assert events == [
            TextDelta(text="Looking."),
            ToolCallStarted(id="tu-1", name="lookup"),
            ToolCallFinished(id="tu-1", name="lookup", input={"host": "a"}),
            StreamStopped(
                stop_reason=StopReason.TOOL_USE,
                usage=Usage(
                    input_tokens=11, output_tokens=7, cache_read_tokens=5, cache_write_tokens=0
                ),
            ),
        ]

    def test_a_response_without_usage_is_a_protocol_failure(self) -> None:
        with pytest.raises(ProviderError) as raised:
            events_from_converse_response(
                {"output": {"message": {"content": []}}, "stopReason": "end_turn"}
            )
        assert raised.value.failure is ProviderFailure.PROTOCOL
        assert raised.value.__cause__ is None and raised.value.__context__ is None


class TestProviderState:
    """Until the adapter round-trips its own state, every kind's state is left out."""

    @pytest.mark.parametrize("kind", ["gemini", "bedrock"])
    def test_provider_state_is_not_sent(self, kind: str) -> None:
        state = ProviderStateBlock(
            provider_kind=kind, state_type="reasoning", payload=b"zz-state-marker-zz"
        )
        params = build_converse_request(
            request(
                messages=(
                    Message.user("Hello"),
                    Message(role="assistant", content=(state, TextBlock(text="Hi."))),
                    Message.user("Again"),
                )
            ),
            prompt_caching=False,
        )
        assert params["messages"][1]["content"] == [{"text": "Hi."}]
        assert "zz-state-marker-zz" not in repr(params)
