"""Synthetic Bedrock recordings, and a fake ``bedrock-runtime`` client that replays them.

The recordings in ``tests/recordings/bedrock/`` are written by this module through
``conduit.testing.replay.save_recording``, which refuses anything unscrubbed. They are
synthetic: the prompts, hosts and identifiers are invented, and the responses follow
the Converse and ConverseStream wire shapes. Regenerate them with

    python -m tests.unit.providers.bedrock.recordings

A test checks that the committed files equal what this module builds.

The non-Anthropic recordings (``VENDORS``) follow the Converse wire shapes the Bedrock
user guide documents; they are not captured from live calls, so those models are
recorded-response tested until the next live verification pass.

Each recorded request is ``{"operation", "region", "params"}``: the region the client
was built for and the exact Converse parameters, so a recording also proves the
request shape (a bare model id called in-region, for example).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, Replay, save_recording

DIRECTORY = Path(__file__).resolve().parents[3] / "recordings" / "bedrock"

SYSTEM = "You summarise incident notes for the operations team at example.com."
NOTES = "Incident notes: host-a.example.com stopped answering on 192.0.2.10 at 09:00."
TOOL_SPEC = {
    "toolSpec": {
        "name": "lookup_host",
        "description": "Look a host up in the inventory.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"],
            }
        },
    }
}
CACHE_POINT = {"cachePoint": {"type": "default"}}


def _params(model: str, messages: list[dict[str, Any]], *, cached: bool) -> dict[str, Any]:
    system: list[dict[str, Any]] = [{"text": SYSTEM}]
    tools: list[dict[str, Any]] = [TOOL_SPEC]
    if cached:
        system.append(CACHE_POINT)
        tools.append(CACHE_POINT)
    return {
        "modelId": model,
        "system": system,
        "messages": messages,
        "toolConfig": {"tools": tools},
        "inferenceConfig": {"maxTokens": 1024},
    }


FIRST_TURN: list[dict[str, Any]] = [{"role": "user", "content": [{"text": NOTES}]}]
TOOL_TURN: list[dict[str, Any]] = [
    *FIRST_TURN,
    {
        "role": "assistant",
        "content": [
            {"text": "Checking the inventory."},
            {
                "toolUse": {
                    "toolUseId": "tooluse_example0001",
                    "name": "lookup_host",
                    "input": {"host": "host-a.example.com"},
                }
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "tooluse_example0001",
                    "content": [{"text": "host-a.example.com: web tier, owner team-a"}],
                    "status": "success",
                }
            }
        ],
    },
]


def _tool_stream(cache_write: int) -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Checking "}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "the inventory."}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tooluse_example0001", "name": "lookup_host"}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"host": "host-a'}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '.example.com"}'}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 52,
                    "outputTokens": 31,
                    "totalTokens": 52 + 31 + cache_write,
                    "cacheReadInputTokens": 0,
                    "cacheWriteInputTokens": cache_write,
                },
                "metrics": {"latencyMs": 812},
            }
        },
    ]


ANSWER_STREAM = [
    {"messageStart": {"role": "assistant"}},
    {
        "contentBlockDelta": {
            "delta": {"text": "host-a.example.com (web tier, team-a) "},
            "contentBlockIndex": 0,
        }
    },
    {
        "contentBlockDelta": {
            "delta": {"text": "stopped answering at 09:00."},
            "contentBlockIndex": 0,
        }
    },
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"messageStop": {"stopReason": "end_turn"}},
    {
        "metadata": {
            "usage": {
                "inputTokens": 61,
                "outputTokens": 17,
                "totalTokens": 61 + 17 + 1200,
                "cacheReadInputTokens": 1200,
                "cacheWriteInputTokens": 0,
            },
            "metrics": {"latencyMs": 640},
        }
    },
]


def sonnet_4_6_in_region() -> Recording:
    """Two ConverseStream turns on the bare Sonnet 4.6 id in eu-west-2, with cache points."""
    model = "anthropic.claude-sonnet-4-6"
    return Recording(
        name="sonnet-4-6-in-region-stream",
        provider="bedrock",
        interactions=(
            Interaction(
                request={
                    "operation": "ConverseStream",
                    "region": "eu-west-2",
                    "params": _params(model, FIRST_TURN, cached=True),
                },
                response=_tool_stream(cache_write=1200),
            ),
            Interaction(
                request={
                    "operation": "ConverseStream",
                    "region": "eu-west-2",
                    "params": _params(model, TOOL_TURN, cached=True),
                },
                response=ANSWER_STREAM,
            ),
        ),
    )


def sonnet_5_in_region() -> Recording:
    """One Converse call on the bare Sonnet 5 id in ap-southeast-1."""
    model = "anthropic.claude-sonnet-5"
    return Recording(
        name="sonnet-5-in-region-converse",
        provider="bedrock",
        interactions=(
            Interaction(
                request={
                    "operation": "Converse",
                    "region": "ap-southeast-1",
                    "params": _params(model, FIRST_TURN, cached=True),
                },
                response={
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"text": "Checking the inventory."},
                                {
                                    "toolUse": {
                                        "toolUseId": "tooluse_example0001",
                                        "name": "lookup_host",
                                        "input": {"host": "host-a.example.com"},
                                    }
                                },
                            ],
                        }
                    },
                    "stopReason": "tool_use",
                    "usage": {
                        "inputTokens": 52,
                        "outputTokens": 31,
                        "totalTokens": 1283,
                        "cacheReadInputTokens": 0,
                        "cacheWriteInputTokens": 1200,
                    },
                    "metrics": {"latencyMs": 790},
                },
            ),
        ),
    )


def uncached_profile() -> Recording:
    """A ConverseStream turn on a model whose catalogue entry has no prompt caching."""
    model = "eu.anthropic.claude-sonnet-4-6"
    return Recording(
        name="uncached-profile-stream",
        provider="bedrock",
        interactions=(
            Interaction(
                request={
                    "operation": "ConverseStream",
                    "region": "eu-west-2",
                    "params": _params(model, FIRST_TURN, cached=False),
                },
                response=_tool_stream(cache_write=0),
            ),
        ),
    )


#: A reasoning block as non-Anthropic models stream it: text, and no signature.
REASONING = "The notes name one host and one time."


def _vendor_params(
    model: str, *, tools: bool, system_cache: bool, tools_cache: bool
) -> dict[str, Any]:
    system: list[dict[str, Any]] = [{"text": SYSTEM}]
    if system_cache:
        system.append(CACHE_POINT)
    params: dict[str, Any] = {"modelId": model, "system": system, "messages": FIRST_TURN}
    if tools:
        params["toolConfig"] = {"tools": [TOOL_SPEC, *([CACHE_POINT] if tools_cache else [])]}
    params["inferenceConfig"] = {"maxTokens": 1024}
    return params


def _usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
    }


def _answer_stream(*, reasoning: bool, usage: dict[str, Any]) -> list[dict[str, Any]]:
    """A ConverseStream answer: optionally a reasoning block, then text in two pieces."""
    events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
    index = 0
    if reasoning:
        events += [
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": REASONING[:18]}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": REASONING[18:]}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ]
        index = 1
    events += [
        {
            "contentBlockDelta": {
                "delta": {"text": "host-a.example.com "},
                "contentBlockIndex": index,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"text": "stopped answering at 09:00."},
                "contentBlockIndex": index,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": index}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": usage, "metrics": {"latencyMs": 420}}},
    ]
    return events


def _vendor(
    name: str,
    model: str,
    region: str,
    response: Any,
    *,
    operation: str = "ConverseStream",
    tools: bool = False,
    system_cache: bool = False,
) -> Recording:
    params = _vendor_params(model, tools=tools, system_cache=system_cache, tools_cache=False)
    return Recording(
        name=name,
        provider="bedrock",
        interactions=(
            Interaction(
                request={"operation": operation, "region": region, "params": params},
                response=response,
            ),
        ),
    )


def nova_lite_tools() -> Recording:
    """Amazon Nova Lite in-region: a streamed tool call; its tools take no cache point."""
    return _vendor(
        "nova-lite-in-region-tool-stream",
        "amazon.nova-lite-v1:0",
        "us-east-1",
        _tool_stream(cache_write=1200),
        tools=True,
        system_cache=True,
    )


def mistral_large_tools() -> Recording:
    """Mistral Large (24.02): tools go through Converse, since ConverseStream refuses them."""
    return _vendor(
        "mistral-large-2402-tool-converse",
        "mistral.mistral-large-2402-v1:0",
        "eu-west-2",
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_example0002",
                                "name": "lookup_host",
                                "input": {"host": "host-a.example.com"},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": _usage(64, 22),
            "metrics": {"latencyMs": 530},
        },
        operation="Converse",
        tools=True,
    )


def qwen_tools() -> Recording:
    """Qwen3 235B in-region: a streamed tool call, no cache points."""
    return _vendor(
        "qwen3-235b-tool-stream",
        "qwen.qwen3-235b-a22b-2507-v1:0",
        "eu-west-2",
        _tool_stream(cache_write=0),
        tools=True,
    )


def gpt_oss_reasoning() -> Recording:
    """OpenAI gpt-oss-120b in-region: reasoning without a signature, then text."""
    return _vendor(
        "gpt-oss-120b-reasoning-stream",
        "openai.gpt-oss-120b-1:0",
        "us-west-2",
        _answer_stream(reasoning=True, usage=_usage(40, 35)),
    )


def gpt_profile_text() -> Recording:
    """OpenAI GPT-5.6 Terra through its India profile: text only."""
    return _vendor(
        "gpt-5-6-terra-in-profile-stream",
        "in.openai.gpt-5.6-terra",
        "ap-south-1",
        _answer_stream(reasoning=False, usage=_usage(38, 12)),
    )


def nemotron_reasoning() -> Recording:
    """NVIDIA Nemotron Super 3 through Converse: a whole reasoning block, then text."""
    return _vendor(
        "nemotron-super-3-reasoning-converse",
        "nvidia.nemotron-super-3-120b",
        "eu-west-2",
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": REASONING}}},
                        {"text": "host-a.example.com stopped answering at 09:00."},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": _usage(41, 30),
            "metrics": {"latencyMs": 610},
        },
        operation="Converse",
    )


def _text_only(name: str, model: str, region: str) -> Recording:
    return _vendor(name, model, region, _answer_stream(reasoning=False, usage=_usage(37, 11)))


def deepseek_text() -> Recording:
    return _text_only("deepseek-v3-2-text-stream", "deepseek.v3.2", "ap-southeast-2")


def minimax_text() -> Recording:
    return _text_only("minimax-m2-5-text-stream", "minimax.minimax-m2.5", "eu-west-2")


def kimi_text() -> Recording:
    return _text_only("kimi-k2-5-text-stream", "moonshotai.kimi-k2.5", "ap-southeast-2")


def palmyra_text() -> Recording:
    return _text_only("palmyra-vision-7b-text-stream", "writer.palmyra-vision-7b", "us-east-1")


def glm_text() -> Recording:
    return _text_only("glm-5-text-stream", "zai.glm-5", "us-west-2")


#: One recording at least per non-Anthropic vendor the catalogue declares: Amazon,
#: Mistral AI, Qwen, OpenAI, NVIDIA, DeepSeek, MiniMax, Moonshot AI, Writer and Z.AI.
VENDORS = (
    nova_lite_tools,
    mistral_large_tools,
    qwen_tools,
    gpt_oss_reasoning,
    gpt_profile_text,
    nemotron_reasoning,
    deepseek_text,
    minimax_text,
    kimi_text,
    palmyra_text,
    glm_text,
)

ALL = (sonnet_4_6_in_region, sonnet_5_in_region, uncached_profile, *VENDORS)


def path(recording: Recording) -> Path:
    return DIRECTORY / f"{recording.name}.json"


class RecordedRuntime:
    """A fake ``bedrock-runtime`` client answering from a recording."""

    def __init__(self, recording: Recording, *, region: str) -> None:
        self.replay = Replay(recording)
        self.region = region
        self.closed = False

    def _next(self, operation: str, params: dict[str, Any]) -> Any:
        return self.replay.next({"operation": operation, "region": self.region, "params": params})

    def converse(self, **params: Any) -> Any:
        return self._next("Converse", params)

    def converse_stream(self, **params: Any) -> dict[str, Iterator[Any]]:
        return {"stream": iter(self._next("ConverseStream", params))}

    def close(self) -> None:
        self.closed = True


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for build in ALL:
        recording = build()
        save_recording(recording, path(recording))


if __name__ == "__main__":
    main()
