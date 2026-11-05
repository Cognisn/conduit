"""Synthetic Bedrock recordings, and a fake ``bedrock-runtime`` client that replays them.

The recordings in ``tests/recordings/bedrock/`` are written by this module through
``conduit.testing.replay.save_recording``, which refuses anything unscrubbed. They are
synthetic: the prompts, hosts and identifiers are invented, and the responses follow
the Converse and ConverseStream wire shapes. Regenerate them with

    python -m tests.unit.providers.bedrock.recordings

A test checks that the committed files equal what this module builds.

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


ALL = (sonnet_4_6_in_region, sonnet_5_in_region, uncached_profile)


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
