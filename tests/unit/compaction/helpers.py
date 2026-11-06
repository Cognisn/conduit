"""Shared fixtures for the compaction tests: a stub runtime, conversations and recorders.

Everything here is synthetic. The runtime is started through ``conduit.core.runtime.start``
with stub providers, so every model call (the summariser's included) passes through the
real registry, metering and assignment checks. Three task classes are declared:

- ``chat`` runs on ``stub-small``, a stub model with a 4,000-token context window, so a
  conversation of a few kilobytes reaches a compaction threshold;
- ``summary`` runs on ``stub-summary`` (a 16,000-token window) on its own provider, so the
  summariser's calls are visible apart from the conversation's;
- ``tiny`` runs on ``stub-tiny`` (a 600-token window), for floor and size refusals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from conduit.core.events import CapturePolicy, Event, EventSink
from conduit.core.runtime import ConduitSettings, Runtime, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import CapabilityFloor, ResolvedAssignment, TaskClass
from conduit.core.types import (
    Message,
    ModelRequest,
    SystemBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.stub import StubFactory, StubScript, StubTurn, echo

CHAT = TaskClass(name="chat", floor=CapabilityFloor(tools=True))
SUMMARY = TaskClass(name="summary")
TINY = TaskClass(name="tiny")

SMALL_WINDOW = 4_000


def _entry(model_id: str, window: int, output: int) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "provider": "stub",
        "kind": "chat",
        "residency_scope": "local",
        "capabilities": {"tools": True, "prompt_caching": True},
        "context_window": window,
        "output_limit": output,
    }


def settings(capture: CapturePolicy | None = None) -> ConduitSettings:
    return ConduitSettings.model_validate(
        {
            "capture": (capture or CapturePolicy()).model_dump(),
            "providers": {
                "primary": {"kind": "stub", "script": "primary", "not_for_production": True},
                "summariser": {
                    "kind": "stub",
                    "script": "summariser",
                    "not_for_production": True,
                },
            },
            "assignments": {
                "chat": {"provider": "primary", "model": "stub-small", "max_output_tokens": 256},
                "summary": {
                    "provider": "summariser",
                    "model": "stub-summary",
                    "max_output_tokens": 512,
                },
                "tiny": {"provider": "summariser", "model": "stub-tiny", "max_output_tokens": 64},
            },
            "catalogue": [
                {
                    "name": "compaction-tests",
                    "add": [
                        _entry("stub-small", SMALL_WINDOW, 1024),
                        _entry("stub-summary", 16_000, 1024),
                        _entry("stub-tiny", 600, 128),
                    ],
                }
            ],
        }
    )


class ScriptLog:
    """A stub script that records every request it serves, then answers from ``script``."""

    def __init__(self, script: StubScript = echo) -> None:
        self._script = script
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest, index: int) -> StubTurn:
        self.requests.append(request)
        return self._script(request, index)


def summary_script(text: str = "SUMMARY: the example.com release was discussed.") -> ScriptLog:
    return ScriptLog(lambda request, index: StubTurn(text=text))


async def runtime(
    primary: StubScript = echo,
    summariser: StubScript | None = None,
    *,
    events: EventSink | None = None,
    capture: CapturePolicy | None = None,
) -> Runtime:
    scripts: dict[str, StubScript] = {
        "primary": primary,
        "summariser": summariser if summariser is not None else summary_script(),
    }
    return await start(
        settings(capture),
        task_classes=[CHAT, SUMMARY, TINY],
        factories=[StubFactory(scripts=scripts)],
        secrets=LiteralSecretResolver({}),
        events=events,
    )


class Sink:
    """An event sink that keeps every event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    def of[E: Event](self, kind: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, kind)]


@dataclass(frozen=True, slots=True)
class Request:
    """The shape of the runner's ``CompactionRequest``, for driving the hook directly."""

    run_id: str
    task_class: str
    session_id: str | None
    assignment: ResolvedAssignment
    system: tuple[SystemBlock, ...]
    tools: tuple[ToolSpec, ...]
    messages: tuple[Message, ...]
    usage: Usage
    last_usage: Usage | None
    model_calls: int


def request(
    assignment: ResolvedAssignment,
    messages: Sequence[Message],
    *,
    session_id: str | None = None,
    system: Sequence[SystemBlock] = (),
    tools: Sequence[ToolSpec] = (),
    last_usage: Usage | None = None,
) -> Request:
    return Request(
        run_id="run_test",
        task_class=assignment.task_class.name,
        session_id=session_id,
        assignment=assignment,
        system=tuple(system),
        tools=tuple(tools),
        messages=tuple(messages),
        usage=Usage.zero(),
        last_usage=last_usage,
        model_calls=0,
    )


def filler(n: int, word: str = "lorem") -> str:
    """About ``n`` characters of synthetic text."""
    return (f"{word} " * (n // (len(word) + 1) + 1))[:n]


def tool_turn(index: int, result_chars: int = 400) -> tuple[Message, Message]:
    """An assistant tool call and the user message carrying its result."""
    use = ToolUseBlock(id=f"toolu_{index}", name="lookup", input={"q": f"item {index}"})
    result = ToolResultBlock(
        tool_use_id=f"toolu_{index}",
        content=(TextBlock(text=f"RESULT-{index} " + filler(result_chars, "data")),),
    )
    return (
        Message(role="assistant", content=(TextBlock(text=f"Looking up item {index}."), use)),
        Message(role="user", content=(result,)),
    )


def conversation(turns: int, *, result_chars: int = 400) -> tuple[Message, ...]:
    """A user question, ``turns`` tool turns, and nothing after: it ends with a user message."""
    messages: list[Message] = [Message.user("Please check the example.com release items.")]
    for i in range(turns):
        messages.extend(tool_turn(i, result_chars))
    return tuple(messages)


def oversized() -> tuple[Message, ...]:
    """About 5,100 tokens: more than ``stub-small``'s window holds with its output budget,
    and a kept last turn of about 360 tokens."""
    return conversation(14, result_chars=1_200)


def chat(turns: int, chars: int = 300) -> tuple[Message, ...]:
    """Plain chat: ``turns`` user and assistant pairs, then a closing user message."""
    messages: list[Message] = []
    for i in range(turns):
        messages.append(Message.user(f"Q{i} " + filler(chars, "ask")))
        messages.append(Message.assistant(f"A{i} " + filler(chars, "say")))
    messages.append(Message.user("And finally?"))
    return tuple(messages)
