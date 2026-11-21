"""Shared fixtures for the agent runner tests: a stub runtime, tools and recorders.

Everything here is synthetic. The runtime is started through ``conduit.core.runtime.start``
with stub providers, so every model call passes through the real registry, metering and
assignment checks. Two providers are configured: ``primary`` serves the ``chat`` task class
on ``stub-echo``, and ``specialist`` serves ``review`` on ``stub-large`` (a stub model added
by a catalogue overlay), so a hand-off is visible as another provider and another model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from conduit.core.backends import ApprovalDecision, ApprovalRequest
from conduit.core.events import Event, EventSink
from conduit.core.runtime import ConduitSettings, Runtime, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.tools import (
    TaskTools,
    ToolCallContext,
    ToolCatalogue,
    ToolExecutor,
    ToolOutcome,
)
from conduit.core.types import ModelRequest, TextBlock, ToolResultBlock
from conduit.providers.stub import StubFactory, StubScript, StubTurn

CHAT = TaskClass(name="chat", floor=CapabilityFloor(tools=True))
REVIEW = TaskClass(name="review", floor=CapabilityFloor(tools=True))
TRIAGE = TaskClass(name="triage")

STUB_LARGE = {
    "model_id": "stub-large",
    "provider": "stub",
    "kind": "chat",
    "residency_scope": "local",
    "capabilities": {"tools": True, "structured_output": True},
    "context_window": 200000,
    "output_limit": 8192,
}


class ScriptLog:
    """A stub script that records every request it serves, then answers from ``script``."""

    def __init__(self, script: StubScript) -> None:
        self._script = script
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest, index: int) -> StubTurn:
        self.requests.append(request)
        return self._script(request, index)


def settings(*, triage: bool = False) -> ConduitSettings:
    assignments: dict[str, Any] = {
        "chat": {"provider": "primary", "model": "stub-echo", "max_output_tokens": 1024},
        "review": {"provider": "specialist", "model": "stub-large", "max_output_tokens": 512},
    }
    if triage:
        assignments["triage"] = {"provider": "primary", "model": "stub-echo"}
    return ConduitSettings.model_validate(
        {
            "providers": {
                "primary": {"kind": "stub", "script": "primary", "not_for_production": True},
                "specialist": {
                    "kind": "stub",
                    "script": "specialist",
                    "not_for_production": True,
                },
            },
            "assignments": assignments,
            "catalogue": [{"name": "agent-tests", "add": [STUB_LARGE]}],
            "require_all_assignments": False,
        }
    )


async def runtime(
    primary: StubScript,
    specialist: StubScript | None = None,
    *,
    events: EventSink | None = None,
    triage: bool = False,
) -> Runtime:
    """A started runtime with stub providers playing ``primary`` and ``specialist``."""
    scripts: dict[str, StubScript] = {"primary": primary}
    scripts["specialist"] = specialist if specialist is not None else primary
    classes = [CHAT, REVIEW, TRIAGE] if triage else [CHAT, REVIEW]
    return await start(
        settings(triage=triage),
        task_classes=classes,
        factories=[StubFactory(scripts=scripts)],
        secrets=LiteralSecretResolver({}),
        events=events,
    )


@dataclass
class SampleTool:
    """A declared tool, as a product would declare one."""

    name: str
    description: str = "A synthetic tool for the agent tests."
    task_classes: frozenset[str] = frozenset({"chat"})
    requires_approval: Any = False
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {"q": {"type": "string"}}}
    )


@dataclass
class Executor:
    """A product's tool executor: answers from ``answers`` and records every call."""

    answers: Mapping[str, str | Callable[[dict[str, Any]], str]] = field(default_factory=dict)
    delay_s: float = 0.0
    fail_with: BaseException | None = None
    calls: list[tuple[str, dict[str, Any], ToolCallContext]] = field(default_factory=list)

    async def __call__(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome:
        self.calls.append((tool.name, dict(arguments), context))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with
        answer = self.answers.get(tool.name, f"result of {tool.name}")
        content = answer(arguments) if callable(answer) else answer
        return ToolOutcome(ok=True, content=content, record_id="rec-1")


def task_tools(
    tools: Sequence[SampleTool], executor: ToolExecutor, task_class: str = "chat"
) -> TaskTools:
    return TaskTools(ToolCatalogue.from_tools(tools).registry_for(task_class), executor)


@dataclass
class Approvals:
    """An approval handler that decides from ``approve`` and records every request."""

    approve: bool | Callable[[ApprovalRequest], bool] = True
    fail_with: BaseException | None = None
    requests: list[ApprovalRequest] = field(default_factory=list)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        approved = self.approve(request) if callable(self.approve) else self.approve
        return ApprovalDecision(approved=approved, decided_by="reviewer@example.com")


class Sink:
    """An event sink keeping every event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    def of(self, kind: type[Event]) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


def result_text(block: ToolResultBlock) -> str:
    """The text of a tool result's first (text) block."""
    first = block.content[0]
    assert isinstance(first, TextBlock)
    return first.text
