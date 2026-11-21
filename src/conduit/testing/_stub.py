"""A started runtime on the stub provider, and tool fixtures, for the approval suite and
the evaluation harness. Internal: products use them through those two.

Every model call passes through the real provider registry, metering and assignment
checks; only the model is the stub, playing one script. One stub provider serves every
task class, so a script's call index runs on across a hand-off.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from conduit.core.events import EventSink
from conduit.core.runtime import ConduitSettings, Runtime, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import TaskClass
from conduit.core.tools import ToolCallContext, ToolOutcome
from conduit.providers.stub import StubFactory, StubScript

#: The configured name of the stub provider, and the catalogue model it serves.
STUB_PROVIDER = "stub"
STUB_MODEL = "stub-echo"
_SCRIPT = "case"
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


async def start_stub_runtime(
    script: StubScript,
    task_classes: Iterable[TaskClass],
    *,
    events: EventSink | None = None,
) -> Runtime:
    """A runtime assigning every one of ``task_classes`` to the stub playing ``script``."""
    classes = list(task_classes)
    settings = ConduitSettings.model_validate(
        {
            "providers": {
                STUB_PROVIDER: {"kind": "stub", "script": _SCRIPT, "not_for_production": True}
            },
            "assignments": {
                task.name: {"provider": STUB_PROVIDER, "model": STUB_MODEL} for task in classes
            },
        }
    )
    return await start(
        settings,
        task_classes=classes,
        factories=[StubFactory(scripts={_SCRIPT: script})],
        secrets=LiteralSecretResolver({}),
        events=events,
    )


def check_tool_name(name: object) -> str:
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise ValueError("a tool name must be lower-case snake_case")
    return name


@dataclass(frozen=True, slots=True)
class DeclaredTool:
    """A tool declaration as the tool catalogue reads one."""

    name: str
    description: str
    task_classes: frozenset[str]
    input_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    requires_approval: bool = False


class RecordingExecutor:
    """Answers each tool from canned responses and records every call it executes."""

    def __init__(self, responses: Mapping[str, str] | None = None) -> None:
        self._responses = dict(responses or {})
        self.executed: list[tuple[str, dict[str, Any]]] = []

    async def __call__(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome:
        self.executed.append((tool.name, dict(arguments)))
        content = self._responses.get(tool.name, '{"status": "ok"}')
        return ToolOutcome(ok=True, content=content, record_id=f"eval-{len(self.executed):04d}")
