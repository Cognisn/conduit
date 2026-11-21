"""The tool protocol, the tool catalogue and per-task tool registries.

A product declares tools; each names the task classes it serves. ``ToolCatalogue``
checks every declaration once, and ``registry_for(task_class)`` gives the registry for
one task class, which resolves exact names only: a tool outside the task's registry
is ``UnknownTool`` before anything runs. ``TaskTools`` pairs a registry with the
product's ``ToolExecutor`` (its own pipeline: validation, authorisation, execution,
recording); this is the shape VOR's tool layer consumes.

Which tools exist, and policy such as read and write separation, are the product's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from conduit.core.errors import ToolRegistrationError, UnknownTool
from conduit.core.types import ToolSpec

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@runtime_checkable
class Tool(Protocol):
    """A declared tool. It may also carry ``input_model`` (a pydantic model) or
    ``input_schema`` (a JSON Schema mapping) describing its arguments."""

    name: str
    description: str
    task_classes: frozenset[str]


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Who a tool call is for and why: the executor records it and authorises on it."""

    task_class: str
    run_id: str
    call_id: str
    correlation_id: str | None = None
    subject_type: str | None = None
    subject_id: str | None = None
    actor_id: str | None = None  # the signed-in user a chat acts for; None if autonomous


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The executor's answer. ``content`` goes back to the model as untrusted content."""

    ok: bool
    content: str
    record_id: str | None = None  # the product's own record of this call
    error_code: str | None = None  # input_invalid, forbidden, timeout and so on


class ToolExecutor(Protocol):
    async def __call__(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome: ...


def input_schema(tool: Any) -> dict[str, Any]:
    model = getattr(tool, "input_model", None)
    if isinstance(model, type) and issubclass(model, BaseModel):
        return model.model_json_schema()
    schema = getattr(tool, "input_schema", None)
    if isinstance(schema, Mapping):
        return dict(schema)
    return {"type": "object"}


def tool_spec(tool: Any) -> ToolSpec:
    """The tool as the model sees it."""
    return ToolSpec(name=tool.name, description=tool.description, input_schema=input_schema(tool))


def _check(tool: Any) -> None:
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ToolRegistrationError(f"tool {tool!r} needs a lower-case snake_case name")
    description = getattr(tool, "description", None)
    if not isinstance(description, str) or not description.strip():
        raise ToolRegistrationError(f"tool {name} needs a description")
    tasks = getattr(tool, "task_classes", None)
    if (
        not isinstance(tasks, frozenset)
        or not tasks
        or not all(isinstance(t, str) and t for t in tasks)
    ):
        raise ToolRegistrationError(f"tool {name} needs a non-empty frozenset of task classes")


class ToolRegistry:
    """The tools one task class may call. Resolution is exact; anything else is refused."""

    def __init__(self, task_class: str, tools: Mapping[str, Any]) -> None:
        self.task_class = task_class
        self._tools = dict(tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def resolve(self, name: str) -> Any:
        if name not in self._tools:
            raise UnknownTool(f"no tool of that name for task class {self.task_class}")
        return self._tools[name]

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool_spec(self._tools[name]) for name in self.names())


class ToolCatalogue:
    """Every declared tool, checked; gives each task class its registry."""

    def __init__(self, tools: Mapping[str, Any]) -> None:
        self._tools = dict(tools)

    @classmethod
    def from_tools(cls, tools: Iterable[Any]) -> ToolCatalogue:
        found: dict[str, Any] = {}
        for tool in tools:
            _check(tool)
            if tool.name in found:
                raise ToolRegistrationError(f"tool {tool.name} is declared twice")
            found[tool.name] = tool
        return cls(found)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def registry_for(self, task_class: str) -> ToolRegistry:
        return ToolRegistry(
            task_class,
            {name: tool for name, tool in self._tools.items() if task_class in tool.task_classes},
        )


class TaskTools:
    """The tools one task may use: its registry plus the product's executor."""

    def __init__(self, registry: ToolRegistry, executor: ToolExecutor) -> None:
        self._registry = registry
        self._executor = executor

    @property
    def task_class(self) -> str:
        return self._registry.task_class

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._registry.specs()

    def resolve(self, name: str) -> Any:
        """The registered tool, or ``UnknownTool``."""
        return self._registry.resolve(name)

    async def call(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome:
        return await self._executor(tool, arguments, context)
