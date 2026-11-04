"""The tool protocol, the tool catalogue and per-task registries."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from conduit.core.errors import ToolRegistrationError, UnknownTool
from conduit.core.tools import (
    TaskTools,
    Tool,
    ToolCallContext,
    ToolCatalogue,
    ToolOutcome,
    tool_spec,
)


class LookupInput(BaseModel):
    host: str


class Lookup:
    name: ClassVar[str] = "lookup_host"
    description: ClassVar[str] = "Look up a host by name."
    task_classes: ClassVar[frozenset[str]] = frozenset({"triage", "chat"})
    input_model: ClassVar[type[BaseModel]] = LookupInput


class Report:
    name: ClassVar[str] = "write_report"
    description: ClassVar[str] = "Write a report."
    task_classes: ClassVar[frozenset[str]] = frozenset({"chat"})
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
    }


def _tool(**attributes: Any) -> type:
    base: dict[str, Any] = {
        "name": "a_tool",
        "description": "d",
        "task_classes": frozenset({"chat"}),
    }
    base.update(attributes)
    return type("Declared", (), base)


class TestCatalogue:
    def test_registries_hold_only_the_task_class_tools(self) -> None:
        catalogue = ToolCatalogue.from_tools([Lookup, Report])
        triage = catalogue.registry_for("triage")
        assert triage.task_class == "triage"
        assert triage.names() == ["lookup_host"]
        assert catalogue.registry_for("chat").names() == ["lookup_host", "write_report"]
        assert catalogue.registry_for("other").names() == []
        assert isinstance(Lookup, Tool)

    def test_resolution_is_exact_and_refuses_other_tools(self) -> None:
        triage = ToolCatalogue.from_tools([Lookup, Report]).registry_for("triage")
        assert triage.resolve("lookup_host") is Lookup
        with pytest.raises(UnknownTool):
            triage.resolve("write_report")
        with pytest.raises(UnknownTool):
            triage.resolve("LOOKUP_HOST")

    @pytest.mark.parametrize(
        "declared",
        [
            pytest.param(_tool(name="Has Caps"), id="bad-name"),
            pytest.param(_tool(name=""), id="empty-name"),
            pytest.param(_tool(description=""), id="no-description"),
            pytest.param(_tool(task_classes=frozenset()), id="no-task-classes"),
            pytest.param(_tool(task_classes={"chat"}), id="not-a-frozenset"),
        ],
    )
    def test_a_badly_declared_tool_is_refused(self, declared: type) -> None:
        with pytest.raises(ToolRegistrationError):
            ToolCatalogue.from_tools([declared])

    def test_a_name_declared_twice_is_refused(self) -> None:
        with pytest.raises(ToolRegistrationError):
            ToolCatalogue.from_tools([Lookup, _tool(name="lookup_host")])


class TestSpecs:
    def test_specs_come_from_the_input_model_or_schema(self) -> None:
        lookup = tool_spec(Lookup)
        assert lookup.name == "lookup_host"
        assert lookup.input_schema["properties"]["host"]["type"] == "string"
        assert tool_spec(Report).input_schema["properties"]["title"] == {"type": "string"}
        assert tool_spec(_tool()).input_schema == {"type": "object"}


class TestTaskTools:
    async def test_calls_go_through_the_executor_with_their_context(self) -> None:
        seen: list[tuple[Any, dict[str, Any], ToolCallContext]] = []

        async def executor(
            tool: Any, arguments: dict[str, Any], context: ToolCallContext
        ) -> ToolOutcome:
            seen.append((tool, arguments, context))
            return ToolOutcome(ok=True, content="192.0.2.10", record_id="rec_1")

        tools = TaskTools(ToolCatalogue.from_tools([Lookup]).registry_for("chat"), executor)
        assert tools.task_class == "chat"
        assert [s.name for s in tools.specs()] == ["lookup_host"]
        context = ToolCallContext(task_class="chat", run_id="run_1", call_id="toolu_1")
        outcome = await tools.call(tools.resolve("lookup_host"), {"host": "example.com"}, context)
        assert outcome.ok and outcome.content == "192.0.2.10"
        assert seen == [(Lookup, {"host": "example.com"}, context)]
