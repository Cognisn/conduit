"""The use_skill tool: a Tool for a ToolCatalogue that discloses skills on request."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.tools import (
    TaskTools,
    Tool,
    ToolCallContext,
    ToolCatalogue,
    ToolOutcome,
    tool_spec,
)
from conduit.skills.disclosure import DisclosureLimits, SkillSession
from conduit.skills.errors import SkillUnavailable
from conduit.skills.memory import MemorySkillSource
from conduit.skills.tool import UseSkillTool, with_use_skill
from tests.unit.skills.helpers import MARKER, skill

CONTEXT = ToolCallContext(task_class="chat", run_id="run_1", call_id="call_1")


def _tool(*skills: Any, **kwargs: Any) -> UseSkillTool:
    return SkillSession(MemorySkillSource(skills), **kwargs).tool(task_classes={"chat"})


async def _product_executor(
    tool: Any, arguments: dict[str, Any], context: ToolCallContext
) -> ToolOutcome:
    return ToolOutcome(ok=True, content=f"product ran {tool.name}")


class TestDeclaration:
    def test_it_is_a_tool_the_catalogue_accepts(self) -> None:
        tool = _tool(skill("a"))
        assert isinstance(tool, Tool)
        catalogue = ToolCatalogue.from_tools([tool])
        assert catalogue.registry_for("chat").names() == ["use_skill"]
        assert catalogue.registry_for("other").names() == []

    def test_its_schema_takes_a_name_and_an_optional_resource(self) -> None:
        schema = tool_spec(_tool(skill("a"))).input_schema
        assert schema["required"] == ["name"]
        assert set(schema["properties"]) == {"name", "resource"}
        assert schema["additionalProperties"] is False

    def test_it_needs_a_task_class(self) -> None:
        session = SkillSession(MemorySkillSource([]))
        from conduit.core.errors import ToolRegistrationError

        with pytest.raises(ToolRegistrationError):
            ToolCatalogue.from_tools([session.tool(task_classes=set())])


class TestCalls:
    async def test_loading_a_skill_answers_with_its_resources_not_its_body(self) -> None:
        tool = _tool(skill("a", f"Body {MARKER}\n", resources={"r.md": "R", "s.md": "S"}))
        outcome = await tool({"name": "a"}, CONTEXT)
        assert outcome.ok
        assert "a" in outcome.content and "r.md" in outcome.content and "s.md" in outcome.content
        assert MARKER not in outcome.content
        assert [s.name for s in tool.session.loaded] == ["a"]

    async def test_a_second_call_says_it_is_already_loaded(self) -> None:
        tool = _tool(skill("a"))
        await tool({"name": "a"}, CONTEXT)
        outcome = await tool({"name": "a"}, CONTEXT)
        assert outcome.ok
        assert "already" in outcome.content
        assert [m.action for m in tool.session.markers] == ["loaded", "used"]

    async def test_a_resource_is_answered_with_its_text(self) -> None:
        tool = _tool(skill("a", resources={"r.md": "Resource text."}))
        await tool({"name": "a"}, CONTEXT)
        outcome = await tool({"name": "a", "resource": "r.md"}, CONTEXT)
        assert outcome == ToolOutcome(ok=True, content="Resource text.")

    async def test_a_resource_before_its_skill_is_loaded_is_refused(self) -> None:
        tool = _tool(skill("a", resources={"r.md": "R"}))
        outcome = await tool({"name": "a", "resource": "r.md"}, CONTEXT)
        assert (outcome.ok, outcome.error_code) == (False, "skill_not_loaded")
        assert outcome.content != "R"

    @pytest.mark.parametrize(
        ("arguments", "code"),
        [
            ({"name": f"{MARKER}-x"}, "skill_unknown"),
            ({"name": f"../{MARKER}"}, "skill_unknown"),
            ({}, "input_invalid"),
            ({"name": 3}, "input_invalid"),
            ({"name": "a", "extra": MARKER}, "input_invalid"),
            ({"name": "a" * 65}, "input_invalid"),
            ({"name": "a", "resource": f"../{MARKER}"}, "path_traversal"),
            ({"name": "a", "resource": f"/{MARKER}"}, "path_absolute"),
            ({"name": "a", "resource": f"{MARKER}.md"}, "resource_missing"),
        ],
    )
    async def test_a_refusal_is_an_error_outcome_that_never_echoes(
        self, arguments: dict[str, Any], code: str
    ) -> None:
        tool = _tool(skill("a", resources={"r.md": "R"}))
        await tool({"name": "a"}, CONTEXT)
        outcome = await tool(arguments, CONTEXT)
        assert (outcome.ok, outcome.error_code) == (False, code)
        assert MARKER not in outcome.content

    async def test_an_unknown_skill_lists_the_available_ones(self) -> None:
        outcome = await _tool(skill("alpha"), skill("beta"))({"name": "gamma"}, CONTEXT)
        assert "alpha" in outcome.content and "beta" in outcome.content
        assert "gamma" not in outcome.content

    async def test_a_limit_is_an_error_outcome(self) -> None:
        tool = _tool(skill("a"), skill("b"), limits=DisclosureLimits(max_loaded=1))
        await tool({"name": "a"}, CONTEXT)
        outcome = await tool({"name": "b"}, CONTEXT)
        assert (outcome.ok, outcome.error_code) == (False, "too_many_loaded")

    async def test_an_unavailable_source_is_an_error_outcome(self) -> None:
        class Down(MemorySkillSource):
            async def load_skill(self, name: str) -> Any:
                raise SkillUnavailable("load the skill", "OSError")

        tool = SkillSession(Down([skill("a")])).tool(task_classes={"chat"})
        outcome = await tool({"name": "a"}, CONTEXT)
        assert (outcome.ok, outcome.error_code) == (False, "skill_unavailable")

    async def test_an_unexpected_failure_is_not_swallowed(self) -> None:
        class Broken(MemorySkillSource):
            async def load_skill(self, name: str) -> Any:
                raise RuntimeError("unexpected")

        tool = SkillSession(Broken([skill("a")])).tool(task_classes={"chat"})
        with pytest.raises(RuntimeError):
            await tool({"name": "a"}, CONTEXT)

    async def test_events_carry_the_call_context(self) -> None:
        from conduit.core.events import MemoryEventSink

        sink = MemoryEventSink()
        tool = _tool(skill("a"), events=sink)
        await tool({"name": "a"}, CONTEXT)
        assert (sink.events[0].run_id, sink.events[0].task_class) == ("run_1", "chat")


class TestExecutor:
    async def test_the_executor_runs_use_skill_and_hands_on_every_other_tool(self) -> None:
        tool = _tool(skill("a"))

        class Other:
            name = "lookup"
            description = "Look something up."
            task_classes = frozenset({"chat"})

        tools = TaskTools(
            ToolCatalogue.from_tools([tool, Other()]).registry_for("chat"),
            with_use_skill(_product_executor),
        )
        loaded = await tools.call(tools.resolve("use_skill"), {"name": "a"}, CONTEXT)
        assert loaded.ok and "a" in loaded.content
        other = await tools.call(tools.resolve("lookup"), {}, CONTEXT)
        assert other.content == "product ran lookup"
