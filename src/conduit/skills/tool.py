"""The ``use_skill`` tool: progressive disclosure as a Tool for a ``ToolCatalogue``.

``UseSkillTool`` is declared like any product tool (a name, a description, the task
classes it serves and an input model), so it registers in a ``ToolCatalogue``. It runs
itself: ``with_use_skill(executor)`` wraps the product's ``ToolExecutor`` so a
``use_skill`` call goes to the tool and every other call to the product's executor.

Called with ``name``, it loads that skill into its ``SkillSession`` and answers that
the skill is loaded, listing its resources; the instructions themselves go into the
system prompt (see ``conduit.skills.disclosure``), never into the tool result. Called
with ``name`` and ``resource``, it answers with that resource's text.

A refusal is an error outcome (``ok=False``) with a fixed message and a fixed
``error_code``: ``input_invalid``, ``skill_unknown``, ``skill_not_loaded``,
``too_many_loaded``, ``loaded_too_large``, a resource path code (``path_traversal``,
``path_absolute``, ``path_invalid``, ``path_reserved``), ``resource_missing``,
``resource_too_large``, ``skill_invalid`` or ``skill_unavailable``. No outcome echoes
the model's arguments; an unknown skill's outcome lists the skills that are available.
Failures that are not the skills layer's own propagate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from conduit.core.tools import ToolCallContext, ToolExecutor, ToolOutcome
from conduit.skills.disclosure import SkillSession
from conduit.skills.errors import (
    SkillError,
    SkillInvalid,
    SkillNotFound,
    SkillRefused,
    SkillResourceRefused,
    SkillUnavailable,
)
from conduit.skills.format import MAX_NAME_LENGTH, MAX_RESOURCE_PATH_LENGTH

__all__ = ["UseSkillInput", "UseSkillTool", "with_use_skill"]

_PATH = "That resource path is not valid: give one of the skill's resource paths as listed."
_MESSAGES = {
    "input_invalid": (
        "use_skill needs the name of a listed skill and, to read one of a loaded skill's "
        "resources, the resource's path."
    ),
    "skill_not_loaded": "Load the skill with use_skill first; its resources can be read then.",
    "too_many_loaded": "No more skills can be loaded in this conversation.",
    "loaded_too_large": "That skill is too large to load alongside the skills already loaded.",
    "path_traversal": _PATH,
    "path_absolute": _PATH,
    "path_invalid": _PATH,
    "path_reserved": _PATH,
    "resource_too_large": "That resource is too large to read.",
    "skill_invalid": "That skill cannot be loaded: its definition is invalid.",
    "skill_unavailable": "The skill could not be read just now.",
}
_REFUSED = "The skill request was refused."


class UseSkillInput(BaseModel):
    """The ``use_skill`` arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(
        min_length=1, max_length=MAX_NAME_LENGTH, description="The name of a listed skill."
    )
    resource: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RESOURCE_PATH_LENGTH,
        description="To read one of a loaded skill's resources: its path, as listed.",
    )


def _code(error: SkillError) -> str:
    if isinstance(error, SkillNotFound):
        return "skill_unknown"
    if isinstance(error, SkillRefused | SkillResourceRefused):
        return error.reason
    if isinstance(error, SkillInvalid):
        return "skill_invalid"
    if isinstance(error, SkillUnavailable):
        return "skill_unavailable"
    return "skill_refused"  # pragma: no cover - every skills error is one of the above


class UseSkillTool:
    """Loads a skill, or reads one of its resources, on the model's request."""

    name = "use_skill"
    description = (
        "Load a skill's instructions by its name, before a task the skill covers; they "
        "then appear in the system prompt. To read one of a loaded skill's resources, "
        "give the skill's name and the resource's path."
    )
    input_model = UseSkillInput

    def __init__(self, session: SkillSession, *, task_classes: Iterable[str]) -> None:
        self.session = session
        self.task_classes = frozenset(task_classes)

    async def __call__(
        self, arguments: Mapping[str, Any], context: ToolCallContext | None = None
    ) -> ToolOutcome:
        parsed: UseSkillInput | None = None
        try:
            parsed = UseSkillInput.model_validate(dict(arguments))
        except ValidationError:
            parsed = None
        if parsed is None:
            return await self._failure("input_invalid", None)
        code: str | None = None
        try:
            if parsed.resource is not None:
                text = await self.session.read_resource(
                    parsed.name, parsed.resource, context=context
                )
                return ToolOutcome(ok=True, content=text)
            already = any(s.name == parsed.name for s in self.session.loaded)
            loaded = await self.session.load(parsed.name, context=context)
        except SkillError as error:
            code = _code(error)
        if code is not None:
            return await self._failure(code, parsed.name)
        version = f" (version {loaded.version})" if loaded.version else ""
        state = (
            "is already loaded; its instructions are in the system prompt."
            if already
            else "is loaded: its instructions are now in the system prompt. Follow them."
        )
        content = f"Skill {loaded.name}{version} {state}"
        if loaded.resources:
            content += (
                " Its resources, which use_skill reads given the skill's name and the path: "
                + ", ".join(loaded.resource_paths)
                + "."
            )
        return ToolOutcome(ok=True, content=content)

    async def _failure(self, code: str, name: str | None) -> ToolOutcome:
        if code == "skill_unknown":
            names = [s.name for s in await self.session.summaries()]
            content = "No skill of that name is available. Available skills: " + (
                ", ".join(names) or "none"
            )
            content += "."
        elif code == "resource_missing":
            loaded = next((s for s in self.session.loaded if s.name == name), None)
            paths = ", ".join(loaded.resource_paths) if loaded is not None else ""
            content = "The skill has no resource at that path. Its resources: " + (paths or "none")
            content += "."
        else:
            content = _MESSAGES.get(code, _REFUSED)
        return ToolOutcome(ok=False, content=content, error_code=code)


def with_use_skill(executor: ToolExecutor) -> ToolExecutor:
    """``executor``, with ``use_skill`` calls run by the tool itself."""

    async def run(tool: Any, arguments: dict[str, Any], context: ToolCallContext) -> ToolOutcome:
        if isinstance(tool, UseSkillTool):
            return await tool(arguments, context)
        return await executor(tool, arguments, context)

    return run
