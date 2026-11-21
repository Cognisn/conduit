"""An in-memory skill source, for tests and for products that build skills in code."""

from __future__ import annotations

from collections.abc import Iterable

from conduit.core.backends import Skill, SkillSummary
from conduit.core.errors import ConfigurationError
from conduit.skills.errors import SkillNotFound
from conduit.skills.format import is_skill_name

__all__ = ["MemorySkillSource"]


class MemorySkillSource:
    """Holds skills given in code. ``load_skill`` raises ``SkillNotFound`` when absent.

    It stores what it is given as it is: validation of names and versions is
    progressive disclosure's, so it applies to every source alike.
    """

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.summary.name in self._skills:
                raise ConfigurationError("a skill name is given twice")
            self._skills[skill.summary.name] = skill

    async def list_skills(self) -> list[SkillSummary]:
        return [self._skills[name].summary for name in sorted(self._skills)]

    async def load_skill(self, name: str) -> Skill:
        if name not in self._skills:
            raise SkillNotFound(name if is_skill_name(name) else None)
        return self._skills[name]
