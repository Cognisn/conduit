"""The skill source conformance suite, against the memory and filesystem sources, and
the suite's own checks: it must fail a source that breaks the contract."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from conduit.core.backends import Skill, SkillSource, SkillSummary
from conduit.skills.filesystem import FilesystemSkillSource
from conduit.skills.memory import MemorySkillSource
from conduit.testing.conformance import SkillSourceSuite
from conduit.testing.conformance.skill import SUITE_SKILLS
from tests.conformance.meta import fails


class TestMemorySkillSource(SkillSourceSuite):
    async def make_source(self, skills: Sequence[Skill]) -> SkillSource:
        return MemorySkillSource(skills)


def _write_skills(root: Path, skills: Sequence[Skill]) -> None:
    for skill in skills:
        folder = root / skill.summary.name
        folder.mkdir(parents=True)
        version = f"version: {skill.summary.version}\n" if skill.summary.version else ""
        head = f"name: {skill.summary.name}\ndescription: {skill.summary.description}\n"
        (folder / "SKILL.md").write_text(
            f"---\n{head}{version}---\n{skill.body}", encoding="utf-8", newline="\n"
        )
        for path, text in skill.resources.items():
            target = folder.joinpath(*path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")


class _TemporaryFilesystemSource(FilesystemSkillSource):
    async def aclose(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


class TestFilesystemSkillSource(SkillSourceSuite):
    async def make_source(self, skills: Sequence[Skill]) -> SkillSource:
        root = Path(tempfile.mkdtemp(prefix="conduit-skills-"))
        _write_skills(root, skills)
        return _TemporaryFilesystemSource(root)


class _ListsNothing(MemorySkillSource):
    async def list_skills(self) -> list[SkillSummary]:
        return []


class _ListsTwice(MemorySkillSource):
    async def list_skills(self) -> list[SkillSummary]:
        listed = await super().list_skills()
        return listed + listed[:1]


class _DropsResources(MemorySkillSource):
    async def load_skill(self, name: str) -> Skill:
        skill = await super().load_skill(name)
        return skill.model_copy(update={"resources": {}})


class _TrimsBodies(MemorySkillSource):
    async def load_skill(self, name: str) -> Skill:
        skill = await super().load_skill(name)
        return skill.model_copy(update={"body": skill.body.strip()})


class _NoneForAbsent(MemorySkillSource):
    async def load_skill(self, name: str) -> Skill:
        try:
            return await super().load_skill(name)
        except LookupError:
            pass
        return None  # type: ignore[return-value]


class _LastAsked(MemorySkillSource):
    """Answers each load with whichever skill was asked for last: a shared slot."""

    def __init__(self, skills: Sequence[Skill]) -> None:
        super().__init__(skills)
        self._slot = ""

    async def load_skill(self, name: str) -> Skill:
        self._slot = name
        await asyncio.sleep(0)
        return await super().load_skill(self._slot)


class _VersionLost(MemorySkillSource):
    async def list_skills(self) -> list[SkillSummary]:
        return [s.model_copy(update={"version": None}) for s in await super().list_skills()]


def _suite(source: type[MemorySkillSource]) -> SkillSourceSuite:
    class Suite(SkillSourceSuite):
        async def make_source(self, skills: Sequence[Skill]) -> SkillSource:
            return source(skills)

    return Suite()


@pytest.mark.parametrize(
    ("broken", "scenario"),
    [
        (_ListsNothing, "test_every_skill_is_listed_once_with_its_summary"),
        (_ListsTwice, "test_every_skill_is_listed_once_with_its_summary"),
        (_VersionLost, "test_every_skill_is_listed_once_with_its_summary"),
        (_DropsResources, "test_every_listed_skill_loads_with_its_body_and_resources"),
        (_TrimsBodies, "test_every_listed_skill_loads_with_its_body_and_resources"),
        (_NoneForAbsent, "test_an_absent_skill_is_a_lookup_error"),
        (_LastAsked, "test_concurrent_loads_return_the_skill_asked_for"),
    ],
)
def test_the_suite_fails_a_broken_source(broken: type[MemorySkillSource], scenario: str) -> None:
    with fails():
        getattr(_suite(broken), scenario)()


def test_the_suite_skills_are_synthetic_and_valid() -> None:
    from conduit.skills.format import is_skill_name, is_skill_version

    for skill in SUITE_SKILLS:
        assert is_skill_name(skill.summary.name)
        assert skill.summary.version is None or is_skill_version(skill.summary.version)
        assert "\n" not in skill.summary.description
