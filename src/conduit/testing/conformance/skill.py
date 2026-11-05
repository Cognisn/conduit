"""The skill source conformance suite."""

# Suites are test code shipped for products to run under pytest: assertions are the point.
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import ClassVar

from conduit.core.backends import Skill, SkillSource, SkillSummary
from conduit.testing.conformance.base import ConformanceSuite

#: The synthetic skills every scenario installs in the source under test.
SUITE_SKILLS: tuple[Skill, ...] = (
    Skill(
        summary=SkillSummary(
            name="release-notes",
            description="Write release notes from merged changes for example.com products.",
            version="1.2.0",
        ),
        body=(
            "# Release notes\n\n"
            "1. Group the changes by kind.\n"
            "2. Name the component first, then the change.\n"
            "3. Keep the naïve reader in mind: no internal jargon.\n"
        ),
        resources={
            "references/style.md": "# Style\n\nUse Australian English.\n",
            "templates/entry.md": "## <version>\n\n### Added\n- <line>\n",
        },
    ),
    Skill(
        summary=SkillSummary(
            name="status-report",
            description="Summarise a week's work for the example.org status page.",
        ),
        body="# Status report\n\nThree short paragraphs: done, in progress, blocked.\n",
    ),
    Skill(
        summary=SkillSummary(
            name="host-triage",
            description="Triage an alert for a host in 192.0.2.0/24.",
            version="3",
        ),
        body="# Host triage\n\nCheck the host's recent changes before its alerts.\n",
        resources={"checklist.md": "- recent changes\n- open alerts\n"},
    ),
)


class SkillSourceSuite(ConformanceSuite):
    """Listing, loading, resources, absent skills and concurrent loads.

    A product subclasses it and overrides ``make_source`` to return its source holding
    exactly the given skills (``SUITE_SKILLS``), each with its body and text resources.
    """

    interface: ClassVar[str] = "skill_source"

    async def make_source(self, skills: Sequence[Skill]) -> SkillSource:
        """A source holding exactly ``skills``. Products override this."""
        raise NotImplementedError("override make_source to return your skill source")

    async def close_source(self, source: SkillSource) -> None:
        """Release a source after a scenario. Override if the source needs closing."""
        closer = getattr(source, "aclose", None)
        if closer is not None:
            await closer()

    def _scenario(self, body: Callable[[SkillSource], Awaitable[None]]) -> None:
        async def scenario() -> None:
            source = await self.make_source(SUITE_SKILLS)
            try:
                await body(source)
            finally:
                await self.close_source(source)

        self.run(scenario)

    def test_every_skill_is_listed_once_with_its_summary(self) -> None:
        async def body(source: SkillSource) -> None:
            listed = list(await source.list_skills())
            assert all(isinstance(s, SkillSummary) for s in listed)
            assert sorted(s.name for s in listed) == sorted(s.summary.name for s in SUITE_SKILLS)
            by_name = {s.name: s for s in listed}
            for skill in SUITE_SKILLS:
                assert by_name[skill.summary.name] == skill.summary

        self._scenario(body)

    def test_every_listed_skill_loads_with_its_body_and_resources(self) -> None:
        async def body(source: SkillSource) -> None:
            for summary in await source.list_skills():
                loaded = await source.load_skill(summary.name)
                expected = next(s for s in SUITE_SKILLS if s.summary.name == summary.name)
                assert loaded.summary == expected.summary
                assert loaded.body == expected.body
                assert dict(loaded.resources) == dict(expected.resources)

        self._scenario(body)

    def test_a_skill_without_a_version_or_resources_loads(self) -> None:
        async def body(source: SkillSource) -> None:
            loaded = await source.load_skill("status-report")
            assert loaded.summary.version is None
            assert dict(loaded.resources) == {}

        self._scenario(body)

    def test_an_absent_skill_is_a_lookup_error(self) -> None:
        async def body(source: SkillSource) -> None:
            for name in ("never-installed", "release-note", "../release-notes", ""):
                raised: BaseException | None = None
                try:
                    await source.load_skill(name)
                except LookupError as error:
                    raised = error
                assert isinstance(raised, LookupError), "load_skill must raise LookupError"

        self._scenario(body)

    def test_concurrent_loads_return_the_skill_asked_for(self) -> None:
        async def body(source: SkillSource) -> None:
            names = [s.summary.name for s in SUITE_SKILLS] * 5
            loaded = await asyncio.gather(*(source.load_skill(n) for n in names))
            assert [s.summary.name for s in loaded] == names
            assert [s.body for s in loaded] == [
                next(k for k in SUITE_SKILLS if k.summary.name == n).body for n in names
            ]

        self._scenario(body)
