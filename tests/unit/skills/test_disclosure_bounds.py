"""Disclosure bounds: one listing snapshot under concurrency, and a prefix whose size the
limits count in full (the rendered skills and the listing)."""

from __future__ import annotations

import asyncio

import pytest

from conduit.core.backends import Skill, SkillSummary
from conduit.core.events import MemoryEventSink
from conduit.core.types import TextBlock
from conduit.skills.disclosure import (
    DisclosureLimits,
    LoadedSkill,
    SkillSession,
    render_listing,
    render_skill,
)
from conduit.skills.errors import SkillNotFound, SkillRefused
from conduit.skills.events import SkillLoadedEvent
from conduit.skills.memory import MemorySkillSource
from tests.unit.skills.helpers import MARKER, skill


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _rendered(item: Skill) -> int:
    return _size(
        render_skill(
            LoadedSkill(
                name=item.summary.name,
                version=item.summary.version,
                body=item.body,
                resources=item.resources,
            )
        )
    )


class _Shifting(MemorySkillSource):
    """Lists alpha on its first call and beta on every later one."""

    def __init__(self) -> None:
        super().__init__([skill("alpha"), skill("beta")])
        self.calls = 0

    async def list_skills(self) -> list[SkillSummary]:
        self.calls += 1
        first = self.calls == 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return [skill("alpha").summary] if first else [skill("beta").summary]


class TestOneSnapshot:
    async def test_concurrent_first_callers_share_one_listing(self) -> None:
        source = _Shifting()
        session = SkillSession(source)
        system, loaded = await asyncio.gather(session.system(), session.load("alpha"))
        assert source.calls == 1
        assert loaded.name == "alpha"
        listing = next(b.text for b in system if isinstance(b, TextBlock))
        assert "alpha" in listing and "beta" not in listing
        after = await session.system()
        assert next(b.text for b in after if isinstance(b, TextBlock)) == listing

    async def test_many_concurrent_callers_fetch_once(self) -> None:
        source = _Shifting()
        session = SkillSession(source)
        results = await asyncio.gather(*(session.summaries() for _ in range(10)))
        assert source.calls == 1
        assert all([s.name for s in r] == ["alpha"] for r in results)

    async def test_a_cancelled_first_fetch_leaves_the_next_caller_to_fetch(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class Slow(MemorySkillSource):
            calls = 0

            async def list_skills(self) -> list[SkillSummary]:
                Slow.calls += 1
                if Slow.calls == 1:
                    started.set()
                    await release.wait()
                return await super().list_skills()

        session = SkillSession(Slow([skill("a")]))
        first = asyncio.ensure_future(session.summaries())
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert [s.name for s in await session.summaries()] == ["a"]
        assert Slow.calls == 2


class TestRenderedSize:
    async def test_the_loaded_size_counts_the_rendered_skill(self) -> None:
        item = skill("a", "Body.\n", resources={"r.md": "R"})
        session = SkillSession(MemorySkillSource([item]))
        await session.load("a")
        assert session.loaded_size == _rendered(item)
        assert session.loaded_size > _size(item.body)

    async def test_a_skill_whose_resource_list_is_too_large_is_refused(self) -> None:
        paths = {f"references/{'r' * 40}-{i:03d}.md": "x" for i in range(30)}
        item = skill("a", "x" * 10, resources=paths)
        assert _size(item.body) < 1000 < _rendered(item)
        session = SkillSession(
            MemorySkillSource([item]), limits=DisclosureLimits(max_loaded_bytes=1000)
        )
        with pytest.raises(SkillRefused) as caught:
            await session.load("a")
        assert caught.value.reason == "loaded_too_large"

    async def test_the_rendered_limit_is_inclusive(self) -> None:
        a, b = skill("a", "x" * 60), skill("b", "y" * 40)
        exact = _rendered(a) + _rendered(b)
        session = SkillSession(
            MemorySkillSource([a, b]), limits=DisclosureLimits(max_loaded_bytes=exact)
        )
        await session.load("a")
        await session.load("b")
        assert session.loaded_size == exact
        short = SkillSession(
            MemorySkillSource([a, b]), limits=DisclosureLimits(max_loaded_bytes=exact - 1)
        )
        await short.load("a")
        with pytest.raises(SkillRefused):
            await short.load("b")

    async def test_a_product_rendering_is_what_is_counted(self) -> None:
        session = SkillSession(
            MemorySkillSource([skill("a", "tiny")]),
            limits=DisclosureLimits(max_loaded_bytes=100),
            render_skill=lambda loaded: "z" * 101,
        )
        with pytest.raises(SkillRefused) as caught:
            await session.load("a")
        assert caught.value.reason == "loaded_too_large"

    async def test_events_report_the_rendered_size(self) -> None:
        sink = MemoryEventSink()
        item = skill("a", "Body.\n", resources={"r.md": "R"})
        session = SkillSession(MemorySkillSource([item]), events=sink)
        await session.load("a")
        event = sink.events[0]
        assert isinstance(event, SkillLoadedEvent)
        assert (event.body_size, event.loaded_size) == (_size(item.body), _rendered(item))


class TestListingBounds:
    async def test_at_most_max_listed_skills_are_listed_by_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = SkillSession(
            MemorySkillSource([skill("c"), skill("a"), skill("b")]),
            limits=DisclosureLimits(max_listed=2),
        )
        assert [s.name for s in await session.summaries()] == ["a", "b"]
        assert "1 skill" in caplog.text
        with pytest.raises(SkillNotFound):
            await session.load("c")

    async def test_the_listing_size_is_bounded_and_inclusive(self) -> None:
        items = [skill("a"), skill("b"), skill("c")]
        two = _size(render_listing([s.summary for s in items[:2]]))
        session = SkillSession(
            MemorySkillSource(items), limits=DisclosureLimits(max_listing_bytes=two)
        )
        assert [s.name for s in await session.summaries()] == ["a", "b"]
        system = await session.system()
        listing = next(b.text for b in system if isinstance(b, TextBlock))
        assert _size(listing) == two
        under = SkillSession(
            MemorySkillSource(items), limits=DisclosureLimits(max_listing_bytes=two - 1)
        )
        assert [s.name for s in await under.summaries()] == ["a"]

    async def test_a_listing_with_no_room_lists_nothing(self) -> None:
        session = SkillSession(
            MemorySkillSource([skill("a")]), limits=DisclosureLimits(max_listing_bytes=1)
        )
        assert await session.summaries() == ()
        assert await session.system() == ()

    async def test_a_product_listing_rendering_is_what_is_counted(self) -> None:
        session = SkillSession(
            MemorySkillSource([skill("a"), skill("b")]),
            limits=DisclosureLimits(max_listing_bytes=10),
            render_listing=lambda summaries: "L" * (5 * len(summaries)),
        )
        assert [s.name for s in await session.summaries()] == ["a", "b"]

    @pytest.mark.parametrize(
        "description", ["two\nlines", "tab\there", "sep\u2028here", "", "d" * 1025]
    )
    async def test_a_summary_with_a_bad_description_is_left_out(self, description: str) -> None:
        session = SkillSession(
            MemorySkillSource([skill("good"), skill("bad", description=f"{description}")])
        )
        if description == "":
            # skill() replaces an empty description with its default; build it directly.
            session = SkillSession(
                MemorySkillSource(
                    [
                        skill("good"),
                        Skill(summary=SkillSummary(name="bad", description=""), body="b"),
                    ]
                )
            )
        assert [s.name for s in await session.summaries()] == ["good"]

    def test_listing_limits_must_be_positive(self) -> None:
        from conduit.core.errors import ConfigurationError

        for field in ("max_listed", "max_listing_bytes"):
            with pytest.raises(ConfigurationError):
                DisclosureLimits(**{field: 0})

    async def test_no_marker_reaches_the_log(self, caplog: pytest.LogCaptureFixture) -> None:
        session = SkillSession(
            MemorySkillSource([skill("a", description=f"{MARKER}\nx"), skill("b")]),
            limits=DisclosureLimits(max_listed=1),
        )
        await session.summaries()
        assert MARKER not in caplog.text
