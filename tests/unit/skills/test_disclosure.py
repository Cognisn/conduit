"""Progressive disclosure: listing, loading on demand, resources, limits and markers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conduit.core.backends import Skill, SkillSummary
from conduit.core.errors import ConfigurationError
from conduit.core.events import MemoryEventSink
from conduit.core.transcript import SkillUse
from conduit.core.types import CacheBreakpoint, TextBlock
from conduit.skills.disclosure import DisclosureLimits, SkillSession
from conduit.skills.errors import (
    SkillInvalid,
    SkillNotFound,
    SkillRefused,
    SkillResourceRefused,
)
from conduit.skills.events import (
    SkillLoadedEvent,
    SkillRefusedEvent,
    SkillUnloadedEvent,
    SkillUsedEvent,
)
from conduit.skills.filesystem import FilesystemSkillSource
from conduit.skills.memory import MemorySkillSource
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.skills.helpers import FIXTURES, MARKER, clean, skill, unchained


def _session(*skills: Skill, **kwargs: object) -> SkillSession:
    return SkillSession(MemorySkillSource(skills), **kwargs)  # type: ignore[arg-type]


class TestListing:
    async def test_summaries_come_from_the_source(self) -> None:
        session = SkillSession(FilesystemSkillSource(FIXTURES))
        assert [s.name for s in await session.summaries()] == ["release-notes", "status-report"]

    async def test_the_listing_is_a_snapshot_until_refreshed(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "SKILL.md").write_text("---\nname: a\ndescription: d\n---\nb\n")
        session = SkillSession(FilesystemSkillSource(tmp_path))
        before = await session.system()
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "SKILL.md").write_text("---\nname: b\ndescription: d\n---\nb\n")
        assert await session.system() == before
        await session.refresh()
        assert [s.name for s in await session.summaries()] == ["a", "b"]

    async def test_an_allow_list_narrows_the_listing(self) -> None:
        session = _session(skill("a"), skill("b"), allowed={"b"})
        assert [s.name for s in await session.summaries()] == ["b"]
        with pytest.raises(SkillNotFound):
            await session.load("a")

    async def test_a_summary_the_format_would_refuse_is_left_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = _session(
            skill("good"),
            skill(f"{MARKER} bad"),
            skill("spaced-version", version="has space"),
        )
        assert [s.name for s in await session.summaries()] == ["good"]
        assert MARKER not in caplog.text

    async def test_the_listing_names_each_skill_with_its_description(self) -> None:
        system = await _session(skill("a", description="Use for alpha.")).system()
        text = "".join(b.text for b in system if isinstance(b, TextBlock))
        assert "a" in text and "Use for alpha." in text
        assert "use_skill" in text

    async def test_no_skills_and_no_base_gives_an_empty_system(self) -> None:
        assert await _session().system() == ()


class TestLoading:
    async def test_a_skill_loads_on_demand(self) -> None:
        session = _session(skill("a", "Alpha body.\n", resources={"r.md": "R"}))
        before = session.loaded
        assert before == ()
        loaded = await session.load("a")
        assert (loaded.name, loaded.version, loaded.body) == ("a", "1", "Alpha body.\n")
        assert loaded.resource_paths == ("r.md",)
        assert [s.name for s in session.loaded] == ["a"]

    async def test_loading_twice_keeps_one_copy_and_records_use(self) -> None:
        session = _session(skill("a"))
        first = await session.load("a")
        second = await session.load("a")
        assert first is second
        assert [m.action for m in session.markers] == ["loaded", "used"]

    async def test_skills_stay_in_the_order_they_were_loaded(self) -> None:
        session = _session(skill("a"), skill("b"), skill("c"))
        for name in ("c", "a", "b"):
            await session.load(name)
        assert [s.name for s in session.loaded] == ["c", "a", "b"]

    async def test_an_unlisted_skill_is_not_loaded(self) -> None:
        class Wider(MemorySkillSource):
            async def list_skills(self) -> list[SkillSummary]:
                return []

        session = SkillSession(Wider([skill("hidden")]))
        with pytest.raises(SkillNotFound) as caught:
            await session.load("hidden")
        assert caught.value.skill is None
        unchained(caught.value)

    @pytest.mark.parametrize("name", ["", "A", "../a", f"{MARKER} x", "a" * 65])
    async def test_an_invalid_name_is_refused_without_echo(self, name: str) -> None:
        with pytest.raises(SkillNotFound) as caught:
            await _session(skill("a")).load(name)
        clean(caught.value, MARKER)

    async def test_a_source_that_answers_with_another_skill_is_refused(self) -> None:
        class Wrong(MemorySkillSource):
            async def load_skill(self, name: str) -> Skill:
                return skill("other")

        with pytest.raises(SkillInvalid) as caught:
            await SkillSession(Wrong([skill("a"), skill("other")])).load("a")
        assert caught.value.reason == "name_mismatch"

    async def test_a_source_that_answers_with_an_invalid_version_is_refused(self) -> None:
        class Odd(MemorySkillSource):
            async def load_skill(self, name: str) -> Skill:
                return skill(name, version=f"{MARKER} x")

        with pytest.raises(SkillInvalid) as caught:
            await SkillSession(Odd([skill("a")])).load("a")
        assert caught.value.reason == "version_invalid"
        clean(caught.value, MARKER)

    async def test_a_source_lookup_error_becomes_not_found_unchained(self) -> None:
        class Gone(MemorySkillSource):
            async def load_skill(self, name: str) -> Skill:
                raise LookupError(MARKER)

        with pytest.raises(SkillNotFound) as caught:
            await SkillSession(Gone([skill("a")])).load("a")
        clean(caught.value, MARKER)

    async def test_cancellation_propagates_unchanged(self) -> None:
        started = asyncio.Event()

        class Slow(MemorySkillSource):
            async def load_skill(self, name: str) -> Skill:
                started.set()
                await asyncio.sleep(10)
                raise AssertionError("not reached")

        session = SkillSession(Slow([skill("a")]))
        task = asyncio.ensure_future(session.load("a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.loaded == ()


class TestLimits:
    async def test_the_number_of_loaded_skills_is_bounded(self) -> None:
        session = _session(
            skill("a"), skill("b"), skill("c"), limits=DisclosureLimits(max_loaded=2)
        )
        await session.load("a")
        await session.load("b")
        with pytest.raises(SkillRefused) as caught:
            await session.load("c")
        assert caught.value.reason == "too_many_loaded"
        assert [s.name for s in session.loaded] == ["a", "b"]
        await session.load("a")  # a loaded skill can still be used

    async def test_the_loaded_size_is_bounded(self) -> None:
        session = _session(
            skill("a", "x" * 60),
            skill("b", "x" * 41),
            limits=DisclosureLimits(max_loaded_bytes=100),
        )
        await session.load("a")
        with pytest.raises(SkillRefused) as caught:
            await session.load("b")
        assert caught.value.reason == "loaded_too_large"

    async def test_the_loaded_size_limit_is_inclusive(self) -> None:
        session = _session(
            skill("a", "x" * 60),
            skill("b", "x" * 40),
            limits=DisclosureLimits(max_loaded_bytes=100),
        )
        await session.load("a")
        await session.load("b")
        assert session.loaded_size == 100

    async def test_a_resource_over_the_disclosure_limit_is_refused(self) -> None:
        session = _session(
            skill("a", resources={"big.md": "x" * 11, "ok.md": "x" * 10}),
            limits=DisclosureLimits(max_resource_bytes=10),
        )
        await session.load("a")
        assert await session.read_resource("a", "ok.md") == "x" * 10
        with pytest.raises(SkillResourceRefused) as caught:
            await session.read_resource("a", "big.md")
        assert caught.value.reason == "resource_too_large"

    def test_limits_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError):
            DisclosureLimits(max_loaded=0)

    async def test_concurrent_loads_respect_the_limits(self) -> None:
        names = [f"s{i}" for i in range(10)]
        session = _session(*(skill(n) for n in names), limits=DisclosureLimits(max_loaded=3))
        results = await asyncio.gather(*(session.load(n) for n in names), return_exceptions=True)
        assert sum(not isinstance(r, BaseException) for r in results) == 3
        assert all(isinstance(r, SkillRefused) for r in results if isinstance(r, BaseException))
        assert len(session.loaded) == 3

    async def test_concurrent_loads_of_one_skill_load_it_once(self) -> None:
        loads = 0

        class Counting(MemorySkillSource):
            async def load_skill(self, name: str) -> Skill:
                nonlocal loads
                loads += 1
                await asyncio.sleep(0)
                return await super().load_skill(name)

        session = SkillSession(Counting([skill("a")]))
        await asyncio.gather(*(session.load("a") for _ in range(10)))
        assert loads == 1
        assert [m.action for m in session.markers].count("loaded") == 1


class TestResources:
    async def test_a_resource_is_read_only_once_its_skill_is_loaded(self) -> None:
        session = _session(skill("a", resources={"r.md": "R"}))
        with pytest.raises(SkillRefused) as caught:
            await session.read_resource("a", "r.md")
        assert caught.value.reason == "skill_not_loaded"
        await session.load("a")
        assert await session.read_resource("a", "r.md") == "R"

    @pytest.mark.parametrize(
        ("path", "reason"),
        [
            ("../r.md", "path_traversal"),
            ("/r.md", "path_absolute"),
            ("C:/r.md", "path_absolute"),
            ("dir\\r.md", "path_invalid"),
            ("absent.md", "resource_missing"),
            (f"{MARKER}.md", "resource_missing"),
        ],
    )
    async def test_a_bad_resource_path_is_refused_without_echo(
        self, path: str, reason: str
    ) -> None:
        session = _session(skill("a", resources={"r.md": "R"}))
        await session.load("a")
        with pytest.raises(SkillResourceRefused) as caught:
            await session.read_resource("a", path)
        assert caught.value.reason == reason
        clean(caught.value, MARKER, path)

    async def test_reading_a_resource_records_use(self) -> None:
        session = _session(skill("a", resources={"r.md": "R"}))
        await session.load("a")
        await session.read_resource("a", "r.md")
        assert [m.action for m in session.markers] == ["loaded", "used"]


class TestUnloading:
    async def test_a_skill_can_be_unloaded(self) -> None:
        session = _session(skill("a"), skill("b"))
        await session.load("a")
        await session.load("b")
        await session.unload("a")
        assert [s.name for s in session.loaded] == ["b"]
        assert [m.action for m in session.markers] == ["loaded", "loaded", "unloaded"]

    async def test_unloading_a_skill_not_loaded_is_refused(self) -> None:
        with pytest.raises(SkillRefused) as caught:
            await _session(skill("a")).unload("a")
        assert caught.value.reason == "skill_not_loaded"

    async def test_an_unloaded_skill_frees_its_room(self) -> None:
        session = _session(skill("a"), skill("b"), limits=DisclosureLimits(max_loaded=1))
        await session.load("a")
        await session.unload("a")
        await session.load("b")


class TestMarkers:
    async def test_markers_go_to_the_transcript_store(self) -> None:
        store = MemoryTranscriptStore()
        session = _session(
            skill("a", resources={"r.md": "R"}), transcript=store, session_id="sess-1"
        )
        await session.load("a")
        await session.read_resource("a", "r.md")
        await session.unload("a")
        stored = (await store.load("sess-1")).entries
        assert [(e.skill, e.version, e.action) for e in stored if isinstance(e, SkillUse)] == [
            ("a", "1", "loaded"),
            ("a", "1", "used"),
            ("a", "1", "unloaded"),
        ]

    def test_a_transcript_needs_a_session_id(self) -> None:
        with pytest.raises(ConfigurationError):
            _session(transcript=MemoryTranscriptStore())

    async def test_a_failing_store_leaves_the_skill_unloaded(self) -> None:
        class Broken(MemoryTranscriptStore):
            async def append(self, session_id: str, entries: object) -> tuple[object, ...]:  # type: ignore[override]
                raise OSError("store down")

        session = _session(skill("a"), transcript=Broken(), session_id="sess-1")
        with pytest.raises(OSError):
            await session.load("a")
        assert session.loaded == ()

    async def test_a_restored_session_has_its_skills_in_their_order(self) -> None:
        store = MemoryTranscriptStore()
        source = MemorySkillSource([skill("a"), skill("b"), skill("c")])
        first = SkillSession(source, transcript=store, session_id="s")
        for name in ("b", "a", "c"):
            await first.load(name)
        await first.unload("a")
        transcript = await store.load("s")
        again = SkillSession(source, transcript=store, session_id="s")
        await again.restore(transcript)
        assert [s.name for s in again.loaded] == ["b", "c"]
        assert await again.system() == await first.system()
        assert len((await store.load("s")).entries) == len(transcript.entries)

    async def test_a_restored_skill_that_is_gone_is_marked_unloaded(self) -> None:
        store = MemoryTranscriptStore()
        first = SkillSession(
            MemorySkillSource([skill("a"), skill("b")]), transcript=store, session_id="s"
        )
        await first.load("a")
        await first.load("b")
        again = SkillSession(MemorySkillSource([skill("b")]), transcript=store, session_id="s")
        await again.restore(await store.load("s"))
        assert [s.name for s in again.loaded] == ["b"]
        last = (await store.load("s")).entries[-1]
        assert isinstance(last, SkillUse)
        assert (last.skill, last.action) == ("a", "unloaded")

    async def test_restoring_after_loading_is_refused(self) -> None:
        session = _session(skill("a"))
        await session.load("a")
        with pytest.raises(SkillRefused) as caught:
            await session.restore(None)  # type: ignore[arg-type]
        assert caught.value.reason == "already_started"


class TestEvents:
    async def test_events_carry_names_versions_and_sizes_only(self) -> None:
        sink = MemoryEventSink()
        body = f"Body {MARKER}.\n"
        session = _session(
            skill("a", body, resources={"r.md": f"R {MARKER}"}),
            events=sink,
            run_id="run_1",
            task_class="chat",
        )
        await session.load("a")
        await session.read_resource("a", "r.md")
        await session.unload("a")
        loaded, used, unloaded = sink.events
        assert isinstance(loaded, SkillLoadedEvent)
        assert (loaded.skill, loaded.version, loaded.body_size, loaded.resource_count) == (
            "a",
            "1",
            len(body.encode()),
            1,
        )
        assert (loaded.loaded_count, loaded.loaded_size) == (1, len(body.encode()))
        assert (loaded.run_id, loaded.task_class) == ("run_1", "chat")
        assert isinstance(used, SkillUsedEvent)
        assert (used.resource, used.resource_size) == ("r.md", len(f"R {MARKER}".encode()))
        assert isinstance(unloaded, SkillUnloadedEvent)
        assert (unloaded.loaded_count, unloaded.loaded_size) == (0, 0)
        for event in sink.events:
            assert MARKER not in event.model_dump_json()

    async def test_a_refusal_is_an_event_naming_only_a_listed_skill(self) -> None:
        sink = MemoryEventSink()
        session = _session(
            skill("a"), skill("b"), events=sink, limits=DisclosureLimits(max_loaded=1)
        )
        await session.load("a")
        with pytest.raises(SkillRefused):
            await session.load("b")
        with pytest.raises(SkillNotFound):
            await session.load(f"{MARKER}-x")
        with pytest.raises(SkillRefused):
            await session.read_resource(f"{MARKER}-x", "r.md")
        with pytest.raises(SkillRefused):
            await session.read_resource("b", "r.md")
        refusals = [e for e in sink.events if isinstance(e, SkillRefusedEvent)]
        assert [(e.skill, e.reason) for e in refusals] == [
            ("b", "too_many_loaded"),
            (None, "skill_unknown"),
            (None, "skill_not_loaded"),
            ("b", "skill_not_loaded"),
        ]
        assert MARKER not in "".join(e.model_dump_json() for e in sink.events)


class TestPlacement:
    async def test_loaded_skills_sit_before_one_closing_breakpoint(self) -> None:
        session = _session(skill("a", "Alpha.\n"), skill("b", "Beta.\n"))
        base = (TextBlock(text="System prompt."),)
        await session.load("b")
        await session.load("a")
        system = await session.system(base)
        assert system[0] == base[0]
        assert isinstance(system[-1], CacheBreakpoint)
        assert sum(isinstance(b, CacheBreakpoint) for b in system) == 1
        texts = [b.text for b in system if isinstance(b, TextBlock)]
        assert texts.index(next(t for t in texts if "Beta." in t)) < texts.index(
            next(t for t in texts if "Alpha." in t)
        )

    async def test_the_system_is_identical_for_the_same_skills(self) -> None:
        one, two = _session(skill("a"), skill("b")), _session(skill("a"), skill("b"))
        for session in (one, two):
            await session.load("a")
        assert await one.system() == await two.system()

    async def test_a_product_may_render_the_listing_and_skills(self) -> None:
        session = _session(
            skill("a", "Alpha.\n"),
            render_listing=lambda summaries: "LIST " + ",".join(s.name for s in summaries),
            render_skill=lambda loaded: f"SKILL {loaded.name}",
        )
        await session.load("a")
        assert await session.system() == (
            TextBlock(text="LIST a"),
            TextBlock(text="SKILL a"),
            CacheBreakpoint(),
        )
