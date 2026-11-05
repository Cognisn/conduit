"""The transcript store conformance suite."""

# Suites are test code shipped for products to run under pytest: assertions are the point.
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from typing import ClassVar

from conduit.core.transcript import (
    CompactionPoint,
    MessageEntry,
    SkillUse,
    TranscriptEntry,
    TranscriptStore,
)
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from conduit.testing.conformance.base import ConformanceSuite


def _text(entry: TranscriptEntry) -> str:
    if isinstance(entry, MessageEntry):
        block = entry.message.content[0]
        return block.text if isinstance(block, TextBlock) else ""
    return ""


def _message(text: str) -> MessageEntry:
    return MessageEntry(message=Message.user(text))


def _every_kind() -> list[TranscriptEntry]:
    return [
        MessageEntry(
            message=Message(
                role="user",
                content=(
                    TextBlock(text="Review the alert for host-01.example.com."),
                    ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\n\x00\xff"),
                    CacheBreakpoint(),
                ),
            )
        ),
        MessageEntry(
            message=Message(
                role="assistant",
                content=(ToolUseBlock(id="toolu_1", name="lookup", input={"ip": "192.0.2.10"}),),
            ),
            model="stub-echo",
            usage=Usage(
                input_tokens=12, output_tokens=4, cache_read_tokens=0, cache_write_tokens=8
            ),
        ),
        MessageEntry(
            message=Message(
                role="user",
                content=(
                    ToolResultBlock(
                        tool_use_id="toolu_1", content=(TextBlock(text="no match"),), is_error=True
                    ),
                ),
            )
        ),
        SkillUse(skill="incident-notes", version="2", action="loaded"),
        CompactionPoint(
            summary=Message.user("Summary: the alert was reviewed."),
            replaces_through=3,
            strategy="summary",
            tokens_before=900,
            tokens_after=40,
        ),
    ]


class TranscriptStoreSuite(ConformanceSuite):
    """Round trip, ordering, compaction across reloads, and concurrent sessions."""

    interface: ClassVar[str] = "transcript_store"
    #: Sessions and appends per session in the concurrency scenarios.
    concurrent_sessions: ClassVar[int] = 20
    appends_per_session: ClassVar[int] = 10

    async def make_store(self) -> TranscriptStore:
        """A new, empty store for one scenario. Products override this."""
        raise NotImplementedError("override make_store to return your transcript store")

    async def reopen_store(self, store: TranscriptStore) -> TranscriptStore:
        """The same backend reached afresh (a new connection or instance). Durable stores
        override this; the default returns ``store`` itself."""
        return store

    async def close_store(self, store: TranscriptStore) -> None:
        """Release a store after a scenario. Override if the store needs closing."""
        closer = getattr(store, "aclose", None)
        if closer is not None:
            await closer()

    def _scenario(self, body: object) -> None:
        async def scenario() -> None:
            store = await self.make_store()
            try:
                await body(store)  # type: ignore[operator]
            finally:
                await self.close_store(store)

        self.run(scenario)

    # Round trip

    def test_every_entry_kind_round_trips(self) -> None:
        async def body(store: TranscriptStore) -> None:
            entries = _every_kind()
            stored = await store.append("sess_round_trip", entries)
            loaded = await (await self.reopen_store(store)).load("sess_round_trip")
            assert loaded.session_id == "sess_round_trip"
            assert loaded.entries == stored
            for original, kept in zip(entries, loaded.entries, strict=True):
                assert kept.model_copy(update={"seq": None}) == original

        self._scenario(body)

    def test_an_unknown_session_loads_empty(self) -> None:
        async def body(store: TranscriptStore) -> None:
            loaded = await store.load("sess_never_written")
            assert loaded.entries == ()

        self._scenario(body)

    # Ordering

    def test_sequence_numbers_are_assigned_in_append_order(self) -> None:
        async def body(store: TranscriptStore) -> None:
            first = await store.append("sess_order", [_message("a"), _message("b")])
            preset = _message("c").model_copy(update={"seq": 99})
            second = await store.append("sess_order", [preset])
            assert [e.seq for e in first + second] == [1, 2, 3]
            loaded = await store.load("sess_order")
            assert [_text(e) for e in loaded.entries] == ["a", "b", "c"]
            assert [e.seq for e in loaded.entries] == [1, 2, 3]

        self._scenario(body)

    def test_sessions_are_listed_and_deleted(self) -> None:
        async def body(store: TranscriptStore) -> None:
            await store.append("sess_keep", [_message("k")])
            await store.append("sess_drop", [_message("d")])
            assert {"sess_keep", "sess_drop"} <= set(await store.list_sessions())
            await store.delete("sess_drop")
            assert "sess_drop" not in set(await store.list_sessions())
            assert (await store.load("sess_drop")).entries == ()
            assert len((await store.load("sess_keep")).entries) == 1
            await store.delete("sess_drop")  # deleting what is absent is not an error

        self._scenario(body)

    # Compaction

    def test_a_compaction_point_survives_a_reload(self) -> None:
        async def body(store: TranscriptStore) -> None:
            await store.append("sess_compact", [_message("one"), _message("two")])
            await store.append(
                "sess_compact",
                [
                    CompactionPoint(
                        summary=Message.user("Summary of one and two."),
                        replaces_through=2,
                        strategy="summary",
                    )
                ],
            )
            await store.append("sess_compact", [_message("three")])
            loaded = await (await self.reopen_store(store)).load("sess_compact")
            point = loaded.latest_compaction()
            assert point is not None and point.seq == 3 and point.replaces_through == 2
            texts = [m.content[0].text for m in loaded.active_messages()]  # type: ignore[union-attr]
            assert texts == ["Summary of one and two.", "three"]

        self._scenario(body)

    def test_an_append_that_would_not_load_is_refused_whole(self) -> None:
        """A compaction point may replace only earlier entries. An append that breaks
        the rule raises ``ValueError`` and stores nothing, so the session stays
        loadable (and a session it would have started does not appear)."""

        async def refused(
            store: TranscriptStore, session: str, batch: list[TranscriptEntry]
        ) -> None:
            try:
                await store.append(session, batch)
            except ValueError:
                return
            raise AssertionError("an append replacing later entries was accepted")

        def ahead(through: int) -> CompactionPoint:
            return CompactionPoint(
                summary=Message.user("Summary of what has not happened."),
                replaces_through=through,
                strategy="summary",
            )

        async def body(store: TranscriptStore) -> None:
            await store.append("sess_guarded", [_message("one")])
            await refused(store, "sess_guarded", [_message("two"), ahead(9)])
            await refused(store, "sess_guarded", [ahead(2)])  # it would itself be seq 2
            await refused(store, "sess_unborn", [ahead(1)])
            reopened = await self.reopen_store(store)
            loaded = await reopened.load("sess_guarded")
            assert [_text(e) for e in loaded.entries] == ["one"]
            assert "sess_unborn" not in set(await reopened.list_sessions())
            assert (await reopened.load("sess_unborn")).entries == ()
            stored = await reopened.append("sess_guarded", [_message("two"), ahead(2)])
            assert [e.seq for e in stored] == [2, 3]
            assert len((await reopened.load("sess_guarded")).active_messages()) == 1

        self._scenario(body)

    # Concurrency

    def test_concurrent_sessions_do_not_interfere(self) -> None:
        sessions, appends = self.concurrent_sessions, self.appends_per_session

        async def write(store: TranscriptStore, session: int) -> None:
            for i in range(appends):
                await store.append(f"sess_c{session}", [_message(f"{session}:{i}")])
                await asyncio.sleep(0)

        async def body(store: TranscriptStore) -> None:
            await asyncio.gather(*(write(store, s) for s in range(sessions)))
            for s in range(sessions):
                loaded = await store.load(f"sess_c{s}")
                assert [_text(e) for e in loaded.entries] == [f"{s}:{i}" for i in range(appends)]
                assert [e.seq for e in loaded.entries] == list(range(1, appends + 1))

        self._scenario(body)

    def test_concurrent_appends_to_one_session_stay_whole(self) -> None:
        batches, size = self.appends_per_session, 3

        async def body(store: TranscriptStore) -> None:
            await asyncio.gather(
                *(
                    store.append("sess_one", [_message(f"{b}:{i}") for i in range(size)])
                    for b in range(batches)
                )
            )
            loaded = await store.load("sess_one")
            assert [e.seq for e in loaded.entries] == list(range(1, batches * size + 1))
            texts = [_text(e) for e in loaded.entries]
            for start in range(0, len(texts), size):
                batch = {t.split(":")[0] for t in texts[start : start + size]}
                assert len(batch) == 1, "a batch was interleaved with another"

        self._scenario(body)
