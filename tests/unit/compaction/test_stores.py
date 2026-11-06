"""Compaction points written through each reference store reload to the compacted view.

Each store is opened afresh for the reload (a new JSON store on the same directory, a
new SQL store on the same database), so the view comes from what was persisted, not
from memory. Postgres runs where the suite runs it (``CONDUIT_POSTGRES_TESTS``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest

from conduit.compaction import Compactor, KeepLastTurns
from conduit.core.transcript import CompactionPoint, MessageEntry, TranscriptStore
from conduit.core.types import Message
from conduit.stores.json import JsonTranscriptStore
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.compaction.helpers import conversation, request, runtime

Opener = Callable[[], TranscriptStore]


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    pytest.importorskip("sqlalchemy", reason="the sql extra is not installed")
    from tests.unit.stores.sql.postgres import postgres_server

    with postgres_server() as url:
        yield url


@pytest.fixture(params=["memory", "json", "sqlite", "postgres"])
async def opener(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Opener]:
    if request.param == "memory":
        store = MemoryTranscriptStore()
        yield lambda: store
        return
    if request.param == "json":
        yield lambda: JsonTranscriptStore(tmp_path / "transcripts")
        return
    pytest.importorskip("sqlalchemy", reason="the sql extra is not installed")
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from conduit.stores.sql import SCHEMA, SqlTranscriptStore, migrate

    if request.param == "sqlite":
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'conduit.db'}")
    else:
        engine = create_async_engine(request.getfixturevalue("postgres_url"))
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
            await conn.execute(text("DROP TABLE IF EXISTS public.alembic_version"))
    try:
        await migrate(engine)
        yield lambda: SqlTranscriptStore(engine)
    finally:
        await engine.dispose()


async def test_two_compactions_reload_to_the_compacted_view(opener: Opener) -> None:
    assignment = (await runtime()).assignments["chat"]
    store = opener()
    messages = conversation(4)
    await store.append("s-1", [MessageEntry(message=m) for m in messages])
    compactor = Compactor(
        [KeepLastTurns(threshold=0.01, keep_turns=1, notice="NOTICE")], transcripts=store
    )
    first = await compactor.before_model_call(request(assignment, messages, session_id="s-1"))
    assert first is not None
    more = conversation(6)[len(messages) :]
    await store.append("s-1", [MessageEntry(message=m) for m in more])
    second = await compactor.before_model_call(
        request(assignment, (*first, *more), session_id="s-1")
    )
    assert second is not None

    reloaded = await opener().load("s-1")
    active = reloaded.active_messages()
    assert [m.model_dump_json() for m in active] == [m.model_dump_json() for m in second]
    points = [e for e in reloaded.entries if isinstance(e, CompactionPoint)]
    assert [p.strategy for p in points] == ["keep_last_turns", "keep_last_turns"]
    assert all(p.tokens_before is not None and p.tokens_after is not None for p in points)
    assert all((p.tokens_after or 0) < (p.tokens_before or 0) for p in points)
    assert points[0].replaces_through < points[1].replaces_through
    # Every message is still stored; only the view is compacted.
    stored = [e.message for e in reloaded.entries if isinstance(e, MessageEntry)]
    assert stored == [*messages, *more]
    assert active[0] == second[0]
    assert active[0].content[0] == Message.user("NOTICE").content[0]
