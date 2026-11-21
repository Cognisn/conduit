"""The sql extra: migrations, schema placement and the stores, on SQLite and Postgres."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("sqlalchemy", reason="the sql extra is not installed")

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from pydantic import SecretStr, ValidationError
from sqlalchemy import BLANK_SCHEMA, Connection, MetaData, inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from conduit.core.backends import TokenSet, TokenStore
from conduit.core.errors import ConfigurationError
from conduit.core.transcript import CompactionPoint, MessageEntry, TranscriptStore
from conduit.core.types import Message
from conduit.stores.sql import (
    SCHEMA,
    VERSION_TABLE,
    SqlTokenStore,
    SqlTranscriptStore,
    current_revision,
    downgrade,
    head_revision,
    migrate,
    revisions,
)
from conduit.stores.sql.models import Base
from tests.unit.stores.sql.postgres import postgres_server, watchdog

TABLES = {"conduit_transcript_sessions", "conduit_transcript_entries", "conduit_tokens"}


@pytest.fixture(autouse=True)
def _watchdog() -> Iterator[None]:
    with watchdog():
        yield


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with postgres_server() as url:
        yield url


@pytest.fixture(params=["sqlite", "postgres"])
async def engine(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    if request.param == "sqlite":
        made = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'conduit.db'}")
    else:
        made = create_async_engine(request.getfixturevalue("postgres_url"))
        async with made.begin() as conn:  # a clean server for every test
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
            await conn.execute(text("DROP TABLE IF EXISTS public.alembic_version"))
            await conn.execute(text("DROP TABLE IF EXISTS public.product_table"))
    try:
        yield made
    finally:
        await made.dispose()


def _postgres(engine: AsyncEngine) -> bool:
    return engine.dialect.name == "postgresql"


async def _tables(engine: AsyncEngine) -> dict[str | None, set[str]]:
    """Table names per schema: ``conduit`` and ``public`` on Postgres, ``None`` on SQLite."""

    def read(conn: Connection) -> dict[str | None, set[str]]:
        inspector = inspect(conn)
        if conn.dialect.name == "postgresql":
            schemas = set(inspector.get_schema_names())
            return {
                s: set(inspector.get_table_names(schema=s))
                for s in (SCHEMA, "public")
                if s in schemas
            }
        return {None: set(inspector.get_table_names())}

    async with engine.connect() as conn:
        return await conn.run_sync(read)


#: No schema, for SQLite (the stubs do not admit None, which SQLAlchemy accepts).
_NONE: Any = None


def _no_schema(*_: object) -> Any:
    return BLANK_SCHEMA


def _message(text: str) -> MessageEntry:
    return MessageEntry(message=Message.user(text))


# Migrations


async def test_the_chain_is_linear_and_named() -> None:
    chain = revisions()
    assert [down for _, down in chain] == [None, *[rev for rev, _ in chain[:-1]]]
    assert chain[-1][0] == head_revision()


async def test_migrate_puts_tables_and_the_version_table_in_conduit_s_own_place(
    engine: AsyncEngine,
) -> None:
    await migrate(engine)
    tables = await _tables(engine)
    if _postgres(engine):
        assert tables[SCHEMA] == TABLES | {VERSION_TABLE}
        assert tables.get("public", set()) == set()
    else:
        assert tables[None] == TABLES | {VERSION_TABLE}
    assert await current_revision(engine) == head_revision()


async def test_migrate_never_touches_the_product_s_alembic_version(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        await conn.execute(text("INSERT INTO alembic_version VALUES ('product_0042')"))
        await conn.execute(text("CREATE TABLE product_table (id INTEGER PRIMARY KEY)"))
    await migrate(engine)
    await downgrade(engine, "base")
    await migrate(engine)
    async with engine.connect() as conn:
        rows = (await conn.execute(text("SELECT version_num FROM alembic_version"))).all()
    assert [r[0] for r in rows] == ["product_0042"]
    names = set().union(*(await _tables(engine)).values())
    assert {"alembic_version", "product_table"} <= names


async def test_migrate_is_idempotent(engine: AsyncEngine) -> None:
    await migrate(engine)
    await migrate(engine)
    assert await current_revision(engine) == head_revision()


_MIGRATE_TOGETHER = """
import asyncio, sys
from sqlalchemy.ext.asyncio import create_async_engine
from conduit.stores.sql import migrate

async def main(url: str) -> None:
    engines = [create_async_engine(url) for _ in range(4)]
    await asyncio.gather(*(migrate(e) for e in engines))
    for engine in engines:
        await engine.dispose()

asyncio.run(main(sys.argv[1]))
"""


async def test_runs_in_one_process_take_turns(engine: AsyncEngine) -> None:
    # Alembic's running context is global, so concurrent runs in one process must take
    # turns. Without that they can deadlock the event loop itself, which no asyncio
    # timeout can interrupt: the runs go in a child process with a hard time limit.
    if not _postgres(engine):
        pytest.skip("covered on Postgres; the lock is the same for SQLite")
    url = engine.url.render_as_string(hide_password=False)
    process = subprocess.Popen(
        [sys.executable, "-c", _MIGRATE_TOGETHER, url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, err = await asyncio.to_thread(process.communicate, timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        await asyncio.to_thread(process.communicate)
        pytest.fail("concurrent migrations in one process did not finish in 120 seconds")
    assert process.returncode == 0, err[-400:]
    assert await current_revision(engine) == head_revision()


_MIGRATE_AT = """
import asyncio, sys, time
from sqlalchemy.ext.asyncio import create_async_engine
from conduit.stores.sql import migrate

async def main(url: str, at: float) -> None:
    engine = create_async_engine(url)
    async with engine.connect():
        pass
    time.sleep(max(0.0, at - time.time()))
    await migrate(engine)
    await engine.dispose()

asyncio.run(main(sys.argv[1], float(sys.argv[2])))
"""


async def test_separate_processes_migrating_at_once_take_turns(engine: AsyncEngine) -> None:
    if not _postgres(engine):
        pytest.skip("SQLite databases are migrated by one process")
    url = engine.url.render_as_string(hide_password=False)
    at = time.time() + 8
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _MIGRATE_AT, url, str(at)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    outcomes = [await asyncio.to_thread(p.communicate, timeout=120) for p in processes]
    failures = [err[-400:] for p, (_, err) in zip(processes, outcomes, strict=True) if p.returncode]
    assert failures == []
    assert await current_revision(engine) == head_revision()


async def test_an_unmigrated_database_has_no_revision(engine: AsyncEngine) -> None:
    assert await current_revision(engine) is None


async def test_every_migration_is_reversible(engine: AsyncEngine) -> None:
    for revision, down in revisions():
        await migrate(engine, revision)
        after_up = await _tables(engine)
        await downgrade(engine, down or "base")
        assert await current_revision(engine) == down
        await migrate(engine, revision)
        assert await _tables(engine) == after_up
        assert await current_revision(engine) == revision


async def test_downgrade_to_base_removes_every_conduit_table(engine: AsyncEngine) -> None:
    await migrate(engine)
    await downgrade(engine, "base")
    tables = await _tables(engine)
    names = set().union(*tables.values())
    assert names & TABLES == set()
    assert await current_revision(engine) is None
    # As documented: the (empty) version table stays, in the conduit schema on Postgres.
    assert tables[SCHEMA if _postgres(engine) else None] == {VERSION_TABLE}


async def test_the_models_match_the_migrated_schema(engine: AsyncEngine) -> None:
    await migrate(engine)

    def diff(conn: Connection) -> list[Any]:
        options: dict[str, Any] = {"version_table": VERSION_TABLE, "compare_type": True}
        if conn.dialect.name == "postgresql":  # compare the conduit schema only
            options["include_schemas"] = True
            options["version_table_schema"] = SCHEMA
            options["include_name"] = lambda name, kind, _: kind != "schema" or name == SCHEMA
        context = MigrationContext.configure(conn, opts=options)
        metadata = Base.metadata
        if conn.dialect.name == "sqlite":  # the same tables, without the schema
            metadata = MetaData()
            for table in Base.metadata.sorted_tables:
                table.to_metadata(metadata, schema=_NONE, referred_schema_fn=_no_schema)
        return list(compare_metadata(context, metadata))

    async with engine.connect() as conn:
        assert await conn.run_sync(diff) == []


# Stores


async def test_the_stores_meet_their_protocols_and_refuse_other_databases(
    engine: AsyncEngine,
) -> None:
    assert isinstance(SqlTranscriptStore(engine), TranscriptStore)
    assert isinstance(SqlTokenStore(engine), TokenStore)
    other: Any = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
    for make in (SqlTranscriptStore, SqlTokenStore):
        with pytest.raises(ConfigurationError, match="Postgres or SQLite"):
            make(other)


async def test_entries_are_stored_in_the_conduit_tables(engine: AsyncEngine) -> None:
    await migrate(engine)
    store = SqlTranscriptStore(engine)
    await store.append("sess_1", [_message("one"), _message("two")])
    table = (
        f"{SCHEMA}.conduit_transcript_entries"
        if _postgres(engine)
        else ("conduit_transcript_entries")
    )
    async with engine.connect() as conn:
        rows = (
            await conn.execute(text(f"SELECT session_id, seq, entry_type, body FROM {table}"))  # noqa: S608
        ).all()
    assert [(r[0], r[1], r[2]) for r in sorted(rows, key=lambda r: r[1])] == [
        ("sess_1", 1, "message"),
        ("sess_1", 2, "message"),
    ]
    assert all(json.loads(r[3])["type"] == "message" for r in rows)


async def test_a_compaction_replacing_later_entries_is_refused(engine: AsyncEngine) -> None:
    await migrate(engine)
    store = SqlTranscriptStore(engine)
    await store.append("sess_1", [_message("one"), _message("two")])
    bad = CompactionPoint(summary=Message.user("s"), replaces_through=4, strategy="summary")
    with pytest.raises(ValidationError):
        await store.append("sess_1", [_message("three"), bad])
    # Nothing of the refused batch is kept, and numbering carries on from 2.
    assert len((await store.load("sess_1")).entries) == 2
    assert [e.seq for e in await store.append("sess_1", [_message("three")])] == [3]


async def test_a_session_id_the_transcript_refuses_is_refused(engine: AsyncEngine) -> None:
    await migrate(engine)
    store = SqlTranscriptStore(engine)
    with pytest.raises(ValueError, match="session id"):
        await store.append("x" * 257, [_message("one")])
    assert await store.list_sessions() == []


async def test_an_empty_append_creates_no_session(engine: AsyncEngine) -> None:
    await migrate(engine)
    store = SqlTranscriptStore(engine)
    assert await store.append("sess_1", []) == ()
    assert await store.list_sessions() == []


async def test_two_engines_append_to_one_session_without_clashing(
    engine: AsyncEngine,
) -> None:
    if not _postgres(engine):
        pytest.skip("SQLite stores serialise within one store object (documented)")
    await migrate(engine)
    other = create_async_engine(engine.url.render_as_string(hide_password=False))
    try:
        first, second = SqlTranscriptStore(engine), SqlTranscriptStore(other)
        await asyncio.gather(
            *(
                s.append("sess_1", [_message(f"{i}a"), _message(f"{i}b")])
                for i in range(10)
                for s in (first, second)
            )
        )
        loaded = await first.load("sess_1")
        assert [e.seq for e in loaded.entries] == list(range(1, 41))
        texts = [e.message.content[0].text for e in loaded.entries]  # type: ignore[union-attr]
        assert all(texts[n][:-1] == texts[n + 1][:-1] for n in range(0, 40, 2))
    finally:
        await other.dispose()


async def test_delete_removes_entries_and_the_session(engine: AsyncEngine) -> None:
    await migrate(engine)
    store = SqlTranscriptStore(engine)
    await store.append("sess_1", [_message("one")])
    await store.delete("sess_1")
    assert await store.list_sessions() == []
    assert [e.seq for e in await store.append("sess_1", [_message("again")])] == [1]


async def test_tokens_are_stored_only_through_to_storage(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await migrate(engine)
    store = SqlTokenStore(engine)
    tokens = TokenSet(
        access_token=SecretStr("access-0001-example"),
        refresh_token=SecretStr("refresh-0001-example"),
    )
    calls: list[int] = []
    original = TokenSet.to_storage

    def counting(self: TokenSet) -> dict[str, Any]:
        calls.append(1)
        return original(self)

    monkeypatch.setattr(TokenSet, "to_storage", counting)
    await store.put("mcp:example", tokens)
    assert calls == [1]
    table = f"{SCHEMA}.conduit_tokens" if _postgres(engine) else "conduit_tokens"
    async with engine.connect() as conn:
        rows = (await conn.execute(text(f"SELECT key, data FROM {table}"))).all()  # noqa: S608
    assert [(r[0], json.loads(r[1])) for r in rows] == [("mcp:example", original(tokens))]
    assert await store.get("mcp:example") == tokens
