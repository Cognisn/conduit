"""SQL transcript and token stores, on Postgres (asyncpg) or SQLite (aiosqlite).

The product supplies an ``AsyncEngine`` and runs ``migrate`` first. The stores never
create, dispose of or reconfigure the engine; on SQLite they reach the tables without
a schema through SQLAlchemy's schema translation.

Appends are atomic per session and numbered in order. Each append claims its session's
row in one statement (``INSERT ... ON CONFLICT DO UPDATE ... RETURNING``), which on
Postgres also locks the row, so appends and deletes from any number of engines and
processes are numbered without gaps or clashes. SQLite cannot lock a row, so a SQLite
store serialises its own calls; use one store object per SQLite database in a process.

A database error is raised as ``StoreError``, naming the operation and the error class
only. Drivers print a statement's parameters in their messages, and those parameters
hold transcript content and token values, so the original error is never chained.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import TypeAdapter
from sqlalchemy import Table, delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from conduit.core.backends import TokenSet
from conduit.core.errors import ConfigurationError
from conduit.core.transcript import Transcript, TranscriptEntry
from conduit.stores.errors import StoreError
from conduit.stores.sql.models import (
    SCHEMA,
    SESSION_ID_LENGTH,
    TOKEN_KEY_LENGTH,
    TokenRow,
    TranscriptEntryRow,
    TranscriptSessionRow,
)

__all__ = ["SqlTokenStore", "SqlTranscriptStore"]

_ENTRY: TypeAdapter[TranscriptEntry] = TypeAdapter(TranscriptEntry)
_SESSIONS: Table = TranscriptSessionRow.__table__  # type: ignore[assignment]
_ENTRIES: Table = TranscriptEntryRow.__table__  # type: ignore[assignment]
_TOKENS: Table = TokenRow.__table__  # type: ignore[assignment]
_DIALECTS = ("postgresql", "sqlite")


class _SqlStore:
    _name: ClassVar[str]

    def __init__(self, engine: AsyncEngine) -> None:
        dialect = engine.dialect.name
        if dialect not in _DIALECTS:
            raise ConfigurationError("the SQL stores support Postgres or SQLite only")
        self._postgres = dialect == "postgresql"
        self._engine = (
            engine
            if self._postgres
            else engine.execution_options(schema_translate_map={SCHEMA: None})
        )
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    async def _guarded[T](self, operation: str, action: Callable[[], Awaitable[T]]) -> T:
        """Run ``action``; a database error becomes a ``StoreError`` with nothing chained."""
        failure: str | None = None
        try:
            return await action()
        except StatementError as error:  # DBAPIError and every wrapped driver error
            failure = type(error).__name__
            driver = getattr(error, "orig", None)
            if driver is not None:
                kind = type(driver)
                failure += f" from {kind.__module__}.{kind.__qualname__}"
        # Raised here, after the handler, so the driver's error (and the statement
        # parameters in its message) is neither the cause nor the context.
        raise StoreError(self._name, operation, failure)

    @asynccontextmanager
    async def _begin(self) -> AsyncIterator[AsyncConnection]:
        """A transaction; on SQLite, also this store's lock."""
        if self._postgres:
            async with self._engine.begin() as connection:
                yield connection
            return
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = self._locks[loop] = asyncio.Lock()
        async with lock, self._engine.begin() as connection:
            yield connection

    def _insert(self, table: Table) -> Any:
        return postgres_insert(table) if self._postgres else sqlite_insert(table)


class SqlTranscriptStore(_SqlStore):
    """A transcript store over conduit's transcript tables."""

    _name = "SQL transcript store"

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        _check_session_id(session_id)
        if not entries:
            return ()
        return await self._guarded("append to a session", lambda: self._append(session_id, entries))

    async def _append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        async with self._begin() as connection:
            # Create or claim the session row and read its last sequence number in one
            # statement: there is no moment at which a delete can remove it unseen.
            claim = (
                self._insert(_SESSIONS)
                .values(session_id=session_id, last_seq=0)
                .on_conflict_do_update(
                    index_elements=["session_id"], set_={"last_seq": _SESSIONS.c.last_seq}
                )
                .returning(_SESSIONS.c.last_seq)
            )
            last: int = (await connection.execute(claim)).scalar_one()
            numbered = tuple(
                entry.model_copy(update={"seq": last + n}) for n, entry in enumerate(entries, 1)
            )
            # A compaction point may only replace earlier entries; refusing here rolls
            # the whole batch back.
            Transcript(session_id=session_id, entries=numbered)
            await connection.execute(
                _ENTRIES.insert(),
                [
                    {
                        "session_id": session_id,
                        "seq": entry.seq,
                        "entry_id": entry.entry_id,
                        "entry_type": entry.type,
                        "created_at": entry.created_at,
                        "body": _ENTRY.dump_json(entry).decode("utf-8"),
                    }
                    for entry in numbered
                ],
            )
            await connection.execute(
                update(_SESSIONS)
                .where(_SESSIONS.c.session_id == session_id)
                .values(last_seq=last + len(numbered), updated_at=datetime.now(UTC))
            )
        return numbered

    async def load(self, session_id: str) -> Transcript:
        _check_session_id(session_id)

        async def read() -> Sequence[str]:
            async with self._begin() as connection:
                rows = await connection.execute(
                    select(_ENTRIES.c.body)
                    .where(_ENTRIES.c.session_id == session_id)
                    .order_by(_ENTRIES.c.seq)
                )
                return rows.scalars().all()

        bodies = await self._guarded("load a session", read)
        decoded = True
        try:
            entries = tuple(_ENTRY.validate_json(body) for body in bodies)
        except ValueError:  # pydantic's error text would repeat the stored content
            decoded = False
        if not decoded:
            raise StoreError(self._name, "decode a stored session")
        return Transcript(session_id=session_id, entries=entries)

    async def list_sessions(self) -> Sequence[str]:
        async def read() -> list[str]:
            async with self._begin() as connection:
                rows = await connection.execute(
                    select(_SESSIONS.c.session_id).order_by(_SESSIONS.c.session_id)
                )
                return list(rows.scalars().all())

        return await self._guarded("list sessions", read)

    async def delete(self, session_id: str) -> None:
        async def remove() -> None:
            async with self._begin() as connection:
                # Lock the session first, so a concurrent append finishes before or after.
                await connection.execute(
                    select(_SESSIONS.c.session_id)
                    .where(_SESSIONS.c.session_id == session_id)
                    .with_for_update()
                )
                await connection.execute(
                    delete(_ENTRIES).where(_ENTRIES.c.session_id == session_id)
                )
                await connection.execute(
                    delete(_SESSIONS).where(_SESSIONS.c.session_id == session_id)
                )

        await self._guarded("delete a session", remove)


def _check_session_id(session_id: str) -> None:
    # The transcript model's own error would repeat the id.
    if not 1 <= len(session_id) <= SESSION_ID_LENGTH:
        raise ValueError(f"refused session id: it must be 1 to {SESSION_ID_LENGTH} characters")


def _check_key(key: str) -> None:
    if len(key) > TOKEN_KEY_LENGTH:
        raise ValueError(f"refused token key: it must be at most {TOKEN_KEY_LENGTH} characters")


class SqlTokenStore(_SqlStore):
    """A token store over conduit's token table. Token values are written only through
    ``TokenSet.to_storage``; protect the database as you would any credential store."""

    _name = "SQL token store"

    async def get(self, key: str) -> TokenSet | None:
        _check_key(key)

        async def read() -> str | None:
            async with self._begin() as connection:
                found: str | None = (
                    await connection.execute(select(_TOKENS.c.data).where(_TOKENS.c.key == key))
                ).scalar_one_or_none()
                return found

        data = await self._guarded("get a token set", read)
        if data is None:
            return None
        tokens: TokenSet | None = None
        # The decoder's error would repeat the stored token values.
        with contextlib.suppress(ValueError, TypeError, KeyError, AttributeError):
            tokens = TokenSet.from_storage(json.loads(data))
        if tokens is None:
            raise StoreError(self._name, "decode a stored token set")
        return tokens

    async def put(self, key: str, tokens: TokenSet) -> None:
        _check_key(key)
        data = json.dumps(tokens.to_storage(), ensure_ascii=False)
        now = datetime.now(UTC)
        statement = self._insert(_TOKENS).values(key=key, data=data, updated_at=now)
        statement = statement.on_conflict_do_update(
            index_elements=["key"], set_={"data": data, "updated_at": now}
        )

        async def write() -> None:
            async with self._begin() as connection:
                await connection.execute(statement)

        await self._guarded("put a token set", write)

    async def delete(self, key: str) -> None:
        _check_key(key)

        async def remove() -> None:
            async with self._begin() as connection:
                await connection.execute(delete(_TOKENS).where(_TOKENS.c.key == key))

        await self._guarded("delete a token set", remove)
