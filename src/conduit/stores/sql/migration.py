"""Running conduit's own Alembic chain against a product's database.

A product calls ``await migrate(engine)`` at start-up, before it uses a SQL store. The
chain is conduit's alone: its scripts ship inside the package, and its version table
is ``conduit_alembic_version`` (in the ``conduit`` schema on Postgres), so it never
reads or writes a product's own Alembic history. Every migration is reversible.

Alembic keeps its running context in module globals, so runs in one process take
turns: on one event loop through an asyncio lock, across threads through a thread
lock. Across processes, the Postgres run holds an advisory lock (see migrations/env.py).
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Callable
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from conduit.stores.sql.models import SCHEMA, VERSION_TABLE

__all__ = ["current_revision", "downgrade", "head_revision", "migrate", "revisions"]

MIGRATIONS = Path(__file__).with_name("migrations")

_THREAD_LOCK = threading.Lock()
_LOOP_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


def _loop_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _LOOP_LOCKS.get(loop)
    if lock is None:
        lock = _LOOP_LOCKS[loop] = asyncio.Lock()
    return lock


async def _run(engine: AsyncEngine, action: Callable[[Connection], None]) -> None:
    """Run an Alembic command on one transaction, one run at a time in this process."""

    def locked(connection: Connection) -> None:
        with _THREAD_LOCK:
            action(connection)

    async with _loop_lock(), engine.begin() as connection:
        await connection.run_sync(locked)


def _config(connection: Connection | None = None) -> Config:
    config = Config()
    # The option goes through configparser interpolation, so a percent sign is doubled.
    config.set_main_option("script_location", str(MIGRATIONS).replace("%", "%%"))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def _scripts() -> ScriptDirectory:
    return ScriptDirectory.from_config(_config())


def revisions() -> list[tuple[str, str | None]]:
    """Every revision as ``(revision, down_revision)``, oldest first."""
    chain: list[tuple[str, str | None]] = []
    for script in _scripts().walk_revisions():
        down = script.down_revision
        if isinstance(down, tuple | list):  # pragma: no cover - the chain has no merges
            raise RuntimeError("conduit's migration chain must be linear")
        chain.append((script.revision, down))
    return list(reversed(chain))


def head_revision() -> str:
    head = _scripts().get_current_head()
    if head is None:  # pragma: no cover - the chain is never empty
        raise RuntimeError("conduit's migration chain is empty")
    return head


async def migrate(engine: AsyncEngine, revision: str = "head") -> None:
    """Upgrade conduit's tables to ``revision`` (the latest by default), in one transaction."""

    await _run(engine, lambda connection: command.upgrade(_config(connection), revision))


async def downgrade(engine: AsyncEngine, revision: str) -> None:
    """Downgrade conduit's tables to ``revision`` (``"base"`` removes them all)."""

    await _run(engine, lambda connection: command.downgrade(_config(connection), revision))


async def current_revision(engine: AsyncEngine) -> str | None:
    """The revision conduit's tables are at, or ``None`` before the first migration."""

    def read(connection: Connection) -> str | None:
        postgres = connection.dialect.name == "postgresql"
        context = MigrationContext.configure(
            connection,
            opts={
                "version_table": VERSION_TABLE,
                "version_table_schema": SCHEMA if postgres else None,
            },
        )
        return context.get_current_revision()

    async with engine.connect() as connection:
        return await connection.run_sync(read)
