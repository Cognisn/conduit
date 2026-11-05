"""The transcript and token store conformance suites, against the SQL stores on SQLite
and on Postgres (a throwaway container; see tests/unit/stores/sql/postgres.py)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("sqlalchemy", reason="the sql extra is not installed")

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from conduit.core.backends import TokenStore
from conduit.core.transcript import TranscriptStore
from conduit.stores.sql import SCHEMA, SqlTokenStore, SqlTranscriptStore, migrate
from conduit.testing.conformance import TokenStoreSuite, TranscriptStoreSuite
from tests.unit.stores.sql.postgres import postgres_server, watchdog


@pytest.fixture(autouse=True)
def _watchdog() -> Iterator[None]:
    with watchdog():
        yield


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with postgres_server() as url:
        yield url


class _SqlBackend:
    """Makes a fresh, migrated database for each scenario and disposes of its engines."""

    backend: ClassVar[str]
    url: str

    @pytest.fixture(autouse=True)
    def _database(self, request: pytest.FixtureRequest, tmp_path: Path) -> None:
        if self.backend == "sqlite":
            self.url = f"sqlite+aiosqlite:///{tmp_path / 'conduit.db'}"
        else:
            self.url = request.getfixturevalue("postgres_url")
        self.engines: list[AsyncEngine] = []

    async def _engine(self) -> AsyncEngine:
        engine = create_async_engine(self.url)
        self.engines.append(engine)
        return engine

    async def _fresh(self) -> AsyncEngine:
        engine = await self._engine()
        if self.backend == "postgres":
            async with engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
        await migrate(engine)
        return engine

    async def close_store(self, store: Any) -> None:
        for engine in self.engines:
            await engine.dispose()
        self.engines.clear()


class _SqlTranscriptSuite(_SqlBackend, TranscriptStoreSuite):
    async def make_store(self) -> TranscriptStore:
        return SqlTranscriptStore(await self._fresh())

    async def reopen_store(self, store: TranscriptStore) -> TranscriptStore:
        return SqlTranscriptStore(await self._engine())


class _SqlTokenSuite(_SqlBackend, TokenStoreSuite):
    async def make_store(self) -> TokenStore:
        return SqlTokenStore(await self._fresh())

    async def reopen_store(self, store: TokenStore) -> TokenStore:
        return SqlTokenStore(await self._engine())


class TestSqliteTranscriptStore(_SqlTranscriptSuite):
    backend = "sqlite"


class TestSqliteTokenStore(_SqlTokenSuite):
    backend = "sqlite"


class TestPostgresTranscriptStore(_SqlTranscriptSuite):
    backend = "postgres"


class TestPostgresTokenStore(_SqlTokenSuite):
    backend = "postgres"
