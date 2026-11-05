"""SQL store errors never carry stored values, ids or keys; appends survive a racing delete."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlalchemy", reason="the sql extra is not installed")

from pydantic import SecretStr
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from conduit.core.backends import TokenSet
from conduit.core.transcript import MessageEntry
from conduit.core.types import Message
from conduit.stores.errors import StoreError
from conduit.stores.sql import SCHEMA, SqlTokenStore, SqlTranscriptStore, migrate
from conduit.stores.sql.models import TOKEN_KEY_LENGTH
from tests.unit.stores.sql.postgres import postgres_server, watchdog

SECRET = "SECRET-value-0001-example"
SECRET_KEY = "mcp:SECRET-key-example"
SECRET_SESSION = "sess_SECRET_session_example"


@pytest.fixture(autouse=True)
def _watchdog() -> Iterator[None]:
    with watchdog():
        yield


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with postgres_server() as url:
        yield url


@pytest.fixture(params=["sqlite", "postgres"])
async def unmigrated(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """An engine on a database conduit's migrations have not run on."""
    if request.param == "sqlite":
        made = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unmigrated.db'}")
    else:
        made = create_async_engine(request.getfixturevalue("postgres_url"))
        async with made.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
    try:
        yield made
    finally:
        await made.dispose()


def _tokens() -> TokenSet:
    return TokenSet(access_token=SecretStr(SECRET), refresh_token=SecretStr(SECRET + "-r"))


def _chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _assert_clean(error: BaseException) -> None:
    for link in _chain(error):
        for text_ in (str(link), repr(link), *map(str, link.args)):
            for secret in ("SECRET",):
                assert secret not in text_, f"{type(link).__name__} carries a stored value"
    assert error.__cause__ is None and error.__context__ is None


async def test_a_failing_put_carries_no_token_value_or_key(unmigrated: AsyncEngine) -> None:
    store = SqlTokenStore(unmigrated)
    with pytest.raises(StoreError) as raised:
        await store.put(SECRET_KEY, _tokens())
    _assert_clean(raised.value)
    assert "put" in str(raised.value)


async def test_failing_gets_and_deletes_carry_no_key(unmigrated: AsyncEngine) -> None:
    store = SqlTokenStore(unmigrated)
    for call in (store.get(SECRET_KEY), store.delete(SECRET_KEY)):
        with pytest.raises(StoreError) as raised:
            await call
        _assert_clean(raised.value)


async def test_failing_transcript_calls_carry_no_content_or_id(unmigrated: AsyncEngine) -> None:
    store = SqlTranscriptStore(unmigrated)
    entry = MessageEntry(message=Message.user(f"content {SECRET}"))
    calls: list[Any] = [
        store.append(SECRET_SESSION, [entry]),
        store.load(SECRET_SESSION),
        store.delete(SECRET_SESSION),
        store.list_sessions(),
    ]
    for call in calls:
        with pytest.raises(StoreError) as raised:
            await call
        _assert_clean(raised.value)


async def test_a_token_key_over_the_column_length_is_refused_up_front(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'conduit.db'}")
    try:
        await migrate(engine)
        store = SqlTokenStore(engine)
        long_key = "mcp:SECRET" + "x" * TOKEN_KEY_LENGTH
        calls: list[Any] = [store.put(long_key, _tokens()), store.get(long_key)]
        calls.append(store.delete(long_key))
        for call in calls:
            with pytest.raises(ValueError, match="token key") as raised:
                await call
            assert "SECRET" not in str(raised.value)
        await store.put("k" * TOKEN_KEY_LENGTH, _tokens())  # the limit itself is fine
    finally:
        await engine.dispose()


async def test_an_append_survives_a_delete_between_its_statements(postgres_url: str) -> None:
    # Deterministic: a hook runs a delete, on another thread and engine, right after the
    # append has made sure of its session row and before it reads the sequence number.
    first = create_async_engine(postgres_url)
    other = create_async_engine(postgres_url)
    try:
        async with first.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
        await migrate(first)
        store = SqlTranscriptStore(first)
        await store.append("sess_race", [MessageEntry(message=Message.user("one"))])

        state = {"after_session": False, "fired": False}
        deleted = threading.Event()

        def delete_elsewhere() -> None:
            asyncio.run(SqlTranscriptStore(other).delete("sess_race"))
            deleted.set()

        def hook(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
            if state["after_session"] and not state["fired"]:
                state["fired"] = True
                threading.Thread(target=delete_elsewhere, daemon=True).start()
                # A store that holds the session row makes the delete wait: give up
                # waiting after a second and carry on.
                deleted.wait(1.0)
            if "conduit_transcript_sessions" in statement and statement.startswith("INSERT"):
                state["after_session"] = True

        event.listen(first.sync_engine, "before_cursor_execute", hook)
        stored = await store.append("sess_race", [MessageEntry(message=Message.user("two"))])
        event.remove(first.sync_engine, "before_cursor_execute", hook)
        assert state["fired"]
        assert [e.seq for e in stored] == [2]
        await asyncio.to_thread(deleted.wait, 30)
        assert (await store.load("sess_race")).entries == ()
    finally:
        await first.dispose()
        await other.dispose()
