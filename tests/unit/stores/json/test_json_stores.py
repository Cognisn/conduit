"""The JSON file stores: layout, atomic writes, refusals and what reaches the disk."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from conduit.core.backends import TokenSet, TokenStore
from conduit.core.transcript import CompactionPoint, MessageEntry, TranscriptStore
from conduit.core.types import Message
from conduit.stores import json as json_stores
from conduit.stores.errors import StoreError
from conduit.stores.json import JsonTokenStore, JsonTranscriptStore


def _message(text: str) -> MessageEntry:
    return MessageEntry(message=Message.user(text))


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


async def test_the_stores_meet_their_protocols(tmp_path: Path) -> None:
    assert isinstance(JsonTranscriptStore(tmp_path / "t"), TranscriptStore)
    assert isinstance(JsonTokenStore(tmp_path / "k"), TokenStore)


async def test_the_directory_is_created_on_first_write(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path / "a" / "b")
    assert (await store.load("sess_1")).entries == ()
    assert await store.list_sessions() == []
    assert not (tmp_path / "a").exists()
    await store.append("sess_1", [_message("one")])
    assert len(_files(tmp_path / "a" / "b")) == 1


async def test_files_are_lf_json_named_by_digest(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    await store.append("sess:Mixed.Case", [_message("one\ntwo")])
    [path] = _files(tmp_path)
    assert path.suffix == ".json" and len(path.stem) == 64 and path.stem == path.stem.lower()
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    document = json.loads(raw)
    assert document["format"] == "conduit.transcript/1"
    assert document["session_id"] == "sess:Mixed.Case"
    assert [entry["seq"] for entry in document["entries"]] == [1]


async def test_case_differing_ids_are_kept_apart(tmp_path: Path) -> None:
    # Windows and macOS file systems ignore case; the file names must not depend on it.
    store = JsonTranscriptStore(tmp_path)
    await store.append("Sess", [_message("upper")])
    await store.append("sess", [_message("lower")])
    assert len(_files(tmp_path)) == 2
    assert sorted(await store.list_sessions()) == ["Sess", "sess"]


async def test_an_empty_append_writes_nothing(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    assert await store.append("sess_1", []) == ()
    assert not tmp_path.exists() or _files(tmp_path) == []


UNSAFE_SESSIONS = [
    "",
    ".",
    "..",
    "../escape",
    "..\\escape",
    "a/b",
    "a\\b",
    "/abs",
    "C:\\abs",
    "nul\x00byte",
    "line\nbreak",
    "x" * 257,
]


@pytest.mark.parametrize("unsafe", UNSAFE_SESSIONS)
async def test_a_path_like_session_id_is_refused(tmp_path: Path, unsafe: str) -> None:
    store = JsonTranscriptStore(tmp_path / "store")
    calls: list[Awaitable[Any]] = [
        store.append(unsafe, [_message("x")]),
        store.load(unsafe),
        store.delete(unsafe),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="session id"):
            await call
    assert not (tmp_path / "store").exists()
    assert not any(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("unsafe", ["", "..", "../escape", "a/b", "a\\b", "tab\tkey", "x" * 513])
async def test_a_path_like_token_key_is_refused(tmp_path: Path, unsafe: str) -> None:
    store = JsonTokenStore(tmp_path / "store")
    tokens = TokenSet(access_token=SecretStr("access-0001-example"))
    calls: list[Awaitable[Any]] = [
        store.put(unsafe, tokens),
        store.get(unsafe),
        store.delete(unsafe),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="token key"):
            await call
    assert not (tmp_path / "store").exists()


async def test_the_refusal_does_not_echo_the_id(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    with pytest.raises(ValueError, match="session id") as raised:
        await store.load("../secret-looking-name")
    assert "secret-looking-name" not in str(raised.value)


async def test_ordinary_ids_and_keys_are_accepted(tmp_path: Path) -> None:
    transcripts, tokens = JsonTranscriptStore(tmp_path / "t"), JsonTokenStore(tmp_path / "k")
    sessions = ["sess_01", "conv-2026.11.04", "user@example.com:42", "x" * 256, "..x", "a b"]
    for session in sessions:
        await transcripts.append(session, [_message("ok")])
    assert sorted(await transcripts.list_sessions()) == sorted(sessions)
    await tokens.put("mcp:server-01.example.com", TokenSet(access_token=SecretStr("a-1")))
    assert await tokens.get("mcp:server-01.example.com") is not None


async def test_a_compaction_replacing_later_entries_is_refused(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    await store.append("sess_1", [_message("one"), _message("two")])
    before = _files(tmp_path)[0].read_bytes()
    # Stored as seq 4, it would replace through 4: itself, not only earlier entries.
    bad = CompactionPoint(summary=Message.user("s"), replaces_through=4, strategy="summary")
    with pytest.raises(ValidationError):
        await store.append("sess_1", [_message("three"), bad])
    # Nothing of the refused batch is kept, and the session still loads and appends.
    assert _files(tmp_path)[0].read_bytes() == before
    assert len((await store.load("sess_1")).entries) == 2
    stored = await store.append("sess_1", [_message("three")])
    assert [e.seq for e in stored] == [3]


async def test_a_write_goes_through_a_temporary_file_and_a_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonTranscriptStore(tmp_path)
    await store.append("sess_1", [_message("one")])
    [path] = _files(tmp_path)
    before = path.read_bytes()
    replaced: list[tuple[str, str]] = []

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replaced.append((os.path.basename(src), os.path.basename(dst)))
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        await store.append("sess_1", [_message("two")])
    # The target was never written in place, and the temporary file was cleaned up.
    assert replaced and replaced[0][1] == path.name and replaced[0][0] != path.name
    assert path.read_bytes() == before
    assert _files(tmp_path) == [path]


async def test_a_reader_never_sees_a_partial_file(tmp_path: Path) -> None:
    store, reader = JsonTranscriptStore(tmp_path), JsonTranscriptStore(tmp_path)

    async def write() -> None:
        for i in range(30):
            await store.append("sess_1", [_message(f"m{i}" * 50)])

    async def read() -> None:
        for _ in range(60):
            loaded = await reader.load("sess_1")
            assert [e.seq for e in loaded.entries] == list(range(1, len(loaded.entries) + 1))
            await asyncio.sleep(0)

    await asyncio.gather(write(), read())


async def test_two_store_objects_on_one_directory_share_the_session_lock(
    tmp_path: Path,
) -> None:
    first, second = JsonTranscriptStore(tmp_path), JsonTranscriptStore(tmp_path)
    await asyncio.gather(
        *(s.append("sess_1", [_message(str(i))]) for i in range(10) for s in (first, second))
    )
    loaded = await first.load("sess_1")
    assert [e.seq for e in loaded.entries] == list(range(1, 21))


async def test_other_files_in_the_directory_are_ignored(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    await store.append("sess_1", [_message("one")])
    (tmp_path / "README.txt").write_text("not a transcript\n", encoding="utf-8")
    (tmp_path / ("0" * 64 + ".json")).write_text('{"format": "other"}\n', encoding="utf-8")
    (tmp_path / ("1" * 64 + ".json")).write_text("not json\n", encoding="utf-8")
    assert await store.list_sessions() == ["sess_1"]


async def test_delete_removes_the_file(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    await store.append("sess_1", [_message("one")])
    await store.delete("sess_1")
    assert _files(tmp_path) == []


async def test_tokens_reach_the_disk_only_through_to_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonTokenStore(tmp_path)
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
    [path] = _files(tmp_path)
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert json.loads(raw) == {
        "format": "conduit.tokens/1",
        "key": "mcp:example",
        "tokens": original(tokens),
    }
    assert await store.get("mcp:example") == tokens
    await store.delete("mcp:example")
    assert _files(tmp_path) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
async def test_token_files_are_private_to_the_owner(tmp_path: Path) -> None:
    store = JsonTokenStore(tmp_path)
    await store.put("mcp:example", TokenSet(access_token=SecretStr("a-1")))
    [path] = _files(tmp_path)
    assert path.stat().st_mode & 0o077 == 0


def test_appends_from_two_threads_with_their_own_loops_are_all_kept(tmp_path: Path) -> None:
    # A product mixing async calls with the sync facade has two event loops in one
    # process; asyncio locks alone do not serialise them.
    def worker(tag: str) -> None:
        async def run() -> None:
            store = JsonTranscriptStore(tmp_path)
            for i in range(30):
                await store.append("sess_1", [_message(f"{tag}{i}")])

        asyncio.run(run())

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in "AB"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    loaded = asyncio.run(JsonTranscriptStore(tmp_path).load("sess_1"))
    assert [e.seq for e in loaded.entries] == list(range(1, 61))


def test_tokens_from_two_threads_with_their_own_loops_are_all_kept(tmp_path: Path) -> None:
    def worker(tag: str) -> None:
        async def run() -> None:
            store = JsonTokenStore(tmp_path)
            for i in range(20):
                await store.put(f"mcp:{tag}{i}", TokenSet(access_token=SecretStr(f"a-{tag}{i}")))

        asyncio.run(run())

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in "AB"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(_files(tmp_path)) == 40


async def test_the_directory_is_synced_after_a_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Directory fsync exists on POSIX only; the flag lets the order be checked anywhere.
    synced: list[str] = []
    opened: dict[int, str] = {}
    real_open, real_fsync, real_close = os.open, os.fsync, os.close

    def fake_open(path: Any, flags: int, *args: Any) -> int:
        if Path(path) == tmp_path:
            opened[-7] = "directory"
            return -7
        return real_open(path, flags, *args)

    def fake_fsync(fd: int) -> None:
        if fd == -7:
            synced.append("directory")
        else:
            synced.append("file")
            real_fsync(fd)

    def fake_close(fd: int) -> None:
        if fd != -7:
            real_close(fd)

    monkeypatch.setattr(json_stores, "_SYNC_DIRECTORY", True)
    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "close", fake_close)
    await JsonTranscriptStore(tmp_path).append("sess_1", [_message("one")])
    await JsonTokenStore(tmp_path).put("mcp:example", TokenSet(access_token=SecretStr("a-1")))
    assert synced == ["file", "directory", "file", "directory"]


async def test_a_token_file_that_is_not_a_record_is_refused_without_echo(tmp_path: Path) -> None:
    store = JsonTokenStore(tmp_path)
    await store.put("mcp:example", TokenSet(access_token=SecretStr("a-1")))
    [path] = _files(tmp_path)
    for content in ('["SECRET-content"]\n', "SECRET-content not json\n", '"SECRET-content"\n'):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(StoreError) as raised:
            await store.get("mcp:example")
        assert "SECRET" not in str(raised.value) and "SECRET" not in repr(raised.value)
        assert raised.value.__cause__ is None and raised.value.__context__ is None


async def test_a_damaged_transcript_file_is_refused_without_echo(tmp_path: Path) -> None:
    store = JsonTranscriptStore(tmp_path)
    await store.append("sess_1", [_message("one")])
    [path] = _files(tmp_path)
    for content in (
        '{"format": "conduit.transcript/1", "session_id": "sess_1", '
        '"entries": [{"type": "message", "message": "SECRET-content"}]}\n',
        "SECRET-content not json\n",
    ):
        path.write_text(content, encoding="utf-8")
        for call in (store.load("sess_1"), store.append("sess_1", [_message("two")])):
            with pytest.raises(StoreError) as raised:
                await call
            assert "SECRET" not in str(raised.value) and "SECRET" not in repr(raised.value)
            assert raised.value.__cause__ is None and raised.value.__context__ is None
