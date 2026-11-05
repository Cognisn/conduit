"""Local JSON file stores for transcripts and MCP tokens.

Each store keeps its records in a directory the product chooses, one file per session
or token key. A file is named by the SHA-256 of its session id or key, so the name is
safe on every file system (no separators, no case collisions on Windows or macOS) and
the id is kept inside the file. Ids and keys that look like paths are refused anyway,
so a mistake is reported rather than silently hashed.

Every write goes to a temporary file in the same directory, is flushed to disk, and
then replaces the record in one step: a reader sees the old record or the new one,
never part of either. On POSIX systems the directory is synced too, so the replace
itself survives a crash.

Calls are serialised per record across every store object, task, event loop and thread
in the process: an asyncio lock per loop keeps tasks in order, and a thread lock held
in the worker thread around each read, read-modify-write and removal keeps loops in
different threads (the sync facade's, for one) from losing each other's writes. The
stores do not lock across processes.

File I/O runs in a worker thread, so the event loop is not blocked. A record that
cannot be read raises ``StoreError``, which never repeats the file's content.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
import threading
import time
import unicodedata
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from conduit.core.backends import TokenSet
from conduit.core.transcript import Transcript, TranscriptEntry
from conduit.stores.errors import StoreError

__all__ = ["JsonTokenStore", "JsonTranscriptStore"]

_TRANSCRIPT_FORMAT = "conduit.transcript/1"
_TOKENS_FORMAT = "conduit.tokens/1"
_SESSION_ID_LIMIT = 256
_TOKEN_KEY_LIMIT = 512
# On Windows a file another handle has open cannot be replaced or opened while a
# replace is pending; such a clash is brief, so it is retried.
_RETRIES = 20
# Directories can be opened and synced on POSIX systems only.
_SYNC_DIRECTORY = os.name == "posix"


def _check_name(value: str, *, what: str, limit: int) -> None:
    """Refuse an id or key that could reach outside the store's directory.

    The message never repeats the value: it may be user-supplied.
    """
    if (
        not value
        or len(value) > limit
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or any(unicodedata.category(c) == "Cc" for c in value)
    ):
        raise ValueError(
            f"refused {what}: it must be 1 to {limit} characters, contain no path "
            "separator or control character, and not be '.' or '..'"
        )


def _file_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() + ".json"


def _is_record_name(name: str) -> bool:
    stem, dot, suffix = name.partition(".")
    return (
        dot == "."
        and suffix == "json"
        and len(stem) == 64
        and all(c in "0123456789abcdef" for c in stem)
    )


# Locks, per event loop, per record. Weak values: a lock lives while a task holds or
# waits on it, so the table never grows past the records in use.
_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.WeakValueDictionary[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _lock_for(path: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _LOCKS.get(loop)
    if locks is None:
        locks = _LOCKS[loop] = weakref.WeakValueDictionary()
    key = os.path.normcase(str(path))
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock


# Thread locks, striped by record: a fixed set, so the table never grows. Two records
# may share a stripe, which only serialises them; no code holds two at once.
_STRIPES = tuple(threading.Lock() for _ in range(64))


def _thread_lock(path: Path) -> threading.Lock:
    digest = hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).digest()
    return _STRIPES[digest[0] % len(_STRIPES)]


async def _in_thread[T](path: Path, action: Callable[..., T], *args: Any) -> T:
    """Run ``action`` in a worker thread, holding the record's asyncio and thread locks."""

    def locked() -> T:
        with _thread_lock(path):
            return action(*args)

    async with _lock_for(path):
        return await asyncio.to_thread(locked)


def _retrying[T](action: Callable[..., T], *args: Any) -> T:
    for attempt in range(_RETRIES):
        try:
            result: T = action(*args)
            return result
        except PermissionError:
            if os.name != "nt" or attempt == _RETRIES - 1:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise AssertionError("unreachable")  # pragma: no cover


def _read(path: Path) -> bytes | None:
    try:
        return _retrying(path.read_bytes)
    except FileNotFoundError:
        return None


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and one replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates the file readable and writable by its owner only.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _retrying(os.replace, temporary, path)
        if _SYNC_DIRECTORY:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _remove(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        _retrying(path.unlink)


class _TranscriptFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["conduit.transcript/1"] = "conduit.transcript/1"
    session_id: str
    entries: tuple[TranscriptEntry, ...] = ()


class JsonTranscriptStore:
    """A transcript store keeping one JSON file per session under ``directory``.

    The directory is created on the first write. Each append rewrites the session's
    file, so this store suits development, tests and modest transcripts.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory).resolve()

    @property
    def directory(self) -> Path:
        return self._directory

    def _path(self, session_id: str) -> Path:
        _check_name(session_id, what="session id", limit=_SESSION_ID_LIMIT)
        return self._directory / _file_name(session_id)

    def _held(self, session_id: str, path: Path) -> tuple[TranscriptEntry, ...]:
        raw = _read(path)
        if raw is None:
            return ()
        held: _TranscriptFile | None = None
        # Raised below, after the error is dropped, so nothing is chained to it.
        with contextlib.suppress(ValueError):
            held = _TranscriptFile.model_validate_json(raw)
        if held is None:
            raise StoreError("JSON transcript store", "read a session's file")
        return held.entries if held.session_id == session_id else ()

    def _append(
        self, session_id: str, path: Path, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        held = self._held(session_id, path)
        last = (held[-1].seq or 0) if held else 0
        numbered = tuple(
            entry.model_copy(update={"seq": last + n}) for n, entry in enumerate(entries, 1)
        )
        # Validate the whole transcript before writing: a compaction point may only
        # replace earlier entries, and a refused batch leaves the file untouched.
        transcript = Transcript(session_id=session_id, entries=held + numbered)
        document = _TranscriptFile(session_id=session_id, entries=transcript.entries)
        _write_atomic(path, document.model_dump_json() + "\n")
        return numbered

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        path = self._path(session_id)
        if not entries:
            return ()
        return await _in_thread(path, self._append, session_id, path, list(entries))

    async def load(self, session_id: str) -> Transcript:
        path = self._path(session_id)
        entries = await _in_thread(path, self._held, session_id, path)
        return Transcript(session_id=session_id, entries=entries)

    def _list(self) -> list[str]:
        try:
            names = os.listdir(self._directory)
        except FileNotFoundError:
            return []
        sessions: list[str] = []
        for name in names:
            if not _is_record_name(name):
                continue
            raw = _read(self._directory / name)
            try:
                document = json.loads(raw) if raw is not None else None
            except ValueError:
                continue
            if (
                isinstance(document, dict)
                and document.get("format") == _TRANSCRIPT_FORMAT
                and isinstance(document.get("session_id"), str)
            ):
                sessions.append(document["session_id"])
        return sorted(sessions)

    async def list_sessions(self) -> Sequence[str]:
        return await asyncio.to_thread(self._list)

    async def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        await _in_thread(path, _remove, path)


class JsonTokenStore:
    """A token store keeping one JSON file per key under ``directory``.

    Token values are written only through ``TokenSet.to_storage``. On POSIX systems the
    files are readable by their owner only; on any system, protect the directory as you
    would any file holding credentials.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory).resolve()

    @property
    def directory(self) -> Path:
        return self._directory

    def _path(self, key: str) -> Path:
        _check_name(key, what="token key", limit=_TOKEN_KEY_LIMIT)
        return self._directory / _file_name(key)

    @staticmethod
    def _get(key: str, path: Path) -> TokenSet | None:
        raw = _read(path)
        if raw is None:
            return None
        tokens: TokenSet | None = None
        foreign = False
        try:
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("not a record")
            if document.get("format") != _TOKENS_FORMAT or document.get("key") != key:
                foreign = True
            else:
                tokens = TokenSet.from_storage(document["tokens"])
        except (ValueError, KeyError, TypeError):
            pass  # raised below, outside the handler, so nothing is chained
        if foreign:
            return None
        if tokens is None:
            raise StoreError("JSON token store", "read a token file")
        return tokens

    async def get(self, key: str) -> TokenSet | None:
        path = self._path(key)
        return await _in_thread(path, self._get, key, path)

    async def put(self, key: str, tokens: TokenSet) -> None:
        path = self._path(key)
        document = {"format": _TOKENS_FORMAT, "key": key, "tokens": tokens.to_storage()}
        text = json.dumps(document, ensure_ascii=False) + "\n"
        await _in_thread(path, _write_atomic, path, text)

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await _in_thread(path, _remove, path)
