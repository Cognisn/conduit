"""In-memory transcript and token stores, for tests and short-lived processes.

Entries are held as JSON text, so anything the memory store accepts a durable store
can hold too, and a caller can never change a stored entry through a reference.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from pydantic import TypeAdapter

from conduit.core.backends import TokenSet
from conduit.core.transcript import Transcript, TranscriptEntry

_ENTRY: TypeAdapter[TranscriptEntry] = TypeAdapter(TranscriptEntry)


class MemoryTranscriptStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        async with self._lock:
            held = self._sessions.setdefault(session_id, [])
            stored: list[TranscriptEntry] = []
            for entry in entries:
                numbered = entry.model_copy(update={"seq": len(held) + 1})
                held.append(_ENTRY.dump_json(numbered).decode("utf-8"))
                stored.append(_ENTRY.validate_json(held[-1]))
            return tuple(stored)

    async def load(self, session_id: str) -> Transcript:
        async with self._lock:
            held = list(self._sessions.get(session_id, []))
        return Transcript(
            session_id=session_id, entries=tuple(_ENTRY.validate_json(text) for text in held)
        )

    async def list_sessions(self) -> Sequence[str]:
        async with self._lock:
            return sorted(self._sessions)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    def raw(self, session_id: str) -> list[str]:
        """The stored JSON, for tests."""
        return list(self._sessions.get(session_id, []))


class MemoryTokenStore:
    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}

    async def get(self, key: str) -> TokenSet | None:
        text = self._tokens.get(key)
        return None if text is None else TokenSet.from_storage(json.loads(text))

    async def put(self, key: str, tokens: TokenSet) -> None:
        self._tokens[key] = json.dumps(tokens.to_storage())

    async def delete(self, key: str) -> None:
        self._tokens.pop(key, None)
