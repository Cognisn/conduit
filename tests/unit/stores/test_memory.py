"""The in-memory transcript and token stores (the conformance suite covers the contract)."""

from __future__ import annotations

from pydantic import SecretStr

from conduit.core.backends import TokenSet, TokenStore
from conduit.core.transcript import MessageEntry, TranscriptStore
from conduit.core.types import Message
from conduit.stores.memory import MemoryTokenStore, MemoryTranscriptStore


async def test_the_transcript_store_meets_the_protocol_and_assigns_sequence() -> None:
    store = MemoryTranscriptStore()
    assert isinstance(store, TranscriptStore)
    stored = await store.append("s", [MessageEntry(message=Message.user("a"))])
    assert [e.seq for e in stored] == [1]
    stored = await store.append("s", [MessageEntry(message=Message.user("b"))])
    assert [e.seq for e in stored] == [2]
    assert [e.seq for e in (await store.load("s")).entries] == [1, 2]
    assert await store.list_sessions() == ["s"]


async def test_entries_are_held_serialised() -> None:
    # The memory store keeps JSON, so anything it accepts a durable store can hold too.
    store = MemoryTranscriptStore()
    entry = MessageEntry(message=Message.user("a"))
    await store.append("s", [entry])
    assert store.raw("s")[0].startswith("{")


async def test_the_token_store_round_trips_and_deletes() -> None:
    store = MemoryTokenStore()
    assert isinstance(store, TokenStore)
    tokens = TokenSet(access_token=SecretStr("access-0000"))
    assert await store.get("mcp:example") is None
    await store.put("mcp:example", tokens)
    assert await store.get("mcp:example") == tokens
    await store.delete("mcp:example")
    assert await store.get("mcp:example") is None
    await store.delete("mcp:example")  # deleting what is absent is not an error
