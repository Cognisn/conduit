"""The transcript model and the transcript store protocol.

A transcript is a session's ordered entries: messages, compaction points and skill-use
markers. It is serialisable to JSON as it is, so a store can hold it in any backend.
The store assigns each entry its sequence number (1, 2, 3 within a session) when it is
appended; a stored transcript's sequence numbers strictly increase.

A ``CompactionPoint`` records that the entries up to ``replaces_through`` are
summarised by ``summary``; the active conversation is that summary followed by every
later message. The compaction layer writes them; the transcript keeps them, so a
compaction survives a reload.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from conduit.core.identifiers import new_id
from conduit.core.types import Message, Usage


def _now() -> datetime:
    return datetime.now(UTC)


class _Entry(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, ser_json_bytes="base64", val_json_bytes="base64"
    )

    entry_id: str = Field(default_factory=lambda: new_id("ent"), min_length=1, max_length=256)
    seq: PositiveInt | None = None  # assigned by the store
    created_at: datetime = Field(default_factory=_now)


class MessageEntry(_Entry):
    type: Literal["message"] = "message"
    message: Message
    model: str | None = None
    usage: Usage | None = None


class CompactionPoint(_Entry):
    type: Literal["compaction"] = "compaction"
    summary: Message
    replaces_through: NonNegativeInt  # the sequence number of the last entry summarised
    strategy: str = Field(min_length=1, max_length=128)
    tokens_before: NonNegativeInt | None = None
    tokens_after: NonNegativeInt | None = None


class SkillUse(_Entry):
    type: Literal["skill_use"] = "skill_use"
    skill: str = Field(min_length=1, max_length=256)
    version: str | None = None
    action: Literal["loaded", "used", "unloaded"]


TranscriptEntry = Annotated[MessageEntry | CompactionPoint | SkillUse, Field(discriminator="type")]


class Transcript(BaseModel):
    """A session's stored entries, in sequence order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1, max_length=256)
    entries: tuple[TranscriptEntry, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        previous = 0
        for entry in self.entries:
            if entry.seq is None:
                raise ValueError("a stored entry needs a sequence number")
            if entry.seq <= previous:
                raise ValueError("sequence numbers must strictly increase")
            if isinstance(entry, CompactionPoint) and entry.replaces_through >= entry.seq:
                raise ValueError("a compaction point can only replace earlier entries")
            previous = entry.seq
        return self

    def latest_compaction(self) -> CompactionPoint | None:
        for entry in reversed(self.entries):
            if isinstance(entry, CompactionPoint):
                return entry
        return None

    def active_messages(self) -> tuple[Message, ...]:
        """The conversation as the model should see it: the latest summary, then later messages."""
        point = self.latest_compaction()
        messages: list[Message] = []
        since = 0
        if point is not None:
            messages.append(point.summary)
            since = point.replaces_through
        messages.extend(
            e.message for e in self.entries if isinstance(e, MessageEntry) and (e.seq or 0) > since
        )
        return tuple(messages)

    def loaded_skills(self) -> frozenset[str]:
        loaded: set[str] = set()
        for entry in self.entries:
            if isinstance(entry, SkillUse):
                if entry.action == "unloaded":
                    loaded.discard(entry.skill)
                else:
                    loaded.add(entry.skill)
        return frozenset(loaded)


@runtime_checkable
class TranscriptStore(Protocol):
    """Holds transcripts. Supplied by the product; conformance suite in conduit.testing.

    ``append`` stores the entries in order, assigning each the next sequence number for
    the session (any sequence number the caller set is replaced), and returns them as
    stored. Appends to one session are atomic with respect to each other. ``load`` of a
    session never written returns an empty transcript.
    """

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]: ...

    async def load(self, session_id: str) -> Transcript: ...

    async def list_sessions(self) -> Sequence[str]: ...

    async def delete(self, session_id: str) -> None: ...
