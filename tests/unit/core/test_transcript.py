"""The transcript model and the backend protocols products supply."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr, ValidationError

from conduit.core.backends import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Skill,
    SkillSource,
    SkillSummary,
    TokenSet,
    TokenStore,
)
from conduit.core.transcript import (
    CompactionPoint,
    MessageEntry,
    SkillUse,
    Transcript,
    TranscriptEntry,
    TranscriptStore,
)
from conduit.core.types import Message, Usage


def _numbered(*entries: TranscriptEntry) -> tuple[TranscriptEntry, ...]:
    return tuple(e.model_copy(update={"seq": i}) for i, e in enumerate(entries, start=1))


class TestTranscript:
    def test_round_trips_through_json_with_every_entry_kind(self) -> None:
        transcript = Transcript(
            session_id="sess_1",
            entries=_numbered(
                MessageEntry(message=Message.user("hello")),
                MessageEntry(
                    message=Message.assistant("hi"),
                    model="stub-echo",
                    usage=Usage.zero(),
                ),
                SkillUse(skill="incident-notes", version="1", action="loaded"),
                CompactionPoint(
                    summary=Message.user("Earlier: greetings."),
                    replaces_through=2,
                    strategy="summary",
                ),
            ),
        )
        again = Transcript.model_validate_json(transcript.model_dump_json())
        assert again == transcript

    def test_active_messages_start_from_the_latest_compaction_summary(self) -> None:
        transcript = Transcript(
            session_id="s",
            entries=_numbered(
                MessageEntry(message=Message.user("one")),
                MessageEntry(message=Message.assistant("two")),
                CompactionPoint(
                    summary=Message.user("Summary of one and two."),
                    replaces_through=2,
                    strategy="summary",
                ),
                MessageEntry(message=Message.user("three")),
            ),
        )
        assert [m.content[0].text for m in transcript.active_messages()] == [  # type: ignore[union-attr]
            "Summary of one and two.",
            "three",
        ]
        latest = transcript.latest_compaction()
        assert latest is not None and latest.seq == 3

    def test_without_compaction_every_message_is_active(self) -> None:
        transcript = Transcript(
            session_id="s",
            entries=_numbered(
                MessageEntry(message=Message.user("one")),
                SkillUse(skill="k", action="loaded"),
                MessageEntry(message=Message.assistant("two")),
            ),
        )
        assert len(transcript.active_messages()) == 2
        assert transcript.loaded_skills() == frozenset({"k"})

    def test_skill_markers_track_loading_and_unloading(self) -> None:
        transcript = Transcript(
            session_id="s",
            entries=_numbered(
                SkillUse(skill="a", action="loaded"),
                SkillUse(skill="b", action="loaded"),
                SkillUse(skill="a", action="unloaded"),
            ),
        )
        assert transcript.loaded_skills() == frozenset({"b"})

    def test_sequence_numbers_must_increase(self) -> None:
        one = MessageEntry(message=Message.user("x"), seq=2)
        two = MessageEntry(message=Message.user("y"), seq=1)
        with pytest.raises(ValidationError):
            Transcript(session_id="s", entries=(one, two))

    def test_a_compaction_point_replaces_only_earlier_entries(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(
                session_id="s",
                entries=(
                    CompactionPoint(
                        summary=Message.user("s"), replaces_through=5, strategy="x", seq=3
                    ),
                ),
            )

    def test_stored_entries_need_a_sequence_number(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(session_id="s", entries=(MessageEntry(message=Message.user("x")),))

    def test_entries_have_ids_and_times(self) -> None:
        entry = MessageEntry(message=Message.user("x"))
        assert entry.entry_id.startswith("ent_")
        assert entry.created_at.tzinfo is not None

    def test_the_store_is_a_protocol(self) -> None:
        assert hasattr(TranscriptStore, "append") and hasattr(TranscriptStore, "load")


class TestBackendModels:
    def test_token_sets_hide_tokens_but_store_them_explicitly(self) -> None:
        tokens = TokenSet(
            access_token=SecretStr("access-0000"),
            refresh_token=SecretStr("refresh-0000"),
            expires_at=datetime(2026, 11, 4, tzinfo=UTC),
            scope="read",
        )
        assert "access-0000" not in repr(tokens)
        assert "access-0000" not in tokens.model_dump_json()
        stored = tokens.to_storage()
        assert stored["access_token"] == "access-0000"
        assert TokenSet.from_storage(stored) == tokens

    def test_skill_and_approval_models(self) -> None:
        summary = SkillSummary(name="incident-notes", description="Write incident notes.")
        skill = Skill(summary=summary, body="# Incident notes\n")
        assert skill.summary.name == "incident-notes"
        request = ApprovalRequest(
            request_id="apr_1",
            run_id="run_1",
            task_class="chat",
            tool="send_report",
            arguments={"to": "team@example.com"},
            reason="Sends mail outside the deployment.",
        )
        decision = ApprovalDecision(approved=False, reason="Not today.", decided_by="analyst")
        assert request.tool == "send_report" and not decision.approved

    def test_the_protocols_are_runtime_checkable(self) -> None:
        class Tokens:
            async def get(self, key: str) -> TokenSet | None:
                return None

            async def put(self, key: str, tokens: TokenSet) -> None:
                return None

            async def delete(self, key: str) -> None:
                return None

        class Skills:
            async def list_skills(self) -> list[SkillSummary]:
                return []

            async def load_skill(self, name: str) -> Skill:
                raise LookupError(name)

        class Approvals:
            async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
                return ApprovalDecision(approved=True)

        assert isinstance(Tokens(), TokenStore)
        assert isinstance(Skills(), SkillSource)
        assert isinstance(Approvals(), ApprovalHandler)
