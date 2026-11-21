"""Transcript sessions and entries.

Revision ID: 0001
Revises: (none)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def _schema() -> str | None:
    return "conduit" if op.get_bind().dialect.name == "postgresql" else None


def upgrade() -> None:
    schema = _schema()
    prefix = f"{schema}." if schema else ""
    op.create_table(
        "conduit_transcript_sessions",
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("last_seq", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_conduit_transcript_sessions"),
        schema=schema,
    )
    op.create_table(
        "conduit_transcript_entries",
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("seq", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("entry_id", sa.String(256), nullable=False),
        sa.Column("entry_type", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            [f"{prefix}conduit_transcript_sessions.session_id"],
            name="fk_conduit_transcript_entries_session_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", "seq", name="pk_conduit_transcript_entries"),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_table("conduit_transcript_entries", schema=schema)
    op.drop_table("conduit_transcript_sessions", schema=schema)
