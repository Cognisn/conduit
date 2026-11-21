"""The SQLAlchemy models of conduit's own tables.

On Postgres the tables live in their own schema, ``conduit``; on SQLite, which has no
schemas, they live in the main database. Either way every table name starts with
``conduit_``, so on SQLite they cannot clash with a product's tables, and conduit's
Alembic version table is ``conduit_alembic_version``, never the product's.

The migrations under ``migrations/versions`` create these tables; a test checks that
the models and the migrated schema agree.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, MetaData, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: The Postgres schema conduit's tables live in.
SCHEMA = "conduit"
#: conduit's own Alembic version table, in that schema on Postgres.
VERSION_TABLE = "conduit_alembic_version"

SESSION_ID_LENGTH = 256
ENTRY_ID_LENGTH = 256
ENTRY_TYPE_LENGTH = 32
TOKEN_KEY_LENGTH = 512

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    # Postgres names are at most 63 characters: the referred table is left out.
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA, naming_convention=NAMING_CONVENTION)


class TranscriptSessionRow(Base):
    """One row per session: the last sequence number, locked while appending."""

    __tablename__ = "conduit_transcript_sessions"

    session_id: Mapped[str] = mapped_column(String(SESSION_ID_LENGTH), primary_key=True)
    last_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TranscriptEntryRow(Base):
    """One transcript entry, held as the entry's JSON."""

    __tablename__ = "conduit_transcript_entries"

    session_id: Mapped[str] = mapped_column(
        String(SESSION_ID_LENGTH),
        ForeignKey(f"{SCHEMA}.conduit_transcript_sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    entry_id: Mapped[str] = mapped_column(String(ENTRY_ID_LENGTH), nullable=False)
    entry_type: Mapped[str] = mapped_column(String(ENTRY_TYPE_LENGTH), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)


class TokenRow(Base):
    """One MCP token set by key, held as ``TokenSet.to_storage()`` JSON."""

    __tablename__ = "conduit_tokens"

    key: Mapped[str] = mapped_column(String(TOKEN_KEY_LENGTH), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
