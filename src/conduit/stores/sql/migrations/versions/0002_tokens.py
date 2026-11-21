"""MCP token sets.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def _schema() -> str | None:
    return "conduit" if op.get_bind().dialect.name == "postgresql" else None


def upgrade() -> None:
    op.create_table(
        "conduit_tokens",
        sa.Column("key", sa.String(512), nullable=False),
        sa.Column("data", sa.Text(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("key", name="pk_conduit_tokens"),
        schema=_schema(),
    )


def downgrade() -> None:
    op.drop_table("conduit_tokens", schema=_schema())
