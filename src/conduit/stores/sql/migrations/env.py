"""Alembic environment for conduit's own migrations.

Run only through ``conduit.stores.sql.migrate``, which passes the connection in. On
Postgres the run holds a transaction-scoped advisory lock, so several processes
migrating at start-up take turns, and creates the ``conduit`` schema; the version
table is ``conduit_alembic_version`` in that schema (in the main database on SQLite),
never the product's ``alembic_version``.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection, text

from conduit.stores.sql.models import SCHEMA, VERSION_TABLE

#: The advisory lock key: "conduit" in ASCII.
_LOCK_KEY = 0x636F6E64756974


def _run(connection: Connection) -> None:
    postgres = connection.dialect.name == "postgresql"
    if postgres:
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _LOCK_KEY})
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
    context.configure(
        connection=connection,
        target_metadata=None,
        version_table=VERSION_TABLE,
        version_table_schema=SCHEMA if postgres else None,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


_connection = context.config.attributes.get("connection")
if not isinstance(_connection, Connection):
    raise RuntimeError("conduit's migrations run only through conduit.stores.sql.migrate")
_run(_connection)
