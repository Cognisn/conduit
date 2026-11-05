"""SQL storage backends (SQLAlchemy, Postgres and SQLite), behind the sql extra."""

# The names below load on first use, so this package imports without the extra; using
# one without it raises ImportError naming the missing library. A product's start-up:
#
#     from sqlalchemy.ext.asyncio import create_async_engine
#     from conduit.stores.sql import SqlTranscriptStore, migrate
#
#     engine = create_async_engine("postgresql+asyncpg://...")
#     await migrate(engine)
#     transcripts = SqlTranscriptStore(engine)

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conduit.stores.sql.migration import (
        current_revision,
        downgrade,
        head_revision,
        migrate,
        revisions,
    )
    from conduit.stores.sql.models import SCHEMA, VERSION_TABLE
    from conduit.stores.sql.stores import SqlTokenStore, SqlTranscriptStore

__all__ = [
    "SCHEMA",
    "VERSION_TABLE",
    "SqlTokenStore",
    "SqlTranscriptStore",
    "current_revision",
    "downgrade",
    "head_revision",
    "migrate",
    "revisions",
]

_HOMES = {
    "SCHEMA": "models",
    "VERSION_TABLE": "models",
    "SqlTokenStore": "stores",
    "SqlTranscriptStore": "stores",
    "current_revision": "migration",
    "downgrade": "migration",
    "head_revision": "migration",
    "migrate": "migration",
    "revisions": "migration",
}


def __getattr__(name: str) -> Any:
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{home}"), name)
