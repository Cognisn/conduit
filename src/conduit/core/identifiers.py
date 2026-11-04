"""Identifier strings: the only free-standing strings an event may carry.

An ``Identifier`` cannot hold whitespace and is bounded, so prose (a prompt, a tool
result, a provider's error text) cannot pass through one. The event-content tripwire
relies on this: every string field of an event is an ``Identifier``.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import StringConstraints

IDENTIFIER_PATTERN = r"^[A-Za-z0-9_.:/@+=-]{1,256}$"


class IdentifierMark:
    """Metadata marking an ``Identifier``; the event tripwire looks for it."""

    def __repr__(self) -> str:
        return "IdentifierMark()"


IDENTIFIER_MARK = IdentifierMark()

Identifier = Annotated[
    str, StringConstraints(pattern=IDENTIFIER_PATTERN, max_length=256), IDENTIFIER_MARK
]


def new_id(prefix: str) -> str:
    """A fresh identifier such as ``run_3f2a...``."""
    return f"{prefix}_{uuid.uuid4().hex}"
