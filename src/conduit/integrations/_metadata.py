"""An event's metadata, flattened, for the integrations that export events elsewhere.

``metadata_fields`` walks an event's fields and keeps only what is metadata:
identifiers, numbers, flags, enumerations, and the hash, size and kind of each piece of
captured content. The text of a ``CapturedContent`` is never read, whatever the capture
policy allowed the event to carry, and a string that is not a valid identifier is
dropped, so an emitter that bypassed validation cannot leak prose through an exporter.

Shared by the otel and warden integrations; it imports neither extra's library.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from conduit.core.events import CapturedContent, Event
from conduit.core.identifiers import IDENTIFIER_PATTERN

__all__ = ["Scalar", "identifier", "metadata_fields"]

Scalar = str | bool | int | float

_IDENTIFIER = re.compile(IDENTIFIER_PATTERN)
#: Top-level fields never exported: the type names the record, and the time dates it.
_SKIPPED = frozenset({"type", "occurred_at"})


def identifier(value: object) -> str | None:
    """``value`` if it is a string (or string enumeration) that is a valid identifier."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None:
        return value
    return None


def _scalar(value: object) -> Scalar | None:
    if isinstance(value, bool | int | float) and not isinstance(value, Enum):
        return value
    return identifier(value)


def _walk(model: BaseModel, prefix: str, out: dict[str, Scalar]) -> None:
    for name in type(model).model_fields:
        if not prefix and name in _SKIPPED:
            continue
        value = getattr(model, name, None)
        key = f"{prefix}{name}"
        if value is None or isinstance(value, datetime):
            continue
        if isinstance(value, CapturedContent):
            # Metadata only: the text is never read, whatever the policy allowed.
            out[f"{key}.sha256"] = value.sha256
            out[f"{key}.size"] = value.size
            out[f"{key}.kind"] = value.kind.value
            continue
        if isinstance(value, BaseModel):
            _walk(value, f"{key}.", out)
            continue
        scalar = _scalar(value)
        if scalar is not None:
            out[key] = scalar


def metadata_fields(event: Event) -> dict[str, Scalar]:
    """The event's metadata, keyed by dotted field path; never any content."""
    out: dict[str, Scalar] = {}
    _walk(event, "", out)
    return out
