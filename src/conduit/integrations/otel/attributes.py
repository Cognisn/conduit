"""Span and metric attributes taken from an event's metadata, never its content.

``metadata_attributes`` is the event's metadata (``conduit.integrations._metadata``)
under the ``conduit.`` prefix: identifiers, numbers, flags, enumerations, and the hash,
size and kind of each piece of captured content. The text of a ``CapturedContent`` is
never read, whatever the capture policy allowed the event to carry, and a string that
is not a valid identifier is dropped.
"""

from __future__ import annotations

from collections.abc import Mapping

from conduit.core.events import Event
from conduit.integrations._metadata import Scalar, identifier, metadata_fields

__all__ = ["PREFIX", "AttributeValue", "identifier", "metadata_attributes", "pick"]

AttributeValue = Scalar
PREFIX = "conduit"


def metadata_attributes(event: Event) -> dict[str, AttributeValue]:
    """The event's metadata as ``conduit.``-prefixed attributes; never any content."""
    return {f"{PREFIX}.{key}": value for key, value in metadata_fields(event).items()}


def pick(attributes: Mapping[str, AttributeValue], *names: str) -> dict[str, AttributeValue]:
    """The ``conduit.<name>`` attributes present, for a metric's low-cardinality set."""
    return {
        f"{PREFIX}.{name}": attributes[f"{PREFIX}.{name}"]
        for name in names
        if f"{PREFIX}.{name}" in attributes
    }
