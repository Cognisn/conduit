"""Helpers the embedding adapters share: batching and reading provider values.

Readers return ``None`` for a value of the wrong shape, so the adapter can raise its
``ProviderError(PROTOCOL)`` outside any handler.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any


def batches(texts: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    """``texts`` in consecutive batches of at most ``size``."""
    for start in range(0, len(texts), size):
        yield tuple(texts[start : start + size])


def read_vector(values: Any) -> tuple[float, ...] | None:
    """A provider's vector as a tuple of finite floats, or ``None`` if it is not one."""
    if not isinstance(values, list | tuple) or not values:
        return None
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        vector.append(number)
    return tuple(vector)


def read_count(value: Any) -> int | None:
    """A provider's token count, or ``None`` if it is not a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
