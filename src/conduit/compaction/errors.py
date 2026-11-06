"""Errors the compaction layer raises.

A conversation is content, and a summary model's text is content too, so no error here
carries either: each names a fixed reason code, and at most a provider failure class or
the class name of an underlying error. None keeps the exception it was raised for
(D0040): ``__cause__`` and ``__context__`` are always ``None``.
"""

from __future__ import annotations

import re

from conduit.core.errors import ConduitError, ProviderFailure

__all__ = ["REASONS", "CompactionError", "CompactionFailed"]

#: Every reason a ``CompactionFailed`` may give.
REASONS = frozenset(
    {
        "summary_failed",
        "summary_empty",
        "summary_too_large",
        "transcript_mismatch",
        "store_failed",
        "strategy_failed",
    }
)
_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def is_class_name(value: object) -> bool:
    """Whether ``value`` is a dotted class name of at most 128 characters."""
    return isinstance(value, str) and len(value) <= 128 and _CLASS_NAME.fullmatch(value) is not None


class CompactionError(ConduitError):
    """Base class for the compaction layer's errors."""


class CompactionFailed(CompactionError):
    """A compaction could not be completed.

    ``reason`` is a fixed code:

    - ``summary_failed``: the summary model call failed (``failure`` gives its class);
    - ``summary_empty``: the summary model answered with no text;
    - ``summary_too_large``: the conversation to summarise does not fit the summary
      model's context window;
    - ``transcript_mismatch``: the stored transcript's active conversation is not the
      conversation the hook was shown, so no compaction point can be placed;
    - ``store_failed``: the transcript store failed (``error_class`` names the error);
    - ``strategy_failed``: a product strategy raised (``error_class`` names the error).

    A product strategy may raise it too, so nothing here can carry free text: a reason
    that is not one of these codes becomes ``strategy_failed``, and an ``error_class``
    that is not a dotted class name of at most 128 characters is dropped.
    """

    def __init__(
        self,
        reason: str,
        *,
        failure: ProviderFailure | None = None,
        error_class: str | None = None,
    ) -> None:
        reason = reason if isinstance(reason, str) and reason in REASONS else "strategy_failed"
        error_class = error_class if is_class_name(error_class) else None
        failure = failure if isinstance(failure, ProviderFailure) else None
        detail = failure.value if failure is not None else error_class
        super().__init__(f"compaction failed: {reason}" + (f" ({detail})" if detail else ""))
        self.reason = reason
        self.failure = failure
        self.error_class = error_class
