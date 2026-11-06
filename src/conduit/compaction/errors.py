"""Errors the compaction layer raises.

A conversation is content, and a summary model's text is content too, so no error here
carries either: each names a fixed reason code, and at most a provider failure class or
the class name of an underlying error. None keeps the exception it was raised for
(D0040): ``__cause__`` and ``__context__`` are always ``None``.
"""

from __future__ import annotations

from conduit.core.errors import ConduitError, ProviderFailure

__all__ = ["CompactionError", "CompactionFailed"]


class CompactionError(ConduitError):
    """Base class for the compaction layer's errors."""


class CompactionFailed(CompactionError):
    """A compaction could not be completed, so the conversation was left as it was.

    ``reason`` is a fixed code:

    - ``summary_failed``: the summary model call failed (``failure`` gives its class);
    - ``summary_empty``: the summary model answered with no text;
    - ``summary_too_large``: the conversation to summarise does not fit the summary
      model's context window;
    - ``transcript_mismatch``: the stored transcript's active conversation is not the
      conversation the hook was shown, so no compaction point can be placed;
    - ``store_failed``: the transcript store failed (``error_class`` names the error);
    - ``strategy_failed``: a product strategy raised (``error_class`` names the error).
    """

    def __init__(
        self,
        reason: str,
        *,
        failure: ProviderFailure | None = None,
        error_class: str | None = None,
    ) -> None:
        detail = failure.value if failure is not None else error_class
        super().__init__(f"compaction failed: {reason}" + (f" ({detail})" if detail else ""))
        self.reason = reason
        self.failure = failure
        self.error_class = error_class
