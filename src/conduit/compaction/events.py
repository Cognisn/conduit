"""The compaction event: what was compacted, how, and what it saved.

It carries identifiers, counts and an outcome. The summary is content (a model's or a
product's text about the conversation), so it appears only as ``CapturedContent`` of the
prompt kind: its hash and size always, its text only when the capture policy allows
prompts (C19).
"""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeInt, PositiveInt

from conduit.core.events import CapturedContent, Event
from conduit.core.identifiers import Identifier

__all__ = ["CompactionEvent", "CompactionOutcome", "CompactionTrigger"]

CompactionTrigger = Literal["threshold", "forced"]
CompactionOutcome = Literal["compacted", "skipped", "failed"]


class CompactionEvent(Event):
    """One compaction, or one attempt.

    ``outcome`` is ``compacted``; ``skipped``, with ``reason`` ``nothing_to_compact`` (a
    forced compaction found too little to replace), ``prefix_too_large`` (the fixed
    prefix, the kept tail and the summary allowance already reach the threshold) or
    ``not_below_threshold`` (the result would still be at or over it); or ``failed``
    (``reason`` gives the ``CompactionFailed`` code, and the conversation was left as it
    was). ``tokens_before`` is the estimate that was tested
    against the threshold, ``tokens_after`` the estimate of the compacted prompt.
    ``entries_replaced`` counts the messages newly folded into the summary (an earlier
    summary is not counted again), and ``replaces_through`` is the sequence number the
    compaction point replaces through, when it was written to a transcript.
    """

    type: Literal["compaction"] = "compaction"
    strategy: Identifier
    trigger: CompactionTrigger
    outcome: CompactionOutcome
    reason: Identifier | None = None
    context_window: PositiveInt
    tokens_before: NonNegativeInt
    tokens_after: NonNegativeInt | None = None
    messages_before: NonNegativeInt
    messages_after: NonNegativeInt | None = None
    entries_replaced: NonNegativeInt = 0
    replaces_through: NonNegativeInt | None = None
    summary: CapturedContent | None = None
