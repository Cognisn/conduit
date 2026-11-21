"""The default catalogue's Gemini entries, as Google AI Studio's model pages give them.

Sources (checked 2026-11-04): aistudio.google.com/docs/models and the pages for
gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite and
gemini-embedding-001. gemini-2.5-pro and gemini-2.5-flash are deprecated and their
requests are routed to other models from 1 December 2026, so they are not declared.
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import Capabilities, default_catalogue
from conduit.core.residency import ResidencyScope

#: gemini-4-carbon added 2026-11-10 from its page (Stable, 1,048,576 tokens in, 65,536
#: out by default). Left out: gemini-3.1-pro-preview (preview), gemini-3.1-flash-lite
#: (shuts down 7 May 2027), gemini-2.5-flash-lite (existing users only) and
#: gemini-embedding-2 (no default output length stated).
CHAT = (
    "gemini-4-carbon",
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
)


def test_the_gemini_entries_are_exactly_these() -> None:
    declared = {e.model_id for e in default_catalogue() if e.provider == "gemini"}
    assert declared == {*CHAT, "gemini-embedding-001"}


@pytest.mark.parametrize("model", CHAT)
def test_a_chat_entry(model: str) -> None:
    entry = default_catalogue().require("gemini", model)
    assert entry.kind == "chat"
    assert entry.residency_scope is ResidencyScope.GLOBAL
    assert entry.context_window == 1_048_576
    assert entry.output_limit == 65_536
    assert entry.capabilities == Capabilities(
        tools=True, vision=True, prompt_caching=True, structured_output=True
    )


def test_the_embedding_entry() -> None:
    entry = default_catalogue().require("gemini", "gemini-embedding-001")
    assert entry.kind == "embedding"
    assert entry.context_window == 2048
    assert entry.embedding_dimensions == 3072


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-2.5-flash"])
def test_deprecated_models_are_not_declared(model: str) -> None:
    assert default_catalogue().get("gemini", model) is None
