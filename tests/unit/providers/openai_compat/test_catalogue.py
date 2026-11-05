"""The default catalogue's openai and xai entries, as the providers document them.

Checked on 2026-11-04 against developers.openai.com/api/docs/models/<id> and
docs.x.ai/developers/models (see the workstream's session summary for every source).
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import Capabilities, default_catalogue
from conduit.core.residency import ResidencyScope

ALL = Capabilities(tools=True, vision=True, prompt_caching=True, structured_output=True)


@pytest.mark.parametrize(
    ("provider", "model_id", "context_window", "output_limit"),
    [
        ("openai", "gpt-5", 400_000, 128_000),
        ("openai", "gpt-5-mini", 400_000, 128_000),
        ("openai", "gpt-6-astra", 1_050_000, 128_000),
        ("openai", "gpt-6.1-sol", 1_050_000, 128_000),
        ("xai", "grok-4.3", 1_000_000, 128_000),
    ],
)
def test_chat_entries(provider: str, model_id: str, context_window: int, output_limit: int) -> None:
    entry = default_catalogue().require(provider, model_id)
    assert entry.kind == "chat"
    assert entry.residency_scope is ResidencyScope.GLOBAL
    assert entry.capabilities == ALL
    assert entry.context_window == context_window
    assert entry.output_limit == output_limit


def test_the_retired_grok_4_is_not_declared() -> None:
    # grok-4-0709 was retired on 15 May 2026; requests are served by grok-4.3 instead.
    catalogue = default_catalogue()
    assert catalogue.get("xai", "grok-4") is None
    assert catalogue.get("xai", "grok-4-0709") is None


def test_the_embedding_entry_is_unchanged() -> None:
    entry = default_catalogue().require("openai", "text-embedding-3-small")
    assert entry.kind == "embedding"
    assert entry.context_window == 8191
    assert entry.embedding_dimensions == 1536


def test_only_these_openai_and_xai_entries() -> None:
    declared = {
        (e.provider, e.model_id) for e in default_catalogue() if e.provider in {"openai", "xai"}
    }
    assert declared == {
        ("openai", "gpt-5"),
        ("openai", "gpt-5-mini"),
        ("openai", "gpt-6-astra"),
        ("openai", "gpt-6.1-sol"),
        ("openai", "text-embedding-3-small"),
        ("xai", "grok-4.3"),
    }
