"""The default catalogue's Anthropic entries, as Anthropic's model documentation states them.

Sources (checked 4 November 2026):
- https://platform.claude.com/docs/en/about-claude/models/overview
- https://platform.claude.com/docs/en/models/sonnet-4-6/overview
- https://platform.claude.com/docs/en/models/haiku-4-5/overview
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs (supported models)

Rechecked 10 November 2026, adding every model the overview lists as current or as a
legacy model still available, from each model's page
(https://platform.claude.com/docs/en/models/<model>/overview) and the deprecations
page (https://platform.claude.com/docs/en/about-claude/model-deprecations). Claude
Mythos models are left out: they are available only to verified organisations.
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import default_catalogue
from conduit.core.residency import ResidencyScope

#: model id: (context window, output limit)
DOCUMENTED = {
    "claude-fable-5-2": (1_000_000, 128_000),
    "claude-fable-5-1": (1_000_000, 128_000),
    "claude-fable-5": (1_000_000, 128_000),
    "claude-opus-5-5": (1_000_000, 128_000),
    "claude-opus-5": (1_000_000, 128_000),
    "claude-opus-4-8": (1_000_000, 128_000),
    "claude-opus-4-7": (1_000_000, 128_000),
    "claude-opus-4-6": (1_000_000, 128_000),
    "claude-opus-4-5": (200_000, 64_000),
    "claude-sonnet-5-5": (1_000_000, 128_000),
    "claude-sonnet-5": (1_000_000, 128_000),
    "claude-haiku-5-5": (1_000_000, 128_000),
    "claude-sonnet-4-6": (1_000_000, 128_000),
    "claude-haiku-4-5": (200_000, 64_000),
}


def test_the_anthropic_entries_are_the_documented_models() -> None:
    held = {e.model_id for e in default_catalogue() if e.provider == "anthropic"}
    assert held == set(DOCUMENTED)


@pytest.mark.parametrize("model_id", sorted(DOCUMENTED))
def test_limits_and_capabilities(model_id: str) -> None:
    entry = default_catalogue().require("anthropic", model_id)
    assert (entry.context_window, entry.output_limit) == DOCUMENTED[model_id]
    assert entry.kind == "chat"
    # Every current model takes text and images, uses tools and supports prompt
    # caching; each is on the structured outputs page's supported list.
    assert entry.capabilities.tools
    assert entry.capabilities.vision
    assert entry.capabilities.prompt_caching
    assert entry.capabilities.structured_output
    # The direct API routes globally unless a request pins inference_geo, which
    # conduit does not send.
    assert entry.residency_scope is ResidencyScope.GLOBAL
