"""The default catalogue's openai and xai entries, as the providers document them.

Checked on 2026-11-04 against developers.openai.com/api/docs/models/<id> and
docs.x.ai/developers/models (see the workstream's session summary for every source).

Rechecked on 2026-11-10 (fix/0.1.0-catalogue-coverage): every current OpenAI chat model
the model pages mark as supported on Chat Completions with function calling, and
text-embedding-3-large (developers.openai.com/api/docs/guides/embeddings); gpt-6-sol and
gpt-6-luna are left out, since Chat Completions takes their function calls only at
reasoning effort none, which conduit does not send, and gpt-6.1-terra, whose tool calling
requires the Responses API. Every current X.AI chat model (docs.x.ai/docs/models
and each model's page); X.AI states no per-model output limit, so its entries carry the
documented max_completion_tokens default, 128,000 (Skuld Q0054).
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import Capabilities, default_catalogue
from conduit.core.residency import ResidencyScope

ALL = Capabilities(tools=True, vision=True, prompt_caching=True, structured_output=True)
NO_TOOLS = ALL.model_copy(update={"tools": False})
#: The function-calling guide (developers.openai.com/api/docs/guides/function-calling):
#: "GPT-6 Astra and GPT-6.1 Sol require the Responses API for tool calling". conduit's
#: adapter calls Chat Completions, so these serve chat without tools (Skuld decision).
RESPONSES_ONLY_TOOLS = frozenset({"gpt-6-astra", "gpt-6.1-sol"})

CHAT = [
    ("openai", "gpt-5", 400_000, 128_000),
    ("openai", "gpt-5-mini", 400_000, 128_000),
    ("openai", "gpt-6-astra", 1_050_000, 128_000),
    ("openai", "gpt-6.1-sol", 1_050_000, 128_000),
    ("openai", "gpt-5.6-sol", 1_050_000, 128_000),
    ("openai", "gpt-5.6-terra", 1_050_000, 128_000),
    ("openai", "gpt-5.6-luna", 1_050_000, 128_000),
    ("openai", "gpt-5.5", 1_050_000, 128_000),
    ("openai", "gpt-5.4", 1_050_000, 128_000),
    ("openai", "gpt-5.4-mini", 400_000, 128_000),
    ("openai", "gpt-5.2", 400_000, 128_000),
    ("xai", "grok-4.3", 1_000_000, 128_000),
    ("xai", "grok-4.7", 500_000, 128_000),
    ("xai", "grok-4.6", 500_000, 128_000),
    ("xai", "grok-4.5", 500_000, 128_000),
    ("xai", "grok-4.20-0309-reasoning", 1_000_000, 128_000),
    ("xai", "grok-4.20-0309-non-reasoning", 1_000_000, 128_000),
    ("xai", "grok-4.20-multi-agent-0309", 1_000_000, 128_000),
    ("xai", "grok-build-0.1", 256_000, 128_000),
]


@pytest.mark.parametrize(("provider", "model_id", "context_window", "output_limit"), CHAT)
def test_chat_entries(provider: str, model_id: str, context_window: int, output_limit: int) -> None:
    entry = default_catalogue().require(provider, model_id)
    assert entry.kind == "chat"
    assert entry.residency_scope is ResidencyScope.GLOBAL
    assert entry.capabilities == (NO_TOOLS if model_id in RESPONSES_ONLY_TOOLS else ALL)
    assert entry.context_window == context_window
    assert entry.output_limit == output_limit


def test_the_retired_grok_4_is_not_declared() -> None:
    # grok-4-0709 was retired on 15 May 2026; requests are served by grok-4.3 instead.
    catalogue = default_catalogue()
    assert catalogue.get("xai", "grok-4") is None
    assert catalogue.get("xai", "grok-4-0709") is None


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-6-sol",  # function calling on Chat Completions only at reasoning effort none
        "gpt-6-luna",  # the same: "only with `reasoning_effort` set to `none`"
        "gpt-6.1-terra",  # "Tool calling requires the Responses API."
        "gpt-5.5-pro",  # Responses API only
        "chat-latest",  # an alias that moves, with no prompt caching listed
    ],
)
def test_openai_models_left_out(model_id: str) -> None:
    assert default_catalogue().get("openai", model_id) is None


@pytest.mark.parametrize(
    ("model_id", "dimensions"),
    [("text-embedding-3-small", 1536), ("text-embedding-3-large", 3072)],
)
def test_the_embedding_entries(model_id: str, dimensions: int) -> None:
    # The embeddings guide: default lengths 1536 and 3072, max input 8192 tokens each.
    entry = default_catalogue().require("openai", model_id)
    assert entry.kind == "embedding"
    assert entry.context_window == 8192
    assert entry.embedding_dimensions == dimensions


def test_only_these_openai_and_xai_entries() -> None:
    declared = {
        (e.provider, e.model_id) for e in default_catalogue() if e.provider in {"openai", "xai"}
    }
    assert declared == {
        *((provider, model_id) for provider, model_id, _, _ in CHAT),
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-large"),
    }
