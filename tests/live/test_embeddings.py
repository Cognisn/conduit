"""Live: embeddings on every remote provider and the local extra.

Each provider is called at its catalogued dimensions and, where the model offers them,
at shorter ones (Titan Text Embeddings V2 at 512 and 256, OpenAI's text-embedding-3 and
Gemini at a reduced length).
"""

from __future__ import annotations

import importlib

import pytest

from conduit.core.catalogue import CatalogueOverlay, ModelEntry, default_catalogue
from conduit.core.residency import ResidencyScope
from tests.live import needs
from tests.live.harness import check_embeddings, started_embeddings
from tests.live.needs import live_env, present, value
from tests.live.targets import (
    Target,
    bedrock_settings,
    deployment_overlay,
    first_bedrock_auth,
    foundry_settings,
    gemini_settings,
    offered_in,
    ollama_settings,
    openai_settings,
)

pytestmark = pytest.mark.integration

TITAN_V2 = "amazon.titan-embed-text-v2:0"


async def _embed_at(target: Target, own: int, *shorter: int) -> None:
    for length in (own, *shorter):
        async with started_embeddings(
            target, dimensions=None if length == own else length
        ) as binding:
            await check_embeddings(binding, target.subject, length)


def _own(provider: str, model: str) -> int:
    entry = default_catalogue().get(provider, model)
    assert entry is not None and entry.embedding_dimensions is not None
    return entry.embedding_dimensions


@pytest.mark.live_needs(needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH)
async def test_bedrock_titan_v2_dimensions() -> None:
    """Q0004 and the embeddings note: Titan V2 at 1,024, 512 and 256."""
    region = value(live_env(), "BEDROCK_REGION")
    mode = first_bedrock_auth(live_env())
    assert region is not None and mode is not None
    entry = default_catalogue().require("bedrock", TITAN_V2)
    if not offered_in(entry, region):
        pytest.skip(f"{TITAN_V2} is not catalogued in {region}")
    target = Target(
        f"bedrock {region} {mode}", bedrock_settings(region, mode, live_env()), TITAN_V2
    )
    await _embed_at(target, _own("bedrock", TITAN_V2), 512, 256)


@pytest.mark.live_needs(needs.OPENAI)
async def test_openai_text_embedding_3_small() -> None:
    model = "text-embedding-3-small"
    await _embed_at(Target("openai", openai_settings(), model), _own("openai", model), 256)


@pytest.mark.live_needs(needs.FOUNDRY_ENDPOINT, needs.FOUNDRY_ANY_AUTH, needs.FOUNDRY_EMBEDDING)
async def test_foundry_embedding_deployment() -> None:
    deployment = value(live_env(), "FOUNDRY_EMBEDDING_DEPLOYMENT")
    assert deployment is not None
    entra = not present(live_env(), "FOUNDRY_OPENAI_API_KEY")
    overlays = deployment_overlay(live_env(), "openai", deployment, kind="embedding")
    target = Target(
        f"foundry {'entra-id' if entra else 'api-key'}",
        foundry_settings(live_env(), entra=entra),
        deployment,
        overlays,
    )
    own = overlays[0].add[0].embedding_dimensions if overlays else _own("openai", deployment)
    assert own is not None
    await _embed_at(target, own)


@pytest.mark.live_needs(needs.GEMINI)
async def test_gemini_embedding_001() -> None:
    model = "gemini-embedding-001"
    await _embed_at(Target("gemini", gemini_settings(), model), _own("gemini", model), 768)


@pytest.mark.live_needs(needs.OLLAMA)
async def test_ollama_nomic_embed_text() -> None:
    model = value(live_env(), "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    assert model is not None
    await _embed_at(Target("ollama", ollama_settings(live_env()), model), _own("ollama", model))


@pytest.mark.live_needs(needs.LOCAL_EMBEDDINGS)
async def test_local_sentence_transformers() -> None:
    """The local extra on the CPU. Skips, naming the reason, where torch cannot load."""
    loaded = ""
    try:
        importlib.import_module("torch")
    except Exception as error:  # torch's DLL load fails without the VC++ runtime (D0073)
        loaded = type(error).__name__
    if loaded:
        pytest.skip(f"torch cannot load on this host ({loaded}); the local extra needs it")
    model = value(live_env(), "LOCAL_EMBEDDING_MODEL")
    dimensions = value(live_env(), "LOCAL_EMBEDDING_DIMENSIONS")
    assert model is not None and dimensions is not None
    entry = ModelEntry(
        model_id=model,
        provider="sentence_transformers",
        kind="embedding",
        residency_scope=ResidencyScope.LOCAL,
        context_window=int(value(live_env(), "LOCAL_EMBEDDING_CONTEXT", "512") or "512"),
        embedding_dimensions=int(dimensions),
    )
    target = Target(
        "local",
        {"kind": "sentence_transformers"},
        model,
        (CatalogueOverlay(name="live-local", add=(entry,)),),
    )
    await _embed_at(target, int(dimensions))
