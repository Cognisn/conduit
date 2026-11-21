"""Live: model discovery through the registry, for each provider that lists its models.

Runs only with ``-m integration`` and the variables each test's needs name. Each check
lists the provider's models with the shipped adapters and reports counts per outcome
only. Claude in Foundry and Foundry's OpenAI v1 endpoint are reported unsupported, which
needs no network and is checked once each.
"""

from __future__ import annotations

import pytest

from conduit.core.registry import ProviderRegistry
from conduit.providers import builtin_factories
from tests.live import needs
from tests.live.harness import check_discovery
from tests.live.needs import live_env, value
from tests.live.targets import (
    anthropic_direct,
    anthropic_foundry,
    bedrock_settings,
    first_bedrock_auth,
    foundry_settings,
    gemini_settings,
    mantle_auth,
    mantle_settings,
    ollama_settings,
    openai_settings,
    resolver,
    xai_settings,
)

pytestmark = pytest.mark.integration


@pytest.mark.live_needs(needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH)
async def test_bedrock_lists_foundation_models_and_profiles() -> None:
    region = value(live_env(), "BEDROCK_REGION")
    mode = first_bedrock_auth(live_env())
    assert region is not None and mode is not None
    result = await check_discovery(
        f"bedrock {region} {mode}", bedrock_settings(region, mode, live_env())
    )
    assert any(m.label == "SYSTEM_DEFINED" for m in result.models), "no inference profile"


@pytest.mark.live_needs(needs.MANTLE_REGION, needs.MANTLE_ANY_AUTH)
async def test_bedrock_mantle_lists_claude_models() -> None:
    region = value(live_env(), "BEDROCK_MANTLE_REGION")
    chosen = mantle_auth(live_env())
    assert region is not None and chosen is not None
    auth, long_term = chosen
    result = await check_discovery(
        f"bedrock-mantle {region}", mantle_settings(region, auth, long_term)
    )
    assert all(m.model_id.startswith("anthropic.") for m in result.models)


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
async def test_anthropic_lists_models() -> None:
    await check_discovery("anthropic direct", anthropic_direct())


@pytest.mark.live_needs(needs.OPENAI)
async def test_openai_lists_models() -> None:
    await check_discovery("openai", openai_settings())


@pytest.mark.live_needs(needs.XAI)
async def test_xai_lists_models() -> None:
    await check_discovery("xai", xai_settings())


@pytest.mark.live_needs(needs.GEMINI)
async def test_gemini_lists_chat_models() -> None:
    await check_discovery("gemini", gemini_settings())


@pytest.mark.live_needs(needs.OLLAMA)
async def test_ollama_lists_local_models() -> None:
    await check_discovery("ollama", ollama_settings(live_env()))


@pytest.mark.live_needs(needs.FOUNDRY_ANTHROPIC_KEY)
async def test_claude_in_foundry_is_reported_unsupported() -> None:
    registry = ProviderRegistry(secrets=resolver(live_env()), factories=builtin_factories())
    result = await registry.discover("live", anthropic_foundry(live_env(), entra=False))
    assert not result.supported


@pytest.mark.live_needs(needs.FOUNDRY_OPENAI_KEY)
async def test_foundry_openai_is_reported_unsupported() -> None:
    registry = ProviderRegistry(secrets=resolver(live_env()), factories=builtin_factories())
    result = await registry.discover("live", foundry_settings(live_env(), entra=False))
    assert not result.supported
