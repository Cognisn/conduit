"""Providers with no Region choice: Gemini (global, no processing location stated),
Ollama (the deployment's own host) and the stub. Their listings offer no Regions, and
their settings choose and declare nothing.
"""

from __future__ import annotations

import pytest

from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.gemini import GeminiFactory, GeminiSettings
from conduit.providers.ollama import OllamaFactory, OllamaSettings
from conduit.providers.stub import StubFactory

NO_SECRETS = LiteralSecretResolver({})


def _registry() -> ProviderRegistry:
    return ProviderRegistry(
        secrets=NO_SECRETS, factories=[GeminiFactory(), OllamaFactory(), StubFactory()]
    )


def test_gemini_offers_no_region_and_says_it_is_global() -> None:
    [listing] = _registry().regions("gemini")
    assert listing.regions == ()
    assert listing.global_option is not None and "global" in listing.global_option
    assert listing.documentation.startswith("https://")


def test_ollama_offers_no_region() -> None:
    [listing] = _registry().regions("ollama")
    assert listing.regions == () and listing.global_option is None


def test_the_stub_offers_nothing() -> None:
    assert _registry().regions("stub") == ()


@pytest.mark.parametrize(
    "settings",
    [
        GeminiSettings.model_validate({"api_key": "secret://gemini/key"}),
        OllamaSettings.model_validate({}),
    ],
)
def test_settings_choose_and_declare_nothing(settings: GeminiSettings | OllamaSettings) -> None:
    assert settings.region is None
    assert settings.region_choice() is None
    assert settings.declared_residency("any-model") is None
