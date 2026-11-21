"""Gemini settings: the API key only as a secret reference, and a plain https base URL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conduit.core.errors import ConfigurationError
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.providers.gemini import GeminiFactory, GeminiSettings

KEY = "secret://gemini/key"


def test_the_defaults() -> None:
    settings = GeminiSettings(api_key=KEY)  # type: ignore[arg-type]
    assert settings.kind == "gemini"
    assert settings.api_key == SecretRef("gemini/key")
    assert settings.base_url == "https://generativelanguage.googleapis.com"
    assert settings.max_attempts == 3


def test_the_api_key_is_required() -> None:
    with pytest.raises(ValidationError):
        GeminiSettings.model_validate({"kind": "gemini"})


def test_a_literal_api_key_is_refused_without_echoing_it() -> None:
    with pytest.raises(ValidationError) as caught:
        GeminiSettings.model_validate({"kind": "gemini", "api_key": "AIzaLiteralKeyValue"})
    assert "AIzaLiteralKeyValue" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://gemini.example.com",
        "https://user:hunter2@gemini.example.com",
        "https://gemini.example.com/?key=hunter2",
        "https://gemini.example.com/#hunter2",
        "gemini.example.com",
    ],
)
def test_a_bad_base_url_is_refused_without_echoing_it(url: str) -> None:
    with pytest.raises(ValidationError) as caught:
        GeminiSettings.model_validate({"kind": "gemini", "api_key": KEY, "base_url": url})
    assert "hunter2" not in str(caught.value)
    assert "gemini.example.com" not in str(caught.value)


def test_a_proxy_base_url_keeps_its_path_without_a_trailing_slash() -> None:
    settings = GeminiSettings.model_validate(
        {"kind": "gemini", "api_key": KEY, "base_url": "https://gemini.example.com/proxy/"}
    )
    assert settings.base_url == "https://gemini.example.com/proxy"


@pytest.mark.parametrize("attempts", [0, 11])
def test_max_attempts_is_bounded(attempts: int) -> None:
    with pytest.raises(ValidationError):
        GeminiSettings.model_validate({"kind": "gemini", "api_key": KEY, "max_attempts": attempts})


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        GeminiSettings.model_validate({"kind": "gemini", "api_key": KEY, "project": "p"})


def test_the_registry_validates_a_mapping_against_the_adapter_model() -> None:
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[GeminiFactory()])
    settings = registry.validate("gemini", {"kind": "gemini", "api_key": KEY})
    assert isinstance(settings, GeminiSettings)
    with pytest.raises(ConfigurationError):
        registry.validate("gemini", {"kind": "gemini"})
