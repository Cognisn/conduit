"""Ollama settings: a loopback base URL by default, and no credentials in the URL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conduit.core.errors import ConfigurationError
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.ollama import OllamaFactory, OllamaSettings


def test_the_default_base_url_is_loopback() -> None:
    settings = OllamaSettings()
    assert settings.kind == "ollama"
    assert settings.base_url == "http://127.0.0.1:11434"
    assert settings.num_ctx is None


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("https://ollama.example.com/", "https://ollama.example.com"),
        ("http://192.0.2.10:11434/ollama/", "http://192.0.2.10:11434/ollama"),
        ("http://[2001:db8::1]:11434", "http://[2001:db8::1]:11434"),
    ],
)
def test_a_base_url_is_accepted_without_a_trailing_slash(given: str, kept: str) -> None:
    assert OllamaSettings(base_url=given).base_url == kept


@pytest.mark.parametrize(
    "url",
    [
        "ftp://ollama.example.com",
        "ollama.example.com:11434",
        "http://",
        "http://user:hunter2@ollama.example.com",
        "http://user@ollama.example.com",
        "http://ollama.example.com/?key=hunter2",
        "http://ollama.example.com/#hunter2",
    ],
)
def test_a_bad_base_url_is_refused_without_echoing_it(url: str) -> None:
    with pytest.raises(ValidationError) as caught:
        OllamaSettings(base_url=url)
    assert "hunter2" not in str(caught.value)
    assert "ollama.example.com" not in str(caught.value)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        OllamaSettings.model_validate({"kind": "ollama", "api_key": "secret://x"})


def test_num_ctx_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        OllamaSettings(num_ctx=0)


def test_the_registry_validates_a_mapping_against_the_adapter_model() -> None:
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[OllamaFactory()])
    settings = registry.validate("local", {"kind": "ollama", "num_ctx": 8192})
    assert isinstance(settings, OllamaSettings)
    assert settings.num_ctx == 8192
    with pytest.raises(ConfigurationError):
        registry.validate("local", {"kind": "ollama", "base_url": "ftp://example.com"})
