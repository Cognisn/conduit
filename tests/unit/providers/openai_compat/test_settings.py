"""Settings for the OpenAI-compatible adapter: three services, credentials only as references."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.core.settings import ProviderSettings
from conduit.providers.openai_compat import (
    FOUNDRY_SCOPE,
    OPENAI_BASE_URL,
    XAI_BASE_URL,
    FoundryApiKey,
    FoundryEntraId,
    FoundryService,
    OpenAICompatFactory,
    OpenAICompatSettings,
    OpenAIService,
    XAIService,
)

FOUNDRY_ENDPOINT = "https://example-resource.openai.azure.com/openai/v1/"


def _settings(service: dict[str, Any], **fields: Any) -> OpenAICompatSettings:
    return OpenAICompatSettings.model_validate(
        {"kind": "openai_compat", "service": service, **fields}
    )


class TestServices:
    def test_openai_defaults_to_the_public_endpoint(self) -> None:
        settings = _settings({"name": "openai", "api_key": "secret://OPENAI_KEY"})
        assert isinstance(settings.service, OpenAIService)
        assert settings.service.base_url == OPENAI_BASE_URL == "https://api.openai.com/v1"
        assert settings.service.api_key == SecretRef("OPENAI_KEY")
        assert settings.catalogue_provider == "openai"

    def test_xai_defaults_to_its_endpoint(self) -> None:
        settings = _settings({"name": "xai", "api_key": "secret://XAI_KEY"})
        assert isinstance(settings.service, XAIService)
        assert settings.service.base_url == XAI_BASE_URL == "https://api.x.ai/v1"
        assert settings.catalogue_provider == "xai"

    def test_foundry_with_an_api_key(self) -> None:
        settings = _settings(
            {
                "name": "foundry",
                "endpoint": FOUNDRY_ENDPOINT,
                "auth": {"mode": "api_key", "api_key": "secret://FOUNDRY_KEY"},
            }
        )
        assert isinstance(settings.service, FoundryService)
        assert isinstance(settings.service.auth, FoundryApiKey)
        assert settings.catalogue_provider == "openai"

    def test_foundry_with_entra_id_defaults_to_the_foundry_scope(self) -> None:
        settings = _settings(
            {"name": "foundry", "endpoint": FOUNDRY_ENDPOINT, "auth": {"mode": "entra_id"}}
        )
        assert isinstance(settings.service, FoundryService)
        assert isinstance(settings.service.auth, FoundryEntraId)
        assert settings.service.auth.scope == FOUNDRY_SCOPE == "https://ai.azure.com/.default"
        assert settings.service.auth.managed_identity_client_id is None

    def test_foundry_models_may_be_declared_under_xai(self) -> None:
        settings = _settings(
            {
                "name": "foundry",
                "endpoint": FOUNDRY_ENDPOINT,
                "auth": {"mode": "entra_id"},
                "catalogue_provider": "xai",
            }
        )
        assert settings.catalogue_provider == "xai"

    def test_foundry_catalogue_provider_is_openai_or_xai_only(self) -> None:
        with pytest.raises(ValidationError):
            _settings(
                {
                    "name": "foundry",
                    "endpoint": FOUNDRY_ENDPOINT,
                    "auth": {"mode": "entra_id"},
                    "catalogue_provider": "anthropic",
                }
            )

    def test_an_unknown_service_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _settings({"name": "chatgpt", "api_key": "secret://K"})

    def test_retries_are_bounded(self) -> None:
        service = {"name": "openai", "api_key": "secret://K"}
        assert _settings(service).max_retries == 2
        assert _settings(service, max_retries=0).max_retries == 0
        with pytest.raises(ValidationError):
            _settings(service, max_retries=11)


class TestCredentialsAreReferencesOnly:
    @pytest.mark.parametrize(
        "service",
        [
            {"name": "openai", "api_key": "sk-plainvalue0000000000"},
            {"name": "xai", "api_key": "xai-plainvalue"},
            {
                "name": "foundry",
                "endpoint": FOUNDRY_ENDPOINT,
                "auth": {"mode": "api_key", "api_key": "plain-foundry-value"},
            },
        ],
    )
    def test_a_plain_key_is_refused_without_echo(self, service: dict[str, Any]) -> None:
        with pytest.raises(ValidationError) as caught:
            _settings(service)
        plain = service.get("api_key") or service["auth"]["api_key"]
        assert plain not in str(caught.value)

    def test_an_api_key_service_needs_its_key(self) -> None:
        with pytest.raises(ValidationError):
            _settings({"name": "openai"})
        with pytest.raises(ValidationError):
            _settings(
                {"name": "foundry", "endpoint": FOUNDRY_ENDPOINT, "auth": {"mode": "api_key"}}
            )


class TestEndpoints:
    @pytest.mark.parametrize(
        "url",
        ["http://api.openai.com/v1", "ftp://example.com/v1", "api.openai.com/v1", ""],
    )
    def test_base_urls_must_be_https(self, url: str) -> None:
        with pytest.raises(ValidationError):
            _settings({"name": "openai", "api_key": "secret://K", "base_url": url})
        with pytest.raises(ValidationError):
            _settings({"name": "xai", "api_key": "secret://K", "base_url": url})

    def test_a_base_url_may_be_overridden(self) -> None:
        settings = _settings(
            {"name": "openai", "api_key": "secret://K", "base_url": "https://proxy.example.com/v1"}
        )
        assert isinstance(settings.service, OpenAIService)
        assert settings.service.base_url == "https://proxy.example.com/v1"

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://example-resource.openai.azure.com/openai/v1",
            "https://example-resource.openai.azure.com/openai/v1/",
            "https://example-resource.services.ai.azure.com/openai/v1/",
        ],
    )
    def test_foundry_endpoints_are_the_v1_openai_path(self, endpoint: str) -> None:
        settings = _settings(
            {"name": "foundry", "endpoint": endpoint, "auth": {"mode": "entra_id"}}
        )
        assert isinstance(settings.service, FoundryService)
        assert settings.service.endpoint == endpoint

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://example-resource.openai.azure.com/openai/v1/",
            "https://example-resource.openai.azure.com/",
            "https://example-resource.services.ai.azure.com/api/projects/example",
            "https://example-resource.services.ai.azure.com/anthropic",
        ],
    )
    def test_other_foundry_endpoints_are_refused(self, endpoint: str) -> None:
        with pytest.raises(ValidationError):
            _settings({"name": "foundry", "endpoint": endpoint, "auth": {"mode": "entra_id"}})


class TestRegistry:
    def test_generic_settings_are_revalidated_against_the_adapter(self) -> None:
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({}), factories=[OpenAICompatFactory()]
        )
        generic = ProviderSettings.model_validate(
            {"kind": "openai_compat", "service": {"name": "xai", "api_key": "secret://XAI_KEY"}}
        )
        validated = registry.validate("grok", generic)
        assert isinstance(validated, OpenAICompatSettings)
        assert validated.catalogue_provider == "xai"

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            _settings({"name": "openai", "api_key": "secret://K"}, model="gpt-5")
        with pytest.raises(ValidationError):
            _settings({"name": "openai", "api_key": "secret://K", "extra": 1})
