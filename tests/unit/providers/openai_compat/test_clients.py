"""``create_client`` with the real SDKs (the ``openai`` extra). No network is reached:
clients are built and inspected, and the Entra ID token provider is checked with a
stand-in credential."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.providers.openai_compat import OpenAICompatFactory, OpenAICompatSettings
from conduit.providers.openai_compat.factory import OpenAICompatClient

ENDPOINT = "https://example-resource.openai.azure.com/openai/v1/"


def _settings(service: dict[str, Any], **fields: Any) -> OpenAICompatSettings:
    return OpenAICompatSettings.model_validate(
        {"kind": "openai_compat", "service": service, **fields}
    )


def test_importing_the_adapter_imports_no_sdk() -> None:
    code = (
        "import sys\n"
        "import conduit.providers\n"
        "import conduit.providers.openai_compat\n"
        "from conduit.providers import builtin_factories\n"
        "builtin_factories()\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in {'openai', 'azure'})\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


class TestWithTheSdk:
    @pytest.fixture(autouse=True)
    def _sdk(self) -> None:
        pytest.importorskip("openai")
        pytest.importorskip("azure.identity.aio")

    def _client(
        self, settings: OpenAICompatSettings, secrets: Mapping[str, SecretStr]
    ) -> OpenAICompatClient:
        client = OpenAICompatFactory().create_client(settings, secrets)
        assert isinstance(client, OpenAICompatClient)
        return client

    async def test_openai(self) -> None:
        settings = _settings(
            {"name": "openai", "api_key": "secret://K", "organization": "org-example"},
            max_retries=4,
        )
        client = self._client(settings, {"service.api_key": SecretStr("test-openai-value")})
        assert str(client.sdk.base_url).rstrip("/") == "https://api.openai.com/v1"
        assert client.sdk.api_key == "test-openai-value"
        assert client.sdk.organization == "org-example"
        assert client.sdk.max_retries == 4
        assert client.credential is None
        await client.close()

    async def test_xai(self) -> None:
        settings = _settings({"name": "xai", "api_key": "secret://K"})
        client = self._client(settings, {"service.api_key": SecretStr("test-xai-value")})
        assert str(client.sdk.base_url).rstrip("/") == "https://api.x.ai/v1"
        assert client.sdk.api_key == "test-xai-value"
        await client.close()

    async def test_foundry_with_an_api_key(self) -> None:
        settings = _settings(
            {
                "name": "foundry",
                "endpoint": ENDPOINT,
                "auth": {"mode": "api_key", "api_key": "secret://K"},
            }
        )
        client = self._client(settings, {"service.auth.api_key": SecretStr("test-foundry")})
        assert str(client.sdk.base_url).rstrip("/") == ENDPOINT.rstrip("/")
        assert client.sdk.api_key == "test-foundry"
        assert client.credential is None
        await client.close()

    async def test_foundry_with_entra_id_builds_the_token_provider_here(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        identity = pytest.importorskip("azure.identity.aio")
        made: list[dict[str, Any]] = []

        class Token:
            token = "example-entra-token"
            expires_on = 4102444800

        class StandInCredential:
            def __init__(self, **kwargs: Any) -> None:
                made.append(kwargs)
                self.scopes: list[tuple[str, ...]] = []
                self.closed = False

            async def get_token(self, *scopes: str, **kwargs: Any) -> Token:
                self.scopes.append(scopes)
                return Token()

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(identity, "DefaultAzureCredential", StandInCredential)
        settings = _settings(
            {
                "name": "foundry",
                "endpoint": ENDPOINT,
                "auth": {"mode": "entra_id", "managed_identity_client_id": "example-client"},
            }
        )
        client = self._client(settings, {})
        assert made == [{"managed_identity_client_id": "example-client"}]
        credential = client.credential
        assert isinstance(credential, StandInCredential)
        token_provider = client.token_provider
        assert token_provider is not None
        assert await token_provider() == "example-entra-token"
        assert credential.scopes == [("https://ai.azure.com/.default",)]
        await client.close()
        assert credential.closed

    async def test_entra_id_without_a_client_id_passes_no_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        identity = pytest.importorskip("azure.identity.aio")
        made: list[dict[str, Any]] = []

        class StandInCredential:
            def __init__(self, **kwargs: Any) -> None:
                made.append(kwargs)

            async def close(self) -> None:
                return None

        monkeypatch.setattr(identity, "DefaultAzureCredential", StandInCredential)
        settings = _settings(
            {"name": "foundry", "endpoint": ENDPOINT, "auth": {"mode": "entra_id"}}
        )
        await self._client(settings, {}).close()
        assert made == [{}]

    @pytest.mark.parametrize("name", ["openai", "xai"])
    async def test_the_environment_never_supplies_credentials_or_headers(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_ORG_ID", "org-from-environment")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-environment")
        monkeypatch.setenv("OPENAI_ADMIN_KEY", "admin-from-environment")
        monkeypatch.setenv("OPENAI_WEBHOOK_SECRET", "webhook-from-environment")
        settings = _settings({"name": name, "api_key": "secret://K"})
        client = self._client(settings, {"service.api_key": SecretStr("test-value")})
        assert client.sdk.organization is None
        assert client.sdk.project is None
        assert client.sdk.admin_api_key is None
        assert client.sdk.webhook_secret is None
        headers = client.sdk.default_headers
        assert "OpenAI-Organization" not in headers or not isinstance(
            headers["OpenAI-Organization"], str
        )
        await client.close()

    async def test_the_settings_organisation_and_project_are_sent(self) -> None:
        settings = _settings(
            {
                "name": "openai",
                "api_key": "secret://K",
                "organization": "org-example",
                "project": "proj_example",
            }
        )
        client = self._client(settings, {"service.api_key": SecretStr("test-value")})
        assert client.sdk.default_headers["OpenAI-Organization"] == "org-example"
        assert client.sdk.default_headers["OpenAI-Project"] == "proj_example"
        await client.close()

    async def test_the_client_offers_chat_completions(self) -> None:
        settings = _settings({"name": "openai", "api_key": "secret://K"})
        client = self._client(settings, {"service.api_key": SecretStr("test-openai-value")})
        assert client.chat is client.sdk.chat
        await client.close()
