"""The adapter's factory for the provider registry.

``create_client`` is the only place the openai SDK and azure-identity are imported
(through ``importlib``, so importing conduit never needs the ``openai`` extra), and the
only place their clients and the Entra ID token provider are built.

The SDK reads ``OPENAI_ORG_ID``, ``OPENAI_PROJECT_ID``, ``OPENAI_ADMIN_KEY`` and
``OPENAI_WEBHOOK_SECRET`` from the environment when they are not passed. conduit's
credentials come only from secret references, and an OpenAI organisation id must never
be sent to X.AI or Foundry, so those are cleared on every client unless the settings
give the organisation or project.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import Provider
from conduit.core.settings import ProviderSettings
from conduit.providers.openai_compat.provider import OpenAICompatProvider
from conduit.providers.openai_compat.settings import (
    FoundryApiKey,
    FoundryService,
    OpenAICompatSettings,
    OpenAIService,
)


@dataclass(eq=False)
class OpenAICompatClient:
    """The SDK client, with the Entra ID credential and token provider when used."""

    sdk: Any
    credential: Any = None
    token_provider: Callable[[], Awaitable[str]] | None = None

    @property
    def chat(self) -> Any:
        return self.sdk.chat

    async def close(self) -> None:
        try:
            await self.sdk.close()
        finally:
            if self.credential is not None:
                await self.credential.close()


def _secret(secrets: Mapping[str, SecretStr], path: str) -> str:
    try:
        return secrets[path].get_secret_value()
    except KeyError:
        raise ConfigurationError(f"openai_compat: {path} was not resolved") from None


class OpenAICompatFactory:
    """The OpenAI-compatible adapter (OpenAI, X.AI, Microsoft Foundry) for the registry.

    ``catalogue`` decides which models get structured output; it defaults to conduit's
    default catalogue. Pass the applied catalogue when a product's overlays change
    ``structured_output`` (until the registry passes it to ``create_provider``).
    """

    kind = "openai_compat"
    settings_model = OpenAICompatSettings

    def __init__(self, catalogue: Catalogue | None = None) -> None:
        self._catalogue = catalogue

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        if not isinstance(settings, OpenAICompatSettings):
            raise ConfigurationError("openai_compat settings expected")
        openai = importlib.import_module("openai")
        service = settings.service
        credential: Any = None
        token_provider: Callable[[], Awaitable[str]] | None = None
        api_key: str | Callable[[], Awaitable[str]]
        if isinstance(service, FoundryService):
            base_url = service.endpoint
            if isinstance(service.auth, FoundryApiKey):
                api_key = _secret(secrets, "service.auth.api_key")
            else:
                identity = importlib.import_module("azure.identity.aio")
                options: dict[str, Any] = {}
                if service.auth.managed_identity_client_id is not None:
                    options["managed_identity_client_id"] = service.auth.managed_identity_client_id
                credential = identity.DefaultAzureCredential(**options)
                token_provider = identity.get_bearer_token_provider(credential, service.auth.scope)
                api_key = token_provider
        else:
            base_url = service.base_url
            api_key = _secret(secrets, "service.api_key")
        organization = service.organization if isinstance(service, OpenAIService) else None
        project = service.project if isinstance(service, OpenAIService) else None
        sdk = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            max_retries=settings.max_retries,
        )
        # Never the environment's values (see the module docstring).
        sdk.organization = organization
        sdk.project = project
        sdk.admin_api_key = None
        sdk.webhook_secret = None
        return OpenAICompatClient(sdk=sdk, credential=credential, token_provider=token_provider)

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        if not isinstance(settings, OpenAICompatSettings):
            raise ConfigurationError("openai_compat settings expected")
        return OpenAICompatProvider(
            client, catalogue_provider=settings.catalogue_provider, catalogue=self._catalogue
        )
