"""The adapter's factory for the provider registry.

``create_client`` is the only place the openai SDK and azure-identity are imported
(through ``importlib``, so importing conduit never needs the ``openai`` extra), and the
only place their clients and the Entra ID token provider are built.

The SDK reads ``OPENAI_ORG_ID``, ``OPENAI_PROJECT_ID``, ``OPENAI_ADMIN_KEY`` and
``OPENAI_WEBHOOK_SECRET`` from the environment when they are not passed. conduit's
credentials come only from secret references, and an OpenAI organisation id must never
be sent to X.AI or Foundry, so those are cleared on every client unless the settings
give the organisation or project. ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` are never
used: the key and base URL are always passed.

Some environment variables still reach the SDK and its HTTP client, and deployments
should know them:

- ``OPENAI_LOG`` (``info`` or ``debug``) makes the SDK log each request's options,
  including the prompt and tool content, through its own ``openai`` logger. That
  bypasses conduit's content-capture policy, so production deployments should leave it
  unset.
- The HTTP client honours ``HTTPS_PROXY``, ``HTTP_PROXY``, ``ALL_PROXY`` and
  ``NO_PROXY``, so a proxy set in the environment sees every call (the TLS session
  still ends at the provider).

Entra ID comes from ``conduit.providers.openai_compat.entra``, which the Anthropic
adapter's Foundry endpoint shares. If building the SDK client fails after an Entra ID
credential was made, the credential is closed: on the running loop when there is one,
otherwise straight away.

The key (or Entra ID credential) is resolved and the client built in one guarded step
(see ``conduit.providers``): a constructor that raises becomes a fixed
``ConfigurationError`` naming the adapter and the exception's class.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import Provider
from conduit.core.registry import guarded_client
from conduit.core.settings import ProviderSettings
from conduit.providers.openai_compat.entra import (
    close_quietly,
    close_soon,
    entra_id_token_provider,
)
from conduit.providers.openai_compat.provider import OpenAICompatProvider
from conduit.providers.openai_compat.settings import (
    FoundryApiKey,
    FoundryService,
    OpenAICompatSettings,
    OpenAIService,
    XAIService,
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
                await close_quietly(self.credential)  # never replaces the SDK's failure


def _secret(secrets: Mapping[str, SecretStr], path: str) -> str:
    try:
        return secrets[path].get_secret_value()
    except KeyError:
        raise ConfigurationError(f"openai_compat: {path} was not resolved") from None


class OpenAICompatFactory:
    """The OpenAI-compatible adapter (OpenAI, X.AI, Microsoft Foundry) for the registry.

    The catalogue the registry passes to ``create_provider`` (the one it applied the
    product's overlays to) decides which models get structured output. Called without
    one, outside a registry, the factory uses conduit's default catalogue.
    """

    kind = "openai_compat"
    settings_model = OpenAICompatSettings

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        if not isinstance(settings, OpenAICompatSettings):
            raise ConfigurationError("openai_compat settings expected")
        openai = importlib.import_module("openai")
        service = settings.service
        max_retries = settings.max_retries

        def build() -> OpenAICompatClient:
            # The key is resolved here, in the guarded step's own frame.
            credential: Any = None
            token_provider: Callable[[], Awaitable[str]] | None = None
            api_key: str | Callable[[], Awaitable[str]]
            if isinstance(service, FoundryService):
                base_url = service.endpoint
                if isinstance(service.auth, FoundryApiKey):
                    api_key = _secret(secrets, "service.auth.api_key")
                else:
                    credential, token_provider = entra_id_token_provider(service.auth)
                    api_key = token_provider
            else:
                base_url = service.base_url
                api_key = _secret(secrets, "service.api_key")
            organization = service.organization if isinstance(service, OpenAIService) else None
            project = service.project if isinstance(service, OpenAIService) else None
            try:
                sdk = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    organization=organization,
                    project=project,
                    max_retries=max_retries,
                )
            except Exception:
                if credential is not None:
                    close_soon(credential)
                raise
            # Never the environment's values (see the module docstring).
            sdk.organization = organization
            sdk.project = project
            sdk.admin_api_key = None
            sdk.webhook_secret = None
            return OpenAICompatClient(sdk=sdk, credential=credential, token_provider=token_provider)

        return guarded_client(self.kind, build)

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        if not isinstance(settings, OpenAICompatSettings):
            raise ConfigurationError("openai_compat settings expected")
        return OpenAICompatProvider(
            client,
            catalogue_provider=settings.catalogue_provider,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
            reasoning_by_total=isinstance(settings.service, XAIService),
        )
