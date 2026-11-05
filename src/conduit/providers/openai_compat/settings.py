"""Settings for the OpenAI-compatible adapter.

One adapter (kind ``openai_compat``) serves three services, chosen by ``service.name``:

- ``openai``: the OpenAI API (``https://api.openai.com/v1``), with an API key.
- ``xai``: the X.AI API (``https://api.x.ai/v1``), with an API key.
- ``foundry``: Microsoft Foundry's OpenAI v1 endpoint
  (``https://<resource>.openai.azure.com/openai/v1/``), with an API key or Microsoft
  Entra ID (``DefaultAzureCredential`` and a bearer token provider for
  ``https://ai.azure.com/.default``).

API keys are only ever ``secret://`` references. The catalogue provider is ``openai``
or ``xai`` for those services; for Foundry it is ``service.catalogue_provider``
(``openai`` by default, ``xai`` for Grok deployments). A Foundry model id is the
deployment name, which the product declares in a catalogue overlay under that provider
with its own residency (Foundry deployments are regional; the default catalogue's
``openai`` and ``xai`` entries describe the public APIs). Name deployments so they do
not collide with a public model id, or the overlay would change that entry too.
"""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field

from conduit.core.secrets import SecretRef
from conduit.core.settings import ProviderSettings, SettingsModel

OPENAI_BASE_URL = "https://api.openai.com/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
FOUNDRY_SCOPE = "https://ai.azure.com/.default"
FOUNDRY_PATH = "/openai/v1"


def _https(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("must be an https URL")
    return value


def _foundry_endpoint(value: str) -> str:
    _https(value)
    if urlsplit(value).path.rstrip("/") != FOUNDRY_PATH:
        raise ValueError(f"must be the resource's {FOUNDRY_PATH} endpoint")
    return value


HttpsUrl = Annotated[str, AfterValidator(_https)]
FoundryEndpoint = Annotated[str, AfterValidator(_foundry_endpoint)]


class OpenAIService(SettingsModel):
    """The OpenAI API. ``organization`` and ``project`` are sent only when set here."""

    name: Literal["openai"] = "openai"
    api_key: SecretRef
    base_url: HttpsUrl = OPENAI_BASE_URL
    organization: str | None = Field(default=None, min_length=1, max_length=256)
    project: str | None = Field(default=None, min_length=1, max_length=256)


class XAIService(SettingsModel):
    """The X.AI API."""

    name: Literal["xai"] = "xai"
    api_key: SecretRef
    base_url: HttpsUrl = XAI_BASE_URL


class FoundryApiKey(SettingsModel):
    mode: Literal["api_key"] = "api_key"
    api_key: SecretRef


class FoundryEntraId(SettingsModel):
    """Microsoft Entra ID through ``azure.identity.aio.DefaultAzureCredential``.

    The credential chain is azure-identity's (environment, workload identity, managed
    identity, developer sign-ins). ``managed_identity_client_id`` picks a user-assigned
    managed identity.
    """

    mode: Literal["entra_id"] = "entra_id"
    scope: HttpsUrl = FOUNDRY_SCOPE
    managed_identity_client_id: str | None = Field(default=None, min_length=1, max_length=128)


class FoundryService(SettingsModel):
    """Microsoft Foundry's OpenAI v1 endpoint, with an API key or Entra ID."""

    name: Literal["foundry"] = "foundry"
    endpoint: FoundryEndpoint
    auth: FoundryApiKey | FoundryEntraId = Field(discriminator="mode")
    catalogue_provider: Literal["openai", "xai"] = "openai"


Service = Annotated[OpenAIService | XAIService | FoundryService, Field(discriminator="name")]


class OpenAICompatSettings(ProviderSettings):
    """Settings for one configured OpenAI-compatible provider."""

    kind: Literal["openai_compat"] = "openai_compat"
    service: Service
    #: The SDK's own retries for connection errors, 408, 409, 429 and 5xx.
    max_retries: int = Field(default=2, ge=0, le=10)

    @property
    def catalogue_provider(self) -> str:
        if isinstance(self.service, FoundryService):
            return self.service.catalogue_provider
        return self.service.name
