"""Settings for the Anthropic adapter: the direct API, or Claude in Microsoft Foundry."""

from __future__ import annotations

from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from conduit.core.secrets import SecretRef
from conduit.core.settings import ProviderSettings

#: The direct API's endpoint. Always passed explicitly, so ``ANTHROPIC_BASE_URL`` in the
#: environment never redirects a conduit client.
DEFAULT_BASE_URL = "https://api.anthropic.com"

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class AnthropicSettings(ProviderSettings):
    """One configured Anthropic provider.

    ``endpoint: direct`` calls Anthropic's API (``base_url`` optional, for a gateway in
    front of it). ``endpoint: foundry`` calls Claude in Microsoft Foundry through the
    resource's Anthropic endpoint, given as ``base_url`` (for example
    ``https://<resource>.services.ai.azure.com/anthropic/``). Either way the key is a
    secret reference, resolved by the product's secret resolver when the provider is
    built. The SDK never falls back to keys or endpoints from the environment.
    """

    kind: Literal["anthropic"] = "anthropic"
    endpoint: Literal["direct", "foundry"] = "direct"
    api_key: SecretRef
    base_url: str | None = Field(default=None, max_length=2048)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parts = urlsplit(value)
            host = parts.hostname
        except ValueError:
            raise ValueError("base_url must be an https URL") from None
        if parts.username is not None or parts.password is not None:
            raise ValueError("base_url must not carry credentials")
        if not host:
            raise ValueError("base_url must be an https URL")
        if parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK):
            return value
        raise ValueError("base_url must be an https URL (plain http only to loopback)")

    @model_validator(mode="after")
    def _foundry_needs_its_endpoint(self) -> Self:
        if self.endpoint == "foundry" and self.base_url is None:
            raise ValueError("endpoint foundry needs base_url, the resource's Anthropic endpoint")
        return self
