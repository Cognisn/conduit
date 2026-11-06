"""The Ollama provider factory: the only place the ollama layer makes an HTTP client.

``create_client`` imports httpx and builds an ``AsyncClient`` for the configured base
URL. The client ignores the environment (``trust_env=False``): a proxy variable or a
``.netrc`` entry must not route prompts to another host or attach credentials, and it
never follows redirects. Each call sets its own timeout from the request. The client is
built in the guarded step every adapter uses (see ``conduit.providers``): a constructor
that raises becomes a fixed ``ConfigurationError`` naming the adapter and the
exception's class.

Providers read the context window and structured-output support from the catalogue
the registry passes to ``create_provider``: the one it applied the product's overlays
to. Called without one, outside a registry, the factory uses conduit's default
catalogue. ``transport`` replaces httpx's network transport, for tests.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import Provider
from conduit.core.registry import guarded_client
from conduit.core.settings import ProviderSettings
from conduit.providers.ollama.provider import OllamaProvider
from conduit.providers.ollama.settings import OllamaSettings

#: The client's timeout when a call gives none; every call made here gives one.
DEFAULT_TIMEOUT_S = 120.0


def _ollama(settings: ProviderSettings) -> OllamaSettings:
    if not isinstance(settings, OllamaSettings):
        raise ConfigurationError("the Ollama factory needs OllamaSettings")
    return settings


class OllamaFactory:
    """Builds Ollama clients and providers for the provider registry."""

    kind = "ollama"
    settings_model = OllamaSettings

    def __init__(self, *, transport: Any = None) -> None:
        self._transport = transport

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """An httpx ``AsyncClient`` for ``settings``. Imports httpx here only."""
        ollama = _ollama(settings)
        httpx = importlib.import_module("httpx")
        transport = self._transport

        def build() -> Any:
            return httpx.AsyncClient(
                base_url=ollama.base_url,
                timeout=DEFAULT_TIMEOUT_S,
                trust_env=False,
                follow_redirects=False,
                transport=transport,
            )

        return guarded_client(self.kind, build)

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        return OllamaProvider(
            name,
            _ollama(settings),
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
