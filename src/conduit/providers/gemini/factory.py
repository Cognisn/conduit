"""The Gemini provider factory: the only place google-genai is imported.

``create_client`` imports the SDK and builds a ``google.genai.Client`` for the Gemini
API with the resolved API key. Three choices the SDK would otherwise take from the
environment are pinned:

- the key: passed explicitly, so ``GOOGLE_API_KEY`` and ``GEMINI_API_KEY`` are not read;
- the backend: ``vertexai=False``, so ``GOOGLE_GENAI_USE_VERTEXAI`` and
  ``GOOGLE_GENAI_USE_ENTERPRISE`` are not read;
- the endpoint: ``base_url`` passed explicitly, so ``GOOGLE_GEMINI_BASE_URL`` cannot
  redirect prompts.

The rest of the SDK's transport still follows the environment, as any HTTPS client on
the host would: proxy variables and ``.netrc`` (its HTTP clients use ``trust_env``),
``SSL_CERT_FILE`` and ``SSL_CERT_DIR`` for certificate authorities, and
``GOOGLE_GENAI_HTTP_CLIENT`` for the choice of HTTP library. The SDK's retries are
bounded by ``max_attempts``.

``async_client_args`` are passed to the SDK's async HTTP client (a test gives a mock
transport). ``signature_memory`` bounds the thought signatures each provider keeps.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.errors import ConfigurationError
from conduit.core.provider import Provider
from conduit.core.settings import ProviderSettings
from conduit.providers.gemini.provider import DEFAULT_SIGNATURE_MEMORY, GeminiProvider
from conduit.providers.gemini.settings import GeminiSettings


def _gemini(settings: ProviderSettings) -> GeminiSettings:
    if not isinstance(settings, GeminiSettings):
        raise ConfigurationError("the Gemini factory needs GeminiSettings")
    return settings


class GeminiFactory:
    """Builds Gemini clients and providers for the provider registry."""

    kind = "gemini"
    settings_model = GeminiSettings

    def __init__(
        self,
        *,
        async_client_args: Mapping[str, Any] | None = None,
        signature_memory: int = DEFAULT_SIGNATURE_MEMORY,
    ) -> None:
        self._async_client_args = dict(async_client_args or {})
        self._signature_memory = signature_memory

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """A ``google.genai.Client`` for ``settings``. Imports the SDK here only."""
        gemini = _gemini(settings)
        resolved = secrets.get("api_key")
        if resolved is None:
            raise ConfigurationError("secret api_key was not resolved")
        api_key = resolved.get_secret_value()
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
        options: dict[str, Any] = {
            "base_url": gemini.base_url,
            "retry_options": types.HttpRetryOptions(attempts=gemini.max_attempts),
        }
        if self._async_client_args:
            options["async_client_args"] = dict(self._async_client_args)
        return genai.Client(
            vertexai=False,
            api_key=api_key,
            http_options=types.HttpOptions(**options),
        )

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        return GeminiProvider(
            name, _gemini(settings), client, signature_memory=self._signature_memory
        )
