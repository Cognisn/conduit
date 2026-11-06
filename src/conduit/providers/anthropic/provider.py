"""The Anthropic provider and its factory for the provider registry.

The factory builds the SDK's async client: ``AsyncAnthropic`` for the direct API or
``AsyncAnthropicFoundry`` for Claude in Microsoft Foundry. The client takes its key,
endpoint and headers only from conduit's settings:

- the key and endpoint are always passed explicitly, so the SDK never reads a key,
  token, profile or endpoint from the environment;
- the SDK merges ``ANTHROPIC_CUSTOM_HEADERS`` over the headers it is given, including
  the key and API version, so a client is refused while that variable is set;
- for Foundry, the SDK refuses an endpoint alongside ``ANTHROPIC_FOUNDRY_RESOURCE``, so
  that variable is refused with a clear configuration error.

Content capture (C19): the SDK's own OpenTelemetry spans are switched off, since they
can carry request content outside conduit's capture policy. ``ANTHROPIC_LOG=debug``
makes the SDK log whole request bodies on the ``anthropic`` logger, so a client is
refused while it is set (``info`` logs only retries and is allowed). A product that
sets the ``anthropic`` logger to DEBUG itself takes on the same exposure.

The SDK is imported only inside ``create_client``. Errors name the variable, never its
value. The key is resolved and the client built in one guarded step (see
``conduit.providers``): a constructor that raises becomes a fixed
``ConfigurationError`` naming the adapter and the exception's class.

Foundry takes an API key or Microsoft Entra ID (Q0011). Entra ID shares the
OpenAI-compatible adapter's approach (``conduit.providers.openai_compat.entra``):
azure-identity's ``DefaultAzureCredential`` and a bearer token provider, which the SDK
calls for every request, sending ``Authorization: Bearer`` and no key. If the SDK
client cannot be built, the credential is closed before the error goes on; closing the
provider closes the credential after the SDK client. With Entra ID the SDK would read
``ANTHROPIC_FOUNDRY_API_KEY`` from the environment, though never send it while it has
a token, so the key is cleared from the client.

Capability checks (structured output) read the catalogue the registry passes to
``create_provider``: the one it applied the product's overlays to. A factory called
without one, outside a registry, uses conduit's default catalogue.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider, Provider
from conduit.core.registry import guarded_client
from conduit.core.settings import ProviderSettings
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.anthropic.errors import failure_for
from conduit.providers.anthropic.request import build_params
from conduit.providers.anthropic.settings import DEFAULT_BASE_URL, AnthropicSettings
from conduit.providers.anthropic.stream import translate
from conduit.providers.openai_compat.entra import (
    close_quietly,
    close_soon,
    entra_id_token_provider,
)

logger = logging.getLogger("conduit.providers.anthropic")


async def _close(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _close_quietly(resource: Any, what: str) -> None:
    """Close an SDK stream or client; a failing close is logged by class and never raised.

    Only the exception's class name is logged: its message can carry provider text.
    """
    failed: str | None = None
    try:
        await _close(resource)
    except Exception as error:
        failed = type(error).__name__
    if failed is not None:
        logger.warning("closing an anthropic %s failed: %s", what, failed)


class AnthropicProvider(BaseProvider):
    """Claude through the Messages API, streaming.

    ``catalogue`` answers capability checks, looked up under ``catalogue_provider``
    (``anthropic`` unless the settings name another).
    """

    kind = "anthropic"

    def __init__(
        self,
        client: Any,
        *,
        name: str,
        catalogue: Catalogue,
        catalogue_provider: str = "anthropic",
    ) -> None:
        self.name = name
        self._client = client
        self._catalogue = catalogue
        self._catalogue_provider = catalogue_provider

    def structured_output(self, model: str) -> bool:
        """Whether the catalogue says ``model`` supports structured output."""
        entry = self._catalogue.get(self._catalogue_provider, model)
        return entry is not None and entry.capabilities.structured_output

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        supported = self.structured_output(request.model)
        if request.output_schema is not None and not supported:
            raise ProviderError(ProviderFailure.REJECTED)
        params = build_params(request, structured_output=supported)
        sdk_stream: Any = None
        failure: ProviderFailure | None = None
        try:
            sdk_stream = await self._client.messages.create(**params)
            async for event in translate(aiter(sdk_stream)):
                yield event
        except ProviderError as error:
            failure = error.failure
        except Exception as error:
            # The SDK's message can echo the request: only the failure class goes on.
            failure = failure_for(error)
        finally:
            if sdk_stream is not None:
                await _close_quietly(sdk_stream, "stream")
        # Raised here, after the handler, so the failing exception is neither the
        # cause nor the context of the ProviderError.
        if failure is not None:
            raise ProviderError(failure)

    async def aclose(self) -> None:
        await _close_quietly(self._client, "client")

    def __repr__(self) -> str:
        return f"AnthropicProvider(name={self.name!r})"


def _check_environment(endpoint: str) -> None:
    """Refuse variables through which the SDK would override conduit's settings."""
    if "ANTHROPIC_CUSTOM_HEADERS" in os.environ:
        raise ConfigurationError(
            "ANTHROPIC_CUSTOM_HEADERS is set; the anthropic SDK would merge it over the "
            "key and headers conduit configures. Unset it to build an anthropic provider"
        )
    if endpoint == "foundry" and "ANTHROPIC_FOUNDRY_RESOURCE" in os.environ:
        raise ConfigurationError(
            "ANTHROPIC_FOUNDRY_RESOURCE is set; a foundry provider takes its endpoint only "
            "from base_url in its settings. Unset the variable"
        )
    if os.environ.get("ANTHROPIC_LOG") == "debug":
        raise ConfigurationError(
            "ANTHROPIC_LOG=debug makes the anthropic SDK log request content outside "
            "conduit's content-capture policy. Unset it or use info"
        )


@dataclass(eq=False)
class FoundryEntraClient:
    """The Foundry SDK client with the Entra ID credential that signs its requests.

    ``close`` closes the SDK client and then, whatever happened, the credential; a
    failing credential close is logged by class name and never replaces the SDK's.
    """

    sdk: Any
    credential: Any

    @property
    def messages(self) -> Any:
        return self.sdk.messages

    async def close(self) -> None:
        try:
            await self.sdk.close()
        finally:
            await close_quietly(self.credential)  # never replaces the SDK's failure


def _settings(settings: ProviderSettings) -> AnthropicSettings:
    if not isinstance(settings, AnthropicSettings):
        raise ConfigurationError("anthropic settings expected")
    return settings


class AnthropicFactory:
    """The Anthropic adapter, for the provider registry."""

    kind = "anthropic"
    settings_model = AnthropicSettings

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        validated = _settings(settings)
        _check_environment(validated.endpoint)
        sdk: Any = None
        with contextlib.suppress(ImportError):
            sdk = importlib.import_module("anthropic")
        if sdk is None:  # raised after the handler, with nothing chained
            raise ConfigurationError(
                "the anthropic provider needs the anthropic extra: "
                "pip install 'cognisn-conduit[anthropic]'"
            )
        entra_id = validated.entra_id

        def build() -> Any:
            # The key is resolved here, in the guarded step's own frame.
            if validated.endpoint == "direct":
                return sdk.AsyncAnthropic(
                    api_key=secrets["api_key"].get_secret_value(),
                    base_url=validated.base_url or DEFAULT_BASE_URL,
                    max_retries=validated.max_retries,
                    open_telemetry=False,
                )
            if entra_id is None:
                return sdk.AsyncAnthropicFoundry(
                    api_key=secrets["api_key"].get_secret_value(),
                    base_url=validated.base_url,
                    max_retries=validated.max_retries,
                    open_telemetry=False,
                )
            credential, token_provider = entra_id_token_provider(entra_id)
            try:
                client = sdk.AsyncAnthropicFoundry(
                    azure_ad_token_provider=token_provider,
                    base_url=validated.base_url,
                    max_retries=validated.max_retries,
                    open_telemetry=False,
                )
            except Exception:
                close_soon(credential)
                raise
            # The SDK reads a key from the environment when none is given; with Entra ID
            # the token always wins, and the client holds no key at all.
            client.api_key = None
            return FoundryEntraClient(sdk=client, credential=credential)

        return guarded_client(self.kind, build)

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        validated = _settings(settings)
        return AnthropicProvider(
            client,
            name=name,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
            catalogue_provider=validated.catalogue_provider,
        )
