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
value.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import default_catalogue
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider, Provider
from conduit.core.settings import ProviderSettings
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.anthropic.errors import failure_for
from conduit.providers.anthropic.request import build_params
from conduit.providers.anthropic.settings import DEFAULT_BASE_URL, AnthropicSettings
from conduit.providers.anthropic.stream import translate

logger = logging.getLogger("conduit.providers.anthropic")

StructuredOutputCheck = Callable[[str], bool]


def default_structured_output(model: str) -> bool:
    """Whether the default catalogue says ``model`` supports structured output."""
    entry = default_catalogue().get("anthropic", model)
    return entry is not None and entry.capabilities.structured_output


async def _close(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _close_quietly(resource: Any) -> None:
    """Close an SDK stream; a failing close is logged by class and never raised."""
    failed: str | None = None
    try:
        await _close(resource)
    except Exception as error:
        failed = type(error).__name__
    if failed is not None:
        logger.warning("closing an anthropic stream failed: %s", failed)


class AnthropicProvider(BaseProvider):
    """Claude through the Messages API, streaming."""

    kind = "anthropic"

    def __init__(
        self,
        client: Any,
        *,
        name: str,
        structured_output: StructuredOutputCheck = default_structured_output,
    ) -> None:
        self.name = name
        self._client = client
        self._structured_output = structured_output

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        supported = self._structured_output(request.model)
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
                await _close_quietly(sdk_stream)
        # Raised here, after the handler, so the failing exception is neither the
        # cause nor the context of the ProviderError.
        if failure is not None:
            raise ProviderError(failure)

    async def aclose(self) -> None:
        await _close(self._client)

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
        key = secrets["api_key"].get_secret_value()
        if validated.endpoint == "foundry":
            return sdk.AsyncAnthropicFoundry(
                api_key=key,
                base_url=validated.base_url,
                max_retries=validated.max_retries,
                open_telemetry=False,
            )
        return sdk.AsyncAnthropic(
            api_key=key,
            base_url=validated.base_url or DEFAULT_BASE_URL,
            max_retries=validated.max_retries,
            open_telemetry=False,
        )

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        _settings(settings)
        return AnthropicProvider(client, name=name)
