"""The Anthropic provider and its factory for the provider registry.

The factory builds the SDK's async client: ``AsyncAnthropic`` for the direct API or
``AsyncAnthropicFoundry`` for Claude in Microsoft Foundry. The key and endpoint are
always passed explicitly, so the SDK never reads a key, profile or endpoint from the
environment, and the SDK's own OpenTelemetry spans are switched off, since they can
carry request content outside conduit's content-capture policy. The SDK is imported
only inside ``create_client``.
"""

from __future__ import annotations

import importlib
import inspect
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
        try:
            sdk_stream = await self._client.messages.create(**params)
            async for event in translate(aiter(sdk_stream)):
                yield event
        except ProviderError:
            raise
        except Exception as error:
            # The SDK's message can echo the request: only the failure class goes on.
            raise ProviderError(failure_for(error)) from None
        finally:
            if sdk_stream is not None:
                await _close(sdk_stream)

    async def aclose(self) -> None:
        await _close(self._client)

    def __repr__(self) -> str:
        return f"AnthropicProvider(name={self.name!r})"


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
        try:
            sdk = importlib.import_module("anthropic")
        except ImportError:
            raise ConfigurationError(
                "the anthropic provider needs the anthropic extra: "
                "pip install 'cognisn-conduit[anthropic]'"
            ) from None
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
