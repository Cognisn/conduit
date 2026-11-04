"""A fake provider adapter for the registry and start-up tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider, Provider
from conduit.core.secrets import SecretRef
from conduit.core.settings import ProviderSettings
from conduit.core.streaming import StreamEvent, StreamStopped, TextDelta, UsageReported
from conduit.core.types import ModelRequest, StopReason, Usage

USAGE = Usage(input_tokens=11, output_tokens=7, cache_read_tokens=5, cache_write_tokens=3)


class FakeSettings(ProviderSettings):
    kind: Literal["fake"] = "fake"
    api_key: SecretRef
    behaviour: Literal["ok", "fail", "no_stop"] = "ok"


class FakeClient:
    def __init__(self, key: SecretStr) -> None:
        self.key = key
        self.closed = False


class FakeProvider(BaseProvider):
    kind = "fake"

    def __init__(self, name: str, settings: FakeSettings, client: Any) -> None:
        self.name = name
        self.settings = settings
        self.client = client
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if self.settings.behaviour == "fail":
            raise ProviderError(ProviderFailure.UNAVAILABLE)
        yield UsageReported(usage=Usage.zero())
        yield TextDelta(text="echo: ")
        yield TextDelta(text=request.messages[-1].content[0].text)  # type: ignore[union-attr]
        if self.settings.behaviour == "no_stop":
            return
        yield StreamStopped(stop_reason=StopReason.END_TURN, usage=USAGE)

    async def aclose(self) -> None:
        self.closed = True


class FakeFactory:
    kind = "fake"
    settings_model = FakeSettings

    def __init__(self) -> None:
        self.clients: list[FakeClient] = []
        self.providers: list[FakeProvider] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        client = FakeClient(secrets["api_key"])
        self.clients.append(client)
        return client

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        assert isinstance(settings, FakeSettings)
        provider = FakeProvider(name, settings, client)
        self.providers.append(provider)
        return provider


CATALOGUE = Catalogue(
    [
        ModelEntry(
            model_id="fake-large",
            provider="fake",
            residency_scope="local",  # type: ignore[arg-type]
            capabilities={"tools": True},  # type: ignore[arg-type]
            context_window=100_000,
            output_limit=8_000,
        ),
        ModelEntry(
            model_id="fake-small",
            provider="fake",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=4_000,
            output_limit=1_000,
        ),
    ],
    version="fake-1",
)
