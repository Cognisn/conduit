"""A fake provider adapter for the registry and start-up tests."""

from __future__ import annotations

import asyncio
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
PARTIAL = Usage(input_tokens=11, output_tokens=2, cache_read_tokens=5, cache_write_tokens=3)


class FakeSettings(ProviderSettings):
    kind: Literal["fake"] = "fake"
    api_key: SecretRef
    behaviour: Literal["ok", "fail", "no_stop", "slow", "after_stop", "sync_fail"] = "ok"


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
        self.requests: list[ModelRequest] = []
        self.streams_closed = 0
        self.waiting = asyncio.Event()

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self.settings.behaviour == "sync_fail":
            raise RuntimeError("refused before streaming")  # raised when stream() is called
        return self._events(request)

    async def _events(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        behaviour = self.settings.behaviour
        try:
            if behaviour == "fail":
                raise ProviderError(ProviderFailure.UNAVAILABLE)
            yield UsageReported(usage=Usage.zero())
            yield TextDelta(text="echo: ")
            yield UsageReported(usage=PARTIAL)
            if behaviour == "slow":
                self.waiting.set()
                await asyncio.sleep(30)
            yield TextDelta(text=request.messages[-1].content[0].text)  # type: ignore[union-attr]
            if behaviour == "no_stop":
                return
            yield StreamStopped(stop_reason=StopReason.END_TURN, usage=USAGE)
            if behaviour == "after_stop":
                yield TextDelta(text="after the stop")
        finally:
            await asyncio.sleep(0)  # cleanup that awaits, as an SDK's does
            self.streams_closed += 1

    async def aclose(self) -> None:
        self.closed = True


class FakeFactory:
    kind = "fake"
    settings_model = FakeSettings

    def __init__(self) -> None:
        self.clients: list[FakeClient] = []
        self.providers: list[FakeProvider] = []
        self.catalogues: list[Catalogue | None] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        client = FakeClient(secrets["api_key"])
        self.clients.append(client)
        return client

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        assert isinstance(settings, FakeSettings)
        self.catalogues.append(catalogue)
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
