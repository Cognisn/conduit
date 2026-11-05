"""The Ollama provider: /api/chat streamed over an httpx ``AsyncClient``.

The client is made by ``OllamaFactory.create_client``; this module only uses it, and
imports httpx (for its exception classes) only when a call fails, so it loads without
the ollama extra. ``request.timeout_s`` applies to connecting and to the wait for each
next line of the stream, so a stalled model fails with ``TIMEOUT`` rather than hanging.
If the consumer stops early, the response is closed.

Every error becomes ``ProviderError(failure)`` raised ``from None``, so Ollama's text
reaches no log, exception or event.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from typing import Any

from conduit.core.catalogue import Catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.ollama.mapping import (
    CHAT_PATH,
    StreamMapper,
    build_chat_body,
    failure_for_status,
)
from conduit.providers.ollama.settings import OllamaSettings


def _classify(error: Exception) -> ProviderFailure | None:
    """The failure for an httpx error, or ``None`` for anything else."""
    httpx = importlib.import_module("httpx")
    if isinstance(error, httpx.TimeoutException):
        return ProviderFailure.TIMEOUT
    if isinstance(error, httpx.HTTPError | httpx.StreamError):
        return ProviderFailure.UNAVAILABLE
    return None


class OllamaProvider(BaseProvider):
    """One configured Ollama server, around an httpx ``AsyncClient``."""

    kind = "ollama"

    def __init__(
        self, name: str, settings: OllamaSettings, client: Any, *, catalogue: Catalogue
    ) -> None:
        self.name = name
        self.settings = settings
        self.client = client
        self._catalogue = catalogue

    def num_ctx(self, model: str) -> int | None:
        """The context length to ask for: the setting, else the catalogue's window."""
        if self.settings.num_ctx is not None:
            return self.settings.num_ctx
        entry = self._catalogue.get(self.settings.catalogue_provider, model)
        return entry.context_window if entry is not None else None

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        body = build_chat_body(request, num_ctx=self.num_ctx(request.model))
        mapper = StreamMapper()
        try:
            async with self.client.stream(
                "POST", CHAT_PATH, json=body, timeout=request.timeout_s
            ) as response:
                if response.status_code != 200:
                    await response.aread()  # drain; the body is never read further
                    raise ProviderError(failure_for_status(response.status_code))
                async for line in response.aiter_lines():
                    for event in mapper.feed(line):
                        yield event
        except ProviderError:
            raise
        except Exception as error:
            failure = _classify(error)
            if failure is None:
                raise
            raise ProviderError(failure) from None
        for event in mapper.finish():
            yield event

    async def aclose(self) -> None:
        await self.client.aclose()

    def __repr__(self) -> str:
        return f"OllamaProvider(name={self.name!r})"
