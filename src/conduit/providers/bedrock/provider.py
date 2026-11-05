"""The Bedrock provider: Converse and ConverseStream, run off the event loop (C3).

boto3 is synchronous, so each call runs in a worker thread. A ConverseStream call gets
its own daemon thread, which reads the event stream and hands each raw event to the
loop with ``call_soon_threadsafe``; the loop maps it and yields conduit events, so a
slow or stalled stream never blocks the loop. ``request.timeout_s`` bounds the wait for
each next event (and, for Converse, the whole call). If the consumer stops early, the
thread is told to stop and the underlying stream is closed.

Every error becomes ``ProviderError(failure)`` raised ``from None``, so provider text
reaches no log, exception or event.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Mapping
from typing import Any

from conduit.core.catalogue import Catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.bedrock.errors import classify
from conduit.providers.bedrock.mapping import (
    StreamMapper,
    build_converse_request,
    events_from_converse_response,
)
from conduit.providers.bedrock.settings import BedrockSettings

_EVENT, _END, _ERROR = "event", "end", "error"


def _close(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if close is not None:
        # Closing a finished or broken stream may fail; there is nothing more to do.
        with contextlib.suppress(Exception):
            close()


class BedrockProvider(BaseProvider):
    """One configured Bedrock provider around a ``bedrock-runtime`` client."""

    kind = "bedrock"

    def __init__(
        self, name: str, settings: BedrockSettings, client: Any, *, catalogue: Catalogue
    ) -> None:
        self.name = name
        self.settings = settings
        self._client = client
        self._catalogue = catalogue

    def prompt_caching(self, model: str) -> bool:
        """Whether the catalogue says ``model`` supports prompt caching."""
        entry = self._catalogue.get(self.settings.catalogue_provider, model)
        return entry is not None and entry.capabilities.prompt_caching

    def converse_parameters(self, request: ModelRequest) -> dict[str, Any]:
        return build_converse_request(request, prompt_caching=self.prompt_caching(request.model))

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        params = self.converse_parameters(request)
        if self.settings.use_converse_stream:
            async for event in self._converse_stream(params, request.timeout_s):
                yield event
            return
        for event in events_from_converse_response(await self._converse(params, request.timeout_s)):
            yield event

    async def _converse(self, params: dict[str, Any], timeout: float) -> Mapping[str, Any]:
        try:
            response: Mapping[str, Any] = await asyncio.wait_for(
                asyncio.to_thread(lambda: self._client.converse(**params)), timeout
            )
        except Exception as error:
            raise ProviderError(classify(error)) from None
        return response

    async def _converse_stream(
        self, params: dict[str, Any], timeout: float
    ) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        abandoned = threading.Event()
        opened: list[Any] = []

        def hand(kind: str, item: Any) -> None:
            if abandoned.is_set():
                return
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, item))
            except RuntimeError:  # the loop has closed
                abandoned.set()

        def read() -> None:
            try:
                stream = self._client.converse_stream(**params)["stream"]
                opened.append(stream)
                for raw in stream:
                    if abandoned.is_set():
                        break
                    hand(_EVENT, raw)
                hand(_END, None)
            except Exception as error:
                hand(_ERROR, error)
            finally:
                if opened:
                    _close(opened[0])

        threading.Thread(target=read, name="conduit-bedrock-stream", daemon=True).start()
        mapper = StreamMapper()
        try:
            while True:
                try:
                    kind, item = await asyncio.wait_for(queue.get(), timeout)
                except TimeoutError:
                    raise ProviderError(ProviderFailure.TIMEOUT) from None
                if kind == _ERROR:
                    raise ProviderError(classify(item)) from None
                if kind == _END:
                    for event in mapper.finish():
                        yield event
                    return
                for event in mapper.feed(item):
                    yield event
        finally:
            abandoned.set()
            if opened:
                _close(opened[0])

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    def __repr__(self) -> str:
        return f"BedrockProvider(name={self.name!r}, region={self.settings.region!r})"
