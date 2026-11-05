"""The Bedrock provider: Converse and ConverseStream, run off the event loop (C3).

boto3 is synchronous, so each call runs on its own daemon thread, never on the loop or
in its default executor (Q0020). A Converse call's thread hands its answer back to the
loop; if ``request.timeout_s`` passes first, the call fails with ``TIMEOUT`` at once and
the thread's late answer is dropped, so a timed-out call holds no executor thread (the
thread itself lives until botocore's ``read_timeout_s``).

A ConverseStream call's thread reads the event stream and hands each raw event to the
loop through a bounded queue (``STREAM_BUFFER`` events), waiting for space, so it never
reads far ahead of the consumer; the loop maps each event and yields conduit events, so
a slow or stalled stream never blocks the loop. ``request.timeout_s`` bounds the wait for
each next event. If the consumer stops early and closes the stream, the thread is told
to stop and the underlying stream is closed. If the consumer stops iterating without
closing it, the thread waits for space at most ``request.timeout_s`` for any one event,
then gives up and closes the stream; a consumer that comes back reads what was already
queued and then fails with ``TIMEOUT``.

Every error becomes ``ProviderError(failure)``, raised after its handler so that it has
no cause and no context: provider text reaches no log, exception or event.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
import time
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
#: Raw stream events the worker may read ahead of the consumer.
STREAM_BUFFER = 64


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
        loop = asyncio.get_running_loop()
        answered: asyncio.Future[tuple[Any, Exception | None]] = loop.create_future()

        def settle(outcome: tuple[Any, Exception | None]) -> None:
            if not answered.done():  # a late answer, after the timeout, is dropped
                answered.set_result(outcome)

        def call() -> None:
            outcome: tuple[Any, Exception | None]
            try:
                outcome = (self._client.converse(**params), None)
            except Exception as error:
                outcome = (None, error)
            with contextlib.suppress(RuntimeError):  # the loop has closed: nobody waits
                loop.call_soon_threadsafe(settle, outcome)

        threading.Thread(target=call, name="conduit-bedrock-converse", daemon=True).start()
        timed_out = False
        try:
            response, error = await asyncio.wait_for(answered, timeout)
        except TimeoutError:
            timed_out = True
        # Raised outside any handler, so the SDK's error is neither cause nor context.
        if timed_out:
            raise ProviderError(ProviderFailure.TIMEOUT)
        if error is not None:
            raise ProviderError(classify(error))
        result: Mapping[str, Any] = response
        return result

    async def _converse_stream(
        self, params: dict[str, Any], timeout: float
    ) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=STREAM_BUFFER)
        abandoned = threading.Event()
        opened: list[Any] = []

        def hand(kind: str, item: Any) -> None:
            # Wait for space in the bounded queue, giving up if the consumer goes, or
            # stops taking events for longer than the request's timeout.
            if abandoned.is_set():
                return
            try:
                put = asyncio.run_coroutine_threadsafe(queue.put((kind, item)), loop)
            except RuntimeError:  # the loop has closed
                abandoned.set()
                return
            deadline = time.monotonic() + timeout
            while True:
                try:
                    put.result(timeout=0.1)
                    return
                except concurrent.futures.TimeoutError:
                    stalled = time.monotonic() >= deadline
                    if abandoned.is_set() or loop.is_closed() or stalled:
                        put.cancel()
                        abandoned.set()  # the reader stops and closes the stream
                        return
                except concurrent.futures.CancelledError:
                    return

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
                waited = True
                try:
                    kind, item = await asyncio.wait_for(queue.get(), timeout)
                except TimeoutError:
                    waited = False
                if not waited:
                    raise ProviderError(ProviderFailure.TIMEOUT)
                if kind == _ERROR:
                    raise ProviderError(classify(item))
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
