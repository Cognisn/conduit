"""The Ollama provider: /api/chat streamed over an httpx ``AsyncClient``.

The client is made by ``OllamaFactory.create_client``; this module only uses it, and
imports httpx (for its exception classes) only when a call fails, so it loads without
the ollama extra. ``request.timeout_s`` applies to connecting and to the wait for each
next line of the stream, so a stalled model fails with ``TIMEOUT`` rather than hanging.
A line longer than ``MAX_LINE_BYTES`` (or the provider's ``max_line_bytes``), or one
that is not UTF-8, fails with ``PROTOCOL``. A redirect is never followed: any status
but 200 is a failure (3xx and 4xx ``REJECTED``).
If the consumer stops early, the response is closed.

Every error becomes ``ProviderError(failure)``, raised after the handler that caught
httpx's error, so that error is neither its cause nor its context: Ollama's text
reaches no log, exception or event. An error that is not one of httpx's own is
``UNAVAILABLE``. Closing the response never replaces or chains a failure.
"""

from __future__ import annotations

import contextlib
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


#: The longest NDJSON line read. A line carries one piece of the reply (a token, a
#: tool call, the final counts), so this is generous; a longer one is refused rather
#: than buffered without bound.
MAX_LINE_BYTES = 8 << 20


def _decode(line: bytes) -> str:
    text: str | None = None
    with contextlib.suppress(UnicodeDecodeError):
        text = line.rstrip(b"\r").decode("utf-8")
    # Raised here, after the handler, so the decoding error (which quotes the
    # bytes) is neither the cause nor the context.
    if text is None:
        raise ProviderError(ProviderFailure.PROTOCOL)
    return text


async def _lines(chunks: AsyncIterator[bytes], limit: int) -> AsyncIterator[str]:
    """The NDJSON lines of a body, each at most ``limit`` bytes."""
    buffer = bytearray()
    async for chunk in chunks:
        buffer += chunk
        while (end := buffer.find(b"\n")) >= 0:
            if end > limit:
                raise ProviderError(ProviderFailure.PROTOCOL)
            line = bytes(buffer[:end])
            del buffer[: end + 1]
            yield _decode(line)
        if len(buffer) > limit:
            raise ProviderError(ProviderFailure.PROTOCOL)
    if buffer:
        yield _decode(bytes(buffer))


async def _send(
    client: Any, body: dict[str, Any], timeout: float
) -> tuple[Any, ProviderFailure | None]:
    """The streamed response to a chat request, or the failure sending it.

    The failure is returned, not raised: the caller raises its ``ProviderError`` after
    this handler has finished, so httpx's error is neither its cause nor its context.
    """
    try:
        sent = client.build_request("POST", CHAT_PATH, json=body, timeout=timeout)
        return await client.send(sent, stream=True), None
    except Exception as error:
        return None, _classify(error) or ProviderFailure.UNAVAILABLE


async def _next_line(lines: AsyncIterator[str]) -> tuple[str | None, ProviderFailure | None]:
    """The next line, ``(None, None)`` at the end of the body, or the failure."""
    try:
        return await anext(lines), None
    except StopAsyncIteration:
        return None, None
    except ProviderError as error:
        return None, error.failure
    except Exception as error:
        return None, _classify(error) or ProviderFailure.UNAVAILABLE


async def _close(lines: Any, response: Any) -> None:
    """Close the line reader and the response. A failing close never replaces, or
    rides along with, the failure being raised: its error is suppressed."""
    for closable in (lines, response):
        with contextlib.suppress(Exception):
            await closable.aclose()


class OllamaProvider(BaseProvider):
    """One configured Ollama server, around an httpx ``AsyncClient``."""

    kind = "ollama"

    def __init__(
        self,
        name: str,
        settings: OllamaSettings,
        client: Any,
        *,
        catalogue: Catalogue,
        max_line_bytes: int = MAX_LINE_BYTES,
    ) -> None:
        self.name = name
        self.settings = settings
        self.client = client
        self._catalogue = catalogue
        self._max_line_bytes = max_line_bytes

    def num_ctx(self, model: str) -> int | None:
        """The context length to ask for: the setting, else the catalogue's window."""
        if self.settings.num_ctx is not None:
            return self.settings.num_ctx
        entry = self._catalogue.get(self.settings.catalogue_provider, model)
        return entry.context_window if entry is not None else None

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        body = build_chat_body(request, num_ctx=self.num_ctx(request.model))
        mapper = StreamMapper()
        response, failure = await _send(self.client, body, request.timeout_s)
        if failure is not None:
            raise ProviderError(failure)
        lines = _lines(response.aiter_bytes(), self._max_line_bytes)
        try:
            if response.status_code != 200:
                # The body is never read: closing the response discards it.
                raise ProviderError(failure_for_status(response.status_code))
            while True:
                line, failure = await _next_line(lines)
                if failure is not None:
                    raise ProviderError(failure)
                if line is None:
                    break
                for event in mapper.feed(line):
                    yield event
        finally:
            await _close(lines, response)
        for event in mapper.finish():
            yield event

    async def aclose(self) -> None:
        await self.client.aclose()

    def __repr__(self) -> str:
        return f"OllamaProvider(name={self.name!r})"
