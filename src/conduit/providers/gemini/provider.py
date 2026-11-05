"""The Gemini provider: ``client.aio.models.generate_content_stream`` (C3).

The client is a ``google.genai.Client`` made by ``GeminiFactory.create_client``; this
module never imports the SDK. ``request.timeout_s`` bounds the wait for the stream to
open and for each next chunk, so a stalled call fails with ``TIMEOUT``. If the
consumer stops early, the SDK stream is closed.

The catalogue (the registry's, passed through the factory) decides structured output:
a schema for a model it does not mark ``structured_output`` is refused as
``REJECTED`` before any call.

Thought signatures (see ``conduit.providers.gemini.mapping``) are remembered by call
id, the most recent ``signature_memory`` of them, and sent back with their calls.

Every error becomes ``ProviderError(failure)``, raised after the handler that caught
the SDK's error, so that error is neither its cause nor its context: provider text
reaches no log, exception or event. Closing the SDK stream never replaces or chains
a failure: an error from the close is suppressed.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Any

from conduit.core.catalogue import Catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.gemini.errors import classify
from conduit.providers.gemini.mapping import StreamMapper, build_request
from conduit.providers.gemini.settings import GeminiSettings

#: How many thought signatures a provider remembers, most recent first kept.
DEFAULT_SIGNATURE_MEMORY = 4096


def _as_json(chunk: Any) -> Mapping[str, Any] | None:
    """A chunk in JSON form (SDK objects are dumped, mappings pass through), or
    ``None`` for a chunk of any other type, which the caller treats as PROTOCOL."""
    if isinstance(chunk, Mapping):
        return chunk
    dump = getattr(chunk, "model_dump", None)
    if not callable(dump):
        return None
    data: Any = None
    with contextlib.suppress(Exception):
        data = dump(mode="json", exclude_none=True)
    return data if isinstance(data, Mapping) else None


async def _next_chunk(
    chunks: Any, timeout: float
) -> tuple[Mapping[str, Any] | None, ProviderFailure | None]:
    """The next chunk in JSON form, ``(None, None)`` at the end, or the failure.

    The failure is returned, not raised: the caller raises its ``ProviderError`` after
    this handler has finished, so the SDK's error is neither its cause nor its context.
    """
    try:
        raw = await asyncio.wait_for(anext(chunks), timeout)
    except StopAsyncIteration:
        return None, None
    except Exception as error:
        return None, classify(error)
    chunk = _as_json(raw)
    # A chunk the SDK should never yield breaks the stream's contract: PROTOCOL.
    return (chunk, None) if chunk is not None else (None, ProviderFailure.PROTOCOL)


def _feed(
    mapper: StreamMapper, chunk: Mapping[str, Any]
) -> tuple[list[StreamEvent], ProviderFailure | None]:
    """The chunk's events, or the failure: a chunk the mapper cannot read is PROTOCOL."""
    try:
        return mapper.feed(chunk), None
    except ProviderError as error:
        return [], error.failure
    except Exception:
        return [], ProviderFailure.PROTOCOL


async def _close(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if callable(close):
        # Closing a finished or broken stream may fail; there is nothing more to do.
        with contextlib.suppress(Exception):
            await close()


class GeminiProvider(BaseProvider):
    """One configured Gemini API provider, around a ``google.genai.Client``."""

    kind = "gemini"

    def __init__(
        self,
        name: str,
        settings: GeminiSettings,
        client: Any,
        *,
        catalogue: Catalogue,
        signature_memory: int = DEFAULT_SIGNATURE_MEMORY,
    ) -> None:
        self.name = name
        self.settings = settings
        self.client = client
        self._catalogue = catalogue
        self._memory = max(1, signature_memory)
        self._signatures: OrderedDict[str, bytes] = OrderedDict()

    def structured_output(self, model: str) -> bool:
        """Whether the catalogue says ``model`` supports structured output."""
        entry = self._catalogue.get(self.settings.catalogue_provider, model)
        return entry is not None and entry.capabilities.structured_output

    @property
    def signatures(self) -> Mapping[str, bytes]:
        """The thought signatures remembered, by call id (read-only)."""
        return MappingProxyType(self._signatures)

    def _remember(self, found: Mapping[str, bytes]) -> None:
        for call_id, signature in found.items():
            self._signatures[call_id] = signature
            self._signatures.move_to_end(call_id)
        while len(self._signatures) > self._memory:
            self._signatures.popitem(last=False)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if request.output_schema is not None and not self.structured_output(request.model):
            raise ProviderError(ProviderFailure.REJECTED)  # refused before any call
        params = build_request(request, signatures=self._signatures)
        timeout = request.timeout_s
        chunks, failure = await self._open(params, timeout)
        if failure is not None:
            raise ProviderError(failure)
        mapper = StreamMapper()
        try:
            while True:
                chunk, failure = await _next_chunk(chunks, timeout)
                if failure is not None:
                    raise ProviderError(failure)
                if chunk is None:
                    break
                events, failure = _feed(mapper, chunk)
                if failure is not None:
                    raise ProviderError(failure)
                self._remember(mapper.signatures)
                for event in events:
                    yield event
        finally:
            await _close(chunks)
        for event in mapper.finish():
            yield event

    async def _open(
        self, params: dict[str, Any], timeout: float
    ) -> tuple[Any, ProviderFailure | None]:
        """The SDK stream, or the failure opening it (never the error itself)."""
        try:
            stream = await asyncio.wait_for(
                self.client.aio.models.generate_content_stream(**params), timeout
            )
        except Exception as error:
            return None, classify(error)
        return stream, None

    async def aclose(self) -> None:
        await self.client.aio.aclose()

    def __repr__(self) -> str:
        return f"GeminiProvider(name={self.name!r})"
