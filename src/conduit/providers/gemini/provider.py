"""The Gemini provider: ``client.aio.models.generate_content_stream`` (C3).

The client is a ``google.genai.Client`` made by ``GeminiFactory.create_client``; this
module never imports the SDK. ``request.timeout_s`` bounds the wait for the stream to
open and for each next chunk, so a stalled call fails with ``TIMEOUT``. If the
consumer stops early, the SDK stream is closed.

Thought signatures (see ``conduit.providers.gemini.mapping``) are remembered by call
id, the most recent ``signature_memory`` of them, and sent back with their calls.

Every error becomes ``ProviderError(failure)`` raised ``from None``, so provider text
reaches no log, exception or event.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.gemini.errors import classify
from conduit.providers.gemini.mapping import StreamMapper, build_request
from conduit.providers.gemini.settings import GeminiSettings

#: How many thought signatures a provider remembers, most recent first kept.
DEFAULT_SIGNATURE_MEMORY = 4096


def _as_json(chunk: Any) -> Mapping[str, Any]:
    """A chunk in JSON form: SDK objects are dumped, mappings pass through."""
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        data = dump(mode="json", exclude_none=True)
        if isinstance(data, Mapping):
            return data
    if isinstance(chunk, Mapping):
        return chunk
    raise TypeError("unexpected chunk type")


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
        signature_memory: int = DEFAULT_SIGNATURE_MEMORY,
    ) -> None:
        self.name = name
        self.settings = settings
        self.client = client
        self._memory = max(1, signature_memory)
        self._signatures: OrderedDict[str, bytes] = OrderedDict()

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
        params = build_request(request, signatures=self._signatures)
        timeout = request.timeout_s
        try:
            chunks = await asyncio.wait_for(
                self.client.aio.models.generate_content_stream(**params), timeout
            )
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(classify(error)) from None
        mapper = StreamMapper()
        try:
            while True:
                try:
                    chunk = _as_json(await asyncio.wait_for(anext(chunks), timeout))
                except StopAsyncIteration:
                    break
                except Exception as error:
                    raise ProviderError(classify(error)) from None
                try:
                    events = mapper.feed(chunk)
                except ProviderError:
                    raise
                except Exception:
                    # A chunk the mapper cannot read (a malformed shape): never the
                    # raw error, which could carry the chunk's text.
                    raise ProviderError(ProviderFailure.PROTOCOL) from None
                self._remember(mapper.signatures)
                for event in events:
                    yield event
        finally:
            await _close(chunks)
        for event in mapper.finish():
            yield event

    async def aclose(self) -> None:
        await self.client.aio.aclose()

    def __repr__(self) -> str:
        return f"GeminiProvider(name={self.name!r})"
