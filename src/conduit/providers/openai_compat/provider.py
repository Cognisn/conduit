"""The OpenAI-compatible provider: streaming Chat Completions on an async client."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.openai_compat.errors import classify
from conduit.providers.openai_compat.mapping import StreamMapper, build_request


def _mapping(chunk: Any) -> Mapping[str, Any]:
    if isinstance(chunk, Mapping):
        return chunk
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    raise ProviderError(ProviderFailure.PROTOCOL)


class OpenAICompatProvider(BaseProvider):
    """Streams Chat Completions through ``client.chat.completions.create``.

    ``client`` is the SDK client (or a test's stand-in) with an optional async
    ``close``. Structured output is sent only for models the catalogue marks
    ``structured_output``, looked up under ``catalogue_provider``.
    """

    kind = "openai_compat"

    def __init__(
        self, client: Any, *, catalogue_provider: str, catalogue: Catalogue | None = None
    ) -> None:
        self._client = client
        self._catalogue_provider = catalogue_provider
        self._catalogue = catalogue if catalogue is not None else default_catalogue()

    def _structured_output(self, model: str) -> bool:
        entry = self._catalogue.get(self._catalogue_provider, model)
        return entry is not None and entry.capabilities.structured_output

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        body = build_request(request, structured_output=self._structured_output(request.model))
        try:
            chunks = await self._client.chat.completions.create(**body, timeout=request.timeout_s)
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(classify(error)) from None
        mapper = StreamMapper()
        try:
            iterator = aiter(chunks)
            while True:
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                except ProviderError:
                    raise
                except Exception as error:
                    raise ProviderError(classify(error)) from None
                for event in mapper.feed(_mapping(chunk)):
                    yield event
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                await close()
        for event in mapper.finish():
            yield event

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            await close()
