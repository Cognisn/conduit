"""The OpenAI-compatible provider: streaming Chat Completions on an async client.

Every failure leaves as ``ProviderError`` carrying only its failure class. The SDK
builds chunk objects without validating them, so a chunk can hold any shape: anything
the mapper cannot read is ``PROTOCOL``. Each error is raised after its handler has
finished, so the original exception (which may quote provider text) is neither the
cause nor the context of the error that leaves.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from conduit.core.catalogue import Catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import ModelRequest
from conduit.providers.openai_compat.errors import classify
from conduit.providers.openai_compat.mapping import StreamMapper, build_request

_END = object()


def _mapping(chunk: Any) -> Mapping[str, Any]:
    if isinstance(chunk, Mapping):
        return chunk
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        # warnings=False: pydantic's serialiser warnings quote the offending value.
        dumped = dump(mode="json", warnings=False)
        if isinstance(dumped, Mapping):
            return dumped
    raise ProviderError(ProviderFailure.PROTOCOL)


def _protocol[T](read: Callable[..., T], *args: Any) -> T:
    """``read(*args)``, with any failure other than ``ProviderError`` turned into PROTOCOL."""
    failed = False
    try:
        result = read(*args)
    except ProviderError:
        raise
    except Exception:
        failed = True
    if failed:
        raise ProviderError(ProviderFailure.PROTOCOL)
    return result


def _feed(mapper: StreamMapper, chunk: Any) -> list[StreamEvent]:
    return mapper.feed(_mapping(chunk))


async def _close_stream(chunks: Any, *, quiet: bool) -> None:
    """Close the SDK stream. A failure is dropped when ``quiet`` (something else is
    already leaving), and otherwise leaves as an unchained ``ProviderError``."""
    close = getattr(chunks, "close", None)
    if not callable(close):
        return
    failure: ProviderFailure | None = None
    try:
        await close()
    except Exception as error:
        failure = classify(error)
    if failure is not None and not quiet:
        raise ProviderError(failure)


class OpenAICompatProvider(BaseProvider):
    """Streams Chat Completions through ``client.chat.completions.create``.

    ``client`` is the SDK client (or a test's stand-in) with an optional async
    ``close``. Structured output is sent only for models the catalogue marks
    ``structured_output``, looked up under ``catalogue_provider``.
    """

    kind = "openai_compat"

    def __init__(self, client: Any, *, catalogue_provider: str, catalogue: Catalogue) -> None:
        self._client = client
        self._catalogue_provider = catalogue_provider
        self._catalogue = catalogue

    def _structured_output(self, model: str) -> bool:
        entry = self._catalogue.get(self._catalogue_provider, model)
        return entry is not None and entry.capabilities.structured_output

    async def _create(self, body: dict[str, Any], timeout: float) -> Any:
        failure: ProviderFailure | None = None
        try:
            return await self._client.chat.completions.create(**body, timeout=timeout)
        except ProviderError:
            raise
        except Exception as error:
            failure = classify(error)
        raise ProviderError(failure)

    @staticmethod
    async def _next(iterator: AsyncIterator[Any]) -> Any:
        """The next chunk, or ``_END``."""
        failure: ProviderFailure | None = None
        try:
            return await anext(iterator)
        except StopAsyncIteration:
            return _END
        except ProviderError:
            raise
        except Exception as error:
            failure = classify(error)
        raise ProviderError(failure)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        body = build_request(request, structured_output=self._structured_output(request.model))
        chunks = await self._create(body, request.timeout_s)
        mapper = StreamMapper()
        try:
            iterator: AsyncIterator[Any] = _protocol(aiter, chunks)
            while (chunk := await self._next(iterator)) is not _END:
                for event in _protocol(_feed, mapper, chunk):
                    yield event
        except BaseException:
            # Something is already leaving (an error, a cancellation, the consumer
            # closing the stream): close quietly so it is never replaced.
            await _close_stream(chunks, quiet=True)
            raise
        await _close_stream(chunks, quiet=False)
        for event in _protocol(mapper.finish):
            yield event

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            await close()
