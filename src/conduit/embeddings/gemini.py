"""Gemini API embeddings: ``models.embed_content`` through google-genai (the gemini extra).

``GeminiEmbeddingFactory`` is the Gemini chat factory with ``create_embedder`` added:
the same ``GeminiSettings`` (an API key as a secret reference, an explicit base URL)
and the same ``google.genai.Client`` from ``GeminiFactory.create_client``, which pins
the key, the backend and the endpoint against the SDK's environment variables.

A batch is sent as one call per ``MAX_TEXTS`` texts, each bounded by
``request.timeout_s``. ``gemini-embedding-001`` returns 3,072 dimensions by default and
takes ``output_dimensionality`` from 128 to 3,072 (Google recommends 768, 1,536 or
3,072), with an input limit of 2,048 tokens (aistudio.google.com/docs/models/
gemini-embedding-001). Google notes that vectors shorter than 3,072 from
``gemini-embedding-001`` are not normalised; conduit leaves normalising to the product.

The Gemini API reports no token count for embeddings. Usage is the sum of each
embedding's ``statistics.token_count`` when every embedding carries one, and ``None``
otherwise (the event then records no usage).

Every error becomes ``ProviderError(failure)`` (classified as the chat adapter does),
raised after its handler, so it has no cause and no context.
"""

from __future__ import annotations

import asyncio
from typing import Any

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.provider import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_usage,
)
from conduit.core.settings import ProviderSettings
from conduit.embeddings._common import batches, read_count, read_vector
from conduit.providers.gemini import GeminiFactory, GeminiSettings
from conduit.providers.gemini.errors import classify

#: Texts per call. Not stated in the embeddings guide; 100 is the batch limit the
#: Gemini API's batchEmbedContents has been documented with (confirmed by the live suite).
MAX_TEXTS = 100
#: The shortest ``output_dimensionality`` the model card allows.
MIN_DIMENSIONS = 128


def embed_parameters(model: str, texts: tuple[str, ...], dimensions: int | None) -> dict[str, Any]:
    """The ``embed_content`` arguments for one batch."""
    parameters: dict[str, Any] = {"model": model, "contents": list(texts)}
    if dimensions is not None:
        parameters["config"] = {"output_dimensionality": dimensions}
    return parameters


def read_embed_response(
    response: Any, count: int
) -> tuple[list[tuple[float, ...]], int | None] | None:
    """The vectors and, when every embedding reports one, the token count; or ``None``."""
    embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, list) or len(embeddings) != count:
        return None
    vectors: list[tuple[float, ...]] = []
    counts: list[int | None] = []
    for embedding in embeddings:
        vector = read_vector(getattr(embedding, "values", None))
        if vector is None:
            return None
        vectors.append(vector)
        statistics = getattr(embedding, "statistics", None)
        raw = getattr(statistics, "token_count", None) if statistics is not None else None
        tokens = (
            None
            if raw is None
            else read_count(int(raw) if isinstance(raw, float) and raw.is_integer() else raw)
        )
        if raw is not None and tokens is None:
            return None
        counts.append(tokens)
    if any(c is None for c in counts):
        return vectors, None
    return vectors, sum(c for c in counts if c is not None)


class GeminiEmbedder(BaseEmbeddingProvider):
    """One configured Gemini API provider's embeddings."""

    kind = "gemini"

    def __init__(
        self,
        name: str,
        settings: GeminiSettings,
        client: Any,
        *,
        catalogue: Catalogue,
        max_texts: int = MAX_TEXTS,
    ) -> None:
        self.name = name
        self.settings = settings
        self.client = client
        self._catalogue = catalogue
        self._max_texts = max_texts

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return dimensions >= MIN_DIMENSIONS

    async def _call(self, parameters: dict[str, Any], timeout: float) -> Any:
        failure: ProviderFailure | None = None
        try:
            return await asyncio.wait_for(
                self.client.aio.models.embed_content(**parameters), timeout
            )
        except TimeoutError:
            failure = ProviderFailure.TIMEOUT
        except Exception as error:
            failure = classify(error)
        # Raised after the handler: the SDK's error is neither cause nor context.
        raise ProviderError(failure)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        vectors: list[tuple[float, ...]] = []
        tokens: int | None = 0
        for batch in batches(request.texts, self._max_texts):
            parameters = embed_parameters(request.model, batch, request.dimensions)
            response = await self._call(parameters, request.timeout_s)
            read = read_embed_response(response, len(batch))
            if read is None:
                raise ProviderError(ProviderFailure.PROTOCOL)
            vectors.extend(read[0])
            tokens = None if tokens is None or read[1] is None else tokens + read[1]
        usage = embedding_usage(tokens) if tokens is not None else None
        return EmbeddingResponse(model=request.model, vectors=tuple(vectors), usage=usage)

    async def aclose(self) -> None:
        await self.client.aio.aclose()

    def __repr__(self) -> str:
        return f"GeminiEmbedder(name={self.name!r})"


class GeminiEmbeddingFactory(GeminiFactory):
    """The Gemini chat factory, with embeddings: same settings, client and API key."""

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        if not isinstance(settings, GeminiSettings):
            raise ConfigurationError("the Gemini factory needs GeminiSettings")
        return GeminiEmbedder(
            name,
            settings,
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
