"""OpenAI-compatible embeddings: POST /v1/embeddings (the openai extra).

``OpenAICompatEmbeddingFactory`` is the OpenAI-compatible chat factory with
``create_embedder`` added: the same ``OpenAICompatSettings``, services (OpenAI, X.AI,
Microsoft Foundry) and authentication (API keys as secret references, or Entra ID for
Foundry), and the same SDK client from ``OpenAICompatFactory.create_client``, with its
handling of the SDK's environment variables (see that module: ``OPENAI_LOG`` makes the
SDK log request options, including the texts, so production deployments leave it unset).

A batch is sent as one request per ``MAX_INPUTS`` texts (OpenAI allows 2,048 inputs and
300,000 tokens summed across them per request; a request over the token limit is
refused by the API as ``REJECTED``). Usage is the sum of ``usage.prompt_tokens``. The
vectors are read in the order of each item's ``index``.

``dimensions`` (a shorter vector) is offered by ``text-embedding-3`` and later models
only (OpenAI's reference: "Only supported in text-embedding-3 and later models"), so it
is accepted for model ids starting ``text-embedding-3``. A Foundry deployment's model id
is its deployment name, so a shorter length is not offered for it. OpenAI notes that a
vector shortened by hand must be normalised again; one shortened by the API is not
shortened by hand, and conduit leaves normalising to the product.

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
from conduit.providers.openai_compat import OpenAICompatFactory, OpenAICompatSettings
from conduit.providers.openai_compat.errors import classify

#: Inputs per request (OpenAI's limit on the input array).
MAX_INPUTS = 2048
REDUCIBLE_PREFIX = "text-embedding-3"


def embeddings_body(model: str, texts: tuple[str, ...], dimensions: int | None) -> dict[str, Any]:
    """The /v1/embeddings parameters for one batch."""
    body: dict[str, Any] = {"model": model, "input": list(texts), "encoding_format": "float"}
    if dimensions is not None:
        body["dimensions"] = dimensions
    return body


def read_embeddings(response: Any, count: int) -> tuple[list[tuple[float, ...]], int] | None:
    """The vectors (by index) and prompt tokens of a response, or ``None``."""
    data = getattr(response, "data", None)
    usage = getattr(response, "usage", None)
    tokens = read_count(getattr(usage, "prompt_tokens", None))
    if not isinstance(data, list) or len(data) != count or tokens is None:
        return None
    found: dict[int, tuple[float, ...]] = {}
    for item in data:
        index = read_count(getattr(item, "index", None))
        vector = read_vector(getattr(item, "embedding", None))
        if index is None or vector is None or index >= count or index in found:
            return None
        found[index] = vector
    return [found[i] for i in range(count)], tokens


def _sdk(client: Any) -> Any:
    """The SDK client inside ``OpenAICompatClient`` (or the client itself, in tests)."""
    return getattr(client, "sdk", client)


class OpenAICompatEmbedder(BaseEmbeddingProvider):
    """One configured OpenAI-compatible provider's embeddings."""

    kind = "openai_compat"

    def __init__(
        self,
        name: str,
        settings: OpenAICompatSettings,
        client: Any,
        *,
        catalogue: Catalogue,
        max_inputs: int = MAX_INPUTS,
    ) -> None:
        self.name = name
        self.settings = settings
        self._client = client
        self._catalogue = catalogue
        self._max_inputs = max_inputs

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return dimensions >= 1 and model.startswith(REDUCIBLE_PREFIX)

    async def _create(self, body: dict[str, Any], timeout: float) -> Any:
        failure: ProviderFailure | None = None
        try:
            return await _sdk(self._client).embeddings.create(**body, timeout=timeout)
        except Exception as error:
            failure = classify(error)
        # Raised after the handler: the SDK's error is neither cause nor context.
        raise ProviderError(failure)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        vectors: list[tuple[float, ...]] = []
        tokens = 0
        for batch in batches(request.texts, self._max_inputs):
            body = embeddings_body(request.model, batch, request.dimensions)
            response = await self._create(body, request.timeout_s)
            read = read_embeddings(response, len(batch))
            if read is None:
                raise ProviderError(ProviderFailure.PROTOCOL)
            vectors.extend(read[0])
            tokens += read[1]
        return EmbeddingResponse(
            model=request.model, vectors=tuple(vectors), usage=embedding_usage(tokens)
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def __repr__(self) -> str:
        return f"OpenAICompatEmbedder(name={self.name!r})"


class OpenAICompatEmbeddingFactory(OpenAICompatFactory):
    """The OpenAI-compatible chat factory, with embeddings: same settings, client and
    authentication."""

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        if not isinstance(settings, OpenAICompatSettings):
            raise ConfigurationError("openai_compat settings expected")
        return OpenAICompatEmbedder(
            name,
            settings,
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
