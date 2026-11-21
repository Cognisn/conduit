"""The stub embedding provider: deterministic vectors, for tests. Not for production.

It shares the stub chat provider's settings (``kind: stub``, with
``not_for_production: true``) and reaches no network. Each text's vector comes from the
SHA-256 of the text, so the same text always gives the same vector and different texts
give different ones; usage counts each text's whitespace-separated words as its input
tokens. It accepts any shorter length (the vector is cut to it), so dimension handling
can be tested end to end. The default catalogue declares ``stub-embed`` (8 dimensions).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_usage,
)
from conduit.core.settings import ProviderSettings
from conduit.providers.stub import StubSettings

STUB_EMBED_MODEL = "stub-embed"


def stub_vector(text: str, dimensions: int) -> tuple[float, ...]:
    """The stub's vector for ``text``: values in [-1, 1] from the text's SHA-256."""
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    return tuple(values[:dimensions])


class StubEmbedder(BaseEmbeddingProvider):
    """Deterministic embeddings for tests."""

    kind = "stub"

    def __init__(self, catalogue: Catalogue) -> None:
        self._catalogue = catalogue
        self.requests: list[EmbeddingRequest] = []

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return True

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        entry = self._catalogue.require("stub", request.model)
        size = request.dimensions or entry.embedding_dimensions or 0
        return EmbeddingResponse(
            model=request.model,
            vectors=tuple(stub_vector(text, size) for text in request.texts),
            usage=embedding_usage(sum(len(text.split()) for text in request.texts)),
        )


class StubEmbeddingFactory:
    """The stub embedding adapter for the provider registry."""

    kind = "stub"
    settings_model = StubSettings

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        return None  # the stub has no client

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        if not isinstance(settings, StubSettings):
            raise ConfigurationError("stub settings expected")
        return StubEmbedder(catalogue if catalogue is not None else default_catalogue())
