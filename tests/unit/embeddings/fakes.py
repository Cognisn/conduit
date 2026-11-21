"""A fake embedding adapter for the registry, funnel and start-up tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_usage,
)
from conduit.core.secrets import SecretRef
from conduit.core.settings import ProviderSettings

Behaviour = Literal[
    "ok",
    "fail",
    "raise",
    "slow",
    "short_vector",
    "long_vector",
    "missing_vector",
    "extra_vector",
    "no_usage",
    "wrong_usage",
    "nan",
]


class FakeEmbedSettings(ProviderSettings):
    kind: Literal["fakeembed"] = "fakeembed"
    api_key: SecretRef
    behaviour: Behaviour = "ok"


class FakeEmbedClient:
    def __init__(self, key: SecretStr) -> None:
        self.key = key


def vector(text: str, dimensions: int) -> tuple[float, ...]:
    """A deterministic vector for ``text``."""
    return tuple(float((len(text) + i) % 7) / 7 for i in range(dimensions))


class FakeEmbedder(BaseEmbeddingProvider):
    kind = "fakeembed"

    def __init__(self, settings: FakeEmbedSettings, catalogue: Catalogue) -> None:
        self.settings = settings
        self.catalogue = catalogue
        self.requests: list[EmbeddingRequest] = []
        self.closed = False
        self.close_error: Exception | None = None
        self.waiting = asyncio.Event()
        self.unserved: set[str] = set()

    def serves(self, model: str) -> bool:
        return model not in self.unserved

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return dimensions in (4, 8)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        behaviour = self.settings.behaviour
        entry = self.catalogue.require("fakeembed", request.model)
        size = request.dimensions or entry.embedding_dimensions or 0
        if behaviour == "fail":
            raise ProviderError(ProviderFailure.UNAVAILABLE)
        if behaviour == "raise":
            raise RuntimeError("synthetic adapter failure for example.com")
        if behaviour == "slow":
            self.waiting.set()
            await asyncio.sleep(30)
        vectors = [vector(text, size) for text in request.texts]
        if behaviour == "short_vector":
            vectors[-1] = vectors[-1][:-1]
        elif behaviour == "long_vector":
            vectors[0] = (*vectors[0], 0.5)
        elif behaviour == "missing_vector":
            vectors.pop()
        elif behaviour == "extra_vector":
            vectors.append(vector("extra", size))
        elif behaviour == "nan":
            vectors[0] = (float("nan"), *vectors[0][1:])
        tokens = sum(len(text.split()) for text in request.texts)
        usage = None if behaviour == "no_usage" else embedding_usage(tokens)
        if behaviour == "wrong_usage":
            usage = usage.model_copy(update={"output_tokens": 3}) if usage else None
        return EmbeddingResponse(model=request.model, vectors=tuple(vectors), usage=usage)

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeEmbedFactory:
    kind = "fakeembed"
    settings_model = FakeEmbedSettings

    def __init__(self) -> None:
        self.clients: list[FakeEmbedClient] = []
        self.embedders: list[FakeEmbedder] = []
        self.catalogues: list[Catalogue | None] = []
        self.unserved: set[str] = set()

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        client = FakeEmbedClient(secrets["api_key"])
        self.clients.append(client)
        return client

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        assert isinstance(settings, FakeEmbedSettings)
        assert catalogue is not None
        self.catalogues.append(catalogue)
        embedder = FakeEmbedder(settings, catalogue)
        embedder.unserved = set(self.unserved)
        self.embedders.append(embedder)
        return embedder


CATALOGUE = Catalogue(
    [
        ModelEntry(
            model_id="fake-embed",
            provider="fakeembed",
            kind="embedding",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=512,
            embedding_dimensions=8,
        ),
        ModelEntry(
            model_id="fake-embed-small",
            provider="fakeembed",
            kind="embedding",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=512,
            embedding_dimensions=4,
        ),
        ModelEntry(
            model_id="fake-embed-global",
            provider="fakeembed",
            kind="embedding",
            residency_scope="global",  # type: ignore[arg-type]
            context_window=512,
            embedding_dimensions=8,
        ),
        ModelEntry(
            model_id="fake-chat",
            provider="fakeembed",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=4_000,
            output_limit=1_000,
        ),
    ],
    version="fake-embed-1",
)

FAKE = {"kind": "fakeembed", "api_key": "secret://fake/key"}
