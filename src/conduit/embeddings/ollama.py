"""Ollama embeddings: POST /api/embed over the Ollama adapter's httpx client (the ollama extra).

``OllamaEmbeddingFactory`` is the Ollama chat factory with ``create_embedder`` added:
the same ``OllamaSettings`` and the same httpx ``AsyncClient`` from
``OllamaFactory.create_client`` (``trust_env=False``, so proxy variables and ``.netrc``
cannot route texts elsewhere, and no redirect is followed).

A batch is sent as one request per ``MAX_TEXTS`` texts (``{"model", "input", "truncate":
false, "options": {"num_ctx"}}``), each with ``request.timeout_s`` as its timeout.
``truncate`` is false, so a text longer than the context is refused (``REJECTED``)
rather than embedded cut short. ``num_ctx`` is the settings' value, or the catalogue
entry's ``context_window``, as for chat. Usage is the sum of ``prompt_eval_count``.
Ollama's API takes ``dimensions``, but whether a model honours it depends on the model,
so a shorter length is not offered; the model's own is always used.

Any status but 200 is a failure (classified as the chat adapter does). Errors are
recognised by class name, so this module imports no HTTP library; each becomes
``ProviderError(failure)``, raised after its handler, so it has no cause and no context.
"""

from __future__ import annotations

import contextlib
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
from conduit.providers.ollama import OllamaFactory, OllamaSettings
from conduit.providers.ollama.mapping import failure_for_status

EMBED_PATH = "/api/embed"
#: Texts per request.
MAX_TEXTS = 256
_HTTP_MODULES = frozenset({"httpx", "httpcore"})


def classify(error: BaseException) -> ProviderFailure:
    """The failure for an error from an httpx call, by class name: a timeout is
    ``TIMEOUT``, anything else ``UNAVAILABLE``."""
    names = {cls.__name__ for cls in type(error).__mro__}
    if isinstance(error, TimeoutError) or (
        type(error).__module__.split(".")[0] in _HTTP_MODULES and "TimeoutException" in names
    ):
        return ProviderFailure.TIMEOUT
    return ProviderFailure.UNAVAILABLE


def embed_body(model: str, texts: tuple[str, ...], num_ctx: int) -> dict[str, Any]:
    """The /api/embed body for one batch."""
    return {
        "model": model,
        "input": list(texts),
        "truncate": False,
        "options": {"num_ctx": num_ctx},
    }


def read_embed(payload: Any, count: int) -> tuple[list[tuple[float, ...]], int] | None:
    """The vectors and ``prompt_eval_count`` of a response body, or ``None``."""
    if not isinstance(payload, dict):
        return None
    embeddings = payload.get("embeddings")
    tokens = read_count(payload.get("prompt_eval_count"))
    if not isinstance(embeddings, list) or len(embeddings) != count or tokens is None:
        return None
    vectors = [read_vector(values) for values in embeddings]
    if any(vector is None for vector in vectors):
        return None
    return [v for v in vectors if v is not None], tokens


class OllamaEmbedder(BaseEmbeddingProvider):
    """One configured Ollama server's embeddings."""

    kind = "ollama"

    def __init__(
        self,
        name: str,
        settings: OllamaSettings,
        client: Any,
        *,
        catalogue: Catalogue,
        max_texts: int = MAX_TEXTS,
    ) -> None:
        self.name = name
        self.settings = settings
        self._client = client
        self._catalogue = catalogue
        self._max_texts = max_texts

    async def _post(self, body: dict[str, Any], timeout: float) -> tuple[int, Any]:
        failure: ProviderFailure | None = None
        try:
            response = await self._client.post(EMBED_PATH, json=body, timeout=timeout)
        except Exception as error:
            failure = classify(error)
        else:
            payload: Any = None
            if response.status_code == 200:
                with contextlib.suppress(ValueError):
                    payload = response.json()
            return response.status_code, payload
        # Raised after the handler: httpx's error is neither cause nor context.
        raise ProviderError(failure)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        entry = self._catalogue.require(self.settings.catalogue_provider, request.model)
        num_ctx = self.settings.num_ctx or entry.context_window
        vectors: list[tuple[float, ...]] = []
        tokens = 0
        for batch in batches(request.texts, self._max_texts):
            status, payload = await self._post(
                embed_body(request.model, batch, num_ctx), request.timeout_s
            )
            if status != 200:
                raise ProviderError(failure_for_status(status))
            read = read_embed(payload, len(batch))
            if read is None:
                raise ProviderError(ProviderFailure.PROTOCOL)
            vectors.extend(read[0])
            tokens += read[1]
        return EmbeddingResponse(
            model=request.model, vectors=tuple(vectors), usage=embedding_usage(tokens)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"OllamaEmbedder(name={self.name!r})"


class OllamaEmbeddingFactory(OllamaFactory):
    """The Ollama chat factory, with embeddings: same settings and client."""

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        if not isinstance(settings, OllamaSettings):
            raise ConfigurationError("the Ollama factory needs OllamaSettings")
        return OllamaEmbedder(
            name,
            settings,
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
