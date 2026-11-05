"""Local embeddings: sentence-transformers on the CPU, in a worker thread.

``LocalModels`` (made by ``LocalEmbeddingFactory.create_client``, the only place the
library is imported) loads each model once, on its first use, always with
``device="cpu"`` and ``trust_remote_code=False``, and keeps it for the provider's life.
Loading and encoding block, so each call runs on its own daemon thread, never on the
loop; one encoding runs at a time per provider, so concurrent calls do not oversubscribe
the CPU. ``request.timeout_s`` bounds the wait; a timed-out encoding finishes on its
thread and its answer is dropped.

Usage is the number of tokens the model's tokenizer gives the texts, as encoded (texts
longer than the model's ``max_seq_length`` are truncated by sentence-transformers, and
counted as truncated). A shorter vector length is not offered: the model's own is used.

A model that cannot be loaded (absent, not on disk with ``local_files_only``, not a
sentence-transformers model) is ``REJECTED``; a failure while encoding is
``UNAVAILABLE``. Neither keeps the library's exception, which can quote paths or texts.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from typing import Any

from conduit.core.catalogue import Catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.provider import (
    BaseEmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_usage,
)
from conduit.embeddings._common import read_vector
from conduit.embeddings.local.settings import KIND, LocalEmbeddingSettings

#: The only device conduit runs local models on (C13).
DEVICE = "cpu"


class LocalModels:
    """The loaded models of one local provider. Made only by the factory's create_client."""

    def __init__(self, library: Any, settings: LocalEmbeddingSettings, token: str | None) -> None:
        self._library = library
        self._settings = settings
        self._token = token
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()
        #: One encoding at a time.
        self.encoding = threading.Lock()

    def source(self, model: str) -> str:
        return self._settings.model_sources.get(model, model)

    def load(self, model: str) -> Any:
        """The loaded model for a catalogue model id (loaded on first use; thread-safe)."""
        with self._lock:
            found = self._models.get(model)
            if found is None:
                settings = self._settings
                found = self._library.SentenceTransformer(
                    self.source(model),
                    device=DEVICE,
                    cache_folder=settings.cache_folder,
                    local_files_only=settings.local_files_only,
                    revision=settings.revision,
                    token=self._token if self._token is not None else False,
                    trust_remote_code=False,
                )
                self._models[model] = found
            return found

    def close(self) -> None:
        with self._lock:
            self._models.clear()


def encode(
    models: LocalModels, model: str, texts: list[str], settings: LocalEmbeddingSettings
) -> tuple[list[Any], int]:
    """The raw vectors and token count for ``texts``. Blocks: run on a worker thread."""
    loaded = models.load(model)
    with models.encoding:
        vectors = loaded.encode(
            texts,
            batch_size=settings.batch_size,
            normalize_embeddings=settings.normalise,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        features = loaded.tokenize(texts)
    tokens = int(features["attention_mask"].sum())
    return list(vectors.tolist()), tokens


class LocalEmbedder(BaseEmbeddingProvider):
    """One configured local embedding provider."""

    kind = KIND

    def __init__(
        self,
        name: str,
        settings: LocalEmbeddingSettings,
        models: LocalModels,
        *,
        catalogue: Catalogue,
        encoder: Callable[
            [LocalModels, str, list[str], LocalEmbeddingSettings], tuple[list[Any], int]
        ] = encode,
    ) -> None:
        self.name = name
        self.settings = settings
        self._models = models
        self._catalogue = catalogue
        self._encode = encoder

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        loop = asyncio.get_running_loop()
        answered: asyncio.Future[tuple[Any, ProviderFailure | None]] = loop.create_future()
        models, settings, texts = self._models, self.settings, list(request.texts)
        run = self._encode

        def settle(outcome: tuple[Any, ProviderFailure | None]) -> None:
            if not answered.done():  # a late answer, after the timeout, is dropped
                answered.set_result(outcome)

        def work() -> None:
            outcome: tuple[Any, ProviderFailure | None]
            try:
                models.load(request.model)
            except Exception:
                outcome = (None, ProviderFailure.REJECTED)
            else:
                try:
                    outcome = (run(models, request.model, texts, settings), None)
                except Exception:
                    outcome = (None, ProviderFailure.UNAVAILABLE)
            with contextlib.suppress(RuntimeError):  # the loop has closed: nobody waits
                loop.call_soon_threadsafe(settle, outcome)

        threading.Thread(target=work, name="conduit-local-embed", daemon=True).start()
        timed_out = False
        try:
            result, failure = await asyncio.wait_for(answered, request.timeout_s)
        except TimeoutError:
            timed_out = True
        # Raised outside any handler: the library's error is neither cause nor context.
        if timed_out:
            raise ProviderError(ProviderFailure.TIMEOUT)
        if failure is not None:
            raise ProviderError(failure)
        raw, tokens = result
        vectors = [read_vector(row) for row in raw]
        if any(vector is None for vector in vectors) or not isinstance(tokens, int):
            raise ProviderError(ProviderFailure.PROTOCOL)
        return EmbeddingResponse(
            model=request.model,
            vectors=tuple(v for v in vectors if v is not None),
            usage=embedding_usage(tokens),
        )

    async def aclose(self) -> None:
        self._models.close()

    def __repr__(self) -> str:
        return f"LocalEmbedder(name={self.name!r})"
