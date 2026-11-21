"""Amazon Bedrock embeddings: Titan Text Embeddings through InvokeModel (the bedrock extra).

``BedrockEmbeddingFactory`` is the Bedrock chat factory with ``create_embedder`` added:
the same ``BedrockSettings``, the same four authentication modes, and the same
``bedrock-runtime`` client from ``BedrockFactory.create_client`` (so short-term keys are
refreshed, long-term keys are gated, and the process environment's
``AWS_BEARER_TOKEN_BEDROCK`` is never read). The region the client is called in is the
region the residency policy checks.

Titan Text Embeddings takes one text per InvokeModel call, so a batch is one call per
text, at most ``concurrency`` at a time; each call runs on its own daemon thread, never
on the loop or in its default executor, and ``request.timeout_s`` bounds each one. The
first failure stops calls not yet started and is raised once the started ones have
finished. Usage is the sum of each call's ``inputTextTokenCount``.

Models served (by model id):

- ``amazon.titan-embed-text-v2:0`` (Titan Text Embeddings V2): 1,024 dimensions by
  default; 512 and 256 on request (the ``dimensions`` body field). AWS: "Output vector
  size: 1,024 (default), 512, 256"; up to 8,192 tokens or 50,000 characters
  (docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html).
- ``amazon.titan-embed-text-v1`` and ``amazon.titan-embed-g1-text-02`` (Titan Text
  Embeddings G1): their own dimensions only.

Any other Bedrock model (Cohere Embed, Titan Multimodal) has another request shape and
is refused before any call; a product that overlays one gets that refusal at start-up.

Every error becomes ``ProviderError(failure)`` (classified as the chat adapter does),
raised after its handler, so it has no cause and no context.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable
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
from conduit.embeddings._common import read_count, read_vector
from conduit.providers.bedrock import BedrockFactory, BedrockSettings
from conduit.providers.bedrock.errors import classify

TITAN_V2 = "amazon.titan-embed-text-v2"
TITAN_V1 = ("amazon.titan-embed-text-v1", "amazon.titan-embed-g1-text-02")
#: The shorter lengths Titan Text Embeddings V2 offers (1,024 is its own).
TITAN_V2_DIMENSIONS = frozenset({256, 512})
#: InvokeModel calls one batch may have in flight at once.
DEFAULT_CONCURRENCY = 4


def _family(model: str) -> str | None:
    if model.startswith(TITAN_V2):
        return "titan_v2"
    if model.startswith(TITAN_V1):
        return "titan_v1"
    return None


def titan_body(text: str, dimensions: int | None) -> dict[str, Any]:
    """The InvokeModel body for one text."""
    body: dict[str, Any] = {"inputText": text}
    if dimensions is not None:
        body["dimensions"] = dimensions
    return body


def read_titan(payload: Any) -> tuple[tuple[float, ...], int] | None:
    """The vector and token count of a Titan response body, or ``None``."""
    if not isinstance(payload, dict):
        return None
    vector = read_vector(payload.get("embedding"))
    tokens = read_count(payload.get("inputTextTokenCount"))
    if vector is None or tokens is None:
        return None
    return vector, tokens


class BedrockEmbedder(BaseEmbeddingProvider):
    """One configured Bedrock provider's embeddings, around a ``bedrock-runtime`` client."""

    kind = "bedrock"

    def __init__(
        self,
        name: str,
        settings: BedrockSettings,
        client: Any,
        *,
        catalogue: Catalogue,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.name = name
        self.settings = settings
        self._client = client
        self._catalogue = catalogue
        self._concurrency = concurrency

    def serves(self, model: str) -> bool:
        return _family(model) is not None

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return _family(model) == "titan_v2" and dimensions in TITAN_V2_DIMENSIONS

    async def _invoke(self, model: str, body: dict[str, Any], timeout: float) -> Any:
        loop = asyncio.get_running_loop()
        answered: asyncio.Future[tuple[Any, Exception | None]] = loop.create_future()
        client = self._client
        encoded = json.dumps(body)

        def settle(outcome: tuple[Any, Exception | None]) -> None:
            if not answered.done():  # a late answer, after the timeout, is dropped
                answered.set_result(outcome)

        def call() -> None:
            outcome: tuple[Any, Exception | None]
            try:
                response = client.invoke_model(
                    modelId=model,
                    body=encoded,
                    contentType="application/json",
                    accept="application/json",
                )
                outcome = (response["body"].read(), None)
            except Exception as error:
                outcome = (None, error)
            with contextlib.suppress(RuntimeError):  # the loop has closed: nobody waits
                loop.call_soon_threadsafe(settle, outcome)

        threading.Thread(target=call, name="conduit-bedrock-embed", daemon=True).start()
        timed_out = False
        try:
            raw, error = await asyncio.wait_for(answered, timeout)
        except TimeoutError:
            timed_out = True
        # Raised outside any handler, so the SDK's error is neither cause nor context.
        if timed_out:
            raise ProviderError(ProviderFailure.TIMEOUT)
        if error is not None:
            raise ProviderError(classify(error))
        payload: Any = None
        with contextlib.suppress(ValueError, TypeError):
            payload = json.loads(raw)
        read = read_titan(payload)
        if read is None:
            raise ProviderError(ProviderFailure.PROTOCOL)
        return read

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if _family(request.model) is None:
            raise ConfigurationError(f"the bedrock embedding adapter cannot call {request.model!r}")
        gate = asyncio.Semaphore(self._concurrency)
        stopped = asyncio.Event()

        async def one(text: str) -> tuple[tuple[float, ...], int] | ProviderFailure:
            async with gate:
                if stopped.is_set():
                    return ProviderFailure.UNAVAILABLE  # never sent: an earlier call failed
                failure: ProviderFailure | None = None
                try:
                    read: tuple[tuple[float, ...], int] = await self._invoke(
                        request.model, titan_body(text, request.dimensions), request.timeout_s
                    )
                except ProviderError as error:
                    failure = error.failure
                if failure is not None:
                    stopped.set()
                    return failure
                return read

        results = await asyncio.gather(*(one(text) for text in request.texts))
        first = next((r for r in results if isinstance(r, ProviderFailure)), None)
        if first is not None:
            raise ProviderError(first)
        answers = [r for r in results if not isinstance(r, ProviderFailure)]
        return EmbeddingResponse(
            model=request.model,
            vectors=tuple(vector for vector, _ in answers),
            usage=embedding_usage(sum(tokens for _, tokens in answers)),
        )

    async def aclose(self) -> None:
        close: Callable[[], Any] | None = getattr(self._client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    def __repr__(self) -> str:
        return f"BedrockEmbedder(name={self.name!r}, region={self.settings.region!r})"


class BedrockEmbeddingFactory(BedrockFactory):
    """The Bedrock chat factory, with embeddings: same settings, client and authentication."""

    def __init__(self, *, concurrency: int = DEFAULT_CONCURRENCY, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if concurrency < 1:
            raise ConfigurationError("concurrency must be at least 1")
        self._concurrency = concurrency

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        if not isinstance(settings, BedrockSettings):
            raise ConfigurationError("the Bedrock factory needs BedrockSettings")
        return BedrockEmbedder(
            name,
            settings,
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
            concurrency=self._concurrency,
        )
