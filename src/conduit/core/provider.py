"""The provider protocol: async and streaming first.

A provider turns a ``ModelRequest`` into a stream of typed events. ``complete`` is
defined as the collected stream, so streaming and non-streaming calls can never
disagree about content or usage. Providers are built only by the provider registry
(``conduit.core.registry``), which is the single place provider clients are made.

An embedding provider (C13) turns an ``EmbeddingRequest``, a batch of texts for one
embedding model, into an ``EmbeddingResponse`` with one vector per text, in order. It is
built by the same registry, through an ``EmbeddingProviderFactory``, and every call
passes through the registry's ``MeteredEmbedder``, which checks the model, the
dimensions and the vectors, and meters the call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints

from conduit.core.streaming import StreamEvent, collect
from conduit.core.types import ModelRequest, ModelResponse, Usage


@runtime_checkable
class Provider(Protocol):
    """What every provider offers. ``kind`` names its adapter (``bedrock``, ``stub``)."""

    @property
    def kind(self) -> str: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    async def aclose(self) -> None: ...


class BaseProvider(ABC):
    """A convenient base: implement ``stream``; ``complete`` collects it."""

    kind: ClassVar[str]

    @abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """The call's events, ending with exactly one ``StreamStopped``."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await collect(self.stream(request), model=request.model)

    async def aclose(self) -> None:
        return None


class EmbeddingRequest(BaseModel):
    """One embedding call: a batch of texts for one embedding model.

    ``dimensions`` asks for vectors shorter than the model's own (the catalogue's
    ``embedding_dimensions``), where the model accepts that; ``None`` means the model's
    own. An adapter receives ``dimensions`` only when it differs from the model's own.
    The call context (``run_id``, ``correlation_id``, ``task_class``) identifies the
    call in events. The texts are content: a validation error never quotes them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: str = Field(min_length=1, max_length=256)
    texts: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = Field(min_length=1)
    dimensions: PositiveInt | None = None
    timeout_s: float = Field(default=120, gt=0, le=3_600)
    run_id: str | None = None
    correlation_id: str | None = None
    task_class: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """The vectors for an ``EmbeddingRequest``: one per text, in the same order.

    ``usage`` holds the input tokens the provider reported (the other three counts are
    zero), or ``None`` when the provider reports no count for embeddings.
    """

    model: str
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage | None

    @property
    def dimensions(self) -> int:
        """The length of the vectors (the registry checks they all have it)."""
        return len(self.vectors[0]) if self.vectors else 0


def embedding_usage(input_tokens: int) -> Usage:
    """An embedding call's usage: its input tokens, and zero for the other counts."""
    return Usage(
        input_tokens=input_tokens, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
    )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What every embedding provider offers. ``kind`` names its adapter."""

    @property
    def kind(self) -> str: ...

    def serves(self, model: str) -> bool:
        """Whether the adapter knows how to call ``model`` (its request shape)."""
        ...

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        """Whether ``model`` can return vectors of ``dimensions``, shorter than its own."""
        ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    async def aclose(self) -> None: ...


class BaseEmbeddingProvider(ABC):
    """A convenient base: implement ``embed``. Serves any model, at its own dimensions."""

    kind: ClassVar[str]

    def serves(self, model: str) -> bool:
        return True

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return False

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """One vector per text, in order, each of the requested dimensions."""

    async def aclose(self) -> None:
        return None
