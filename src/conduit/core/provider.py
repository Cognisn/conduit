"""The provider protocol: async and streaming first.

A provider turns a ``ModelRequest`` into a stream of typed events. ``complete`` is
defined as the collected stream, so streaming and non-streaming calls can never
disagree about content or usage. Providers are built only by the provider registry
(``conduit.core.registry``), which is the single place provider clients are made.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar, Protocol, runtime_checkable

from conduit.core.streaming import StreamEvent, collect
from conduit.core.types import ModelRequest, ModelResponse


@runtime_checkable
class Provider(Protocol):
    """What every provider offers. ``kind`` names its adapter (``bedrock``, ``stub``)."""

    kind: str

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
