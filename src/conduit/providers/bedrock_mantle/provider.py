"""The bedrock-mantle provider: Claude through the Anthropic Messages API on bedrock-mantle.

Requests and streams are the anthropic adapter's own (``AnthropicProvider``, with
``conduit.providers.anthropic.request.build_params`` and
``conduit.providers.anthropic.stream.translate``): the same mapping, all four usage
counts from the final stream events, and errors classified with no provider text.
Thinking blocks are provider state of kind ``anthropic``, the same shape as on
Anthropic's own API, so they go back unchanged and other kinds' state is left out.

Capabilities are read from the catalogue under the provider name ``bedrock_mantle``:

- structured output, as the anthropic adapter reads it;
- prompt caching: for a model the catalogue does not say caches, every cache
  breakpoint is removed from the request before it is mapped, so no ``cache_control``
  is sent (an undeclared model never caches).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, cast

from conduit.core.catalogue import Catalogue
from conduit.core.provider import BaseProvider
from conduit.core.streaming import StreamEvent
from conduit.core.types import CacheBreakpoint, Message, ModelRequest
from conduit.providers.anthropic import AnthropicProvider

KIND = "bedrock_mantle"


def without_breakpoints(request: ModelRequest) -> ModelRequest:
    """``request`` with every cache breakpoint removed, and no breakpoint after the tools."""
    return request.model_copy(
        update={
            "cache_tools": False,
            "system": tuple(b for b in request.system if not isinstance(b, CacheBreakpoint)),
            "messages": tuple(
                Message(
                    role=m.role,
                    content=tuple(b for b in m.content if not isinstance(b, CacheBreakpoint)),
                )
                for m in request.messages
            ),
        }
    )


class BedrockMantleProvider(BaseProvider):
    """Claude on bedrock-mantle, streaming, through the anthropic adapter's mapping."""

    kind = KIND

    def __init__(self, client: Any, *, name: str, catalogue: Catalogue) -> None:
        self.name = name
        self._catalogue = catalogue
        self._messages = AnthropicProvider(
            client, name=name, catalogue=catalogue, catalogue_provider=KIND
        )

    def structured_output(self, model: str) -> bool:
        """Whether the catalogue says ``model`` supports structured output on mantle."""
        return self._messages.structured_output(model)

    def prompt_caching(self, model: str) -> bool:
        """Whether the catalogue says ``model`` caches prompts on mantle."""
        entry = self._catalogue.get(KIND, model)
        return entry is not None and entry.capabilities.prompt_caching is True

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if not self.prompt_caching(request.model):
            request = without_breakpoints(request)
        # aclosing: an abandoned stream closes the anthropic stream, and with it the
        # SDK's, at once rather than when the generator is collected.
        stream = cast(AsyncGenerator[StreamEvent, None], self._messages.stream(request))
        async with aclosing(stream) as events:
            async for event in events:
                yield event

    async def aclose(self) -> None:
        await self._messages.aclose()

    def __repr__(self) -> str:
        return f"BedrockMantleProvider(name={self.name!r})"
