"""The Anthropic adapter: Claude through the direct API or Microsoft Foundry.

Install with the ``anthropic`` extra. The SDK is imported only when a client is built.
"""

from __future__ import annotations

from conduit.providers.anthropic.provider import AnthropicFactory, AnthropicProvider
from conduit.providers.anthropic.request import MAX_CACHE_BREAKPOINTS
from conduit.providers.anthropic.settings import DEFAULT_BASE_URL, AnthropicSettings

__all__ = [
    "DEFAULT_BASE_URL",
    "MAX_CACHE_BREAKPOINTS",
    "AnthropicFactory",
    "AnthropicProvider",
    "AnthropicSettings",
]
