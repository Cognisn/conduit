"""The OpenAI-compatible adapter: OpenAI, X.AI and Microsoft Foundry (``openai`` extra).

Streaming Chat Completions with usage on the final chunk, tool calls, images, tool
results and structured output where the catalogue allows it. Prompt caching on these
services is automatic, so ``CacheBreakpoint`` is ignored; cached prompt tokens come
back as ``cache_read_tokens``. See ``settings`` for the three services and how Foundry
deployments are catalogued.

Importing this package imports neither the openai SDK nor azure-identity.
"""

from __future__ import annotations

from conduit.providers.openai_compat.factory import OpenAICompatClient, OpenAICompatFactory
from conduit.providers.openai_compat.provider import OpenAICompatProvider
from conduit.providers.openai_compat.settings import (
    FOUNDRY_SCOPE,
    OPENAI_BASE_URL,
    XAI_BASE_URL,
    FoundryApiKey,
    FoundryEntraId,
    FoundryService,
    OpenAICompatSettings,
    OpenAIService,
    XAIService,
)

__all__ = [
    "FOUNDRY_SCOPE",
    "OPENAI_BASE_URL",
    "XAI_BASE_URL",
    "FoundryApiKey",
    "FoundryEntraId",
    "FoundryService",
    "OpenAICompatClient",
    "OpenAICompatFactory",
    "OpenAICompatProvider",
    "OpenAICompatSettings",
    "OpenAIService",
    "XAIService",
]
