"""Embeddings interface, start-up for embedding task classes, and embedding providers."""

# C13.
#
# The interface lives in the core (``conduit.core.provider``: ``EmbeddingRequest``,
# ``EmbeddingResponse``, ``EmbeddingProvider``; ``conduit.core.registry``:
# ``EmbeddingProviderFactory``, ``MeteredEmbedder`` and the two embedding events) and is
# re-exported here. ``start_embeddings`` resolves a product's embedding task classes and
# builds their providers through the provider registry.
#
# Remote providers share their chat adapter's settings, authentication and residency:
# each embedding factory is its chat factory with ``create_embedder`` added.
#
# - ``bedrock``: Titan Text Embeddings through InvokeModel (the bedrock extra).
# - ``openai_compat``: /v1/embeddings for OpenAI and Microsoft Foundry (the openai extra).
# - ``gemini``: ``embed_content`` on the Gemini API (the gemini extra).
# - ``ollama``: /api/embed on an Ollama server (the ollama extra).
# - ``sentence_transformers``: local models on the CPU (the local-embeddings extra).
# - ``stub``: deterministic vectors for tests.
#
# No adapter imports its SDK until a client is made, so importing this package and
# building ``builtin_embedding_factories()`` never needs an extra installed. Vector
# stores and retrieval are the product's.

from __future__ import annotations

from conduit.core.provider import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_usage,
)
from conduit.core.registry import (
    EmbeddingCallFinishedEvent,
    EmbeddingCallStartedEvent,
    EmbeddingProviderFactory,
    MeteredEmbedder,
)
from conduit.embeddings.bedrock import BedrockEmbedder, BedrockEmbeddingFactory
from conduit.embeddings.gemini import GeminiEmbedder, GeminiEmbeddingFactory
from conduit.embeddings.local import LocalEmbeddingFactory, LocalEmbeddingSettings
from conduit.embeddings.ollama import OllamaEmbedder, OllamaEmbeddingFactory
from conduit.embeddings.openai_compat import OpenAICompatEmbedder, OpenAICompatEmbeddingFactory
from conduit.embeddings.runtime import EmbeddingBinding, EmbeddingRuntime, start_embeddings
from conduit.embeddings.stub import StubEmbedder, StubEmbeddingFactory

__all__ = [
    "BaseEmbeddingProvider",
    "BedrockEmbedder",
    "BedrockEmbeddingFactory",
    "EmbeddingBinding",
    "EmbeddingCallFinishedEvent",
    "EmbeddingCallStartedEvent",
    "EmbeddingProvider",
    "EmbeddingProviderFactory",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingRuntime",
    "GeminiEmbedder",
    "GeminiEmbeddingFactory",
    "LocalEmbeddingFactory",
    "LocalEmbeddingSettings",
    "MeteredEmbedder",
    "OllamaEmbedder",
    "OllamaEmbeddingFactory",
    "OpenAICompatEmbedder",
    "OpenAICompatEmbeddingFactory",
    "StubEmbedder",
    "StubEmbeddingFactory",
    "builtin_embedding_factories",
    "embedding_usage",
    "start_embeddings",
]


def builtin_embedding_factories() -> list[EmbeddingProviderFactory]:
    """An embedding factory for every adapter conduit ships."""
    return [
        StubEmbeddingFactory(),
        BedrockEmbeddingFactory(),
        OpenAICompatEmbeddingFactory(),
        GeminiEmbeddingFactory(),
        OllamaEmbeddingFactory(),
        LocalEmbeddingFactory(),
    ]
