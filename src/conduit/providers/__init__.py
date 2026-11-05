"""Model provider adapters, each SDK behind its own extra, plus the stub provider."""

from __future__ import annotations

from conduit.core.registry import ProviderFactory
from conduit.providers.ollama import OllamaFactory
from conduit.providers.stub import StubFactory

__all__ = ["builtin_factories"]


def builtin_factories() -> list[ProviderFactory]:
    """A factory for every adapter conduit ships.

    Each adapter imports its SDK only inside ``create_client``, so building this list
    never needs an extra installed.
    """
    return [StubFactory(), OllamaFactory()]
