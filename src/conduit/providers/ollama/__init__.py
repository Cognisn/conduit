"""The Ollama adapter: /api/chat streamed over httpx (the ollama extra).

httpx is imported only inside ``OllamaFactory.create_client`` and when a call fails,
so importing this package, and building ``conduit.providers.builtin_factories()``,
never needs the extra installed. Models served by Ollama run on the deployment's own
hosts: their catalogue entries have residency scope ``local``.
"""

from __future__ import annotations

from conduit.providers.ollama.factory import OllamaFactory
from conduit.providers.ollama.provider import OllamaProvider
from conduit.providers.ollama.settings import DEFAULT_BASE_URL, OllamaSettings

__all__ = ["DEFAULT_BASE_URL", "OllamaFactory", "OllamaProvider", "OllamaSettings"]
