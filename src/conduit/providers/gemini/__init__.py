"""The Gemini adapter: the Gemini API over google-genai's async client (the gemini extra).

google-genai is imported only inside ``GeminiFactory.create_client``, so importing
this package, and building ``conduit.providers.builtin_factories()``, never needs the
extra installed. Cache breakpoints are ignored: the Gemini API caches repeated
prefixes implicitly and reports the tokens read from its cache.
"""

from __future__ import annotations

from conduit.providers.gemini.factory import GeminiFactory
from conduit.providers.gemini.provider import GeminiProvider
from conduit.providers.gemini.settings import DEFAULT_BASE_URL, GeminiSettings

__all__ = ["DEFAULT_BASE_URL", "GeminiFactory", "GeminiProvider", "GeminiSettings"]
