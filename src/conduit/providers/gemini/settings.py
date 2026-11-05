"""Settings for the Gemini API (Google AI Studio keys)."""

from __future__ import annotations

from typing import Literal
from urllib.parse import SplitResult, urlsplit

from pydantic import Field, field_validator

from conduit.core.secrets import SecretRef
from conduit.core.settings import ProviderSettings

#: The Gemini API's endpoint, set explicitly so the SDK's environment override
#: (GOOGLE_GEMINI_BASE_URL) can never redirect prompts.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiSettings(ProviderSettings):
    """One configured Gemini API provider.

    ``api_key`` is a secret reference, resolved when the provider is built.
    ``base_url`` may name an https proxy in front of the Gemini API; it may not carry
    credentials, a query or a fragment. ``max_attempts`` bounds the SDK's own retries
    of throttled and failed requests. Vertex AI (the Gemini Enterprise Agent
    Platform) is not served by this adapter.
    """

    kind: Literal["gemini"] = "gemini"
    api_key: SecretRef
    base_url: str = DEFAULT_BASE_URL
    max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("base_url")
    @classmethod
    def _plain_https_url(cls, value: str) -> str:
        # Messages never echo the URL: it may carry a credential by mistake.
        parts: SplitResult | None = None
        try:
            parts = urlsplit(value)
            _ = parts.port  # raises for a malformed port
        except ValueError:
            parts = None
        # Raised after the handler: the parser's message can quote the URL.
        if parts is None:
            raise ValueError("base_url is not a valid URL")
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError("base_url must be an https URL naming a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("base_url may not carry credentials")
        if parts.query or parts.fragment or value.endswith(("?", "#")):
            raise ValueError("base_url may not carry a query or a fragment")
        return value.rstrip("/")
