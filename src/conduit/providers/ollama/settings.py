"""Settings for a locally hosted Ollama server."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, PositiveInt, field_validator

from conduit.core.settings import ProviderSettings

#: Ollama's own default address, on loopback so prompts never leave the host.
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaSettings(ProviderSettings):
    """One configured Ollama server.

    ``base_url`` defaults to loopback. A remote server is allowed (the catalogue's
    ``local`` scope means the deployment's own hosts), but the URL may not carry
    credentials, a query or a fragment: conduit's settings hold credentials only as
    secret references. ``num_ctx`` caps the context length requested from Ollama;
    left unset, the catalogue entry's full ``context_window`` is requested (131072 for
    llama3.1:8b), because Ollama's own default is far smaller than the catalogue
    declares. A full window needs memory on the Ollama host to match; set ``num_ctx``
    to request less.
    """

    kind: Literal["ollama"] = "ollama"
    base_url: str = DEFAULT_BASE_URL
    num_ctx: PositiveInt | None = Field(default=None)

    @field_validator("base_url")
    @classmethod
    def _plain_http_url(cls, value: str) -> str:
        # Messages never echo the URL: it may carry a credential by mistake.
        try:
            parts = urlsplit(value)
            _ = parts.port  # raises for a malformed port
        except ValueError:
            raise ValueError("base_url is not a valid URL") from None
        if parts.scheme not in ("http", "https"):
            raise ValueError("base_url must be an http or https URL")
        if not parts.hostname:
            raise ValueError("base_url must name a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("base_url may not carry credentials")
        if parts.query or parts.fragment or value.endswith(("?", "#")):
            raise ValueError("base_url may not carry a query or a fragment")
        return value.rstrip("/")
