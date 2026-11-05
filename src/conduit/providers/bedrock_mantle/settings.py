"""Settings for the bedrock-mantle adapter: region, Bedrock API keys and client options.

bedrock-mantle serves the Anthropic Messages API at
``https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages``. It runs each request
in the Region it is sent to (it has no cross-Region inference profiles), which is what
makes it the route to single-Region Claude where bedrock-runtime offers only profiles.

Authentication is by Bedrock API key, sent as ``Authorization: Bearer``. Two modes,
chosen by ``auth.mode``, shared with the bedrock adapter:

- ``short_term_api_key`` (the default): conduit generates a short-term key from AWS
  credentials (the credential chain or access keys, ``auth.source``) and refreshes it
  before it expires.
- ``long_term_api_key``: a long-term key held as a ``secret://`` reference. AWS
  recommends long-term keys for exploration only, so they are refused unless
  ``allow_long_term_api_keys`` is set.

No setting holds a credential: every credential is a ``SecretRef``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from conduit.core.settings import ProviderSettings
from conduit.providers.bedrock.settings import (
    REGION_PATTERN,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)

#: The mantle endpoint's Anthropic base URL; the SDK appends ``/v1/messages``.
MANTLE_BASE_URL = "https://bedrock-mantle.{region}.api.aws/anthropic"

MantleAuth = Annotated[ShortTermApiKeyAuth | LongTermApiKeyAuth, Field(discriminator="mode")]


def mantle_base_url(region: str) -> str:
    """The Anthropic base URL of the bedrock-mantle endpoint in ``region``."""
    return MANTLE_BASE_URL.format(region=region)


class BedrockMantleSettings(ProviderSettings):
    """One configured bedrock-mantle provider.

    ``region`` is the Region the endpoint is called in, and the region the residency
    policy checks. ``endpoint_url`` replaces the regional endpoint (for example an
    interface VPC endpoint); it must be https and should serve the same Region.
    """

    kind: Literal["bedrock_mantle"] = "bedrock_mantle"
    region: str = Field(pattern=REGION_PATTERN)
    auth: MantleAuth = Field(default_factory=ShortTermApiKeyAuth)
    allow_long_term_api_keys: bool = False
    endpoint_url: str | None = Field(default=None, pattern=r"^https://", max_length=512)
    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def _long_term_keys_gated(self) -> Self:
        if isinstance(self.auth, LongTermApiKeyAuth) and not self.allow_long_term_api_keys:
            raise ValueError(
                "long-term Bedrock API keys are refused unless allow_long_term_api_keys is "
                "set; use short_term_api_key"
            )
        return self
