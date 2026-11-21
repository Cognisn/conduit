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

from pydantic import Field, field_validator, model_validator

from conduit.core.residency import OfferedRegion, RegionListing, ResidencyScope
from conduit.core.settings import ProviderSettings
from conduit.providers.bedrock.settings import (
    BEDROCK_REGIONS,
    REGION_PATTERN,
    REGIONS_AS_OF,
    REGIONS_DOCUMENTATION,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)

#: The Regions where the bedrock-mantle endpoint is offered (REGIONS_DOCUMENTATION,
#: "Regional availability by endpoints"): a subset of the bedrock-runtime Regions. A
#: request is processed in the Region it is sent to. ap-southeast-4 is kept although that
#: page omits it: several Bedrock model cards and Anthropic's Bedrock page list it for
#: bedrock-mantle, and a Region that turns out not to serve the endpoint fails closed at
#: its listing, where leaving out a served one would refuse a working connection.
MANTLE_REGIONS = frozenset(
    {
        "us-east-1",
        "us-east-2",
        "us-west-2",
        "eu-central-1",
        "eu-north-1",
        "eu-south-1",
        "eu-west-1",
        "eu-west-2",
        "ap-northeast-1",
        "ap-south-1",
        "ap-southeast-2",
        "ap-southeast-3",
        "ap-southeast-4",
        "sa-east-1",
        "us-gov-east-1",
        "us-gov-west-1",
    }
)

NOT_SERVED = "region is not a Region where the Bedrock mantle endpoint is offered"

#: The mantle endpoint's Anthropic base URL; the SDK appends ``/v1/messages``.
MANTLE_BASE_URL = "https://bedrock-mantle.{region}.api.aws/anthropic"

MantleAuth = Annotated[ShortTermApiKeyAuth | LongTermApiKeyAuth, Field(discriminator="mode")]


def mantle_base_url(region: str) -> str:
    """The Anthropic base URL of the bedrock-mantle endpoint in ``region``."""
    return MANTLE_BASE_URL.format(region=region)


class BedrockMantleSettings(ProviderSettings):
    """One configured bedrock-mantle provider.

    ``region`` is the Region the endpoint is called in, and the region the residency
    policy checks; it must be one of ``MANTLE_REGIONS``, and every request is processed
    there. ``endpoint_url`` replaces the regional endpoint (for example an interface VPC
    endpoint); it must be https and should serve the same Region.
    """

    kind: Literal["bedrock_mantle"] = "bedrock_mantle"
    region: str = Field(pattern=REGION_PATTERN)
    auth: MantleAuth = Field(default_factory=ShortTermApiKeyAuth)
    allow_long_term_api_keys: bool = False
    endpoint_url: str | None = Field(default=None, pattern=r"^https://", max_length=512)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("region")
    @classmethod
    def _served(cls, value: str) -> str:
        if value not in MANTLE_REGIONS:
            raise ValueError(NOT_SERVED)
        return value

    @classmethod
    def offered_regions(cls) -> tuple[RegionListing, ...]:
        return (
            RegionListing(
                kind="bedrock_mantle",
                service=None,
                regions=tuple(
                    OfferedRegion(
                        code, BEDROCK_REGIONS[code], invocation=True, scope=ResidencyScope.IN_REGION
                    )
                    for code in sorted(MANTLE_REGIONS)
                ),
                global_option=None,
                documentation=REGIONS_DOCUMENTATION,
                as_of=REGIONS_AS_OF,
            ),
        )

    @model_validator(mode="after")
    def _long_term_keys_gated(self) -> Self:
        if isinstance(self.auth, LongTermApiKeyAuth) and not self.allow_long_term_api_keys:
            raise ValueError(
                "long-term Bedrock API keys are refused unless allow_long_term_api_keys is "
                "set; use short_term_api_key"
            )
        return self
