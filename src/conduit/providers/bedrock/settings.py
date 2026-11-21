"""Settings for the Amazon Bedrock adapter: region, authentication mode and client options.

Four authentication modes, chosen by ``auth.mode``:

- ``credential_chain`` (the default): the AWS credential chain, optionally with a named
  profile. Requests are signed with SigV4.
- ``access_keys``: an access key id, secret access key and optional session token, each a
  ``secret://`` reference resolved through the product's secret resolver. SigV4.
- ``short_term_api_key``: conduit generates a short-term Bedrock API key from AWS
  credentials (the credential chain or access keys, ``auth.source``) and sends it as a
  bearer token. Keys are refreshed before they expire.
- ``long_term_api_key``: a long-term Bedrock API key held as a ``secret://`` reference.
  AWS recommends long-term keys for exploration only, so they are refused unless
  ``allow_long_term_api_keys`` is set.

No setting holds a credential: every credential is a ``SecretRef``.

Regions: ``region`` is the Region the provider is called in (the source Region of an
inference profile), and must be one where Amazon Bedrock's bedrock-runtime endpoint is
offered (``BEDROCK_REGIONS``, from the documented list below). ``processing`` is the
person's choice of where requests may be processed: AWS Regions, which must include
``region``, and ``allow_global`` for ``global.`` inference profiles (any supported AWS
commercial Region). Without ``processing`` only the deployment's residency policy
applies, as before. Where a chosen source Region's profiles actually route is the
listing's, never a guess from a profile's prefix (C18).
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from conduit.core.residency import (
    OfferedRegion,
    RegionChoice,
    RegionListing,
    ResidencyScope,
)
from conduit.core.secrets import SecretRef
from conduit.core.settings import FieldsRefused, ProviderSettings, SettingsModel

#: Bedrock's own ceiling on a short-term key's lifetime (12 hours).
MAX_SHORT_TERM_LIFETIME_S = 43_200
MIN_SHORT_TERM_LIFETIME_S = 900

REGION_PATTERN = r"^[a-z]{2}(-gov|-iso[a-z]?)?-[a-z]+-\d$"

#: Where Regions are documented, and when the lists below were read.
REGIONS_DOCUMENTATION = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints-region-availability.html"
)
REGIONS_AS_OF = "2026-11-21"

#: The Regions where the bedrock-runtime endpoint is offered, with AWS's names
#: (REGIONS_DOCUMENTATION, "Regional availability by endpoints"). A Region missing here
#: is refused; a Region AWS adds needs a conduit release.
BEDROCK_REGIONS: dict[str, str] = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "ca-central-1": "Canada (Central)",
    "ca-west-1": "Canada West (Calgary)",
    "mx-central-1": "Mexico (Central)",
    "eu-central-1": "Europe (Frankfurt)",
    "eu-central-2": "Europe (Zurich)",
    "eu-north-1": "Europe (Stockholm)",
    "eu-south-1": "Europe (Milan)",
    "eu-south-2": "Europe (Spain)",
    "eu-west-1": "Europe (Ireland)",
    "eu-west-2": "Europe (London)",
    "eu-west-3": "Europe (Paris)",
    "ap-east-2": "Asia Pacific (Taipei)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-south-2": "Asia Pacific (Hyderabad)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ap-southeast-4": "Asia Pacific (Melbourne)",
    "ap-southeast-5": "Asia Pacific (Malaysia)",
    "ap-southeast-6": "Asia Pacific (New Zealand)",
    "ap-southeast-7": "Asia Pacific (Thailand)",
    "il-central-1": "Israel (Tel Aviv)",
    "me-central-1": "Middle East (UAE)",
    "me-south-1": "Middle East (Bahrain)",
    "af-south-1": "Africa (Cape Town)",
    "sa-east-1": "South America (Sao Paulo)",
    "us-gov-east-1": "AWS GovCloud (US-East)",
    "us-gov-west-1": "AWS GovCloud (US-West)",
}

#: What a global inference profile means (the global cross-Region inference guide).
GLOBAL_OPTION = (
    "global. inference profiles: processed in any supported AWS commercial Region, a set "
    "AWS may grow"
)

NOT_SERVED = "region is not a Region where the Bedrock runtime endpoint is offered"


def bedrock_region(value: str) -> str:
    """``value`` if the bedrock-runtime endpoint is offered there; refused otherwise."""
    if value not in BEDROCK_REGIONS:
        raise ValueError(NOT_SERVED)
    return value


class CredentialChainAuth(SettingsModel):
    """The AWS credential chain (environment, profile, container or instance role)."""

    mode: Literal["credential_chain"] = "credential_chain"
    profile: str | None = Field(default=None, min_length=1, max_length=128)


class AccessKeyAuth(SettingsModel):
    """AWS access keys, each held as a secret reference."""

    mode: Literal["access_keys"] = "access_keys"
    access_key_id: SecretRef
    secret_access_key: SecretRef
    session_token: SecretRef | None = None


AwsCredentials = Annotated[CredentialChainAuth | AccessKeyAuth, Field(discriminator="mode")]


class ShortTermApiKeyAuth(SettingsModel):
    """Short-term Bedrock API keys, generated by conduit and refreshed before expiry.

    A key lasts ``lifetime_s`` (at most 12 hours), or until the credentials it was
    signed with expire if that is sooner. A new key is generated ``refresh_before_s``
    before the current one expires.
    """

    mode: Literal["short_term_api_key"] = "short_term_api_key"
    source: AwsCredentials = Field(default_factory=CredentialChainAuth)
    lifetime_s: int = Field(
        default=MAX_SHORT_TERM_LIFETIME_S,
        ge=MIN_SHORT_TERM_LIFETIME_S,
        le=MAX_SHORT_TERM_LIFETIME_S,
    )
    refresh_before_s: int = Field(default=600, ge=60)

    @model_validator(mode="after")
    def _margin_within_lifetime(self) -> Self:
        if self.refresh_before_s >= self.lifetime_s:
            raise ValueError("refresh_before_s must be shorter than lifetime_s")
        return self


class LongTermApiKeyAuth(SettingsModel):
    """A long-term Bedrock API key. Refused unless the settings explicitly allow it."""

    mode: Literal["long_term_api_key"] = "long_term_api_key"
    api_key: SecretRef


BedrockAuth = Annotated[
    CredentialChainAuth | AccessKeyAuth | ShortTermApiKeyAuth | LongTermApiKeyAuth,
    Field(discriminator="mode"),
]


class BedrockSettings(ProviderSettings):
    """One configured Amazon Bedrock provider.

    ``region`` is the Region the Bedrock runtime endpoint is called in, and the region
    the residency policy checks; it must be one of ``BEDROCK_REGIONS``. ``processing``
    optionally chooses where requests may be processed (Regions that include
    ``region``, and ``allow_global``). ``use_converse_stream`` chooses ConverseStream (the
    default) or Converse for every call; both produce the same conduit stream events.
    """

    kind: Literal["bedrock"] = "bedrock"
    region: str = Field(pattern=REGION_PATTERN)
    auth: BedrockAuth = Field(default_factory=CredentialChainAuth)
    allow_long_term_api_keys: bool = False
    endpoint_url: str | None = Field(default=None, pattern=r"^https://", max_length=512)
    connect_timeout_s: float = Field(default=10, gt=0, le=120)
    read_timeout_s: float = Field(default=300, gt=0, le=3_600)
    max_attempts: int = Field(default=3, ge=1, le=10)
    use_converse_stream: bool = True
    processing: RegionChoice | None = None

    @field_validator("region")
    @classmethod
    def _served(cls, value: str) -> str:
        return bedrock_region(value)

    @model_validator(mode="after")
    def _long_term_keys_gated(self) -> Self:
        if isinstance(self.auth, LongTermApiKeyAuth) and not self.allow_long_term_api_keys:
            raise ValueError(
                "long-term Bedrock API keys are refused unless allow_long_term_api_keys is "
                "set; use short_term_api_key, access_keys or the credential chain"
            )
        return self

    @model_validator(mode="after")
    def _choice_within_bedrock(self) -> Self:
        if self.processing is None:
            return self
        if any(region not in BEDROCK_REGIONS for region in self.processing.regions):
            raise FieldsRefused(
                "processing names a Region the Bedrock runtime endpoint does not serve",
                ["processing.regions"],
            )
        if self.region not in self.processing.regions:
            raise FieldsRefused(
                "processing must include region, the Region the provider is called in",
                ["processing.regions"],
            )
        return self

    def region_choice(self) -> RegionChoice | None:
        return self.processing

    @classmethod
    def offered_regions(cls) -> tuple[RegionListing, ...]:
        return (
            RegionListing(
                kind="bedrock",
                service=None,
                regions=tuple(
                    OfferedRegion(
                        code, BEDROCK_REGIONS[code], invocation=True, scope=ResidencyScope.IN_REGION
                    )
                    for code in sorted(BEDROCK_REGIONS)
                ),
                global_option=GLOBAL_OPTION,
                documentation=REGIONS_DOCUMENTATION,
                as_of=REGIONS_AS_OF,
            ),
        )
