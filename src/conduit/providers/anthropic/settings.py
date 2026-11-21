"""Settings for the Anthropic adapter: the direct API, or Claude in Microsoft Foundry.

Regions. On the direct API, ``inference_geo`` chooses where inference runs
(https://platform.claude.com/docs/en/manage-claude/data-residency): ``us`` keeps it on
US infrastructure (``anthropic:us``) and is sent on every request, for the models that
support it (``INFERENCE_GEO_MODELS``, Claude 4.6 and later; others answer 400, so they
are ``global`` here and refused by the choice); ``global`` lets it run in any geography.
Claude in Foundry has no ``inference_geo``: its processing is the deployment type,
declared with ``deployment`` as for the OpenAI-compatible adapter's Foundry service.
``region`` is set from the choice and may not be given otherwise.
"""

from __future__ import annotations

from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from conduit.core.residency import (
    OfferedRegion,
    RegionChoice,
    RegionListing,
    Residency,
    ResidencyScope,
)
from conduit.core.secrets import SecretRef
from conduit.core.settings import FieldsRefused, ProviderSettings
from conduit.providers.openai_compat.settings import (
    FoundryEntraId,
    FoundryProcessing,
    foundry_listing,
)

#: The data residency documentation, read on REGIONS_AS_OF.
DATA_RESIDENCY_DOCUMENTATION = "https://platform.claude.com/docs/en/manage-claude/data-residency"
REGIONS_AS_OF = "2026-11-21"
#: The models that take ``inference_geo`` (the page's supported models), and their dated
#: snapshots. Any other model answers a request carrying it with a 400.
INFERENCE_GEO_MODELS = frozenset(
    {
        "claude-fable-5-2",
        "claude-mythos-5-2",
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }
)
GLOBAL_OPTION = "inference_geo global: inference may run in any geography Anthropic serves"
US_CODE = "anthropic:us"


def supports_inference_geo(model_id: str) -> bool:
    """Whether ``model_id`` (or the model it is a dated snapshot of) takes
    ``inference_geo``."""
    if model_id in INFERENCE_GEO_MODELS:
        return True
    head, _, tail = model_id.rpartition("-")
    return head in INFERENCE_GEO_MODELS and tail.isdigit() and len(tail) == 8


#: The direct API's endpoint. Always passed explicitly, so ``ANTHROPIC_BASE_URL`` in the
#: environment never redirects a conduit client.
DEFAULT_BASE_URL = "https://api.anthropic.com"

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class AnthropicSettings(ProviderSettings):
    """One configured Anthropic provider.

    ``endpoint: direct`` calls Anthropic's API (``base_url`` optional, for a gateway in
    front of it) with ``api_key``. ``endpoint: foundry`` calls Claude in Microsoft
    Foundry through the resource's Anthropic endpoint, given as ``base_url`` (for
    example ``https://<resource>.services.ai.azure.com/anthropic/``), with exactly one
    of ``api_key`` and ``entra_id``. ``entra_id`` is the OpenAI-compatible adapter's
    ``FoundryEntraId`` (azure-identity's ``DefaultAzureCredential``, an optional
    user-assigned managed identity, and the token scope). A key is a secret reference,
    resolved by the product's secret resolver when the provider is built. The SDK never
    falls back to keys or endpoints from the environment.
    """

    kind: Literal["anthropic"] = "anthropic"
    endpoint: Literal["direct", "foundry"] = "direct"
    api_key: SecretRef | None = None
    entra_id: FoundryEntraId | None = None
    base_url: str | None = Field(default=None, max_length=2048)
    max_retries: int = Field(default=2, ge=0, le=10)
    #: Where inference runs on the direct API; sent on every request when set.
    inference_geo: Literal["us", "global"] | None = None
    #: A Foundry deployment's type and Azure region (``endpoint: foundry`` only).
    deployment: FoundryProcessing | None = None

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parts = None
        host: str | None = None
        try:
            parts = urlsplit(value)
            host = parts.hostname
        except ValueError:
            pass  # refused below, with nothing chained
        if parts is None:
            raise ValueError("base_url must be an https URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("base_url must not carry credentials")
        if not host:
            raise ValueError("base_url must be an https URL")
        if parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK):
            return value
        raise ValueError("base_url must be an https URL (plain http only to loopback)")

    @model_validator(mode="after")
    def _foundry_needs_its_endpoint(self) -> Self:
        if self.endpoint == "foundry" and self.base_url is None:
            raise ValueError("endpoint foundry needs base_url, the resource's Anthropic endpoint")
        return self

    @model_validator(mode="after")
    def _one_credential(self) -> Self:
        if self.endpoint == "direct":
            if self.entra_id is not None:
                raise ValueError("entra_id is only for endpoint foundry")
            if self.api_key is None:
                raise ValueError("endpoint direct needs api_key")
        elif (self.api_key is None) == (self.entra_id is None):
            raise ValueError("endpoint foundry needs exactly one of api_key and entra_id")
        return self

    @model_validator(mode="after")
    def _regions(self) -> Self:
        if self.inference_geo is not None and self.endpoint != "direct":
            raise FieldsRefused(
                "inference_geo is only for endpoint direct; a Foundry deployment's "
                "processing is its deployment type",
                ["inference_geo"],
            )
        if self.inference_geo == "us" and self.base_url is not None:
            raise FieldsRefused(
                "inference_geo us cannot be guaranteed through a base_url gateway",
                ["inference_geo"],
            )
        if self.deployment is not None and self.endpoint != "foundry":
            raise FieldsRefused("deployment is only for endpoint foundry", ["deployment"])
        chosen = self._invocation
        if chosen is None:
            return self
        if self.region is not None and self.region != chosen:
            raise FieldsRefused(
                "region is set by the inference_geo or deployment choice; leave it unset",
                ["region"],
            )
        object.__setattr__(self, "region", chosen)  # frozen: set once, while validating
        return self

    @property
    def _invocation(self) -> str | None:
        if self.inference_geo == "us":
            return US_CODE
        if self.deployment is not None:
            return self.deployment.invocation
        return None

    def region_choice(self) -> RegionChoice | None:
        if self.inference_geo == "us":
            return RegionChoice(regions=frozenset({US_CODE}))
        if self.deployment is not None:
            return self.deployment.choice()
        return None

    def declared_residency(self, model_id: str) -> Residency | None:
        if self.inference_geo == "us" and supports_inference_geo(model_id):
            return Residency(ResidencyScope.CROSS_REGION, frozenset({US_CODE}))
        if self.inference_geo is not None:
            return Residency(ResidencyScope.GLOBAL)
        if self.deployment is not None:
            return self.deployment.residency()
        return None

    @classmethod
    def offered_regions(cls) -> tuple[RegionListing, ...]:
        direct = RegionListing(
            kind="anthropic",
            service="direct",
            regions=(
                OfferedRegion(
                    US_CODE, "United States", invocation=True, scope=ResidencyScope.CROSS_REGION
                ),
            ),
            global_option=GLOBAL_OPTION,
            documentation=DATA_RESIDENCY_DOCUMENTATION,
            as_of=REGIONS_AS_OF,
        )
        return (direct, foundry_listing("anthropic"))
