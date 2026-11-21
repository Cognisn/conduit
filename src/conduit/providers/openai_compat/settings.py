"""Settings for the OpenAI-compatible adapter.

One adapter (kind ``openai_compat``) serves three services, chosen by ``service.name``:

- ``openai``: the OpenAI API (``https://api.openai.com/v1``), with an API key.
- ``xai``: the X.AI API (``https://api.x.ai/v1``), with an API key.
- ``foundry``: Microsoft Foundry's OpenAI v1 endpoint
  (``https://<resource>.openai.azure.com/openai/v1/``), with an API key or Microsoft
  Entra ID (``DefaultAzureCredential`` and a bearer token provider for
  ``https://ai.azure.com/.default``).

API keys are only ever ``secret://`` references. The catalogue provider is ``openai``
or ``xai`` for those services; for Foundry it is ``service.catalogue_provider``
(``openai`` by default, ``xai`` for Grok deployments). A Foundry model id is the
deployment name, which the product declares in a catalogue overlay under that provider
with its own residency (Foundry deployments are regional; the default catalogue's
``openai`` and ``xai`` entries describe the public APIs). Name deployments so they do
not collide with a public model id, or the overlay would change that entry too.

Regions. Each service may choose where its requests are processed, and its endpoint's
documented guarantee becomes the residency of every model it serves:

- ``openai``: ``data_residency`` picks a project region's host
  (``https://<code>.api.openai.com/v1``). Regions with regional processing (us, eu, au,
  jp, in, ae) are ``cross_region`` in ``openai:<code>``, except that au, jp and in
  process only ``gpt-6-luna`` in-region and every other model there is ``global``; the
  storage-only regions (ca, sg, kr, gb) keep stored data there but process anywhere, so
  every model is ``global`` and nothing is chosen.
- ``xai``: ``endpoint_region: us`` picks ``https://us.api.x.ai/v1``, which keeps
  processing in the US (``xai:us``).
- ``foundry``: ``deployment`` declares the deployment type and the resource's Azure
  region for every deployment the connection serves (Foundry's data-plane endpoint does
  not report a deployment's type): ``global`` (any Azure region), ``data_zone`` (the
  region's documented data zone, ``azure:zone-<zone>``) or ``standard`` (the region's
  Azure geography, ``azure:geo-<geography>``). This is the product's declaration,
  which conduit cannot verify.

``region`` is set from the choice (``openai:eu``, ``xai:us``, ``azure:<region>``) and
may not be given otherwise. A service that chooses nothing keeps its old meaning.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, field_validator, model_validator

from conduit.core.residency import (
    OfferedRegion,
    RegionChoice,
    RegionListing,
    Residency,
    ResidencyScope,
)
from conduit.core.secrets import SecretRef
from conduit.core.settings import FieldsRefused, ProviderSettings, SettingsModel

OPENAI_BASE_URL = "https://api.openai.com/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
FOUNDRY_SCOPE = "https://ai.azure.com/.default"
FOUNDRY_PATH = "/openai/v1"


def _https(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("must be an https URL")
    return value


def _foundry_endpoint(value: str) -> str:
    _https(value)
    if urlsplit(value).path.rstrip("/") != FOUNDRY_PATH:
        raise ValueError(f"must be the resource's {FOUNDRY_PATH} endpoint")
    return value


HttpsUrl = Annotated[str, AfterValidator(_https)]
FoundryEndpoint = Annotated[str, AfterValidator(_foundry_endpoint)]

# Regions -----------------------------------------------------------------------------

#: OpenAI's data residency guide, read on REGIONS_AS_OF.
OPENAI_RESIDENCY_DOCUMENTATION = "https://developers.openai.com/api/docs/guides/data-residency"
#: X.AI's regions page.
XAI_REGIONS_DOCUMENTATION = "https://docs.x.ai/docs/key-information/regions"
#: Foundry's deployment types, its data zone tables and Azure's region list.
FOUNDRY_DEPLOYMENT_DOCUMENTATION = (
    "https://learn.microsoft.com/azure/foundry/foundry-models/concepts/deployment-types"
)
REGIONS_AS_OF = "2026-11-21"

#: OpenAI's data residency regions and their names ("Support by region").
OPENAI_RESIDENCY_REGIONS: dict[str, str] = {
    "us": "United States",
    "eu": "Europe (EEA and Switzerland)",
    "au": "Australia",
    "ca": "Canada",
    "jp": "Japan",
    "in": "India",
    "sg": "Singapore",
    "kr": "South Korea",
    "gb": "United Kingdom",
    "ae": "United Arab Emirates",
}
#: The regions with regional processing; the rest keep stored data only.
OPENAI_PROCESSING_REGIONS = frozenset({"us", "eu", "au", "jp", "in", "ae"})
#: Regions that process in-region only for these models (and their dated snapshots).
OPENAI_PROCESSING_MODELS: dict[str, frozenset[str]] = {
    "au": frozenset({"gpt-6-luna"}),
    "jp": frozenset({"gpt-6-luna"}),
    "in": frozenset({"gpt-6-luna"}),
}
OPENAI_GLOBAL_OPTION = (
    "the default host, a storage-only region, or a model a region does not process "
    "in-region: processing is not bounded"
)
XAI_GLOBAL_OPTION = "the default host https://api.x.ai may route requests between regions"
FOUNDRY_GLOBAL_OPTION = (
    "a Global deployment: processed in any Azure region where the model is deployed"
)

#: Azure public cloud regions: programmatic name, display name and geography (Azure's
#: regions list, https://learn.microsoft.com/azure/reliability/regions-list).
AZURE_REGIONS: dict[str, tuple[str, str]] = {
    "australiacentral": ("Australia Central", "Australia"),
    "australiacentral2": ("Australia Central 2", "Australia"),
    "australiaeast": ("Australia East", "Australia"),
    "australiasoutheast": ("Australia Southeast", "Australia"),
    "austriaeast": ("Austria East", "Austria"),
    "belgiumcentral": ("Belgium Central", "Belgium"),
    "brazilsouth": ("Brazil South", "Brazil"),
    "brazilsoutheast": ("Brazil Southeast", "Brazil"),
    "canadacentral": ("Canada Central", "Canada"),
    "canadaeast": ("Canada East", "Canada"),
    "centralindia": ("Central India", "India"),
    "centralus": ("Central US", "United States"),
    "chilecentral": ("Chile Central", "Chile"),
    "denmarkeast": ("Denmark East", "Denmark"),
    "eastasia": ("East Asia", "Asia Pacific"),
    "eastus": ("East US", "United States"),
    "eastus2": ("East US 2", "United States"),
    "francecentral": ("France Central", "France"),
    "francesouth": ("France South", "France"),
    "germanynorth": ("Germany North", "Germany"),
    "germanywestcentral": ("Germany West Central", "Germany"),
    "indiasouthcentral": ("India South Central", "India"),
    "indonesiacentral": ("Indonesia Central", "Indonesia"),
    "israelcentral": ("Israel Central", "Israel"),
    "italynorth": ("Italy North", "Italy"),
    "japaneast": ("Japan East", "Japan"),
    "japanwest": ("Japan West", "Japan"),
    "koreacentral": ("Korea Central", "Korea"),
    "koreasouth": ("Korea South", "Korea"),
    "malaysiawest": ("Malaysia West", "Malaysia"),
    "mexicocentral": ("Mexico Central", "Mexico"),
    "newzealandnorth": ("New Zealand North", "New Zealand"),
    "northcentralus": ("North Central US", "United States"),
    "northeurope": ("North Europe", "Europe"),
    "norwayeast": ("Norway East", "Norway"),
    "norwaywest": ("Norway West", "Norway"),
    "polandcentral": ("Poland Central", "Poland"),
    "qatarcentral": ("Qatar Central", "Qatar"),
    "saudiarabiaeast": ("Saudi Arabia East", "Saudi Arabia"),
    "southafricanorth": ("South Africa North", "South Africa"),
    "southafricawest": ("South Africa West", "South Africa"),
    "southcentralus": ("South Central US", "United States"),
    "southindia": ("South India", "India"),
    "southeastasia": ("Southeast Asia", "Asia Pacific"),
    "southeastasia3": ("Southeast Asia 3", "Malaysia"),
    "spaincentral": ("Spain Central", "Spain"),
    "swedencentral": ("Sweden Central", "Sweden"),
    "switzerlandnorth": ("Switzerland North", "Switzerland"),
    "switzerlandwest": ("Switzerland West", "Switzerland"),
    "taiwannorth": ("Taiwan North", "Taiwan"),
    "uaecentral": ("UAE Central", "UAE"),
    "uaenorth": ("UAE North", "UAE"),
    "uksouth": ("UK South", "United Kingdom"),
    "ukwest": ("UK West", "United Kingdom"),
    "westcentralus": ("West Central US", "United States"),
    "westeurope": ("West Europe", "Europe"),
    "westindia": ("West India", "India"),
    "westus": ("West US", "United States"),
    "westus2": ("West US 2", "United States"),
    "westus3": ("West US 3", "United States"),
}

#: The regions Foundry's data zone tables place in each data zone ("Region availability
#: for Foundry Models sold by Azure", Data Zone Standard). A data zone deployment in a
#: region not listed here is refused; Microsoft may add regions to a zone.
AZURE_DATA_ZONES: dict[str, frozenset[str]] = {
    "us": frozenset(
        {"centralus", "eastus", "eastus2", "northcentralus", "southcentralus", "westus", "westus3"}
    ),
    "eu": frozenset(
        {
            "francecentral",
            "germanywestcentral",
            "italynorth",
            "polandcentral",
            "spaincentral",
            "swedencentral",
            "westeurope",
        }
    ),
    "apac": frozenset(
        {"australiaeast", "japaneast", "japanwest", "koreacentral", "southeastasia", "southindia"}
    ),
}
_ZONE_NAMES = {"us": "United States", "eu": "European Union", "apac": "Asia Pacific"}


def _slug(words: str) -> str:
    return words.lower().replace(" ", "-")


def azure_code(region: str) -> str:
    return f"azure:{region}"


def openai_host(code: str) -> str:
    return f"https://{code}.api.openai.com/v1"


XAI_US_BASE_URL = "https://us.api.x.ai/v1"


class FoundryProcessing(SettingsModel):
    """A Foundry deployment type and the resource's Azure region, declared for every
    deployment a connection serves."""

    type: Literal["global", "data_zone", "standard"]
    azure_region: str = Field(min_length=1, max_length=64)

    @field_validator("azure_region")
    @classmethod
    def _known(cls, value: str) -> str:
        if value not in AZURE_REGIONS:
            raise ValueError("azure_region is not an Azure public cloud region")
        return value

    @model_validator(mode="after")
    def _zoned(self) -> Self:
        if self.type == "data_zone" and self.zone is None:
            raise ValueError("azure_region is in no documented Foundry data zone")
        return self

    @property
    def zone(self) -> str | None:
        for zone, regions in AZURE_DATA_ZONES.items():
            if self.azure_region in regions:
                return zone
        return None

    @property
    def invocation(self) -> str:
        return azure_code(self.azure_region)

    def residency(self) -> Residency:
        if self.type == "global":
            return Residency(ResidencyScope.GLOBAL)
        if self.type == "data_zone":
            bound = f"azure:zone-{self.zone}"
        else:
            bound = f"azure:geo-{_slug(AZURE_REGIONS[self.azure_region][1])}"
        return Residency(ResidencyScope.CROSS_REGION, frozenset({self.invocation, bound}))

    def choice(self) -> RegionChoice:
        residency = self.residency()
        if residency.scope is ResidencyScope.GLOBAL:
            return RegionChoice(regions=frozenset({self.invocation}), allow_global=True)
        return RegionChoice(regions=residency.regions)


def foundry_listing(kind: str) -> RegionListing:
    """The Azure regions a Foundry resource may be in, and the data zones."""
    regions = [
        OfferedRegion(azure_code(code), name, invocation=True, scope=ResidencyScope.CROSS_REGION)
        for code, (name, _) in sorted(AZURE_REGIONS.items())
    ]
    regions += [
        OfferedRegion(
            f"azure:zone-{zone}",
            f"Data Zone {_ZONE_NAMES[zone]}",
            invocation=False,
            scope=ResidencyScope.CROSS_REGION,
            members=frozenset(azure_code(code) for code in members),
        )
        for zone, members in AZURE_DATA_ZONES.items()
    ]
    return RegionListing(
        kind=kind,
        service="foundry",
        regions=tuple(regions),
        global_option=FOUNDRY_GLOBAL_OPTION,
        documentation=FOUNDRY_DEPLOYMENT_DOCUMENTATION,
        as_of=REGIONS_AS_OF,
    )


def _openai_residency(code: str, model_id: str) -> Residency:
    if code not in OPENAI_PROCESSING_REGIONS:
        return Residency(ResidencyScope.GLOBAL)
    only = OPENAI_PROCESSING_MODELS.get(code)
    if only is not None and not any(
        model_id == model or model_id.startswith(f"{model}-") for model in only
    ):
        return Residency(ResidencyScope.GLOBAL)
    return Residency(ResidencyScope.CROSS_REGION, frozenset({f"openai:{code}"}))


def _regional_host(data: Any, chosen: str, regional: str, default: str) -> Any:
    """``data`` with ``base_url`` set to the regional host the choice picks; a custom
    ``base_url`` beside a choice is refused."""
    if not isinstance(data, dict) or data.get(chosen) is None:
        return data
    given = data.get("base_url")
    if given not in (None, default, regional):
        raise ValueError(f"{chosen} picks the regional host, so base_url must be left unset")
    return {**data, "base_url": regional}


class OpenAIService(SettingsModel):
    """The OpenAI API. ``organization`` and ``project`` are sent only when set here."""

    name: Literal["openai"] = "openai"
    api_key: SecretRef
    base_url: HttpsUrl = OPENAI_BASE_URL
    organization: str | None = Field(default=None, min_length=1, max_length=256)
    project: str | None = Field(default=None, min_length=1, max_length=256)
    #: A data residency region: the project's regional host is called.
    data_residency: Literal["us", "eu", "au", "ca", "jp", "in", "sg", "kr", "gb", "ae"] | None = (
        None
    )

    @model_validator(mode="before")
    @classmethod
    def _host(cls, data: Any) -> Any:
        code = data.get("data_residency") if isinstance(data, dict) else None
        if code not in OPENAI_RESIDENCY_REGIONS:
            return data  # no choice, or one the field itself refuses
        return _regional_host(data, "data_residency", openai_host(code), OPENAI_BASE_URL)


class XAIService(SettingsModel):
    """The X.AI API."""

    name: Literal["xai"] = "xai"
    api_key: SecretRef
    base_url: HttpsUrl = XAI_BASE_URL
    #: ``us``: the US host, which keeps processing in the US.
    endpoint_region: Literal["us"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _host(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("endpoint_region") != "us":
            return data
        return _regional_host(data, "endpoint_region", XAI_US_BASE_URL, XAI_BASE_URL)


class FoundryApiKey(SettingsModel):
    mode: Literal["api_key"] = "api_key"
    api_key: SecretRef


class FoundryEntraId(SettingsModel):
    """Microsoft Entra ID through ``azure.identity.aio.DefaultAzureCredential``.

    The credential chain is azure-identity's (environment, workload identity, managed
    identity, developer sign-ins). ``managed_identity_client_id`` picks a user-assigned
    managed identity.
    """

    mode: Literal["entra_id"] = "entra_id"
    scope: HttpsUrl = FOUNDRY_SCOPE
    managed_identity_client_id: str | None = Field(default=None, min_length=1, max_length=128)


class FoundryService(SettingsModel):
    """Microsoft Foundry's OpenAI v1 endpoint, with an API key or Entra ID."""

    name: Literal["foundry"] = "foundry"
    endpoint: FoundryEndpoint
    auth: FoundryApiKey | FoundryEntraId = Field(discriminator="mode")
    catalogue_provider: Literal["openai", "xai"] = "openai"
    #: The deployment type and Azure region of every deployment this connection serves.
    deployment: FoundryProcessing | None = None


Service = Annotated[OpenAIService | XAIService | FoundryService, Field(discriminator="name")]


class OpenAICompatSettings(ProviderSettings):
    """Settings for one configured OpenAI-compatible provider."""

    kind: Literal["openai_compat"] = "openai_compat"
    service: Service
    #: The SDK's own retries for connection errors, 408, 409, 429 and 5xx.
    max_retries: int = Field(default=2, ge=0, le=10)

    @property
    def catalogue_provider(self) -> str:
        if isinstance(self.service, FoundryService):
            return self.service.catalogue_provider
        return self.service.name

    @property
    def _invocation(self) -> str | None:
        """The Region code the service's choice calls, or ``None`` with no choice."""
        service = self.service
        if isinstance(service, OpenAIService) and service.data_residency is not None:
            return f"openai:{service.data_residency}"
        if isinstance(service, XAIService) and service.endpoint_region is not None:
            return f"xai:{service.endpoint_region}"
        if isinstance(service, FoundryService) and service.deployment is not None:
            return service.deployment.invocation
        return None

    @model_validator(mode="after")
    def _region_from_choice(self) -> Self:
        chosen = self._invocation
        if chosen is None:
            return self
        if self.region is not None and self.region != chosen:
            raise FieldsRefused("region is set by the service's choice; leave it unset", ["region"])
        object.__setattr__(self, "region", chosen)  # frozen: set once, while validating
        return self

    def region_choice(self) -> RegionChoice | None:
        service = self.service
        if isinstance(service, FoundryService) and service.deployment is not None:
            return service.deployment.choice()
        if isinstance(service, OpenAIService):
            code = service.data_residency
            if code is None or code not in OPENAI_PROCESSING_REGIONS:
                return None  # no choice, or storage only: processing is not chosen
            return RegionChoice(regions=frozenset({f"openai:{code}"}))
        if isinstance(service, XAIService) and service.endpoint_region is not None:
            return RegionChoice(regions=frozenset({f"xai:{service.endpoint_region}"}))
        return None

    def declared_residency(self, model_id: str) -> Residency | None:
        service = self.service
        if isinstance(service, OpenAIService) and service.data_residency is not None:
            return _openai_residency(service.data_residency, model_id)
        if isinstance(service, XAIService) and service.endpoint_region is not None:
            return Residency(ResidencyScope.CROSS_REGION, frozenset({"xai:us"}))
        if isinstance(service, FoundryService) and service.deployment is not None:
            return service.deployment.residency()
        return None

    @classmethod
    def offered_regions(cls) -> tuple[RegionListing, ...]:
        openai = RegionListing(
            kind="openai_compat",
            service="openai",
            regions=tuple(
                OfferedRegion(
                    f"openai:{code}",
                    name,
                    invocation=True,
                    scope=(
                        ResidencyScope.CROSS_REGION
                        if code in OPENAI_PROCESSING_REGIONS
                        else ResidencyScope.GLOBAL
                    ),
                    stores_only=code not in OPENAI_PROCESSING_REGIONS,
                )
                for code, name in OPENAI_RESIDENCY_REGIONS.items()
            ),
            global_option=OPENAI_GLOBAL_OPTION,
            documentation=OPENAI_RESIDENCY_DOCUMENTATION,
            as_of=REGIONS_AS_OF,
        )
        xai = RegionListing(
            kind="openai_compat",
            service="xai",
            regions=(
                OfferedRegion(
                    "xai:us", "United States", invocation=True, scope=ResidencyScope.CROSS_REGION
                ),
            ),
            global_option=XAI_GLOBAL_OPTION,
            documentation=XAI_REGIONS_DOCUMENTATION,
            as_of=REGIONS_AS_OF,
        )
        return (openai, xai, foundry_listing("openai_compat"))
