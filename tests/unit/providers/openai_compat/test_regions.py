"""OpenAI, X.AI and Foundry Regions: each service's processing choice, the residency its
endpoint declares, the host it is called on, and the Region listings.

OpenAI's regional hosts process in-region only in some regions and only for some models
there; X.AI's US host keeps processing in the US; a Foundry deployment's processing is
its deployment type. Everything here is synthetic settings; nothing is called.
"""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.catalogue import ModelEntry
from conduit.core.errors import ResidencyRefused, SettingsRefused
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import RegionChoice, Residency, ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.openai_compat import OpenAICompatFactory
from conduit.providers.openai_compat.settings import (
    AZURE_DATA_ZONES,
    AZURE_REGIONS,
    OPENAI_BASE_URL,
    OPENAI_PROCESSING_REGIONS,
    OPENAI_RESIDENCY_REGIONS,
    XAI_BASE_URL,
    FoundryProcessing,
    FoundryService,
    OpenAICompatSettings,
)

KEY = "secret://openai/key"
NO_SECRETS = LiteralSecretResolver({})
GLOBAL = Residency(ResidencyScope.GLOBAL)


def _openai(**service: Any) -> OpenAICompatSettings:
    return OpenAICompatSettings.model_validate(
        {"service": {"name": "openai", "api_key": KEY, **service}}
    )


def _xai(**service: Any) -> OpenAICompatSettings:
    return OpenAICompatSettings.model_validate(
        {"service": {"name": "xai", "api_key": "secret://xai/key", **service}}
    )


def _foundry(deployment: dict[str, Any] | None, **fields: Any) -> OpenAICompatSettings:
    service: dict[str, Any] = {
        "name": "foundry",
        "endpoint": "https://example-resource.openai.azure.com/openai/v1/",
        "auth": {"mode": "api_key", "api_key": "secret://foundry/key"},
    }
    if deployment is not None:
        service["deployment"] = deployment
    return OpenAICompatSettings.model_validate({"service": service, **fields})


def _base(settings: OpenAICompatSettings) -> str:
    service = settings.service
    assert not isinstance(service, FoundryService)
    return service.base_url


def _refused(mapping: dict[str, Any]) -> list[str]:
    registry = ProviderRegistry(secrets=NO_SECRETS, factories=[OpenAICompatFactory()])
    with pytest.raises(SettingsRefused) as refused:
        registry.validate("connection", {"kind": "openai_compat", **mapping})
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    return list(refused.value.reasons)


class TestNoChoiceKeepsItsMeaning:
    @pytest.mark.parametrize("build", [_openai, _xai])
    def test_an_old_setting_declares_and_chooses_nothing(self, build: Any) -> None:
        settings = build()
        assert settings.region is None
        assert settings.region_choice() is None
        assert settings.declared_residency("gpt-6-sol") is None

    def test_the_default_hosts_are_unchanged(self) -> None:
        assert _base(_openai()) == OPENAI_BASE_URL
        assert _base(_xai()) == XAI_BASE_URL

    def test_a_foundry_service_without_a_deployment_declares_nothing(self) -> None:
        settings = _foundry(None)
        assert settings.region is None and settings.region_choice() is None


class TestOpenAIDataResidency:
    def test_the_documented_regions(self) -> None:
        assert set(OPENAI_RESIDENCY_REGIONS) == {
            "us",
            "eu",
            "au",
            "ca",
            "jp",
            "in",
            "sg",
            "kr",
            "gb",
            "ae",
        }
        assert {"us", "eu", "au", "jp", "in", "ae"} == OPENAI_PROCESSING_REGIONS

    def test_a_region_picks_the_regional_host(self) -> None:
        settings = _openai(data_residency="eu")
        assert _base(settings) == "https://eu.api.openai.com/v1"
        assert settings.region == "openai:eu"

    def test_a_processing_region_declares_and_chooses_its_geography(self) -> None:
        settings = _openai(data_residency="eu")
        eu = Residency(ResidencyScope.CROSS_REGION, frozenset({"openai:eu"}))
        assert settings.declared_residency("gpt-6-sol") == eu
        assert settings.region_choice() == RegionChoice(regions=frozenset({"openai:eu"}))

    @pytest.mark.parametrize("code", ["ca", "sg", "kr", "gb"])
    def test_a_storage_only_region_processes_anywhere(self, code: str) -> None:
        settings = _openai(data_residency=code)
        assert settings.declared_residency("gpt-6-sol") == GLOBAL
        assert settings.region_choice() is None  # stored there; processing not chosen

    @pytest.mark.parametrize("code", ["au", "jp", "in"])
    def test_a_luna_only_region_processes_only_luna_in_region(self, code: str) -> None:
        settings = _openai(data_residency=code)
        inside = Residency(ResidencyScope.CROSS_REGION, frozenset({f"openai:{code}"}))
        assert settings.declared_residency("gpt-6-luna") == inside
        assert settings.declared_residency("gpt-6-luna-2026-09-22") == inside
        assert settings.declared_residency("gpt-6-sol") == GLOBAL
        assert settings.declared_residency("gpt-6-lunar") == GLOBAL

    def test_a_luna_only_choice_refuses_the_other_models(self) -> None:
        settings = _openai(data_residency="au")
        entry = ModelEntry(
            model_id="gpt-6-sol",
            provider="openai",
            residency_scope=ResidencyScope.GLOBAL,
            context_window=128_000,
            output_limit=16_000,
        )
        with pytest.raises(ResidencyRefused, match="global processing is outside"):
            ResidencyPolicy().check(entry, settings.region, task="triage", provider=settings)

    def test_an_unknown_region_is_refused(self) -> None:
        _refused({"service": {"name": "openai", "api_key": KEY, "data_residency": "zz"}})

    def test_a_custom_base_url_with_a_region_is_refused(self) -> None:
        reasons = _refused(
            {
                "service": {
                    "name": "openai",
                    "api_key": KEY,
                    "data_residency": "eu",
                    "base_url": "https://gateway.example.com/v1",
                }
            }
        )
        assert any("base_url must be left unset" in reason for reason in reasons)

    def test_a_region_that_disagrees_is_refused(self) -> None:
        reasons = _refused(
            {
                "region": "openai:us",
                "service": {"name": "openai", "api_key": KEY, "data_residency": "eu"},
            }
        )
        assert any("region is set by the service's choice" in reason for reason in reasons)

    def test_the_settings_round_trip(self) -> None:
        settings = _openai(data_residency="eu")
        again = OpenAICompatSettings.model_validate(settings.model_dump())
        assert again == settings


class TestXAIEndpointRegion:
    def test_us_picks_the_us_host_and_declares_the_us(self) -> None:
        settings = _xai(endpoint_region="us")
        assert _base(settings) == "https://us.api.x.ai/v1"
        assert settings.region == "xai:us"
        us = Residency(ResidencyScope.CROSS_REGION, frozenset({"xai:us"}))
        assert settings.declared_residency("grok-5") == us
        assert settings.region_choice() == RegionChoice(regions=frozenset({"xai:us"}))

    def test_an_unknown_region_is_refused(self) -> None:
        _refused({"service": {"name": "xai", "api_key": KEY, "endpoint_region": "eu"}})

    def test_a_custom_base_url_with_a_region_is_refused(self) -> None:
        reasons = _refused(
            {
                "service": {
                    "name": "xai",
                    "api_key": KEY,
                    "endpoint_region": "us",
                    "base_url": "https://gateway.example.com/v1",
                }
            }
        )
        assert any("base_url must be left unset" in reason for reason in reasons)


class TestFoundryProcessing:
    def test_a_global_deployment_is_global(self) -> None:
        settings = _foundry({"type": "global", "azure_region": "swedencentral"})
        assert settings.region == "azure:swedencentral"
        assert settings.declared_residency("my-deployment") == GLOBAL
        assert settings.region_choice() == RegionChoice(
            regions=frozenset({"azure:swedencentral"}), allow_global=True
        )

    def test_a_data_zone_deployment_is_its_zone(self) -> None:
        settings = _foundry({"type": "data_zone", "azure_region": "swedencentral"})
        zone = frozenset({"azure:swedencentral", "azure:zone-eu"})
        assert settings.declared_residency("my-deployment") == Residency(
            ResidencyScope.CROSS_REGION, zone
        )
        assert settings.region_choice() == RegionChoice(regions=zone)

    def test_a_standard_deployment_is_its_geography(self) -> None:
        settings = _foundry({"type": "standard", "azure_region": "australiaeast"})
        geo = frozenset({"azure:australiaeast", "azure:geo-australia"})
        assert settings.declared_residency("my-deployment") == Residency(
            ResidencyScope.CROSS_REGION, geo
        )

    def test_an_unknown_azure_region_is_refused(self) -> None:
        with pytest.raises(ValueError):
            FoundryProcessing.model_validate({"type": "standard", "azure_region": "nowhere"})

    def test_a_data_zone_needs_a_region_in_a_documented_zone(self) -> None:
        assert "brazilsouth" in AZURE_REGIONS
        assert not any("brazilsouth" in zone for zone in AZURE_DATA_ZONES.values())
        with pytest.raises(ValueError):
            FoundryProcessing.model_validate({"type": "data_zone", "azure_region": "brazilsouth"})

    def test_an_unknown_type_is_refused(self) -> None:
        with pytest.raises(ValueError):
            FoundryProcessing.model_validate({"type": "regional", "azure_region": "eastus"})

    def test_a_deployment_is_only_for_foundry(self) -> None:
        _refused(
            {
                "service": {
                    "name": "openai",
                    "api_key": KEY,
                    "deployment": {"type": "global", "azure_region": "eastus"},
                }
            }
        )


class TestListings:
    def test_one_listing_per_service(self) -> None:
        listings = ProviderRegistry(secrets=NO_SECRETS, factories=[OpenAICompatFactory()]).regions(
            "openai_compat"
        )
        by_service = {listing.service: listing for listing in listings}
        assert set(by_service) == {"openai", "xai", "foundry"}
        openai = {r.code: r for r in by_service["openai"].regions}
        assert set(openai) == {f"openai:{code}" for code in OPENAI_RESIDENCY_REGIONS}
        assert openai["openai:ca"].stores_only and not openai["openai:eu"].stores_only
        assert [r.code for r in by_service["xai"].regions] == ["xai:us"]
        foundry = {r.code: r for r in by_service["foundry"].regions}
        assert foundry["azure:zone-eu"].members >= {"azure:swedencentral"}
        assert foundry["azure:swedencentral"].invocation
        assert all(listing.global_option for listing in listings)
        assert all(listing.documentation.startswith("https://") for listing in listings)
