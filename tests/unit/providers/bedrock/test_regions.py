"""Bedrock and bedrock-mantle Regions: the invocation Region, the processing choice, the
Region listing, and an old single-Region setting that keeps its meaning.

Every listing is a synthetic recording; no client reaches AWS.
"""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ResidencyRefused, SettingsRefused
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import RegionChoice, ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.bedrock import BedrockFactory, BedrockSettings
from conduit.providers.bedrock.factory import BedrockClients
from conduit.providers.bedrock.settings import BEDROCK_REGIONS
from conduit.providers.bedrock_mantle import BedrockMantleFactory, BedrockMantleSettings
from conduit.providers.bedrock_mantle.settings import MANTLE_REGIONS
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

from .test_discovery import FakeControl

SYDNEY, MELBOURNE = recordings.REGION, recordings.OTHER_REGION
NO_SECRETS = LiteralSecretResolver({})


def _bedrock(**fields: Any) -> BedrockSettings:
    return BedrockSettings.model_validate({"region": SYDNEY, **fields})


def _refused(model: type[Any], mapping: dict[str, Any]) -> list[str]:
    registry = ProviderRegistry(
        secrets=NO_SECRETS, factories=[BedrockFactory(), BedrockMantleFactory()]
    )
    with pytest.raises(SettingsRefused) as refused:
        registry.validate("connection", mapping)
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    return list(refused.value.reasons)


class TestBedrockInvocationRegion:
    def test_an_old_single_region_setting_keeps_its_meaning(self) -> None:
        settings = BedrockSettings.model_validate({"kind": "bedrock", "region": "eu-west-2"})
        assert settings.region == "eu-west-2"
        assert settings.processing is None
        assert settings.region_choice() is None
        assert settings.declared_residency("anthropic.claude-sonnet-4-6") is None

    @pytest.mark.parametrize("region", ["us-east-1", "ap-southeast-4", "us-gov-west-1"])
    def test_a_region_bedrock_runtime_serves_is_accepted(self, region: str) -> None:
        assert _bedrock(region=region).region == region

    @pytest.mark.parametrize("region", ["cn-north-1", "xx-nowhere-1", "eu-east-9"])
    def test_a_region_bedrock_runtime_does_not_serve_is_refused(self, region: str) -> None:
        reasons = _refused(BedrockSettings, {"kind": "bedrock", "region": region})
        assert any("Bedrock runtime" in reason for reason in reasons)

    def test_the_documented_regions_are_listed(self) -> None:
        assert {"ap-southeast-2", "ap-southeast-4", "eu-west-2", "us-east-1"} <= set(
            BEDROCK_REGIONS
        )
        assert "cn-north-1" not in BEDROCK_REGIONS


class TestBedrockProcessingChoice:
    def test_a_choice_inside_bedrock_regions_is_accepted(self) -> None:
        settings = _bedrock(processing={"regions": [SYDNEY, MELBOURNE]})
        assert settings.region_choice() == RegionChoice(regions=frozenset({SYDNEY, MELBOURNE}))

    def test_a_choice_must_include_the_invocation_region(self) -> None:
        reasons = _refused(
            BedrockSettings,
            {"kind": "bedrock", "region": SYDNEY, "processing": {"regions": [MELBOURNE]}},
        )
        assert any("must include region" in reason for reason in reasons)

    def test_a_choice_naming_a_region_bedrock_does_not_serve_is_refused(self) -> None:
        reasons = _refused(
            BedrockSettings,
            {
                "kind": "bedrock",
                "region": SYDNEY,
                "processing": {"regions": [SYDNEY, "openai:eu"]},
            },
        )
        assert any("does not serve" in reason for reason in reasons)

    def test_allow_global_is_kept(self) -> None:
        settings = _bedrock(processing={"regions": [SYDNEY], "allow_global": True})
        choice = settings.region_choice()
        assert choice is not None and choice.allow_global


def _replay(name: str) -> Replay:
    return Replay(load_recording(recordings.ROOT / name))


async def _discover(
    settings: dict[str, Any], residency: ResidencyPolicy | None = None
) -> DiscoveryResult:
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}), factories=[BedrockFactory()], residency=residency
    )
    replay = _replay("bedrock/discovery-listing.json")
    clients = BedrockClients(runtime=object(), control=FakeControl(replay))
    result = await registry.discover(
        "bedrock_sydney", {"kind": "bedrock", **settings}, client=clients
    )
    replay.assert_exhausted()
    return result


def _outcome(result: DiscoveryResult, model_id: str) -> DiscoveryOutcome:
    found = result.get(model_id)
    assert found is not None, model_id
    return found.outcome


class TestDiscoveryFollowsTheChoice:
    async def test_without_a_choice_the_listing_is_unchanged(self) -> None:
        result = await _discover({"region": SYDNEY})
        assert result.choice is None
        assert _outcome(result, "au.anthropic.claude-opus-5-5") is DiscoveryOutcome.ALLOWED
        assert _outcome(result, "global.anthropic.claude-opus-5-5") is DiscoveryOutcome.ALLOWED

    async def test_a_geographic_profile_outside_the_choice_is_refused(self) -> None:
        result = await _discover({"region": SYDNEY, "processing": {"regions": [SYDNEY]}})
        assert _outcome(result, "anthropic.claude-opus-5-5") is DiscoveryOutcome.ALLOWED
        assert _outcome(result, "au.anthropic.claude-opus-5-5") is DiscoveryOutcome.REFUSED
        refused = result.get("au.anthropic.claude-opus-5-5")
        assert refused is not None and MELBOURNE in (refused.reason or "")

    async def test_a_geographic_profile_inside_the_choice_is_allowed(self) -> None:
        result = await _discover({"region": SYDNEY, "processing": {"regions": [SYDNEY, MELBOURNE]}})
        assert _outcome(result, "au.anthropic.claude-opus-5-5") is DiscoveryOutcome.ALLOWED
        assert _outcome(result, "global.anthropic.claude-opus-5-5") is DiscoveryOutcome.REFUSED

    async def test_a_global_profile_needs_allow_global(self) -> None:
        result = await _discover(
            {"region": SYDNEY, "processing": {"regions": [SYDNEY], "allow_global": True}}
        )
        assert _outcome(result, "global.anthropic.claude-opus-5-5") is DiscoveryOutcome.ALLOWED
        assert _outcome(result, "au.anthropic.claude-opus-5-5") is DiscoveryOutcome.REFUSED

    async def test_the_deployment_policy_still_bounds_the_choice(self) -> None:
        sydney_only = ResidencyPolicy(regions=frozenset({SYDNEY}))
        result = await _discover(
            {"region": SYDNEY, "processing": {"regions": [SYDNEY, MELBOURNE]}}, sydney_only
        )
        assert _outcome(result, "au.anthropic.claude-opus-5-5") is DiscoveryOutcome.REFUSED


class TestBedrockBuild:
    async def test_build_refuses_a_profile_outside_the_choice(self) -> None:
        result = await _discover({"region": SYDNEY, "processing": {"regions": [SYDNEY]}})
        registry = ProviderRegistry(secrets=NO_SECRETS, factories=[BedrockFactory()])
        settings = {"kind": "bedrock", "region": SYDNEY, "processing": {"regions": [SYDNEY]}}
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await registry.build(
                "bedrock_sydney",
                settings,
                client=BedrockClients(runtime=object(), control=object()),
                assigned={"triage": "au.anthropic.claude-opus-5-5"},
                catalogue=result.listed_catalogue(registry.catalogue),
            )


class TestBedrockListing:
    def test_one_listing_of_bedrock_runtime_regions(self) -> None:
        [listing] = ProviderRegistry(secrets=NO_SECRETS, factories=[BedrockFactory()]).regions(
            "bedrock"
        )
        codes = [region.code for region in listing.regions]
        assert codes == sorted(BEDROCK_REGIONS)
        assert all(region.invocation for region in listing.regions)
        assert {region.scope for region in listing.regions} == {ResidencyScope.IN_REGION}
        assert listing.global_option is not None and "commercial" in listing.global_option
        assert listing.documentation.startswith("https://docs.aws.amazon.com/bedrock/")
        sydney = next(r for r in listing.regions if r.code == SYDNEY)
        assert sydney.name == "Asia Pacific (Sydney)"


class TestMantleRegions:
    def test_an_old_mantle_setting_keeps_its_meaning(self) -> None:
        settings = BedrockMantleSettings.model_validate({"region": "us-east-1"})
        assert settings.region == "us-east-1"
        assert settings.region_choice() is None

    @pytest.mark.parametrize("region", ["ap-southeast-1", "us-west-1", "cn-north-1"])
    def test_a_region_without_the_mantle_endpoint_is_refused(self, region: str) -> None:
        reasons = _refused(BedrockMantleSettings, {"kind": "bedrock_mantle", "region": region})
        assert any("mantle endpoint" in reason for reason in reasons)

    def test_the_mantle_regions_are_a_subset_of_bedrock(self) -> None:
        assert set(BEDROCK_REGIONS) >= MANTLE_REGIONS
        assert SYDNEY in MANTLE_REGIONS and "ap-southeast-1" not in MANTLE_REGIONS
        # Listed by the model cards and Anthropic's Bedrock page, though not by AWS's
        # endpoint table: kept, so a served Region is never refused.
        assert MELBOURNE in MANTLE_REGIONS

    def test_the_mantle_listing_is_in_region_only(self) -> None:
        [listing] = ProviderRegistry(
            secrets=NO_SECRETS, factories=[BedrockMantleFactory()]
        ).regions("bedrock_mantle")
        assert {r.code for r in listing.regions} == MANTLE_REGIONS
        assert {r.scope for r in listing.regions} == {ResidencyScope.IN_REGION}
        assert listing.global_option is None
