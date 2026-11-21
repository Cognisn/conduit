"""The catalogue pass's Bedrock and bedrock-mantle routes, proven offline (default run).

Routes come from each configured Region's listing, held to the run's Region choice:
every allowed, catalogued, listed id is a route, through that id. Nothing is taken from
Regions in catalogue entries. Every listing here is synthetic: built from discovered
models, or read by the real Bedrock adapter from a synthetic recording through a fake
control-plane client. No client reaches AWS.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.discovery import (
    CatalogueDrift,
    DiscoveredModel,
    DiscoveryResult,
    DriftReason,
    join_discovered,
)
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.bedrock import BedrockFactory, BedrockSettings
from conduit.providers.bedrock.factory import BedrockClients
from conduit.testing.replay import Replay, load_recording
from tests.live.needs import LiveEnv
from tests.live.report import REPORT, Verdict
from tests.live.targets import Target, bedrock_settings, listed_routes, processing_choice
from tests.live.test_catalogue import connections, record_destinations, record_drift
from tests.unit.providers import discovery_recordings as recordings
from tests.unit.providers.bedrock.test_discovery import FakeControl

SYDNEY, MELBOURNE = recordings.REGION, recordings.OTHER_REGION
CHAIN = {"CONDUIT_LIVE_BEDROCK_CREDENTIAL_CHAIN": "1"}


def _env(values: dict[str, str]) -> LiveEnv:
    return LiveEnv(values)


@pytest.fixture(autouse=True)
def _report_left_as_found() -> Iterator[None]:
    kept = len(REPORT.lines)
    yield
    del REPORT.lines[kept:]


def _entry(model_id: str, **fields: Any) -> ModelEntry:
    return ModelEntry.model_validate(
        {
            "model_id": model_id,
            "provider": "bedrock",
            "context_window": 200_000,
            "output_limit": 64_000,
            **fields,
        }
    )


#: Capabilities and limits only, keyed by base model, plus an entry that still carries
#: Regions (from before wave 13): those Regions must never make a route.
CATALOGUE = Catalogue(
    [
        _entry("vendor.model-a"),
        _entry("vendor.model-b"),
        _entry(
            "geo.vendor.model-stale",
            residency_scope="cross_region",
            regions=frozenset({SYDNEY, MELBOURNE}),
        ),
    ],
    version="routes-1",
)


def _profile(model_id: str, base: str, *regions: str) -> DiscoveredModel:
    return DiscoveredModel(
        model_id,
        scope=ResidencyScope.CROSS_REGION,
        regions=frozenset(regions),
        base_model_id=base,
    )


LISTED = (
    DiscoveredModel(
        "vendor.model-a",
        scope=ResidencyScope.IN_REGION,
        regions=frozenset({SYDNEY}),
        base_model_id="vendor.model-a",
    ),
    _profile("au.vendor.model-a", "vendor.model-a", SYDNEY, MELBOURNE),
    _profile("apac.vendor.model-a", "vendor.model-a", SYDNEY, "ap-northeast-1"),
    DiscoveredModel(
        "global.vendor.model-b", scope=ResidencyScope.GLOBAL, base_model_id="vendor.model-b"
    ),
    _profile("au.vendor.model-new", "vendor.model-new", SYDNEY, MELBOURNE),
)


def _listing(
    choice: dict[str, Any] | None, listed: tuple[DiscoveredModel, ...] = LISTED
) -> DiscoveryResult:
    settings = BedrockSettings.model_validate(
        {"region": SYDNEY, **({"processing": choice} if choice else {})}
    )
    models, drift = join_discovered(
        listed,
        CATALOGUE,
        catalogue_provider="bedrock",
        region=SYDNEY,
        residency=ResidencyPolicy(),
        provider="live",
        connection=settings,
    )
    return DiscoveryResult(
        provider="live",
        kind="bedrock",
        catalogue_provider="bedrock",
        region=SYDNEY,
        supported=True,
        models=models,
        drift=drift,
        choice=settings.processing,
    )


def _ids(listing: DiscoveryResult) -> list[str]:
    return [route.target.model for route in listed_routes("bedrock", {}, SYDNEY, listing)]


class TestListedRoutes:
    def test_without_a_choice_every_catalogued_listed_id_is_a_route(self) -> None:
        assert _ids(_listing(None)) == [
            "vendor.model-a",
            "au.vendor.model-a",
            "apac.vendor.model-a",
            "global.vendor.model-b",
        ]

    def test_the_choice_keeps_only_routes_inside_it(self) -> None:
        assert _ids(_listing({"regions": [SYDNEY, MELBOURNE]})) == [
            "vendor.model-a",
            "au.vendor.model-a",
        ]

    def test_allow_global_adds_the_global_profile(self) -> None:
        listing = _listing({"regions": [SYDNEY], "allow_global": True})
        assert _ids(listing) == ["vendor.model-a", "global.vendor.model-b"]

    def test_an_uncatalogued_listed_model_is_not_routed(self) -> None:
        assert "au.vendor.model-new" not in _ids(_listing(None))

    def test_catalogue_entry_regions_never_make_a_route(self) -> None:
        assert "geo.vendor.model-stale" not in _ids(_listing(None))

    def test_nothing_listed_gives_no_route(self) -> None:
        assert _ids(_listing(None, ())) == []

    def test_a_route_carries_the_listed_entry_and_residency(self) -> None:
        [_, profile] = listed_routes(
            "bedrock x", {"kind": "bedrock"}, SYDNEY, _listing({"regions": [SYDNEY, MELBOURNE]})
        )
        assert profile.target.subject == f"bedrock x {SYDNEY} au.vendor.model-a"
        assert (profile.scope, profile.regions) == (
            ResidencyScope.CROSS_REGION,
            frozenset({SYDNEY, MELBOURNE}),
        )
        assert profile.entry.output_limit == 64_000


class TestTheRealAdapter:
    async def test_routes_from_a_recorded_listing_follow_the_choice(self) -> None:
        replay = Replay(load_recording(recordings.ROOT / "bedrock/discovery-listing.json"))
        settings = {"kind": "bedrock", "region": SYDNEY, "processing": {"regions": [SYDNEY]}}
        registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[BedrockFactory()])
        clients = BedrockClients(runtime=object(), control=FakeControl(replay))
        listing = await registry.discover("live", settings, client=clients)
        ids = [route.target.model for route in listed_routes("bedrock", settings, SYDNEY, listing)]
        assert "anthropic.claude-opus-5-5" in ids
        assert "au.anthropic.claude-opus-5-5" not in ids  # reaches ap-southeast-4
        assert "global.anthropic.claude-opus-5-5" not in ids


class TestTheRunsChoice:
    def test_no_regions_chosen_is_no_choice(self) -> None:
        assert processing_choice(_env({})) is None

    def test_the_chosen_regions_and_global(self) -> None:
        environ = {
            "CONDUIT_LIVE_BEDROCK_PROCESSING_REGIONS": f"{SYDNEY}, {MELBOURNE}",
            "CONDUIT_LIVE_BEDROCK_ALLOW_GLOBAL": "1",
        }
        assert processing_choice(_env(environ)) == {
            "regions": [SYDNEY, MELBOURNE],
            "allow_global": True,
        }

    def test_the_settings_carry_the_choice(self) -> None:
        choice = {"regions": [SYDNEY]}
        settings = bedrock_settings(SYDNEY, "credential_chain", _env(CHAIN), processing=choice)
        assert settings["processing"] == choice
        assert "processing" not in bedrock_settings(SYDNEY, "credential_chain", _env(CHAIN))

    def test_each_configured_region_is_a_connection(self) -> None:
        environ = {
            **CHAIN,
            "CONDUIT_LIVE_BEDROCK_REGIONS": f"{SYDNEY},{MELBOURNE}",
            "CONDUIT_LIVE_BEDROCK_PROCESSING_REGIONS": f"{SYDNEY},{MELBOURNE}",
        }
        found = connections("bedrock", _env(environ))
        assert [region for _, _, region in found] == [SYDNEY, MELBOURNE]
        assert all(
            settings["processing"]["regions"] == [SYDNEY, MELBOURNE] for _, settings, _ in found
        )

    def test_a_mantle_connection_per_configured_region(self) -> None:
        environ = {**CHAIN, "CONDUIT_LIVE_BEDROCK_MANTLE_REGIONS": "us-east-1,ap-southeast-2"}
        found = connections("bedrock_mantle", _env(environ))
        assert [(s["kind"], region) for _, s, region in found] == [
            ("bedrock_mantle", "us-east-1"),
            ("bedrock_mantle", "ap-southeast-2"),
        ]


def _lines(subject: str) -> list[tuple[str, Verdict, str]]:
    return [
        (line.check, line.verdict, line.detail) for line in REPORT.lines if line.subject == subject
    ]


class TestRecording:
    def test_drift_is_recorded_as_findings(self) -> None:
        record_drift("bedrock x", _listing(None))
        lines = _lines("bedrock x")
        assert (
            "uncatalogued au.vendor.model-new",
            Verdict.FINDING,
            "listed by the provider, with no catalogue entry",
        ) in lines
        assert any(check == "unlisted geo.vendor.model-stale" for check, _, _ in lines)
        assert all(verdict is Verdict.FINDING for _, verdict, _ in lines)

    def test_a_listing_without_drift_records_nothing(self) -> None:
        empty = DiscoveryResult("live", "bedrock", "bedrock", SYDNEY, True)
        record_drift("bedrock y", empty)
        assert _lines("bedrock y") == []

    def test_profile_destinations_are_recorded(self) -> None:
        routes = listed_routes("bedrock", {}, SYDNEY, _listing(None))
        record_destinations("bedrock z", routes)
        lines = {check: detail for check, _, detail in _lines("bedrock z")}
        assert lines["destinations au.vendor.model-a"] == f"{SYDNEY}, {MELBOURNE}"
        assert lines["widest geographic profile"] == "2 destination Regions"
        assert "destinations vendor.model-a" not in lines

    def test_drift_details_are_fixed_phrases(self) -> None:
        drift = CatalogueDrift(
            "m", DriftReason.LIMITS_DIFFER, "output limit 1 listed, 2 in the catalogue"
        )
        listing = DiscoveryResult("live", "bedrock", "bedrock", SYDNEY, True, drift=(drift,))
        record_drift("bedrock w", listing)
        assert _lines("bedrock w") == [
            ("limits_differ m", Verdict.FINDING, "output limit 1 listed, 2 in the catalogue")
        ]


class TestOtherProvidersListing:
    """The other providers' entries are verified only when the provider lists them."""

    async def test_an_entry_the_listing_names_is_verified(self) -> None:
        from tests.live import test_catalogue

        target = Target("offline listed", {"kind": "stub"}, "vendor.model-a")
        test_catalogue._LISTINGS[target.label] = _listing(None)
        try:
            assert await test_catalogue.listed_by(target, _env({}))
            missing = Target("offline listed", {"kind": "stub"}, "vendor.model-gone")
            assert not await test_catalogue.listed_by(missing, _env({}))
        finally:
            del test_catalogue._LISTINGS[target.label]

    async def test_a_provider_that_cannot_list_verifies_every_entry(self) -> None:
        from tests.live import test_catalogue

        target = Target("offline unlisted", {"kind": "stub"}, "anything")
        test_catalogue._LISTINGS[target.label] = DiscoveryResult(
            "live", "stub", "stub", None, False, reason="no listing"
        )
        try:
            assert await test_catalogue.listed_by(target, _env({}))
        finally:
            del test_catalogue._LISTINGS[target.label]
