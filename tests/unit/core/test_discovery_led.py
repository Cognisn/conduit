"""Discovery-led availability (C18): the provider's listing decides which models exist
and where a request is processed; the catalogue supplies only capabilities and limits.

Every listing here is synthetic. The four cases of catalogue drift against a live
listing are covered: a profile the catalogue mis-Regions, a model the provider lists
with no entry, an entry the provider no longer lists, and a provider with no Region
notion. Residency is allowed and refused on the Regions the listing reports, and a
listing that fails at start-up fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from conduit.core.catalogue import (
    Capabilities,
    Catalogue,
    CatalogueOverlay,
    ModelEntry,
    ModelKey,
)
from conduit.core.discovery import (
    FLOOR_CONTEXT_WINDOW,
    FLOOR_OUTPUT_LIMIT,
    CatalogueDrift,
    DiscoveredModel,
    DiscoveryOutcome,
    DiscoveryResult,
    DriftReason,
    ReportedCapabilities,
    join_discovered,
)
from conduit.core.errors import (
    AssignmentRefused,
    CapabilityRefused,
    ConfigurationError,
    ProviderError,
    ProviderFailure,
    ResidencyRefused,
)
from conduit.core.events import DiscoveryFinishedEvent, MemoryEventSink
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import AssignmentSettings, ProviderSettings
from conduit.core.tasks import CapabilityFloor, TaskClass, resolve_assignments
from tests.unit.core.fakes import FakeFactory
from tests.unit.core.test_discovery import ListingFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
HOME, AWAY, EXTRA = "xx-home-1", "xx-away-1", "xx-extra-1"
PROVIDER = {"kind": "fake", "api_key": "secret://fake/key", "region": HOME}


def _entry(model_id: str, **fields: Any) -> ModelEntry:
    base: dict[str, Any] = {
        "model_id": model_id,
        "provider": "fake",
        "context_window": 200_000,
        "output_limit": 64_000,
        "capabilities": {"tools": True, "vision": True, "prompt_caching": True},
    }
    base.update(fields)
    return ModelEntry.model_validate(base)


#: The default catalogue: capabilities and limits only, keyed by base model id.
DEFAULT = Catalogue(
    [
        _entry("vendor.model-a"),
        _entry("vendor.model-gone"),
        _entry("vendor.model-small", context_window=32_000, output_limit=4_000),
    ],
    version="led-1",
)

#: An overlay that declares a profile with a Region the provider never routes to.
MIS_REGIONED = CatalogueOverlay(
    name="product",
    add=(
        _entry(
            "geo.vendor.model-a",
            residency_scope="cross_region",
            regions=frozenset({HOME, AWAY, EXTRA}),
        ),
    ),
)


def _profile(model_id: str, base: str, *regions: str) -> DiscoveredModel:
    return DiscoveredModel(
        model_id,
        label="SYSTEM_DEFINED",
        scope=ResidencyScope.CROSS_REGION,
        regions=frozenset(regions),
        base_model_id=base,
    )


def _bare(model_id: str, region: str = HOME) -> DiscoveredModel:
    return DiscoveredModel(
        model_id,
        scope=ResidencyScope.IN_REGION,
        regions=frozenset({region}),
        base_model_id=model_id,
    )


#: The provider's listing in HOME: a geographic profile to HOME and AWAY, an active
#: profile with no entry, an in-Region model, and no listing of vendor.model-gone.
LISTING = (
    _profile("geo.vendor.model-a", "vendor.model-a", HOME, AWAY),
    _profile("geo.vendor.model-new", "vendor.model-new", HOME, AWAY),
    _bare("vendor.model-small"),
)

GEO_POLICY = ResidencyPolicy(
    scopes=frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION}),
    regions=frozenset({HOME, AWAY}),
)


def _join(
    listing: Sequence[DiscoveredModel] = LISTING,
    catalogue: Catalogue | None = None,
    residency: ResidencyPolicy | None = None,
) -> tuple[Any, tuple[CatalogueDrift, ...]]:
    models, drift = join_discovered(
        listing,
        catalogue if catalogue is not None else DEFAULT,
        catalogue_provider="fake",
        region=HOME,
        residency=residency or ResidencyPolicy(),
        provider="main",
    )
    return {model.model_id: model for model in models}, drift


def _reasons(drift: Sequence[CatalogueDrift]) -> dict[str, set[DriftReason]]:
    found: dict[str, set[DriftReason]] = {}
    for item in drift:
        found.setdefault(item.model_id, set()).add(item.reason)
    return found


class TestDiscoveredModel:
    def test_defaults_are_unknown_residency_and_no_reports(self) -> None:
        model = DiscoveredModel("vendor.model-a")
        assert model.scope is None and model.regions == frozenset()
        assert model.base_model_id is None and model.kind is None
        assert model.context_window is None and model.output_limit is None
        assert model.capabilities is None

    @pytest.mark.parametrize(
        ("scope", "regions"),
        [
            (ResidencyScope.LOCAL, {HOME}),
            (ResidencyScope.GLOBAL, {HOME}),
            (None, {HOME}),
            (ResidencyScope.IN_REGION, set()),
            (ResidencyScope.CROSS_REGION, set()),
            (ResidencyScope.IN_REGION, {HOME, AWAY}),
        ],
    )
    def test_scope_and_regions_must_agree(
        self, scope: ResidencyScope | None, regions: set[str]
    ) -> None:
        with pytest.raises(ValueError):
            DiscoveredModel("m", scope=scope, regions=frozenset(regions))

    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "8000"])
    def test_reported_limits_are_positive_integers(self, value: Any) -> None:
        with pytest.raises(ValueError):
            DiscoveredModel("m", context_window=value)
        with pytest.raises(ValueError):
            DiscoveredModel("m", output_limit=value)

    def test_reported_capabilities_hold_booleans_or_nothing(self) -> None:
        reported = ReportedCapabilities(vision=True)
        assert (reported.tools, reported.vision) == (None, True)
        with pytest.raises(ValueError):
            ReportedCapabilities(vision="yes")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DiscoveredModel("m", capabilities={"vision": True})  # type: ignore[arg-type]


class TestCatalogueProvenance:
    def test_the_default_entries_are_not_overlaid(self) -> None:
        assert DEFAULT.overlaid == frozenset()
        assert DEFAULT.withdrawn == frozenset()

    def test_overlay_entries_are_marked_and_withdrawals_remembered(self) -> None:
        applied = DEFAULT.apply(
            MIS_REGIONED,
            CatalogueOverlay(
                name="other",
                override=(_entry("vendor.model-small", context_window=16_000),),
                withdraw=(ModelKey(provider="fake", model_id="vendor.model-gone"),),
            ),
        )
        assert applied.overlaid == {
            ModelKey(provider="fake", model_id="geo.vendor.model-a"),
            ModelKey(provider="fake", model_id="vendor.model-small"),
        }
        assert applied.withdrawn == {ModelKey(provider="fake", model_id="vendor.model-gone")}

    def test_a_model_the_catalogue_lacks_may_be_withdrawn(self) -> None:
        key = ModelKey(provider="fake", model_id="vendor.model-new")
        applied = DEFAULT.apply(CatalogueOverlay(withdraw=(key,)))
        assert key in applied.withdrawn
        assert len(applied) == len(DEFAULT)

    def test_adding_a_withdrawn_model_back_clears_the_withdrawal(self) -> None:
        key = ModelKey(provider="fake", model_id="vendor.model-gone")
        applied = DEFAULT.apply(
            CatalogueOverlay(name="one", withdraw=(key,)),
            CatalogueOverlay(name="two", add=(_entry("vendor.model-gone"),)),
        )
        assert key not in applied.withdrawn
        assert key in applied.overlaid

    def test_an_entry_need_not_declare_residency(self) -> None:
        entry = _entry("vendor.model-a")
        assert entry.residency_scope is None and entry.regions == frozenset()
        with pytest.raises(ValueError):
            _entry("x", regions=frozenset({HOME}))


class TestMisRegionedProfile:
    def test_the_listed_regions_are_used_not_the_overlay(self) -> None:
        models, drift = _join(catalogue=DEFAULT.apply(MIS_REGIONED), residency=GEO_POLICY)
        profile = models["geo.vendor.model-a"]
        assert profile.outcome is DiscoveryOutcome.ALLOWED
        assert profile.scope is ResidencyScope.CROSS_REGION
        assert profile.regions == {HOME, AWAY}
        assert profile.entry is not None and profile.entry.regions == {HOME, AWAY}
        assert profile.catalogued
        assert DriftReason.REGIONS_DECLARED in _reasons(drift)["geo.vendor.model-a"]

    def test_a_default_entry_matches_a_profile_by_its_base_model(self) -> None:
        models, drift = _join(residency=GEO_POLICY)
        profile = models["geo.vendor.model-a"]
        assert profile.outcome is DiscoveryOutcome.ALLOWED
        assert profile.catalogued
        assert profile.entry is not None
        assert profile.entry.model_id == "geo.vendor.model-a"
        assert profile.entry.capabilities.prompt_caching
        assert profile.entry.context_window == 200_000
        assert "geo.vendor.model-a" not in _reasons(drift)

    def test_a_policy_without_a_listed_region_refuses_it(self) -> None:
        narrow = ResidencyPolicy(scopes=GEO_POLICY.scopes, regions=frozenset({HOME, EXTRA}))
        models, _ = _join(residency=narrow)
        profile = models["geo.vendor.model-a"]
        assert profile.outcome is DiscoveryOutcome.REFUSED
        assert profile.reason is not None and AWAY in profile.reason

    def test_the_refusal_is_the_residency_policy_reason(self) -> None:
        in_region = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({HOME})
        )
        models, _ = _join(residency=in_region)
        assert models["vendor.model-small"].outcome is DiscoveryOutcome.ALLOWED
        refused = models["geo.vendor.model-a"]
        assert refused.reason == (
            "main: model 'geo.vendor.model-a' (cross_region): scope cross_region is not "
            "allowed (in_region)"
        )


class TestUncatalogued:
    def test_a_listed_model_with_no_entry_is_usable_on_the_floor(self) -> None:
        models, drift = _join(residency=GEO_POLICY)
        new = models["geo.vendor.model-new"]
        assert new.outcome is DiscoveryOutcome.ALLOWED
        assert not new.catalogued
        assert new.entry is not None
        assert new.entry.kind == "chat"
        assert new.entry.capabilities == Capabilities()
        assert new.entry.context_window == FLOOR_CONTEXT_WINDOW
        assert new.entry.output_limit == FLOOR_OUTPUT_LIMIT
        assert new.entry.regions == {HOME, AWAY}
        assert _reasons(drift)["geo.vendor.model-new"] == {DriftReason.UNCATALOGUED}

    def test_the_floor_constants_are_conservative(self) -> None:
        assert (FLOOR_CONTEXT_WINDOW, FLOOR_OUTPUT_LIMIT) == (8_192, 4_096)

    def test_reported_limits_and_capabilities_are_used(self) -> None:
        listed = DiscoveredModel(
            "vendor.model-reported",
            scope=ResidencyScope.GLOBAL,
            context_window=300_000,
            output_limit=32_000,
            capabilities=ReportedCapabilities(vision=True, structured_output=False),
        )
        models, _ = _join([listed])
        entry = models["vendor.model-reported"].entry
        assert entry is not None
        assert (entry.context_window, entry.output_limit) == (300_000, 32_000)
        assert entry.capabilities == Capabilities(vision=True)

    def test_an_uncatalogued_embedding_model_is_refused(self) -> None:
        listed = DiscoveredModel("vendor.embed", kind="embedding", scope=ResidencyScope.GLOBAL)
        models, drift = _join([listed])
        embed = models["vendor.embed"]
        assert embed.outcome is DiscoveryOutcome.REFUSED
        assert embed.entry is None
        assert embed.reason == (
            "main: model 'vendor.embed' is an embedding model with no catalogue entry, so "
            "its dimensions are not known"
        )
        assert _reasons(drift)["vendor.embed"] == {DriftReason.UNCATALOGUED}

    def test_a_listed_kind_is_kept(self) -> None:
        listed = DiscoveredModel("vendor.chat", kind="chat", scope=ResidencyScope.GLOBAL)
        models, _ = _join([listed])
        assert models["vendor.chat"].entry is not None
        assert models["vendor.chat"].entry.kind == "chat"


class TestUnlisted:
    def test_an_entry_the_provider_does_not_list_is_drift(self) -> None:
        _, drift = _join()
        gone = [d for d in drift if d.model_id == "vendor.model-gone"]
        assert [d.reason for d in gone] == [DriftReason.UNLISTED]
        assert gone[0].detail == "declared in the catalogue, not listed by the provider"

    def test_an_entry_matched_through_a_profile_is_not_unlisted(self) -> None:
        _, drift = _join()
        assert "vendor.model-a" not in _reasons(drift)
        assert "vendor.model-small" not in _reasons(drift)

    def test_other_providers_entries_are_not_drift(self) -> None:
        other = Catalogue([*DEFAULT, _entry("elsewhere", provider="other")], version="led-2")
        _, drift = _join(catalogue=other)
        assert "elsewhere" not in _reasons(drift)


class TestNoRegionNotion:
    LISTED = (DiscoveredModel("vendor.model-a", scope=ResidencyScope.GLOBAL),)

    def test_global_is_refused_when_regions_are_restricted(self) -> None:
        models, _ = _join(self.LISTED, residency=GEO_POLICY)
        assert models["vendor.model-a"].outcome is DiscoveryOutcome.REFUSED

    def test_global_is_allowed_without_a_region_restriction(self) -> None:
        models, _ = _join(self.LISTED)
        assert models["vendor.model-a"].outcome is DiscoveryOutcome.ALLOWED
        assert models["vendor.model-a"].scope is ResidencyScope.GLOBAL

    def test_an_unknown_scope_is_treated_as_global(self) -> None:
        models, _ = _join((DiscoveredModel("vendor.model-a"),), residency=GEO_POLICY)
        model = models["vendor.model-a"]
        assert model.scope is ResidencyScope.GLOBAL
        assert model.outcome is DiscoveryOutcome.REFUSED

    def test_local_is_allowed_under_a_region_restriction(self) -> None:
        local = ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL}), regions=frozenset({HOME}))
        listed = (DiscoveredModel("vendor.model-a", scope=ResidencyScope.LOCAL),)
        models, _ = _join(listed, residency=local)
        assert models["vendor.model-a"].outcome is DiscoveryOutcome.ALLOWED


class TestAttributes:
    def test_listed_limits_win_over_a_default_entry_and_are_drift(self) -> None:
        listed = (
            DiscoveredModel("vendor.model-a", scope=ResidencyScope.GLOBAL, context_window=100_000),
        )
        models, drift = _join(listed)
        entry = models["vendor.model-a"].entry
        assert entry is not None and entry.context_window == 100_000
        assert entry.output_limit == 64_000
        [item] = drift_for(drift, "vendor.model-a")
        assert item.reason is DriftReason.LIMITS_DIFFER
        assert item.detail == "context window 100000 listed, 200000 in the catalogue"

    def test_an_overlay_wins_outright(self) -> None:
        overlay = CatalogueOverlay(
            override=(_entry("vendor.model-a", context_window=50_000, labels={"tier:x"}),)
        )
        listed = (
            DiscoveredModel(
                "vendor.model-a",
                scope=ResidencyScope.GLOBAL,
                context_window=100_000,
                capabilities=ReportedCapabilities(vision=False),
            ),
        )
        models, drift = _join(listed, catalogue=DEFAULT.apply(overlay))
        entry = models["vendor.model-a"].entry
        assert entry is not None
        assert entry.context_window == 50_000
        assert entry.capabilities.vision
        assert entry.labels == {"tier:x"}
        assert {d.reason for d in drift_for(drift, "vendor.model-a")} == {
            DriftReason.LIMITS_DIFFER,
            DriftReason.CAPABILITIES_DIFFER,
        }

    def test_a_capability_needs_an_assertion_and_no_denial(self) -> None:
        listed = (
            DiscoveredModel(
                "vendor.model-small",
                scope=ResidencyScope.GLOBAL,
                capabilities=ReportedCapabilities(vision=False, structured_output=True, tools=True),
            ),
        )
        models, drift = _join(listed)
        caps = models["vendor.model-small"].entry.capabilities
        assert caps.tools  # both assert
        assert not caps.vision  # the listing denies
        assert not caps.structured_output  # the catalogue denies
        assert caps.prompt_caching  # the catalogue asserts, the listing is silent
        details = {d.detail for d in drift_for(drift, "vendor.model-small")}
        assert details == {
            "vision not supported listed, supported in the catalogue",
            "structured output supported listed, not supported in the catalogue",
        }

    def test_labels_come_from_the_matched_entry(self) -> None:
        labelled = Catalogue([_entry("vendor.model-a", labels={"tier:top"})], version="v")
        models, _ = _join(LISTING[:1], catalogue=labelled)
        assert models["geo.vendor.model-a"].entry.labels == {"tier:top"}

    def test_an_exact_entry_wins_over_the_base_entry(self) -> None:
        both = Catalogue(
            [_entry("vendor.model-a"), _entry("geo.vendor.model-a", context_window=1_000)],
            version="v",
        )
        models, _ = _join(LISTING[:1], catalogue=both)
        assert models["geo.vendor.model-a"].entry.context_window == 1_000


class TestWithdrawn:
    def test_a_withdrawn_model_is_refused_although_listed(self) -> None:
        overlay = CatalogueOverlay(
            withdraw=(ModelKey(provider="fake", model_id="vendor.model-new"),)
        )
        models, _ = _join(catalogue=DEFAULT.apply(overlay))
        new = models["geo.vendor.model-new"]
        assert new.outcome is DiscoveryOutcome.REFUSED
        assert new.entry is None
        assert (
            new.reason == "main: model 'geo.vendor.model-new' is withdrawn by a catalogue overlay"
        )

    def test_withdrawing_the_listed_id_also_refuses_it(self) -> None:
        overlay = CatalogueOverlay(
            withdraw=(ModelKey(provider="fake", model_id="geo.vendor.model-a"),)
        )
        models, _ = _join(catalogue=DEFAULT.apply(overlay))
        assert models["geo.vendor.model-a"].outcome is DiscoveryOutcome.REFUSED


def drift_for(drift: Sequence[CatalogueDrift], model_id: str) -> list[CatalogueDrift]:
    return [item for item in drift if item.model_id == model_id]


class TestResult:
    def test_views_and_the_listed_catalogue(self) -> None:
        models, drift = _join(residency=GEO_POLICY)
        result = DiscoveryResult(
            provider="main",
            kind="fake",
            catalogue_provider="fake",
            region=HOME,
            supported=True,
            models=tuple(models.values()),
            drift=drift,
        )
        assert [m.model_id for m in result.uncatalogued] == ["geo.vendor.model-new"]
        assert len(result.allowed) == 3 and result.refused == ()
        listed = result.listed_catalogue(DEFAULT)
        assert sorted(e.model_id for e in listed) == sorted(models)
        assert listed.version == DEFAULT.version
        assert listed.get("fake", "vendor.model-gone") is None


TRIAGE = TaskClass(name="triage")
TOOLS = TaskClass(name="tools", floor=CapabilityFloor(tools=True))


def _settings(model: str, **extra: Any) -> ConduitSettings:
    return ConduitSettings.model_validate(
        {
            "providers": {"main": PROVIDER},
            "assignments": {"triage": {"provider": "main", "model": model}},
            "residency": {
                "scopes": ["in_region", "cross_region"],
                "regions": [HOME, AWAY],
            },
            **extra,
        }
    )


async def _start(
    settings: ConduitSettings,
    factory: FakeFactory,
    *,
    tasks: Sequence[TaskClass] = (TRIAGE,),
    sink: MemoryEventSink | None = None,
    catalogue: Catalogue = DEFAULT,
) -> Any:
    return await start(
        settings,
        task_classes=tasks,
        factories=[factory],
        secrets=SECRETS,
        events=sink,
        catalogue=catalogue,
    )


class TestStartUp:
    async def test_a_profile_is_usable_on_its_listed_regions(self) -> None:
        sink = MemoryEventSink()
        factory = ListingFactory(LISTING)
        runtime = await _start(
            _settings("geo.vendor.model-a", catalogue=[MIS_REGIONED.model_dump()]),
            factory,
            sink=sink,
        )
        try:
            resolved = runtime.assignments.for_task("triage")
            assert resolved.entry.regions == {HOME, AWAY}
            assert runtime.discovery["main"].supported
            # The provider is built on the listed catalogue, not the applied one.
            built = factory.catalogues[-1]
            assert built is not None and built.get("fake", "geo.vendor.model-new") is not None
            [event] = [e for e in sink.events if isinstance(e, DiscoveryFinishedEvent)]
            assert (event.outcome, event.regions_declared, event.uncatalogued) == ("ok", 1, 1)
            assert event.unlisted == 1
        finally:
            await runtime.aclose()

    async def test_an_uncatalogued_model_is_usable_below_a_floor(self) -> None:
        runtime = await _start(_settings("geo.vendor.model-new"), ListingFactory(LISTING))
        try:
            assert runtime.assignments.for_task("triage").entry.output_limit == 4_096
        finally:
            await runtime.aclose()

    async def test_an_uncatalogued_model_is_refused_by_a_tools_floor(self) -> None:
        settings = ConduitSettings.model_validate(
            {
                "providers": {"main": PROVIDER},
                "assignments": {"tools": {"provider": "main", "model": "geo.vendor.model-new"}},
            }
        )
        with pytest.raises(CapabilityRefused, match="lacks tools"):
            await _start(settings, ListingFactory(LISTING), tasks=[TOOLS])

    async def test_an_unlisted_model_is_refused(self) -> None:
        with pytest.raises(AssignmentRefused) as refused:
            await _start(_settings("vendor.model-gone"), ListingFactory(LISTING))
        assert str(refused.value) == (
            "triage: model 'vendor.model-gone' is not listed by provider 'main'"
        )

    async def test_a_residency_breach_on_listed_regions_is_refused(self) -> None:
        settings = _settings("geo.vendor.model-a")
        narrow = settings.model_copy(
            update={
                "residency": ResidencyPolicy(scopes=GEO_POLICY.scopes, regions=frozenset({HOME}))
            }
        )
        factory = ListingFactory(LISTING)
        with pytest.raises(ResidencyRefused, match=AWAY):
            await _start(narrow, factory)
        assert [p.closed for p in factory.providers] == [True]  # only the listing's

    async def test_a_failed_listing_fails_closed(self) -> None:
        sink = MemoryEventSink()
        factory = ListingFactory(failure=ProviderError(ProviderFailure.REJECTED))
        with pytest.raises(ProviderError) as failed:
            await _start(_settings("vendor.model-small"), factory, sink=sink)
        assert failed.value.__cause__ is None and failed.value.__context__ is None
        assert all(p.closed for p in factory.providers)
        assert len(factory.providers) == 1  # nothing built after the listing
        [event] = [e for e in sink.events if isinstance(e, DiscoveryFinishedEvent)]
        assert event.outcome == "failed"

    async def test_catalogue_mode_skips_the_listing(self) -> None:
        overlay = CatalogueOverlay(
            add=(_entry("vendor.declared", residency_scope="in_region", regions={HOME}),)
        )
        factory = ListingFactory(failure=ProviderError(ProviderFailure.REJECTED))
        runtime = await _start(
            _settings(
                "vendor.declared",
                discovery={"main": "catalogue"},
                catalogue=[overlay.model_dump()],
            ),
            factory,
        )
        try:
            assert factory.discovered_with == []
            assert "main" not in runtime.discovery
        finally:
            await runtime.aclose()

    async def test_catalogue_mode_treats_an_undeclared_scope_as_global(self) -> None:
        with pytest.raises(ResidencyRefused, match="global"):
            await _start(
                _settings("vendor.model-small", discovery={"main": "catalogue"}),
                ListingFactory(LISTING),
            )

    async def test_a_factory_without_discovery_uses_the_catalogue(self) -> None:
        overlay = CatalogueOverlay(
            add=(_entry("vendor.declared", residency_scope="in_region", regions={HOME}),)
        )
        runtime = await _start(
            _settings("vendor.declared", catalogue=[overlay.model_dump()]), FakeFactory()
        )
        try:
            assert runtime.discovery == {}
        finally:
            await runtime.aclose()

    async def test_discovery_settings_must_name_configured_providers(self) -> None:
        with pytest.raises(ConfigurationError, match="nowhere"):
            await _start(
                _settings("vendor.model-small", discovery={"nowhere": "catalogue"}),
                ListingFactory(LISTING),
            )

    def test_discovery_modes_are_listing_or_catalogue(self) -> None:
        with pytest.raises(ValueError):
            _settings("x", discovery={"main": "cached"})


class TestResolveWithListings:
    def _listing(self, residency: ResidencyPolicy) -> DiscoveryResult:
        models, drift = _join(residency=residency)
        return DiscoveryResult(
            provider="main",
            kind="fake",
            catalogue_provider="fake",
            region=HOME,
            supported=True,
            models=tuple(models.values()),
            drift=drift,
        )

    def _resolve(self, model: str, listing: DiscoveryResult) -> Any:
        providers: dict[str, ProviderSettings] = {
            "main": FakeFactory.settings_model.model_validate(PROVIDER)
        }
        return resolve_assignments(
            [TRIAGE],
            {"triage": AssignmentSettings(provider="main", model=model)},
            providers,
            DEFAULT,
            GEO_POLICY,
            listings={"main": listing},
        )

    def test_the_listed_entry_is_resolved(self) -> None:
        resolved = self._resolve("geo.vendor.model-a", self._listing(GEO_POLICY))
        assert resolved.for_task("triage").entry.regions == {HOME, AWAY}

    def test_a_withdrawn_listed_model_carries_its_reason(self) -> None:
        overlay = CatalogueOverlay(
            withdraw=(ModelKey(provider="fake", model_id="vendor.model-new"),)
        )
        models, drift = _join(catalogue=DEFAULT.apply(overlay))
        listing = DiscoveryResult(
            provider="main",
            kind="fake",
            catalogue_provider="fake",
            region=HOME,
            supported=True,
            models=tuple(models.values()),
            drift=drift,
        )
        with pytest.raises(AssignmentRefused, match="withdrawn by a catalogue overlay"):
            self._resolve("geo.vendor.model-new", listing)


class TestRegistryBuild:
    async def test_build_uses_the_catalogue_it_is_given(self) -> None:
        factory = ListingFactory(LISTING)
        registry = ProviderRegistry(secrets=SECRETS, catalogue=DEFAULT, factories=[factory])
        listed = Catalogue([_entry("only.this")], version="x")
        provider = await registry.build(
            "main", PROVIDER, assigned={"triage": "only.this"}, catalogue=listed
        )
        try:
            assert factory.catalogues[-1] is listed
        finally:
            await registry.aclose()
        assert provider.assigned_models == frozenset({"only.this"})
