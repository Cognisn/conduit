"""A connection's Region choice: a second, per-connection residency check.

The deployment's ``ResidencyPolicy`` stays the outer bound; a connection's
``RegionChoice`` (built by its adapter's settings from the person's choice) can only
narrow it. A connection may also declare the residency its endpoint guarantees for a
model (OpenAI's regional hosts, X.AI's US host, Anthropic's ``inference_geo``, a Foundry
deployment type), which replaces the residency a listing or an entry states. A
connection that chooses nothing behaves exactly as before.

Every Region code, listing and setting here is synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pytest

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.discovery import DiscoveredModel, DiscoveryOutcome, join_discovered
from conduit.core.errors import ConfigurationError, ResidencyRefused
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import (
    OfferedRegion,
    RegionChoice,
    RegionListing,
    Residency,
    ResidencyPolicy,
    ResidencyScope,
)
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import AssignmentSettings, ProviderSettings
from conduit.core.tasks import TaskClass, resolve_assignments
from conduit.core.types import Message, ModelRequest
from tests.unit.core.fakes import FakeSettings
from tests.unit.core.test_discovery import ListingFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
HOME, NEAR, AWAY = "xx-home-1", "xx-near-1", "xx-away-1"
TRIAGE = TaskClass(name="triage")

#: What choosing nothing means everywhere: a policy with no restriction at all.
OPEN = ResidencyPolicy()


class ChoosingSettings(FakeSettings):
    """The fake adapter's settings, with a Region choice and declared residency."""

    processing: RegionChoice | None = None
    declares: Literal["nothing", "home", "global"] = "nothing"

    def region_choice(self) -> RegionChoice | None:
        return self.processing

    def declared_residency(self, model_id: str) -> Residency | None:
        if self.declares == "home" and model_id.startswith("declared."):
            return Residency(ResidencyScope.CROSS_REGION, frozenset({HOME, NEAR}))
        if self.declares == "global":
            return Residency(ResidencyScope.GLOBAL)
        return None

    @classmethod
    def offered_regions(cls) -> tuple[RegionListing, ...]:
        return (
            RegionListing(
                kind="fake",
                service=None,
                regions=(
                    OfferedRegion(HOME, "Home", invocation=True, scope=ResidencyScope.IN_REGION),
                    OfferedRegion(
                        "xx:zone",
                        "Zone",
                        invocation=False,
                        scope=ResidencyScope.CROSS_REGION,
                        members=frozenset({HOME, NEAR}),
                    ),
                ),
                global_option="any Region the synthetic provider serves",
                documentation="https://example.com/regions",
                as_of="2026-11-21",
            ),
        )


class ChoosingFactory(ListingFactory):
    settings_model = ChoosingSettings


def _settings(**fields: Any) -> ChoosingSettings:
    base: dict[str, Any] = {"api_key": "secret://fake/key", "region": HOME}
    base.update(fields)
    return ChoosingSettings.model_validate(base)


def _choice(*regions: str, allow_global: bool = False) -> dict[str, Any]:
    return {"regions": list(regions), "allow_global": allow_global}


def _entry(model_id: str, scope: str | None, *regions: str) -> ModelEntry:
    return ModelEntry.model_validate(
        {
            "model_id": model_id,
            "provider": "fake",
            "residency_scope": scope,
            "regions": list(regions),
            "context_window": 100_000,
            "output_limit": 8_000,
        }
    )


IN_HOME = _entry("vendor.in-home", "in_region", HOME)
GEO_NEAR = _entry("geo.vendor.near", "cross_region", HOME, NEAR)
GEO_AWAY = _entry("geo.vendor.away", "cross_region", HOME, AWAY)
GLOBAL = _entry("global.vendor.any", "global")
UNSTATED = _entry("vendor.unstated", None)
LOCAL = _entry("vendor.local", "local")
DECLARED = _entry("declared.vendor.model", "global")

CATALOGUE = Catalogue(
    [IN_HOME, GEO_NEAR, GEO_AWAY, GLOBAL, UNSTATED, LOCAL, DECLARED], version="choice-1"
)


def _profile(model_id: str, *regions: str) -> DiscoveredModel:
    return DiscoveredModel(model_id, scope=ResidencyScope.CROSS_REGION, regions=frozenset(regions))


#: The provider's listing in HOME: in-Region, a geographic profile inside the choice, one
#: reaching outside it, and a global profile.
LISTING = (
    DiscoveredModel("vendor.in-home", scope=ResidencyScope.IN_REGION, regions=frozenset({HOME})),
    _profile("geo.vendor.near", HOME, NEAR),
    _profile("geo.vendor.away", HOME, AWAY),
    DiscoveredModel("global.vendor.any", scope=ResidencyScope.GLOBAL),
    DiscoveredModel("declared.vendor.model", scope=ResidencyScope.GLOBAL),
)


class TestRegionChoice:
    def test_a_choice_names_at_least_one_region(self) -> None:
        with pytest.raises(ValueError):
            RegionChoice.model_validate({"regions": []})

    @pytest.mark.parametrize("code", ["", "Upper-Case", "has space", "x" * 65])
    def test_a_region_code_is_a_short_lower_case_token(self, code: str) -> None:
        with pytest.raises(ValueError):
            RegionChoice.model_validate({"regions": [code]})

    def test_prefixed_geography_codes_are_accepted(self) -> None:
        choice = RegionChoice.model_validate({"regions": ["openai:eu", "azure:zone-eu", HOME]})
        assert choice.regions == {"openai:eu", "azure:zone-eu", HOME}
        assert choice.allow_global is False

    def test_a_choice_is_immutable(self) -> None:
        choice = RegionChoice.model_validate(_choice(HOME))
        with pytest.raises(ValueError):
            choice.allow_global = True


class TestChoiceCheck:
    @staticmethod
    def check(entry: ModelEntry, choice: dict[str, Any] | None, region: str | None = HOME) -> None:
        settings = _settings(processing=choice, region=region)
        OPEN.check(entry, region, task="triage", provider=settings)

    def test_no_choice_changes_nothing(self) -> None:
        for entry in CATALOGUE:
            self.check(entry, None)

    def test_local_always_passes_the_choice(self) -> None:
        self.check(LOCAL, _choice(AWAY), region=AWAY)
        # Even from a Region outside the choice: a local model never leaves the host.
        self.check(LOCAL, _choice(AWAY), region=HOME)

    def test_in_region_inside_the_choice_is_allowed(self) -> None:
        self.check(IN_HOME, _choice(HOME))

    def test_a_geographic_profile_inside_the_choice_is_allowed(self) -> None:
        self.check(GEO_NEAR, _choice(HOME, NEAR))

    def test_a_geographic_profile_outside_the_choice_is_refused(self) -> None:
        with pytest.raises(ResidencyRefused, match=r"xx-away-1.*outside the connection"):
            self.check(GEO_AWAY, _choice(HOME, NEAR))

    def test_global_is_refused_unless_the_choice_allows_it(self) -> None:
        with pytest.raises(ResidencyRefused, match="global processing is outside"):
            self.check(GLOBAL, _choice(HOME))
        self.check(GLOBAL, _choice(HOME, allow_global=True))

    def test_an_unstated_scope_is_global_to_the_choice(self) -> None:
        with pytest.raises(ResidencyRefused, match="global processing is outside"):
            self.check(UNSTATED, _choice(HOME))

    def test_allowing_global_still_bounds_the_others(self) -> None:
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            self.check(GEO_AWAY, _choice(HOME, NEAR, allow_global=True))

    def test_the_invocation_region_must_be_chosen(self) -> None:
        settings = _settings(processing=None, region=AWAY)
        chosen = settings.model_copy(
            update={"processing": RegionChoice.model_validate(_choice(HOME))}
        )
        with pytest.raises(ResidencyRefused, match="called in xx-away-1"):
            OPEN.check(IN_HOME, AWAY, task="triage", provider=chosen)

    def test_a_breach_names_the_task_and_model(self) -> None:
        with pytest.raises(ResidencyRefused, match=r"^triage: model 'geo\.vendor\.away'"):
            self.check(GEO_AWAY, _choice(HOME, NEAR))

    def test_the_policy_still_applies_when_the_choice_allows(self) -> None:
        policy = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({HOME})
        )
        settings = _settings(processing=_choice(HOME, NEAR))
        with pytest.raises(ResidencyRefused, match="scope cross_region is not allowed"):
            policy.check(GEO_NEAR, HOME, task="triage", provider=settings)

    def test_the_choice_applies_when_the_policy_allows(self) -> None:
        policy = ResidencyPolicy(regions=frozenset({HOME, NEAR, AWAY}))
        settings = _settings(processing=_choice(HOME, NEAR))
        policy.check(GEO_AWAY, HOME, task="triage")  # the policy alone allows it
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            policy.check(GEO_AWAY, HOME, task="triage", provider=settings)

    def test_the_refusal_keeps_no_exception(self) -> None:
        with pytest.raises(ResidencyRefused) as refused:
            self.check(GEO_AWAY, _choice(HOME, NEAR))
        assert refused.value.__cause__ is None and refused.value.__context__ is None


class TestResidency:
    @pytest.mark.parametrize("scope", [ResidencyScope.LOCAL, ResidencyScope.GLOBAL])
    def test_only_a_region_bound_scope_names_regions(self, scope: ResidencyScope) -> None:
        with pytest.raises(ValueError):
            Residency(scope, frozenset({HOME}))
        assert Residency(scope).regions == frozenset()

    def test_a_region_bound_scope_keeps_its_regions(self) -> None:
        residency = Residency(ResidencyScope.CROSS_REGION, frozenset({HOME, NEAR}))
        assert residency.regions == {HOME, NEAR}


class TestDeclaredResidency:
    def test_a_declaration_replaces_the_entry_scope(self) -> None:
        settings = _settings(declares="home", processing=_choice(HOME, NEAR))
        # The entry says global; the connection declares HOME and NEAR, inside the choice.
        OPEN.check(DECLARED, HOME, task="triage", provider=settings)

    def test_a_declaration_is_held_to_the_policy(self) -> None:
        policy = ResidencyPolicy(regions=frozenset({HOME}))
        settings = _settings(declares="home")
        with pytest.raises(ResidencyRefused, match="xx-near-1"):
            policy.check(DECLARED, HOME, task="triage", provider=settings)

    def test_a_global_declaration_is_never_narrower(self) -> None:
        settings = _settings(declares="global", processing=_choice(HOME))
        with pytest.raises(ResidencyRefused, match="global processing is outside"):
            OPEN.check(IN_HOME, HOME, task="triage", provider=settings)

    def test_the_default_settings_declare_nothing(self) -> None:
        plain = ProviderSettings.model_validate({"kind": "anything", "region": HOME})
        assert plain.region_choice() is None
        assert plain.declared_residency("any.model") is None
        assert ProviderSettings.offered_regions() == ()


def _join(settings: ChoosingSettings) -> dict[str, Any]:
    models, _ = join_discovered(
        LISTING,
        CATALOGUE,
        catalogue_provider="fake",
        region=HOME,
        residency=OPEN,
        provider="main",
        connection=settings,
    )
    return {model.model_id: model for model in models}


class TestDiscoveryFollowsTheChoice:
    def test_without_a_choice_every_listed_model_is_allowed(self) -> None:
        joined = _join(_settings())
        assert {m.outcome for m in joined.values()} == {DiscoveryOutcome.ALLOWED}

    def test_listed_models_are_held_to_the_choice(self) -> None:
        joined = _join(_settings(processing=_choice(HOME, NEAR)))
        outcomes = {name: model.outcome for name, model in joined.items()}
        assert outcomes == {
            "vendor.in-home": DiscoveryOutcome.ALLOWED,
            "geo.vendor.near": DiscoveryOutcome.ALLOWED,
            "geo.vendor.away": DiscoveryOutcome.REFUSED,
            "global.vendor.any": DiscoveryOutcome.REFUSED,
            "declared.vendor.model": DiscoveryOutcome.REFUSED,
        }
        assert "outside the connection" in (joined["geo.vendor.away"].reason or "")

    def test_a_global_profile_is_allowed_with_allow_global(self) -> None:
        joined = _join(_settings(processing=_choice(HOME, NEAR, allow_global=True)))
        assert joined["global.vendor.any"].outcome is DiscoveryOutcome.ALLOWED
        assert joined["geo.vendor.away"].outcome is DiscoveryOutcome.REFUSED

    def test_a_declaration_replaces_the_listed_residency(self) -> None:
        joined = _join(_settings(declares="home", processing=_choice(HOME, NEAR)))
        model = joined["declared.vendor.model"]
        assert model.outcome is DiscoveryOutcome.ALLOWED
        assert (model.scope, model.regions) == (ResidencyScope.CROSS_REGION, {HOME, NEAR})
        assert model.entry is not None and model.entry.regions == {HOME, NEAR}


def _registry(
    factory: ListingFactory, residency: ResidencyPolicy | None = None
) -> ProviderRegistry:
    return ProviderRegistry(
        secrets=SECRETS, catalogue=CATALOGUE, residency=residency, factories=[factory]
    )


class TestRegistry:
    async def test_discover_follows_the_choice_and_reports_it(self) -> None:
        registry = _registry(ChoosingFactory(LISTING))
        settings = _settings(processing=_choice(HOME, NEAR))
        result = await registry.discover("main", settings)
        assert result.choice == settings.processing
        assert {m.model_id for m in result.allowed} == {"vendor.in-home", "geo.vendor.near"}
        await registry.aclose()

    async def test_discover_reports_no_choice(self) -> None:
        registry = _registry(ChoosingFactory(LISTING))
        result = await registry.discover("main", _settings())
        assert result.choice is None
        await registry.aclose()

    async def test_build_refuses_an_assignment_outside_the_choice(self) -> None:
        registry = _registry(ChoosingFactory(LISTING))
        settings = _settings(processing=_choice(HOME, NEAR))
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await registry.build("main", settings, assigned={"triage": "geo.vendor.away"})
        built = await registry.build("main", settings, assigned={"triage": "geo.vendor.near"})
        assert built.assigned_models == frozenset({"geo.vendor.near"})
        await registry.aclose()

    async def test_an_unassigned_provider_refuses_each_call_outside_the_choice(self) -> None:
        factory = ChoosingFactory(LISTING)
        registry = _registry(factory)
        built = await registry.build("main", _settings(processing=_choice(HOME, NEAR)))
        request = ModelRequest(
            model="geo.vendor.away", messages=(Message.user("hello"),), max_output_tokens=10
        )
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await built.complete(request)
        assert factory.providers[-1].requests == []  # refused before any client call
        await registry.aclose()

    def test_regions_lists_what_the_adapter_offers(self) -> None:
        registry = _registry(ChoosingFactory(LISTING))
        [listing] = registry.regions("fake")
        assert [region.code for region in listing.regions] == [HOME, "xx:zone"]
        assert listing.regions[1].members == {HOME, NEAR}
        assert listing.source == "static"

    def test_regions_of_an_adapter_with_no_regions_is_empty(self) -> None:
        registry = _registry(ListingFactory(LISTING))
        assert registry.regions("fake") == ()

    def test_regions_of_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="no adapter"):
            _registry(ListingFactory(LISTING)).regions("nothing_registered")


def _conduit(model: str, processing: dict[str, Any] | None, **extra: Any) -> ConduitSettings:
    provider: dict[str, Any] = {"kind": "fake", "api_key": "secret://fake/key", "region": HOME}
    if processing is not None:
        provider["processing"] = processing
    return ConduitSettings.model_validate(
        {
            "providers": {"main": provider},
            "assignments": {"triage": {"provider": "main", "model": model}},
            **extra,
        }
    )


async def _start(settings: ConduitSettings, factory: ListingFactory) -> Any:
    return await start(
        settings, task_classes=[TRIAGE], factories=[factory], secrets=SECRETS, catalogue=CATALOGUE
    )


class TestStartUp:
    async def test_a_listed_model_inside_the_choice_starts(self) -> None:
        runtime = await _start(
            _conduit("geo.vendor.near", _choice(HOME, NEAR)), ChoosingFactory(LISTING)
        )
        try:
            assert runtime.assignments.for_task("triage").entry.regions == {HOME, NEAR}
            assert runtime.discovery["main"].choice is not None
        finally:
            await runtime.aclose()

    async def test_a_listed_model_outside_the_choice_is_refused(self) -> None:
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await _start(_conduit("geo.vendor.away", _choice(HOME, NEAR)), ChoosingFactory(LISTING))

    async def test_without_a_choice_start_up_is_unchanged(self) -> None:
        runtime = await _start(_conduit("geo.vendor.away", None), ChoosingFactory(LISTING))
        await runtime.aclose()

    async def test_catalogue_mode_is_held_to_the_choice(self) -> None:
        settings = _conduit("geo.vendor.away", _choice(HOME, NEAR), discovery={"main": "catalogue"})
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await _start(settings, ChoosingFactory(LISTING))


class TestResolveAssignments:
    def _resolve(self, model: str, settings: ProviderSettings) -> Any:
        return resolve_assignments(
            [TRIAGE],
            {"triage": AssignmentSettings(provider="main", model=model)},
            {"main": settings},
            CATALOGUE,
            OPEN,
        )

    def test_an_assignment_outside_the_choice_is_refused(self) -> None:
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            self._resolve("geo.vendor.away", _settings(processing=_choice(HOME, NEAR)))

    def test_an_assignment_inside_the_choice_resolves(self) -> None:
        resolved = self._resolve("geo.vendor.near", _settings(processing=_choice(HOME, NEAR)))
        assert resolved.for_task("triage").model == "geo.vendor.near"

    def test_a_declaration_applies_to_a_catalogued_assignment(self) -> None:
        resolved = self._resolve(
            "declared.vendor.model", _settings(declares="home", processing=_choice(HOME, NEAR))
        )
        assert resolved.for_task("triage").model == "declared.vendor.model"


def _codes(listings: Sequence[RegionListing]) -> list[str]:
    return [region.code for listing in listings for region in listing.regions]


class TestOfferedRegion:
    def test_an_offered_region_is_validated(self) -> None:
        with pytest.raises(ValueError):
            OfferedRegion("", "Empty", invocation=True, scope=ResidencyScope.IN_REGION)
        with pytest.raises(ValueError):
            OfferedRegion(
                "xx",
                "No members",
                invocation=False,
                scope=ResidencyScope.CROSS_REGION,
                members=frozenset({""}),
            )

    def test_a_listing_reports_its_codes(self) -> None:
        assert _codes(ChoosingSettings.offered_regions()) == [HOME, "xx:zone"]
