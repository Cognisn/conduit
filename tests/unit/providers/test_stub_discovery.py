"""The stub's model discovery: a configurable listing, for products' tests.

Without a listing the stub has nothing to list, so discovery is unsupported and start-up
serves the catalogue's stub models, as before; with one, start-up takes the listing as
the provider's own (C18), each listed id local unless the product says otherwise.
"""

from __future__ import annotations

from conduit.core.discovery import DiscoveredModel, DiscoveryOutcome
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import TaskClass
from conduit.providers.stub import DEFAULT_STUB_MODELS, StubFactory

STUB = {"kind": "stub", "not_for_production": True}


def _registry(factory: StubFactory) -> ProviderRegistry:
    return ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[factory])


def test_the_stub_factory_discovers() -> None:
    assert isinstance(StubFactory(), DiscoveringFactory)


async def test_without_a_listing_discovery_is_unsupported() -> None:
    result = await _registry(StubFactory()).discover("stub", STUB)
    assert not result.supported
    assert result.reason == "the stub lists models only when it is given a listing"


def test_the_default_listing_names_the_catalogue_stub_models() -> None:
    assert DEFAULT_STUB_MODELS == ("stub-echo", "stub-embed")


async def test_a_product_configures_the_listing() -> None:
    factory = StubFactory(
        models=["stub-echo", DiscoveredModel("stub-new", label="test"), "stub-other"]
    )
    result = await _registry(factory).discover("stub", STUB)
    outcomes = {m.model_id: (m.outcome, m.label, m.catalogued) for m in result.models}
    assert outcomes == {
        "stub-echo": (DiscoveryOutcome.ALLOWED, None, True),
        "stub-new": (DiscoveryOutcome.ALLOWED, "test", False),
        "stub-other": (DiscoveryOutcome.ALLOWED, None, False),
    }


async def test_a_listed_id_is_local_and_a_given_model_keeps_its_own_scope() -> None:
    factory = StubFactory(models=["stub-echo", DiscoveredModel("stub-new")])
    result = await _registry(factory).discover("stub", STUB)
    scopes = {m.model_id: m.scope for m in result.models}
    assert scopes == {"stub-echo": ResidencyScope.LOCAL, "stub-new": ResidencyScope.GLOBAL}


async def test_an_empty_listing() -> None:
    result = await _registry(StubFactory(models=[])).discover("stub", STUB)
    assert result.supported and result.models == ()


async def test_start_up_serves_only_the_listing() -> None:
    settings = ConduitSettings.model_validate(
        {
            "providers": {"local": STUB},
            "assignments": {"chat": {"provider": "local", "model": "stub-new"}},
            "residency": ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL})).model_dump(),
        }
    )
    runtime = await start(
        settings,
        task_classes=[TaskClass(name="chat")],
        factories=[StubFactory(models=["stub-new"])],
        secrets=LiteralSecretResolver({}),
    )
    try:
        assert runtime.assignments.for_task("chat").entry.residency_scope is ResidencyScope.LOCAL
        assert runtime.discovery["local"].supported
    finally:
        await runtime.aclose()
