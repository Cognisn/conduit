"""The stub's model discovery: a configurable listing, for products' tests."""

from __future__ import annotations

from conduit.core.discovery import DiscoveredModel, DiscoveryOutcome
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.stub import DEFAULT_STUB_MODELS, StubFactory

STUB = {"kind": "stub", "not_for_production": True}


def _registry(factory: StubFactory) -> ProviderRegistry:
    return ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[factory])


def test_the_stub_factory_discovers() -> None:
    assert isinstance(StubFactory(), DiscoveringFactory)


async def test_by_default_it_lists_the_stub_models_the_catalogue_declares() -> None:
    result = await _registry(StubFactory()).discover("stub", STUB)
    assert result.supported
    assert [m.model_id for m in result.models] == list(DEFAULT_STUB_MODELS)
    assert all(m.outcome is DiscoveryOutcome.ALLOWED for m in result.models)


async def test_a_product_configures_the_listing() -> None:
    factory = StubFactory(
        models=["stub-echo", DiscoveredModel("stub-new", label="test"), "stub-other"]
    )
    result = await _registry(factory).discover("stub", STUB)
    outcomes = {m.model_id: (m.outcome, m.label) for m in result.models}
    assert outcomes == {
        "stub-echo": (DiscoveryOutcome.ALLOWED, None),
        "stub-new": (DiscoveryOutcome.UNDECLARED, "test"),
        "stub-other": (DiscoveryOutcome.UNDECLARED, None),
    }


async def test_an_empty_listing() -> None:
    result = await _registry(StubFactory(models=[])).discover("stub", STUB)
    assert result.supported and result.models == ()
