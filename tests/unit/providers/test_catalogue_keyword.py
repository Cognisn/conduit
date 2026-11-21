"""Every shipped adapter takes its catalogue from the registry (Q0010, Q0012, D0043).

``create_provider`` declares ``*, catalogue: Catalogue | None = None``, so the registry
passes the catalogue it applied the product's overlays to; no factory takes a catalogue
of its own, and registering a shipped factory draws no deprecation warning.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from conduit.core.registry import ProviderFactory, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers import builtin_factories

FACTORIES = builtin_factories()


def _ids(factory: ProviderFactory) -> str:
    return factory.kind


@pytest.mark.parametrize("factory", FACTORIES, ids=_ids)
def test_create_provider_takes_the_catalogue_keyword(factory: ProviderFactory) -> None:
    parameter = inspect.signature(factory.create_provider).parameters.get("catalogue")
    assert parameter is not None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("factory", FACTORIES, ids=_ids)
def test_no_factory_takes_a_catalogue_of_its_own(factory: ProviderFactory) -> None:
    assert "catalogue" not in inspect.signature(type(factory)).parameters


@pytest.mark.parametrize("factory", FACTORIES, ids=_ids)
def test_registering_a_shipped_factory_draws_no_deprecation_warning(
    factory: ProviderFactory,
) -> None:
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        registry.register(factory)
