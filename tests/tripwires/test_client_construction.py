"""Tripwire: no provider client is constructed, or metered, outside the provider registry.

Never skipped or weakened. A failure names the file, line and call.
"""

from __future__ import annotations

from tests.tripwires.clientscan import (
    LOCAL_MODELS_MODULE,
    METERED_EMBEDDER,
    REGISTRY,
    construction_violations,
    factory_call_violations,
    metered_violations,
    scan_tree,
)


def _report(findings: list[object]) -> str:
    return "\n".join(str(f) for f in findings)


def test_no_client_is_constructed_outside_a_provider_factory() -> None:
    violations = construction_violations(scan_tree())
    assert not violations, _report(list(violations))


def test_only_the_registry_calls_create_client() -> None:
    violations = factory_call_violations(scan_tree().factory_calls)
    assert not violations, _report(list(violations))


def test_the_registry_is_where_clients_are_made() -> None:
    # Guards against a vacuous pass: the scan must see the registry's own call.
    calls = scan_tree().factory_calls
    assert any(call.module == REGISTRY for call in calls)


def test_only_the_registry_wraps_a_provider_in_metering() -> None:
    violations = metered_violations(scan_tree().metered)
    assert not violations, _report(list(violations))


def test_the_registry_is_where_providers_are_metered() -> None:
    # Guards against a vacuous pass: the scan must see the registry's own construction.
    assert any(found.module == REGISTRY for found in scan_tree().metered)


def test_the_registry_is_where_embedders_are_metered() -> None:
    # Guards against a vacuous pass: the scan must see the registry's MeteredEmbedder.
    assert any(
        found.module == REGISTRY and METERED_EMBEDDER in found.what for found in scan_tree().metered
    )


def test_local_models_are_loaded_only_by_the_registry_built_client() -> None:
    # Guards against a vacuous pass: the scan must see the one place a local model is
    # constructed, inside LocalModels (made only by the local factory's create_client).
    found = scan_tree().local_models
    assert [f.module for f in found] == [LOCAL_MODELS_MODULE], _report(list(found))
