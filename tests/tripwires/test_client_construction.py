"""Tripwire: no provider client is constructed outside the provider registry.

Never skipped or weakened. A failure names the file, line and call.
"""

from __future__ import annotations

from tests.tripwires.clientscan import (
    REGISTRY,
    construction_violations,
    factory_call_violations,
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
