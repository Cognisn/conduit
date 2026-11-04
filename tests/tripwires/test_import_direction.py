"""Import-direction tripwires on the real source tree (plan section 4, C2, C10).

These are never skipped or weakened. A failure names the file, line and import.
"""

from __future__ import annotations

from tests.tripwires.importscan import (
    CLI,
    CORE,
    OPTIONAL_LIBRARIES,
    PACKAGE,
    SRC_ROOT,
    cli_violations,
    core_violations,
    module_name,
    optional_library_violations,
    scan_tree,
    source_files,
)


def _report(violations: list[object]) -> str:
    return "\n".join(str(v) for v in violations)


def test_the_scan_covers_the_package() -> None:
    # Guards against a vacuous pass: the scan must see the layers it rules on.
    modules = {module_name(path, SRC_ROOT) for path in source_files()}
    assert {PACKAGE, CORE, CLI} <= modules


def test_core_imports_no_other_conduit_package() -> None:
    violations = core_violations(scan_tree())
    assert not violations, _report(list(violations))


def test_optional_libraries_are_imported_only_by_their_layer() -> None:
    violations = optional_library_violations(scan_tree())
    assert not violations, _report(list(violations))


def test_nothing_outside_cli_imports_cli() -> None:
    violations = cli_violations(scan_tree())
    assert not violations, _report(list(violations))


def test_every_owner_layer_sits_in_an_existing_package() -> None:
    # An owner may be a module a later wave creates, but its parent package must
    # exist now, so a misspelt owner cannot silently admit an import everywhere.
    modules = {module_name(path, SRC_ROOT) for path in source_files()}
    for library, owner in OPTIONAL_LIBRARIES.items():
        parent = owner.rpartition(".")[0]
        assert owner in modules or parent in modules, f"{library}: no package for {owner}"
