"""scripts/licence_audit.py, shown dependency trees it exists to refuse."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata

import pytest
from packaging.requirements import Requirement

from licence_audit import NOT_AUDITED, ROOT, Package, audit, classify, installed, main


@pytest.mark.parametrize(
    ("licences", "verdict"),
    [
        (["MIT"], "permissive"),
        (["Apache-2.0", "License :: OSI Approved :: Apache Software License"], "permissive"),
        (["GPL-3.0-only"], "strong copyleft"),
        (["AGPL-3.0-or-later"], "strong copyleft"),
        (["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"], "strong copyleft"),
        (["LGPL-3.0-only"], "weak copyleft"),
        (["MPL-2.0"], "weak copyleft"),
        (["GPL-2.0-only OR MIT"], "permissive"),
        (["MIT", "LGPL-2.1"], "weak copyleft"),
        ([""], "unknown"),
        (["UNKNOWN", "License :: OSI Approved"], "unknown"),
    ],
)
def test_classify(licences: list[str], verdict: str) -> None:
    assert classify(licences) == verdict


def _lookup(tree: dict[str, Package]) -> Callable[[str], Package]:
    return lambda name: tree[name]


TREE = {
    "cognisn-conduit": Package(
        "cognisn-conduit",
        ["MIT"],
        [
            "pydantic>=2",
            'boto3>=1.35; extra == "bedrock"',
            'gpl-sdk>=1; extra == "otel"',
            'pytest>=8; extra == "dev"',
        ],
        ["bedrock", "otel", "dev"],
    ),
    "pydantic": Package("pydantic", ["MIT"]),
    "boto3": Package("boto3", ["Apache-2.0"], ["lgpl-helper[fast]>=1"]),
    "lgpl-helper": Package("lgpl-helper", ["LGPL-3.0"], ['gpl-speedups; extra == "fast"']),
    "gpl-speedups": Package("gpl-speedups", ["GPL-3.0"]),
    "gpl-sdk": Package("gpl-sdk", ["GPL-3.0"]),
    "pytest": Package("pytest", ["GPL-3.0"]),
}


def test_the_core_install_ignores_every_extra() -> None:
    report = audit(lookup=_lookup(TREE))
    assert [p.name for p in report.checked] == ["cognisn-conduit", "pydantic"]
    assert report.violations == []


def test_an_extra_is_audited_with_its_transitive_extras() -> None:
    report = audit(lookup=_lookup(TREE), extras=["bedrock"])
    assert {p.name for p in report.violations} == {"lgpl-helper", "gpl-speedups"}


def test_a_named_exception_admits_weak_but_never_strong_copyleft() -> None:
    report = audit(lookup=_lookup(TREE), extras=["bedrock"], exceptions={"lgpl-helper": "r"})
    assert [p.name for p in report.excepted] == ["lgpl-helper"]
    assert [p.name for p in report.violations] == ["gpl-speedups"]
    report = audit(lookup=_lookup(TREE), extras=["otel"], exceptions={"gpl-sdk": "r"})
    assert [p.name for p in report.violations] == ["gpl-sdk"]


def test_an_undeclared_licence_is_refused() -> None:
    tree = {**TREE, "pydantic": Package("pydantic", ["UNKNOWN"])}
    assert [p.name for p in audit(lookup=_lookup(tree)).violations] == ["pydantic"]


def test_the_dev_extra_is_never_audited() -> None:
    assert main(["--extra", "dev"]) == 2


def _uninstalled_extra_requirements() -> list[str]:
    """Requirements of a runtime extra that this environment has not installed."""
    root = installed(ROOT)
    missing: list[str] = []
    for spec in root.requires:
        requirement = Requirement(spec)
        marker = requirement.marker
        if marker is None or marker.evaluate({"extra": ""}):
            continue  # a core requirement, always installed
        if not any(marker.evaluate({"extra": e}) for e in root.extras if e not in NOT_AUDITED):
            continue  # a dev requirement
        try:
            metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(requirement.name)
    return sorted(set(missing))


def test_the_installed_package_passes_with_every_extra() -> None:
    # The test job installs only the dev extra; the CI licences job installs every
    # runtime extra and runs the full audit. Here, audit them all when they are present.
    missing = _uninstalled_extra_requirements()
    if missing:
        pytest.skip(f"runtime extras not installed here ({', '.join(missing)})")
    assert main(["--all-extras"]) == 0
