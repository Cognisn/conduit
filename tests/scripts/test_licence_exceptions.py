"""The agreed licence exceptions: certifi only, and only for weak copyleft (Q0016)."""

from __future__ import annotations

from licence_audit import EXCEPTIONS, Package, audit


def _tree(dependency: Package) -> dict[str, Package]:
    root = Package(name="root", licences=["MIT"], requires=[dependency.name], extras=[])
    return {"root": root, dependency.name: dependency}


def test_certifi_is_the_only_named_exception() -> None:
    assert set(EXCEPTIONS) == {"certifi"}
    assert "MPL-2.0" in EXCEPTIONS["certifi"]


def test_certifi_under_mpl_passes_as_an_exception() -> None:
    certifi = Package(name="certifi", licences=["MPL-2.0"], requires=[], extras=[])
    report = audit("root", lookup=_tree(certifi).__getitem__)
    assert report.violations == []
    assert [p.name for p in report.excepted] == ["certifi"]


def test_another_mpl_package_is_still_refused() -> None:
    other = Package(name="other-bundle", licences=["MPL-2.0"], requires=[], extras=[])
    report = audit("root", lookup=_tree(other).__getitem__)
    assert [p.name for p in report.violations] == ["other-bundle"]


def test_certifi_under_strong_copyleft_is_refused_despite_the_exception() -> None:
    certifi = Package(name="certifi", licences=["GPL-3.0-only"], requires=[], extras=[])
    report = audit("root", lookup=_tree(certifi).__getitem__)
    assert [p.name for p in report.violations] == ["certifi"]
