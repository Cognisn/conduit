"""The agreed licence exceptions: certifi (Q0016), tqdm (Q0032), psycopg (Q0036) and fido2
(Q0056), only for weak copyleft."""

from __future__ import annotations

import pytest

from licence_audit import EXCEPTIONS, Package, audit


def _tree(dependency: Package) -> dict[str, Package]:
    root = Package(name="root", licences=["MIT"], requires=[dependency.name], extras=[])
    return {"root": root, dependency.name: dependency}


def test_certifi_tqdm_psycopg_and_fido2_are_the_only_named_exceptions() -> None:
    # Re-pointed for Q0032: tqdm joins certifi, agreed for the local-embeddings extra.
    # Re-pointed for Q0036: psycopg joins them, agreed for the warden extra.
    # Re-pointed for Q0056: fido2 joins them, required by cognisn-warden from 0.1.0b18.
    assert set(EXCEPTIONS) == {"certifi", "tqdm", "psycopg", "fido2"}
    assert "MPL-2.0" in EXCEPTIONS["certifi"]
    assert "MPL-2.0 AND MIT" in EXCEPTIONS["tqdm"]
    assert "Q0032" in EXCEPTIONS["tqdm"]
    assert "LGPL-3.0" in EXCEPTIONS["psycopg"]
    assert "Q0036" in EXCEPTIONS["psycopg"]
    assert "MPL-2.0" in EXCEPTIONS["fido2"]
    assert "Q0056" in EXCEPTIONS["fido2"]


@pytest.mark.parametrize(
    ("name", "licences"),
    [
        ("certifi", ["MPL-2.0"]),
        ("tqdm", ["MPL-2.0 AND MIT"]),
        ("psycopg", ["LGPL-3.0-only"]),
        ("fido2", ["BSD-2-Clause", "Apache-2.0", "MPL-2.0"]),
    ],
)
def test_each_exception_under_its_weak_copyleft_licence_passes(
    name: str, licences: list[str]
) -> None:
    package = Package(name=name, licences=licences, requires=[], extras=[])
    report = audit("root", lookup=_tree(package).__getitem__)
    assert report.violations == []
    assert [p.name for p in report.excepted] == [name]


@pytest.mark.parametrize("licence", ["MPL-2.0", "LGPL-3.0-only"])
def test_another_weak_copyleft_package_is_still_refused(licence: str) -> None:
    other = Package(name="other-bundle", licences=[licence], requires=[], extras=[])
    report = audit("root", lookup=_tree(other).__getitem__)
    assert [p.name for p in report.violations] == ["other-bundle"]


@pytest.mark.parametrize(
    "name",
    [
        "psycopg-binary",
        "psycopg-c",
        "psycopg-pool",
        "psycopg2",
        "certifi-extra",
        "tqdm2",
        "fido2-extra",
        "fido",
    ],
)
def test_an_exception_names_one_package_exactly(name: str) -> None:
    # A package whose name merely starts with an excepted one is not excepted.
    package = Package(name=name, licences=["LGPL-3.0-only"], requires=[], extras=[])
    report = audit("root", lookup=_tree(package).__getitem__)
    assert [p.name for p in report.violations] == [name]
    assert report.excepted == []


@pytest.mark.parametrize("name", ["certifi", "tqdm", "psycopg", "fido2"])
def test_an_exception_under_strong_copyleft_is_refused_despite_it(name: str) -> None:
    package = Package(name=name, licences=["GPL-3.0-only"], requires=[], extras=[])
    report = audit("root", lookup=_tree(package).__getitem__)
    assert [p.name for p in report.violations] == [name]
