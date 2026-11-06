"""The agreed licence exceptions: certifi (Q0016), tqdm (Q0032) and psycopg (Q0036), only
for weak copyleft."""

from __future__ import annotations

import pytest

from licence_audit import EXCEPTIONS, Package, audit


def _tree(dependency: Package) -> dict[str, Package]:
    root = Package(name="root", licences=["MIT"], requires=[dependency.name], extras=[])
    return {"root": root, dependency.name: dependency}


def test_certifi_tqdm_and_psycopg_are_the_only_named_exceptions() -> None:
    # Re-pointed for Q0032: tqdm joins certifi, agreed for the local-embeddings extra.
    # Re-pointed for Q0036: psycopg joins them, agreed for the warden extra.
    assert set(EXCEPTIONS) == {"certifi", "tqdm", "psycopg"}
    assert "MPL-2.0" in EXCEPTIONS["certifi"]
    assert "MPL-2.0 AND MIT" in EXCEPTIONS["tqdm"]
    assert "Q0032" in EXCEPTIONS["tqdm"]
    assert "LGPL-3.0" in EXCEPTIONS["psycopg"]
    assert "Q0036" in EXCEPTIONS["psycopg"]


@pytest.mark.parametrize(
    ("name", "licences"),
    [("certifi", ["MPL-2.0"]), ("tqdm", ["MPL-2.0 AND MIT"]), ("psycopg", ["LGPL-3.0-only"])],
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


@pytest.mark.parametrize("name", ["certifi", "tqdm", "psycopg"])
def test_an_exception_under_strong_copyleft_is_refused_despite_it(name: str) -> None:
    package = Package(name=name, licences=["GPL-3.0-only"], requires=[], extras=[])
    report = audit("root", lookup=_tree(package).__getitem__)
    assert [p.name for p in report.violations] == [name]
