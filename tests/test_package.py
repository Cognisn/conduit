"""The package layout of plan section 4, and the single version source."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import conduit

LAYERS = [
    "conduit.core",
    "conduit.providers",
    "conduit.embeddings",
    "conduit.embeddings.local",
    "conduit.agent",
    "conduit.compaction",
    "conduit.skills",
    "conduit.mcp",
    "conduit.stores",
    "conduit.stores.sql",
    "conduit.integrations",
    "conduit.integrations.konfig",
    "conduit.integrations.warden",
    "conduit.integrations.otel",
    "conduit.testing",
    "conduit.cli",
]

EXTRAS = {
    "bedrock",
    "anthropic",
    "openai",
    "gemini",
    "ollama",
    "all-providers",
    "local-embeddings",
    "mcp",
    "sql",
    "konfig",
    "warden",
    "otel",
    "cli",
}

VERSION_FILE = Path(conduit.__file__).with_name("_version.txt")


@pytest.mark.parametrize("name", LAYERS)
def test_every_layer_imports_with_a_one_line_docstring(name: str) -> None:
    module = importlib.import_module(name)
    doc = module.__doc__
    assert doc, f"{name} has no docstring"
    assert "\n" not in doc.strip(), f"{name} docstring is not one line"


def test_the_package_is_typed() -> None:
    assert Path(conduit.__file__).with_name("py.typed").is_file()


def test_the_version_comes_from_the_version_file() -> None:
    text = VERSION_FILE.read_bytes().decode("utf-8")
    assert text.endswith("\n") and "\r" not in text
    assert conduit.__version__ == text.strip()
    Version(conduit.__version__)  # raises if not PEP 440


def test_the_installed_metadata_matches_the_version_file() -> None:
    # Fails when the editable install is stale: re-install after a stamp.
    assert Version(metadata.version("cognisn-conduit")) == Version(conduit.__version__)


def test_every_planned_extra_is_declared() -> None:
    declared = set(metadata.metadata("cognisn-conduit").get_all("Provides-Extra") or [])
    assert declared - {"dev"} == EXTRAS


def test_the_core_install_depends_on_pydantic_alone() -> None:
    # The core's only agreed runtime dependency is pydantic (Wave 1 kick-off).
    requires = metadata.requires("cognisn-conduit") or []
    core = [Requirement(r) for r in requires if "extra ==" not in r]
    assert [r.name for r in core] == ["pydantic"]
    assert core[0].specifier.contains("2.9") and not core[0].specifier.contains("3.0")
