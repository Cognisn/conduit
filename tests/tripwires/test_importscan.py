"""The scanner and rules, shown imports they exist to catch (synthetic trees)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tripwires.importscan import (
    cli_violations,
    core_violations,
    optional_library_violations,
    scan_source,
    scan_tree,
)


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "src"
    for rel, body in {"conduit/__init__.py": "", **files}.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return src


def _imported(source: str, module: str = "conduit.core.types", package: bool = False) -> set[str]:
    edges = scan_source(source, module, is_package=package, path=Path("x.py"))
    return {edge.imported for edge in edges}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import conduit.agent", "conduit.agent"),
        ("from conduit import agent", "conduit.agent"),
        ("from .. import agent", "conduit.agent"),
        ("from ..agent.runner import Runner", "conduit.agent.runner"),
        ("from . import events", "conduit.core.events"),
        ("def f():\n    import conduit.agent\n", "conduit.agent"),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import anthropic\n",
            "anthropic",
        ),
        ("import importlib\nimportlib.import_module('conduit.agent')", "conduit.agent"),
        ("from importlib import import_module\nimport_module('boto3')", "boto3"),
        ("__import__('openai')", "openai"),
        ("import importlib\nimportlib.import_module('..agent', __package__)", "conduit.agent"),
    ],
)
def test_the_scanner_resolves_every_import_form(source: str, expected: str) -> None:
    assert expected in _imported(source)


def test_relative_imports_resolve_from_a_package_init() -> None:
    assert "conduit.core.events" in _imported(
        "from .events import Event", "conduit.core", package=True
    )
    assert "conduit.agent" in _imported("from .. import agent", "conduit.core", package=True)


def test_core_rule_catches_another_conduit_package(tmp_path: Path) -> None:
    src = _tree(
        tmp_path,
        {
            "conduit/core/__init__.py": "from conduit.core import types\n",
            "conduit/core/types.py": "from ..providers import stub\n",
            "conduit/core/top.py": "import conduit\n",
            "conduit/providers/__init__.py": "from conduit.core import types\n",
        },
    )
    found = {(e.importer, e.imported) for e in core_violations(scan_tree(src))}
    assert found == {
        ("conduit.core.types", "conduit.providers"),
        ("conduit.core.types", "conduit.providers.stub"),
        ("conduit.core.top", "conduit"),
    }


def test_optional_library_rule_admits_only_the_owner(tmp_path: Path) -> None:
    src = _tree(
        tmp_path,
        {
            "conduit/providers/__init__.py": "",
            "conduit/providers/bedrock.py": "import boto3\nfrom botocore import config\n",
            "conduit/providers/anthropic/__init__.py": "import anthropic\n",
            "conduit/agent/__init__.py": "import anthropic.types\n",
            "conduit/core/__init__.py": "def f():\n    import boto3\n",
            "conduit/stores/__init__.py": "from mcp import ClientSession\n",
            "conduit/mcp/__init__.py": "import mcp\nimport sqlalchemy\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {
        ("conduit.agent", "anthropic.types"),
        ("conduit.core", "boto3"),
        ("conduit.stores", "mcp"),
        ("conduit.stores", "mcp.ClientSession"),
        ("conduit.mcp", "sqlalchemy"),
    }


def test_a_library_sharing_a_name_prefix_is_not_confused(tmp_path: Path) -> None:
    src = _tree(tmp_path, {"conduit/agent/__init__.py": "import openai_helpers\nimport mcpx\n"})
    assert optional_library_violations(scan_tree(src)) == []


def test_cli_rule_catches_an_import_from_outside(tmp_path: Path) -> None:
    src = _tree(
        tmp_path,
        {
            "conduit/cli/__init__.py": "from conduit.cli import app\n",
            "conduit/cli/app.py": "from conduit import agent\n",
            "conduit/agent/__init__.py": "from conduit.cli.app import main\n",
            "conduit/testing/__init__.py": "from conduit import cli\n",
        },
    )
    found = {(e.importer, e.imported) for e in cli_violations(scan_tree(src))}
    assert found == {
        ("conduit.agent", "conduit.cli.app"),
        ("conduit.agent", "conduit.cli.app.main"),
        ("conduit.testing", "conduit.cli"),
    }
