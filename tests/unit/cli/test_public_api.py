"""The terminal interface uses conduit's public API only (C10), and rich stays inside it.

Public means: a name without a leading underscore, imported from a conduit module whose
dotted path has no component with a leading underscore, and listed in that module's
``__all__`` when it declares one. The interface also reaches no underscored attribute
of anything but itself. The source is scanned, never imported, so an import in a
function or behind ``TYPE_CHECKING`` counts too.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.tripwires.importscan import (
    OPTIONAL_LIBRARIES,
    SRC_ROOT,
    module_name,
    optional_library_violations,
    scan_tree,
)

CLI_ROOT = SRC_ROOT / "conduit" / "cli"


@dataclass(frozen=True)
class Use:
    path: Path
    line: int
    what: str

    def __str__(self) -> str:
        return f"{self.path.name}:{self.line}: {self.what}"


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _exported(module: str, name: str) -> bool:
    """Whether ``module`` offers ``name`` publicly (its ``__all__``, or a submodule)."""
    try:
        found = importlib.import_module(module)
    except ImportError:
        return True  # an extra that is not installed: the path rule still applies
    listed = getattr(found, "__all__", None)
    if listed is None or name in listed:
        return True
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return False
    return True


def private_uses(source: str, path: Path) -> list[Use]:
    """Every private conduit name or attribute ``source`` reaches."""
    found: list[Use] = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module = node.module
            if module.split(".")[0] != "conduit":
                continue
            if any(_private(part) for part in module.split(".")):
                found.append(Use(path, node.lineno, f"from {module}"))
                continue
            for alias in node.names:
                if _private(alias.name) or not _exported(module, alias.name):
                    found.append(Use(path, node.lineno, f"{module}.{alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.level:
            found.append(Use(path, node.lineno, "a relative import"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "conduit" and any(_private(p) for p in parts):
                    found.append(Use(path, node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.Attribute) and _private(node.attr):
            owner = node.value
            if not (isinstance(owner, ast.Name) and owner.id in {"self", "cls"}):
                found.append(Use(path, node.lineno, f"attribute {node.attr}"))
    return found


def cli_sources() -> list[Path]:
    return sorted(CLI_ROOT.rglob("*.py"))


def test_the_interface_has_sources_to_check() -> None:
    names = {path.stem for path in cli_sources()}
    assert {"app", "chat", "status", "servers", "output"} <= names


@pytest.mark.parametrize("path", cli_sources(), ids=lambda p: p.name)
def test_every_module_uses_the_public_api_only(path: Path) -> None:
    uses = private_uses(path.read_text(encoding="utf-8"), path)
    assert uses == [], "\n".join(str(use) for use in uses)


class TestTheScanCatchesPrivateUse:
    @pytest.mark.parametrize(
        "source",
        [
            "from conduit.agent.runner import _Run",
            "from conduit.core._hidden import thing",
            "import conduit.core._hidden",
            "from conduit.skills import render_listing as r\nx = r._cache",
            "from conduit.agent.untrusted import DEFAULT_MAX_LENGTH, _mark",
            "from conduit.compaction.compactor import logger",
            "from .chat import Chat",
            "def f(provider):\n    return provider._inner",
        ],
    )
    def test_it_is_found(self, source: str) -> None:
        assert private_uses(source, Path("synthetic.py")), source

    @pytest.mark.parametrize(
        "source",
        [
            "from conduit import ConduitSettings",
            "from conduit.agent import AgentRunner",
            "from conduit.agent.events import RunFinishedEvent",
            "from conduit.providers.stub import StubFactory",
            "from rich.console import Console",
            "class A:\n    def f(self):\n        return self._x",
            "x = object().__class__",
        ],
    )
    def test_public_use_passes(self, source: str) -> None:
        assert private_uses(source, Path("synthetic.py")) == []


RICH_OWNERS = {**OPTIONAL_LIBRARIES, "rich": ("conduit.cli",)}


def test_rich_is_imported_only_inside_the_interface() -> None:
    edges = scan_tree()
    assert any(edge.imported.startswith("rich") for edge in edges), "the scan sees no rich"
    assert optional_library_violations(edges, RICH_OWNERS) == []


def test_the_rich_rule_catches_an_import_elsewhere() -> None:
    from tests.tripwires.importscan import scan_source

    path = SRC_ROOT / "conduit" / "agent" / "synthetic.py"
    edges = scan_source(
        "from rich.console import Console",
        module_name(path, SRC_ROOT),
        is_package=False,
        path=path,
    )
    assert optional_library_violations(edges, RICH_OWNERS) == edges
