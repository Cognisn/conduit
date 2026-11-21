"""The terminal interface uses conduit's public API only (C10), and rich stays inside it.

Public means: a name without a leading underscore, imported from a conduit module whose
dotted path has no component with a leading underscore, and listed in that module's
``__all__`` when it declares one. The interface also reaches no underscored attribute
of anything but itself. The source is scanned, never imported, so an import in a
function or behind ``TYPE_CHECKING`` counts too.

It also imports each name from the package that exports it (D0091, stricter reading):
never from ``conduit.core`` or one of its modules, whose names ``conduit`` exports, and
never from a module when a parent package (``conduit`` or a layer's package) exports
the name, or, for a module imported as such, any of its names.
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


def _core(module: str) -> bool:
    return module == "conduit.core" or module.startswith("conduit.core.")


def _covering_package(module: str, name: str) -> str | None:
    """The package above ``module`` that exports ``name`` in its ``__all__``, if any."""
    parts = module.split(".")
    for depth in range(1, len(parts)):
        package = ".".join(parts[:depth])
        try:
            found = importlib.import_module(package)
        except ImportError:
            continue
        if name in getattr(found, "__all__", ()):
            return package
    return None


def _covering_module_package(module: str) -> str | None:
    """The layer package exporting a name the module ``module`` holds, if one does.

    The top level is left to the name rule and the core rule: a layer module may well
    hold a core name conduit exports, imported for its own use.
    """
    try:
        imported = importlib.import_module(module)
    except ImportError:
        return None
    parts = module.split(".")
    for depth in range(2, len(parts)):
        package = ".".join(parts[:depth])
        source = importlib.import_module(package)
        for name in getattr(source, "__all__", ()):
            if getattr(imported, name, None) is getattr(source, name, object()):
                return package
    return None


def _is_module(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _bypass(module: str, name: str | None) -> str | None:
    """Why importing ``name`` from ``module`` (or ``module`` itself) bypasses a package.

    D0091, stricter reading: the core is reached only through ``conduit``, and a name or
    module a parent package covers only through that package.
    """
    target = module if name is None else f"{module}.{name}"
    if _core(module) or _core(target):
        return "import it from conduit"
    if name is not None and not _is_module(target):
        package = _covering_package(module, name)
        return None if package is None else f"import it from {package}"
    package = _covering_module_package(target)
    return None if package is None else f"use the names {package} exports"


def private_uses(source: str, path: Path) -> list[Use]:
    """Every private conduit name or attribute ``source`` reaches, or covered path."""
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
                elif reason := _bypass(module, alias.name):
                    found.append(Use(path, node.lineno, f"{module}.{alias.name}: {reason}"))
        elif isinstance(node, ast.ImportFrom) and node.level:
            found.append(Use(path, node.lineno, "a relative import"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] != "conduit":
                    continue
                if any(_private(p) for p in parts):
                    found.append(Use(path, node.lineno, f"import {alias.name}"))
                elif reason := _bypass(alias.name, None):
                    found.append(Use(path, node.lineno, f"import {alias.name}: {reason}"))
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
        ("source", "reason"),
        [
            ("from conduit.core.tools import TaskTools", "conduit.core.tools.TaskTools"),
            (
                "from conduit.core.events import CapturePolicy as Policy",
                "conduit.core.events.CapturePolicy",
            ),
            (
                "def f():\n    from conduit.core.settings import refusals",
                "conduit.core.settings.refusals",
            ),
            ("from conduit.core import tools", "conduit.core.tools"),
            ("from conduit import core", "conduit.core"),
            ("import conduit.core.tools", "import conduit.core.tools"),
            ("import conduit.core", "import conduit.core"),
            ("import conduit.core.tools as t", "import conduit.core.tools"),
            ("from conduit.stores.json import JsonTranscriptStore", "conduit.stores.json"),
            ("from conduit.stores.memory import MemoryTokenStore as M", "conduit.stores.memory"),
            ("from conduit.agent.events import ToolCallEvent", "conduit.agent.events"),
            ("import conduit.stores.json", "import conduit.stores.json"),
            ("import conduit.stores.json as j", "import conduit.stores.json"),
            ("from conduit.stores import json", "conduit.stores.json"),
        ],
        ids=[
            "core-submodule",
            "core-aliased",
            "core-nested",
            "core-submodule-by-name",
            "core-by-name",
            "import-core-submodule",
            "import-core",
            "import-core-submodule-as",
            "layer-submodule",
            "layer-aliased",
            "top-level-exports-it",
            "import-layer-submodule",
            "import-layer-submodule-as",
            "layer-submodule-by-name",
        ],
    )
    def test_a_submodule_its_package_covers_is_refused(self, source: str, reason: str) -> None:
        # D0091, stricter reading: a name conduit or a layer's package exports is
        # imported from that package, never from the module it lives in, and the core
        # is never imported from directly.
        uses = private_uses(source, Path("synthetic.py"))
        assert any(reason in use.what for use in uses), uses

    @pytest.mark.parametrize(
        "source",
        [
            "from conduit import ConduitSettings",
            "from conduit.agent import AgentRunner",
            "from conduit.agent.events import RunFinishedEvent",
            "from conduit.providers.stub import StubFactory",
            "import conduit.providers.stub",
            "from conduit.stores import JsonTranscriptStore",
            "from conduit import TaskTools, refusals",
            "from conduit.cli.config import CHAT",
            "from rich.console import Console",
            "class A:\n    def f(self):\n        return self._x",
            "x = object().__class__",
        ],
    )
    def test_public_use_passes(self, source: str) -> None:
        assert private_uses(source, Path("synthetic.py")) == []


def test_the_interface_imports_no_core_module() -> None:
    # The source rule directly, so it cannot pass only because a scan found nothing.
    for path in cli_sources():
        text = path.read_text(encoding="utf-8")
        assert "conduit.core" not in text, path.name


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
