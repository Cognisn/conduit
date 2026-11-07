"""Every Python example in the guides, the README and the architecture document runs.

Each ``python`` fenced block of a document is executed, in order, in one namespace per
document (so a later block may use what an earlier one defined), in a fresh temporary
working directory. Examples call the stub provider, local stores and in-process
servers only: nothing here reaches a network or a live model.

The guard has three parts, each of which fails rather than silently checking less:

- **What is checked.** ``EXPECTED_EXAMPLES`` names every document and how many Python
  examples it holds. A document added, removed or missed by discovery, or an example
  added or lost, fails until the table is updated.
- **That each example ran.** Every block is compiled with a counter increment after its
  code, and the count is compared with the table, so a harness that stops executing
  fails.
- **That no example escapes.** A fence is three backticks at the start of a line, with
  ``python`` or a language that is never Python. Indented, tilde and longer fences, which
  the parser would not see, are refused, as are untagged fences, and a never-Python
  fence holding a line that reads as Python (an import, an assert, a definition).

Examples also show the public import path: a name exported from ``conduit``, or from a
layer's package, is never imported from the submodule it lives in, and no example
imports a private module path (a segment with a leading underscore), aliased or not.

A document whose examples import a layer behind an extra that is not installed locally
is skipped, naming the extra; the checks that import nothing (private paths, core
modules, ``import *``) run first, so a skipped document still has them. CI installs
every extra and sets ``CONDUIT_ALL_EXTRAS=require``, the core's switch, which turns
that skip into a failure and makes a check fail when any runtime extra is missing; any
other value of the switch is refused.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.metadata
import re
import sys
import types
from collections.abc import Iterable
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from licence_audit import audit
from tests.tripwires.importscan import OPTIONAL_LIBRARIES
from tests.tripwires.test_event_content import EVERY_EXTRA_SWITCH, every_extra_required

ROOT = Path(__file__).resolve().parents[2]

#: Every document whose examples are run, with its number of Python examples.
EXPECTED_EXAMPLES: dict[str, int] = {
    "README.md": 1,
    "docs/design/architecture.md": 0,
    "docs/guides/README.md": 0,
    "docs/guides/adoption.md": 0,
    "docs/guides/backends.md": 13,
    "docs/guides/compaction.md": 1,
    "docs/guides/embeddings.md": 2,
    "docs/guides/integration.md": 14,
    "docs/guides/mcp.md": 4,
    "docs/guides/observability.md": 3,
    "docs/guides/providers.md": 11,
    "docs/guides/skills.md": 2,
    "docs/guides/terminal.md": 2,
}

#: Documents that must always be checked, whatever discovery finds.
PINNED = ("README.md", "docs/design/architecture.md")

DOCUMENTS = tuple(
    sorted(
        {ROOT / name for name in PINNED} | set((ROOT / "docs" / "guides").glob("*.md")),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
)

#: Fence languages that are never run: shell lines, configuration and output.
NOT_PYTHON = frozenset({"console", "json", "markdown", "powershell", "sh", "text", "toml", "yaml"})

_FENCE = re.compile(r"^```(?P<info>[^\n`]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
#: Any line a Markdown renderer might take as a fence: backticks or tildes, indented or not.
_FENCE_LIKE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
#: The only fence lines allowed: three backticks at the start of the line, then a language.
_FENCE_LINE = re.compile(r"^```[a-z]*$")
#: A line in a never-Python fence that reads as Python.
_PYTHON_LINE = re.compile(
    r"^\s*(?:import \w|from [\w.]+ import |assert |def \w|async def \w|class \w|await \w)"
)

_COUNTER = "__conduit_guide_examples_run__"


def _blocks(text: str) -> list[tuple[str, int, str]]:
    """Every fenced block: its language, the line its code starts on, and its code."""
    found: list[tuple[str, int, str]] = []
    for match in _FENCE.finditer(text):
        language = match.group("info").strip()
        line = text.count("\n", 0, match.start("body")) + 1
        found.append((language, line, match.group("body")))
    return found


def fence_problems(text: str) -> list[str]:
    """Why ``text``'s fences could let an example escape the run; empty when none."""
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE_LIKE.match(line) and not _FENCE_LINE.match(line):
            problems.append(f"line {number}: a fence must be three backticks at line start")
    for language, start, code in _blocks(text):
        if language != "python" and language not in NOT_PYTHON:
            problems.append(
                f"line {start}: a fence must be python or one of {sorted(NOT_PYTHON)}, "
                f"not {language!r}"
            )
        if language != "python":
            for offset, line in enumerate(code.splitlines()):
                if _PYTHON_LINE.match(line):
                    problems.append(
                        f"line {start + offset}: Python in a {language!r} fence; fence it "
                        "as python so it runs"
                    )
    return problems


def _private_path(name: str) -> bool:
    """Whether a dotted name has a segment with a leading underscore (dunders aside)."""
    return any(
        part.startswith("_") and not (part.startswith("__") and part.endswith("__"))
        for part in name.split(".")
    )


@functools.cache
def _exports(package: str) -> frozenset[str]:
    """The names ``package`` exports in its ``__all__``."""
    return frozenset(getattr(importlib.import_module(package), "__all__", ()))


def _submodule(package: str, name: str) -> str | None:
    """``package.name`` when it is a module, else None."""
    if isinstance(getattr(importlib.import_module(package), name, None), types.ModuleType):
        return f"{package}.{name}"
    try:
        importlib.import_module(f"{package}.{name}")
    except ModuleNotFoundError:
        return None
    return f"{package}.{name}"


def _module_problem(module: str) -> str | None:
    """Why importing the module ``module`` itself bypasses the public path, if it does.

    The core and its submodules are never imported as modules: conduit exports their
    names. A layer's submodule is refused when its package exports any of its names.
    """
    if module == "conduit.core" or module.startswith("conduit.core."):
        return f"import {module}: use the names conduit exports"
    parts = module.split(".")
    if parts[0] != "conduit":
        return None
    imported = importlib.import_module(module)
    for depth in range(2, len(parts)):
        package = ".".join(parts[:depth])
        source = importlib.import_module(package)
        if any(
            getattr(imported, name, None) is getattr(source, name) for name in _exports(package)
        ):
            return f"import {module}: use the names {package} exports"
    return None


def _core_path(name: str) -> bool:
    return name == "conduit.core" or name.startswith("conduit.core.")


def import_problems(text: str, *, resolve: bool = True) -> list[str]:
    """Imports in ``text``'s Python examples that bypass a package exporting the name.

    A name exported from ``conduit`` is imported from ``conduit``, and a name a layer's
    package exports (``conduit.stores``) from that package, never from the submodule
    it lives in, so the guides show the public import path (Wave 6 ruling Q0044). The
    same holds for importing such a submodule as a module, and every conduit import
    names what it takes (no ``import *``).

    With ``resolve`` false only the checks that import nothing run: private paths,
    ``import *`` and the core's modules. They hold even where a layer's extra is missing.
    """
    problems: list[str] = []
    for language, start, code in _blocks(text):
        if language != "python":
            continue
        for node in ast.walk(ast.parse(code)):
            where = f"line {start + getattr(node, 'lineno', 1) - 1}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _private_path(alias.name):
                        problems.append(f"{where}: {alias.name} is a private module path")
                        continue
                    if not resolve:
                        if _core_path(alias.name):
                            problems.append(
                                f"{where}: import {alias.name}: use the names conduit exports"
                            )
                        continue
                    problem = _module_problem(alias.name)
                    if problem:
                        problems.append(f"{where}: {problem}")
                continue
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if _private_path(node.module):
                problems.append(f"{where}: {node.module} is a private module path")
                continue
            parts = node.module.split(".")
            for alias in node.names:
                if _private_path(alias.name):
                    problems.append(f"{where}: {node.module}.{alias.name} is a private module path")
            if parts[0] != "conduit":
                continue
            for alias in node.names:
                if _private_path(alias.name):
                    continue
                if alias.name == "*":
                    problems.append(f"{where}: import * from {node.module}: name each import")
                    continue
                if not resolve:
                    if _core_path(f"{node.module}.{alias.name}"):
                        problems.append(
                            f"{where}: {alias.name} from {node.module}: use the names conduit "
                            "exports"
                        )
                    continue
                submodule = _submodule(node.module, alias.name)
                if submodule is not None:
                    problem = _module_problem(submodule)
                    if problem:
                        problems.append(f"{where}: {problem}")
                    continue
                for depth in range(1, len(parts)):
                    package = ".".join(parts[:depth])
                    if alias.name in _exports(package):
                        problems.append(
                            f"{where}: import {alias.name} from {package}, not {node.module}"
                        )
                        break
    return problems


#: The extra each conduit layer behind one needs, by module prefix.
LAYER_EXTRAS: dict[str, str] = {
    "conduit.providers.bedrock": "bedrock",
    "conduit.providers.bedrock_mantle": "bedrock-mantle",
    "conduit.providers.anthropic": "anthropic",
    "conduit.providers.openai_compat": "openai",
    "conduit.providers.gemini": "gemini",
    "conduit.providers.ollama": "ollama",
    "conduit.embeddings.bedrock": "bedrock",
    "conduit.embeddings.openai_compat": "openai",
    "conduit.embeddings.gemini": "gemini",
    "conduit.embeddings.ollama": "ollama",
    "conduit.embeddings.local": "local-embeddings",
    "conduit.mcp": "mcp",
    "conduit.testing": "testing",
    "conduit.stores.sql": "sql",
    "conduit.integrations.konfig": "konfig",
    "conduit.integrations.warden": "warden",
    "conduit.integrations.otel": "otel",
    "conduit.cli": "cli",
}


def _longest_prefix(module: str, prefixes: Iterable[str]) -> str | None:
    matches = [p for p in prefixes if module == p or module.startswith(f"{p}.")]
    return max(matches, key=len, default=None)


def _layer_extra(module: str) -> str | None:
    """The extra the conduit module ``module`` needs, if it is behind one."""
    prefix = _longest_prefix(module, LAYER_EXTRAS)
    return None if prefix is None else LAYER_EXTRAS[prefix]


def _library_extras(module: str) -> tuple[str, ...]:
    """The extras, any one of which installs the optional library ``module`` is in."""
    library = _longest_prefix(module, OPTIONAL_LIBRARIES)
    if library is None:
        return ()
    extras = {_layer_extra(owner) for owner in OPTIONAL_LIBRARIES[library]}
    return tuple(sorted(extra for extra in extras if extra is not None))


def extras_needed(text: str) -> list[tuple[str, ...]]:
    """The extras ``text``'s Python examples import, each as the extras any of which do."""
    modules: set[str] = set()
    for language, _start, code in _blocks(text):
        if language != "python":
            continue
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules.add(node.module)
                modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    needed: set[tuple[str, ...]] = set()
    for module in modules:
        if module.split(".")[0] == "conduit":
            extra = _layer_extra(module)
            if extra is not None:
                needed.add((extra,))
        elif extras := _library_extras(module):
            needed.add(extras)
    return sorted(needed)


@functools.cache
def _conduit_requires() -> tuple[str, ...]:
    return tuple(importlib.metadata.requires("cognisn-conduit") or ())


def extra_installed(extra: str) -> bool:
    """Whether every distribution ``extra`` names is installed (an unknown extra is not)."""
    found = False
    for line in _conduit_requires():
        requirement = Requirement(line)
        marker = requirement.marker
        if marker is None or "extra" not in str(marker) or not marker.evaluate({"extra": extra}):
            continue
        found = True
        if requirement.name == "cognisn-conduit":
            if not all(extra_installed(inner) for inner in requirement.extras):
                return False
            continue
        try:
            importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            return False
    return found


def require_extras(text: str) -> None:
    """Skip, naming the extra, when ``text`` imports a layer whose extra is missing.

    With ``CONDUIT_ALL_EXTRAS=require`` (CI installs every extra) it fails instead, so
    a run that should check every document cannot quietly check fewer. Any other value
    of the switch is refused (the core's rule), so a misspelt switch cannot turn it off.
    """
    required = every_extra_required()
    missing = [
        extras
        for extras in extras_needed(text)
        if not any(extra_installed(extra) for extra in extras)
    ]
    if not missing:
        return
    named = ", ".join(
        f'the {" or ".join(extras)} extra (pip install "cognisn-conduit[{extras[0]}]")'
        for extras in missing
    )
    if required:
        pytest.fail(
            f"needs {named}, not installed, but {EVERY_EXTRA_SWITCH}=require says every "
            "extra is installed"
        )
    pytest.skip(f"needs {named}, not installed")


def _name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# The guard on synthetic text ------------------------------------------------------


def test_blocks_reads_language_line_and_code() -> None:
    text = "intro\n```python\nx = 1\n```\n\n```sh\npip install x\n```\n"
    assert _blocks(text) == [("python", 3, "x = 1\n"), ("sh", 7, "pip install x\n")]


def test_a_clean_document_has_no_problems() -> None:
    text = "```python\nimport os\nassert os\n```\n```sh\npip install x\n```\n"
    assert fence_problems(text) == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("```py\nx = 1\n```\n", "not 'py'"),
        ("```\nx = 1\n```\n", "not ''"),
        ("   ```python\n   x = 1\n   ```\n", "three backticks at line start"),
        ("~~~python\nx = 1\n~~~\n", "three backticks at line start"),
        ("````python\nx = 1\n````\n", "three backticks at line start"),
        ("```text\nimport os\n```\n", "Python in a 'text' fence"),
        ("```console\nassert False\n```\n", "Python in a 'console' fence"),
        ("```sh\nfrom conduit import start\n```\n", "Python in a 'sh' fence"),
        ("```text\nasync def main():\n    pass\n```\n", "Python in a 'text' fence"),
    ],
    ids=[
        "py",
        "untagged",
        "indented",
        "tilde",
        "four-backticks",
        "import-in-text",
        "assert-in-console",
        "from-import-in-sh",
        "def-in-text",
    ],
)
def test_the_guard_refuses_a_fence_an_example_could_escape_through(text: str, reason: str) -> None:
    problems = fence_problems(text)
    assert problems, "the guard accepted a fence an example could escape through"
    assert any(reason in problem for problem in problems), problems


def test_a_public_import_path_has_no_problems() -> None:
    text = (
        "```python\nfrom conduit import TaskTools, start\n"
        "from conduit.stores import MemoryTokenStore\n"
        "from conduit.providers.stub import StubFactory\nimport json\n```\n"
        "```text\nfrom conduit.core.tools import TaskTools\n```\n"
    )
    assert import_problems(text) == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "```python\nfrom conduit.core.tools import TaskTools\n```\n",
            "line 2: import TaskTools from conduit, not conduit.core.tools",
        ),
        (
            "```python\nx = 1\nfrom conduit.core.facade import LoopThread as Loop\n```\n",
            "line 3: import LoopThread from conduit, not conduit.core.facade",
        ),
        (
            "```python\nfrom conduit.core import types\nfrom conduit.core.backends import (\n"
            "    TokenStore,\n    TokenSet,\n)\n```\n",
            "import TokenSet from conduit, not conduit.core.backends",
        ),
        (
            "```python\ndef f():\n    from conduit.core.errors import SecretUnavailable\n```\n",
            "import SecretUnavailable from conduit, not conduit.core.errors",
        ),
        (
            "```python\nfrom conduit.stores.memory import MemoryTranscriptStore\n```\n",
            "import MemoryTranscriptStore from conduit.stores, not conduit.stores.memory",
        ),
    ],
    ids=["core-submodule", "aliased", "parenthesised", "nested", "layer-submodule"],
)
def test_the_guard_refuses_an_import_that_bypasses_the_public_path(text: str, reason: str) -> None:
    problems = import_problems(text)
    assert any(reason in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "```python\nimport conduit.core.tools as t\n```\n",
            "line 2: import conduit.core.tools: use the names conduit exports",
        ),
        (
            "```python\nimport conduit.core.tools\n```\n",
            "import conduit.core.tools: use the names conduit exports",
        ),
        (
            "```python\nimport conduit.core\n```\n",
            "import conduit.core: use the names conduit exports",
        ),
        (
            "```python\nfrom conduit.core import tools\n```\n",
            "import conduit.core.tools: use the names conduit exports",
        ),
        (
            "```python\nfrom conduit import core\n```\n",
            "import conduit.core: use the names conduit exports",
        ),
        (
            "```python\nfrom conduit.core.tools import *\n```\n",
            "import * from conduit.core.tools: name each import",
        ),
        (
            "```python\nfrom conduit.agent import *\n```\n",
            "import * from conduit.agent: name each import",
        ),
        (
            "```python\nimport conduit.stores.memory as memory\n```\n",
            "import conduit.stores.memory: use the names conduit.stores exports",
        ),
        (
            "```python\nfrom conduit.stores import memory\n```\n",
            "import conduit.stores.memory: use the names conduit.stores exports",
        ),
    ],
    ids=[
        "import-core-submodule-as",
        "import-core-submodule",
        "import-core",
        "from-core-import-submodule",
        "from-conduit-import-core",
        "star-from-core",
        "star-from-layer",
        "import-layer-submodule",
        "from-layer-import-submodule",
    ],
)
def test_the_guard_refuses_a_module_import_that_bypasses_the_public_path(
    text: str, reason: str
) -> None:
    problems = import_problems(text)
    assert any(reason in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "```python\nfrom conduit.core._hidden import thing\n```\n",
            "line 2: conduit.core._hidden is a private module path",
        ),
        (
            "```python\nimport conduit.core._hidden\n```\n",
            "conduit.core._hidden is a private module path",
        ),
        (
            "```python\nimport conduit.core._hidden as hidden\n```\n",
            "conduit.core._hidden is a private module path",
        ),
        (
            "```python\nfrom conduit import _version\n```\n",
            "conduit._version is a private module path",
        ),
        (
            "```python\nfrom conduit.agent import _runner as runner\n```\n",
            "conduit.agent._runner is a private module path",
        ),
        (
            "```python\nfrom conduit._private.deep import name as alias\n```\n",
            "conduit._private.deep is a private module path",
        ),
        (
            "```python\ndef f():\n    import json._private\n```\n",
            "json._private is a private module path",
        ),
        (
            "```python\nfrom importlib._bootstrap import spec_from_loader\n```\n",
            "importlib._bootstrap is a private module path",
        ),
    ],
    ids=[
        "from-private",
        "import-private",
        "import-private-as",
        "from-package-import-private",
        "from-package-import-private-as",
        "from-private-deep-as",
        "nested-non-conduit",
        "standard-library-private",
    ],
)
def test_the_guard_refuses_a_private_module_path(text: str, reason: str) -> None:
    # The private path is refused before anything is imported: such a module may not
    # exist, and nothing in it is public whether or not it does.
    problems = import_problems(text)
    assert any(reason in problem for problem in problems), problems


def test_a_dunder_module_is_not_private() -> None:
    assert import_problems("```python\nfrom __future__ import annotations\n```\n") == []


def test_the_guard_finds_a_submodule_not_yet_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    # A submodule nothing has imported is not yet an attribute of its package, so the
    # guard imports it by name to tell it from an exported name.
    require_extras("```python\nimport conduit.mcp\n```\n")
    package = importlib.import_module("conduit.mcp")
    importlib.import_module("conduit.mcp.client")
    monkeypatch.delattr(package, "client")
    monkeypatch.delitem(sys.modules, "conduit.mcp.client")
    assert _submodule("conduit.mcp", "client") == "conduit.mcp.client"
    monkeypatch.delattr(package, "client")
    monkeypatch.delitem(sys.modules, "conduit.mcp.client")
    problems = import_problems("```python\nfrom conduit.mcp import client\n```\n")
    assert any(
        "import conduit.mcp.client: use the names conduit.mcp exports" in problem
        for problem in problems
    ), problems


def test_the_guard_allows_a_layer_module_its_package_does_not_cover() -> None:
    # conduit.providers exports builtin_factories only, so the stub module is the path.
    text = "```python\nimport conduit.providers.stub\nimport conduit.agent\nimport conduit\n```\n"
    assert import_problems(text) == []


# Extras a document needs ----------------------------------------------------------


def _provided_extras() -> set[str]:
    return set(importlib.metadata.metadata("cognisn-conduit").get_all("Provides-Extra") or [])


def test_every_extra_is_mapped_to_its_layer() -> None:
    # A new extra fails here until a document importing its layer can name it.
    assert set(LAYER_EXTRAS.values()) == _provided_extras() - {"dev", "all-providers"}


def test_every_layer_behind_an_optional_library_is_mapped() -> None:
    for library, owners in OPTIONAL_LIBRARIES.items():
        for owner in owners:
            assert _layer_extra(owner) is not None, f"{owner} (for {library}) has no extra"


@pytest.mark.parametrize(
    ("text", "needed"),
    [
        ("```python\nimport json\nfrom conduit import start\n```\n", []),
        ("```python\nfrom conduit.providers.stub import StubFactory\n```\n", []),
        ("```python\nfrom conduit.mcp import McpConnection\n```\n", [("mcp",)]),
        ("```python\nfrom conduit.stores.sql import SqlTranscriptStore\n```\n", [("sql",)]),
        ("```python\nimport conduit.integrations.otel\n```\n", [("otel",)]),
        ("```python\nfrom conduit.integrations import warden\n```\n", [("warden",)]),
        ("```python\nfrom conduit.cli.main import main\n```\n", [("cli",)]),
        ("```python\nfrom warden.audit import AuditRecorder\n```\n", [("warden",)]),
        ("```python\nfrom sqlalchemy.ext.asyncio import create_async_engine\n```\n", [("sql",)]),
        ("```python\ndef f():\n    import httpx\n```\n", [("mcp", "ollama")]),
        ("```text\nfrom conduit.mcp import McpConnection\n```\n", []),
    ],
    ids=[
        "core",
        "stub",
        "layer",
        "nested-layer",
        "module-import",
        "layer-by-name",
        "cli",
        "library",
        "library-submodule",
        "shared-library",
        "not-python",
    ],
)
def test_the_extras_a_document_needs_are_read_from_its_imports(
    text: str, needed: list[tuple[str, ...]]
) -> None:
    assert extras_needed(text) == needed


def test_an_extra_is_installed_when_every_requirement_is(monkeypatch: pytest.MonkeyPatch) -> None:
    requires = (
        "pytest; extra == 'a'",
        "cognisn-conduit[b]; extra == 'a'",
        "pytest>=1; extra == 'b'",
        "no-such-distribution-for-conduit; extra == 'c'",
        "cognisn-conduit[c]; extra == 'd'",
        "pytest; extra == 'e'",
        "no-such-distribution-for-conduit; extra == 'e'",
    )
    monkeypatch.setattr(sys.modules[__name__], "_conduit_requires", lambda: requires)
    assert extra_installed("a")
    assert extra_installed("b")
    assert not extra_installed("c")
    assert not extra_installed("d")
    assert not extra_installed("e")
    assert not extra_installed("unknown"), "an extra with no requirements is not installed"


_OTEL_TEXT = "```python\nfrom conduit.integrations.otel import OtelEventSink\n```\n"


def _outcome(text: str) -> BaseException | None:
    """What ``require_extras`` raised for ``text``, caught here so that a skip it raises
    by mistake fails the calling test instead of skipping it."""
    try:
        require_extras(text)
    except BaseException as stopped:  # skip and fail are BaseExceptions
        return stopped
    return None


def test_a_missing_extra_skips_the_document_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EVERY_EXTRA_SWITCH, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra != "otel")
    stopped = _outcome(_OTEL_TEXT)
    assert isinstance(stopped, pytest.skip.Exception), stopped
    assert re.search(r"the otel extra .*cognisn-conduit\[otel\]", str(stopped)), stopped


def test_a_missing_shared_library_names_each_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EVERY_EXTRA_SWITCH, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: False)
    stopped = _outcome("```python\nimport httpx\n```\n")
    assert isinstance(stopped, pytest.skip.Exception), stopped
    assert "the mcp or ollama extra" in str(stopped)


def test_one_extra_of_a_shared_library_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EVERY_EXTRA_SWITCH, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra == "ollama")
    assert _outcome("```python\nimport httpx\n```\n") is None


def test_a_missing_extra_fails_when_every_extra_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EVERY_EXTRA_SWITCH, "require")
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra != "otel")
    stopped = _outcome(_OTEL_TEXT)
    assert isinstance(stopped, pytest.fail.Exception), stopped
    assert re.search(f"the otel extra .*{EVERY_EXTRA_SWITCH}=require", str(stopped)), stopped


@pytest.mark.parametrize("value", ["1", "true", "Require", "skip"])
def test_any_other_switch_value_is_refused(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # The core's rule (Q0051): a misspelt switch cannot quietly turn the check off.
    monkeypatch.setenv(EVERY_EXTRA_SWITCH, value)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra != "otel")
    stopped = _outcome(_OTEL_TEXT)
    assert isinstance(stopped, ValueError), stopped
    assert EVERY_EXTRA_SWITCH in str(stopped)


def test_the_switch_is_the_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    assert EVERY_EXTRA_SWITCH == "CONDUIT_ALL_EXTRAS"
    monkeypatch.setenv("CONDUIT_REQUIRE_ALL_EXTRAS", "1")
    monkeypatch.delenv(EVERY_EXTRA_SWITCH, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra != "otel")
    assert isinstance(_outcome(_OTEL_TEXT), pytest.skip.Exception), "the old switch is gone"


def test_an_installed_extra_runs_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EVERY_EXTRA_SWITCH, "require")
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: True)
    assert _outcome(_OTEL_TEXT) is None


def test_a_run_requiring_every_extra_has_every_extra() -> None:
    # CI's test jobs install every runtime extra and set CONDUIT_ALL_EXTRAS=require, so
    # this fails there if an extra did not install, and nothing is quietly skipped.
    if not every_extra_required():
        pytest.skip(f"set {EVERY_EXTRA_SWITCH}=require to check that every extra is installed")
    missing = sorted(e for e in _provided_extras() - {"dev"} if not extra_installed(e))
    assert missing == []


def test_a_document_needing_a_missing_extra_is_skipped_before_it_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(EVERY_EXTRA_SWITCH, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra != "warden")
    document = ROOT / "docs" / "guides" / "observability.md"
    with pytest.raises(pytest.skip.Exception, match="the warden extra"):
        test_every_example_imports_from_the_public_path(document)
    with pytest.raises(pytest.skip.Exception, match="the warden extra"):
        test_every_python_example_runs(document, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "line",
    [
        "import conduit.core._hidden as hidden",
        "from conduit.integrations.otel import *",
        "import conduit.core.tools",
        "from conduit.core.tools import TaskTools",
        "from conduit import core",
    ],
    ids=["private-path", "import-star", "core-module", "core-name", "core-by-name"],
)
def test_a_skipped_document_still_has_its_static_import_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, line: str
) -> None:
    # Review M2: the checks that import nothing run before the missing-extra skip, so a
    # private path, an import * or a core module import fails even where the extra is
    # missing.
    monkeypatch.delenv(EVERY_EXTRA_SWITCH, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "extra_installed", lambda extra: extra != "otel")
    document = tmp_path / "guide.md"
    document.write_text(f"```python\n{_OTEL_TEXT.splitlines()[1]}\n{line}\n```\n")
    try:
        test_every_example_imports_from_the_public_path(document)
    except BaseException as stopped:  # skip and fail are BaseExceptions
        assert isinstance(stopped, AssertionError), stopped
    else:
        pytest.fail("the document passed")


def _readme_row(extra: str) -> str:
    rows = [
        line
        for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        if line.startswith(f"| `{extra}` |")
    ]
    assert len(rows) == 1, f"the README's extras table has {len(rows)} rows for {extra}"
    return rows[0]


_WARDEN_TEXT = "```python\nimport conduit.integrations.warden\n```\n"


def test_the_readme_names_the_warden_extras_licence_exception() -> None:
    # The licence audit reads the installed tree, so this runs where warden is installed.
    require_extras(_WARDEN_TEXT)
    report = audit(extras=["warden"])
    assert "psycopg" in {package.name for package in report.excepted}
    row = _readme_row("warden")
    assert "psycopg (LGPL-3.0) under its named licence exception" in row, row


def test_the_readme_counts_what_the_warden_extra_installs() -> None:
    # Platform markers change the tree (keyring's backends, greenlet on arm64), so macOS
    # installs fewer; Linux and Windows install the same count, which the README gives.
    if sys.platform == "darwin":
        pytest.skip("the README counts the warden extra's packages on Linux and Windows")
    require_extras(_WARDEN_TEXT)
    core = len(audit().checked) - 1
    total = len(audit(extras=["warden"]).checked) - 1
    row = _readme_row("warden")
    assert (
        f"installs {total - core} runtime packages on Linux and Windows beyond the core's {core}"
        in row
    ), row
    assert f"({total} in all" in row, row


# The documents ----------------------------------------------------------------------


def test_every_document_is_checked_and_listed() -> None:
    names = {_name(path) for path in DOCUMENTS}
    assert set(PINNED) <= names
    assert all((ROOT / name).is_file() for name in PINNED)
    assert names == set(EXPECTED_EXAMPLES), (
        "a document was added, removed or missed: update EXPECTED_EXAMPLES"
    )
    assert sum(EXPECTED_EXAMPLES.values()) >= 40
    assert EXPECTED_EXAMPLES["README.md"] >= 1


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_no_example_can_escape_the_run(document: Path) -> None:
    assert fence_problems(document.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_example_imports_from_the_public_path(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    # The checks that import nothing first, so a document skipped for a missing extra
    # still has them (review M2).
    assert import_problems(text, resolve=False) == []
    require_extras(text)
    assert import_problems(text) == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_python_example_runs(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = document.read_text(encoding="utf-8")
    require_extras(text)
    monkeypatch.chdir(tmp_path)
    # A real module, registered while it runs, so dataclasses and pickling find it.
    module = types.ModuleType(f"conduit_guide_{document.stem.replace('-', '_')}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__[_COUNTER] = 0
    examples = [b for b in _blocks(text) if b[0] == "python"]
    for _language, line, code in examples:
        # Pad with blank lines so a traceback names the line in the document, and count
        # the example only once its last line has run.
        source = "\n" * (line - 1) + code + f"\n{_COUNTER} += 1\n"
        exec(compile(source, _name(document), "exec"), module.__dict__)  # noqa: S102
    assert module.__dict__[_COUNTER] == len(examples) == EXPECTED_EXAMPLES[_name(document)]
