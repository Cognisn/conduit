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
layer's package, is never imported from the submodule it lives in.
"""

from __future__ import annotations

import ast
import functools
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Every document whose examples are run, with its number of Python examples.
EXPECTED_EXAMPLES: dict[str, int] = {
    "README.md": 1,
    "docs/design/architecture.md": 0,
    "docs/guides/README.md": 0,
    "docs/guides/adoption.md": 0,
    "docs/guides/backends.md": 9,
    "docs/guides/compaction.md": 1,
    "docs/guides/embeddings.md": 2,
    "docs/guides/integration.md": 14,
    "docs/guides/mcp.md": 4,
    "docs/guides/observability.md": 3,
    "docs/guides/providers.md": 9,
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


@functools.cache
def _exports(package: str) -> frozenset[str]:
    """The names ``package`` exports in its ``__all__``."""
    return frozenset(getattr(importlib.import_module(package), "__all__", ()))


def import_problems(text: str) -> list[str]:
    """Imports in ``text``'s Python examples that bypass a package exporting the name.

    A name exported from ``conduit`` is imported from ``conduit``, and a name a layer's
    package exports (``conduit.stores``) from that package, never from the submodule
    it lives in, so the guides show the public import path (Wave 6 ruling Q0044).
    """
    problems: list[str] = []
    for language, start, code in _blocks(text):
        if language != "python":
            continue
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            parts = node.module.split(".")
            if parts[0] != "conduit":
                continue
            for alias in node.names:
                for depth in range(1, len(parts)):
                    package = ".".join(parts[:depth])
                    if alias.name in _exports(package):
                        problems.append(
                            f"line {start + node.lineno - 1}: import {alias.name} from "
                            f"{package}, not {node.module}"
                        )
                        break
    return problems


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
    assert import_problems(document.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_python_example_runs(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # A real module, registered while it runs, so dataclasses and pickling find it.
    module = types.ModuleType(f"conduit_guide_{document.stem.replace('-', '_')}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__[_COUNTER] = 0
    examples = [b for b in _blocks(document.read_text(encoding="utf-8")) if b[0] == "python"]
    for _language, line, code in examples:
        # Pad with blank lines so a traceback names the line in the document, and count
        # the example only once its last line has run.
        source = "\n" * (line - 1) + code + f"\n{_COUNTER} += 1\n"
        exec(compile(source, _name(document), "exec"), module.__dict__)  # noqa: S102
    assert module.__dict__[_COUNTER] == len(examples) == EXPECTED_EXAMPLES[_name(document)]
