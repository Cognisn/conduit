"""Every Python example in the guides, the README and the architecture document runs.

Each ``python`` fenced block of a document is executed, in order, in one namespace per
document (so a later block may use what an earlier one defined), in a fresh temporary
working directory. Examples call the stub provider, local stores and in-process
servers only: nothing here reaches a network or a live model.

Any other fence must name a language that is never Python (shell lines, settings files,
output), so an example cannot slip out of the run by being fenced as ``py`` or left
untagged.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "design" / "architecture.md",
    *sorted((ROOT / "docs" / "guides").glob("*.md")),
)

#: Fence languages that are never run: shell lines, configuration and output.
NOT_PYTHON = frozenset({"console", "json", "markdown", "powershell", "sh", "text", "toml", "yaml"})

_FENCE = re.compile(r"^```(?P<info>[^\n`]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def _blocks(text: str) -> list[tuple[str, int, str]]:
    """Every fenced block: its language, the line its code starts on, and its code."""
    found: list[tuple[str, int, str]] = []
    for match in _FENCE.finditer(text):
        language = match.group("info").strip()
        line = text.count("\n", 0, match.start("body")) + 1
        found.append((language, line, match.group("body")))
    return found


def _name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_the_guides_exist() -> None:
    guides = [p for p in DOCUMENTS if p.parent.name == "guides"]
    assert len(guides) >= 10, "the guides directory is missing guides"


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_fence_is_python_or_never_python(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    for language, line, _code in _blocks(text):
        assert language == "python" or language in NOT_PYTHON, (
            f"{_name(document)}:{line}: a fence must be python (run by this test) or one "
            f"of {sorted(NOT_PYTHON)}, not {language!r}"
        )


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_python_example_runs(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # A real module, registered while it runs, so dataclasses and pickling find it.
    module = types.ModuleType(f"conduit_guide_{document.stem.replace('-', '_')}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for language, line, code in _blocks(document.read_text(encoding="utf-8")):
        if language != "python":
            continue
        # Pad with blank lines so a traceback names the line in the document.
        compiled = compile("\n" * (line - 1) + code, _name(document), "exec")
        exec(compiled, module.__dict__)  # noqa: S102 - the guide's own example, by design
