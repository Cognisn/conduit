"""The local-embeddings libraries in the import tripwire's table, and the embeddings layer's
imports on the real tree.

Removing torch, sentence_transformers or transformers from ``OPTIONAL_LIBRARIES`` would
otherwise pass every tripwire, since nothing in conduit imports torch or transformers by
name. These tests pin each entry and show the rule refusing each library outside
``conduit.embeddings.local``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tripwires.importscan import (
    OPTIONAL_LIBRARIES,
    SRC_ROOT,
    optional_library_violations,
    scan_tree,
    within,
)

LOCAL = "conduit.embeddings.local"
LOCAL_LIBRARIES = ("torch", "sentence_transformers", "transformers")


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "src"
    for rel, body in {"conduit/__init__.py": "", **files}.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
    return src


@pytest.mark.parametrize("library", LOCAL_LIBRARIES)
def test_each_local_library_is_owned_by_the_local_layer_only(library: str) -> None:
    assert OPTIONAL_LIBRARIES[library] == (LOCAL,)


@pytest.mark.parametrize("library", LOCAL_LIBRARIES)
def test_the_rule_refuses_each_local_library_outside_the_local_layer(
    tmp_path: Path, library: str
) -> None:
    src = _tree(
        tmp_path,
        {
            "conduit/embeddings/__init__.py": f"import {library}\n",
            "conduit/embeddings/local/__init__.py": f"import {library}\n",
            "conduit/embeddings/local/factory.py": (
                f"import importlib\nimportlib.import_module('{library}')\n"
            ),
            "conduit/embeddings/localish.py": f"import {library}\n",
            "conduit/core/__init__.py": f"def f():\n    import {library}\n",
            "conduit/providers/__init__.py": f"from {library} import x\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {
        ("conduit.embeddings", library),
        ("conduit.embeddings.localish", library),
        ("conduit.core", library),
        ("conduit.providers", library),
        ("conduit.providers", f"{library}.x"),
    }


def test_the_local_layer_imports_sentence_transformers_by_name() -> None:
    # Guards against a vacuous pass: the scan sees the local layer's import.
    edges = [e for e in scan_tree() if within(e.importer, LOCAL)]
    assert any(e.imported == "sentence_transformers" for e in edges)
    assert (SRC_ROOT / "conduit" / "embeddings" / "local" / "factory.py").is_file()


def test_the_remote_embedding_adapters_import_no_sdk() -> None:
    # They reuse their chat adapter's create_client, so no SDK is imported here at all.
    edges = [
        e
        for e in scan_tree()
        if within(e.importer, "conduit.embeddings") and not within(e.importer, LOCAL)
    ]
    sdks = [e for e in edges if any(within(e.imported, lib) for lib in OPTIONAL_LIBRARIES)]
    assert not sdks, "\n".join(str(e) for e in sdks)
