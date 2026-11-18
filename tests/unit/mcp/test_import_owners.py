"""The MCP SDK is the MCP client's alone, and the client is on the SDK's HTTP library."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tripwires.importscan import (
    OPTIONAL_LIBRARIES,
    SRC_ROOT,
    optional_library_violations,
    scan_tree,
)


@pytest.mark.parametrize("library", ["mcp"])
def test_the_sdk_s_libraries_are_owned_by_the_mcp_client_only(tmp_path: Path, library: str) -> None:
    assert OPTIONAL_LIBRARIES[library] == ("conduit.mcp",)
    src = tmp_path / "src"
    files = {
        "conduit/__init__.py": "",
        "conduit/mcp/__init__.py": f"import {library}\n",
        "conduit/mcp/client.py": f"from {library} import name\n",
        "conduit/providers/__init__.py": "",
        "conduit/providers/ollama/__init__.py": f"import {library}\n",
        "conduit/agent/__init__.py": f"def f():\n    import {library}\n",
    }
    for rel, body in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {("conduit.providers.ollama", library), ("conduit.agent", library)}


def test_the_mcp_client_imports_httpx2_and_not_httpx() -> None:
    imported = {
        e.imported.split(".")[0]
        for e in scan_tree(SRC_ROOT)
        if e.importer == "conduit.mcp" or e.importer.startswith("conduit.mcp.")
    }
    assert "httpx2" in imported
    # The mcp extra no longer installs httpx (SDK 2.x moved to httpx2).
    assert "httpx" not in imported
