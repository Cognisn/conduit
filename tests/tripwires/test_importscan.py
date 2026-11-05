"""The scanner and rules, shown imports they exist to catch (synthetic trees)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tripwires.importscan import (
    OPTIONAL_LIBRARIES,
    check_owners,
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


def test_a_library_with_several_owners_is_admitted_by_each_and_no_other(tmp_path: Path) -> None:
    owners = {"httpx": ("conduit.providers.ollama", "conduit.mcp")}
    src = _tree(
        tmp_path,
        {
            "conduit/providers/__init__.py": "",
            "conduit/providers/ollama/__init__.py": "import httpx\n",
            "conduit/providers/ollama/client.py": "from httpx import AsyncClient\n",
            "conduit/mcp/__init__.py": "def f():\n    import httpx\n",
            "conduit/providers/gemini/__init__.py": "import httpx\n",
            "conduit/providers/ollama_extra.py": "import httpx\n",
            "conduit/core/__init__.py": "import importlib\nimportlib.import_module('httpx')\n",
            "conduit/agent/__init__.py": "from httpx import Client\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src), owners)}
    assert found == {
        ("conduit.providers.gemini", "httpx"),
        ("conduit.providers.ollama_extra", "httpx"),
        ("conduit.core", "httpx"),
        ("conduit.agent", "httpx"),
        ("conduit.agent", "httpx.Client"),
    }


def test_google_genai_is_owned_by_the_gemini_adapter_only(tmp_path: Path) -> None:
    assert OPTIONAL_LIBRARIES["google.genai"] == ("conduit.providers.gemini",)
    src = _tree(
        tmp_path,
        {
            "conduit/providers/__init__.py": "",
            "conduit/providers/gemini/__init__.py": "from google import genai\n",
            "conduit/providers/gemini/factory.py": (
                "import importlib\nimportlib.import_module('google.genai.types')\n"
            ),
            "conduit/providers/ollama/__init__.py": "import google.genai\n",
            "conduit/embeddings/__init__.py": "from google.genai import types\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {
        ("conduit.providers.ollama", "google.genai"),
        ("conduit.embeddings", "google.genai"),
        ("conduit.embeddings", "google.genai.types"),
    }


@pytest.mark.parametrize("driver", ["asyncpg", "aiosqlite"])
def test_the_sql_drivers_are_owned_by_the_sql_store_only(tmp_path: Path, driver: str) -> None:
    # The sql extra's database drivers, like sqlalchemy and alembic, belong to
    # conduit.stores.sql alone: not the memory and JSON stores, not the core.
    assert OPTIONAL_LIBRARIES[driver] == ("conduit.stores.sql",)
    src = _tree(
        tmp_path,
        {
            "conduit/stores/__init__.py": f"import {driver}\n",
            "conduit/stores/sql/__init__.py": f"import {driver}\n",
            "conduit/stores/sql/engine.py": f"def f():\n    from {driver} import connect\n",
            "conduit/stores/json.py": f"import importlib\nimportlib.import_module('{driver}')\n",
            "conduit/core/__init__.py": f"from {driver} import connect\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {
        ("conduit.stores", driver),
        ("conduit.stores.json", driver),
        ("conduit.core", driver),
        ("conduit.core", f"{driver}.connect"),
    }


def test_the_real_table_lists_httpx_for_ollama_and_mcp_only(tmp_path: Path) -> None:
    # Q0006: httpx has two owners. The bedrock-mantle adapter (the workstream Q0013
    # added, which named this owner change) shares the anthropic SDK and botocore,
    # tested below. Every other library keeps exactly one.
    assert OPTIONAL_LIBRARIES["httpx"] == ("conduit.providers.ollama", "conduit.mcp")
    shared = {name for name, owners in OPTIONAL_LIBRARIES.items() if len(owners) != 1}
    assert shared == {"httpx", "anthropic", "botocore"}
    src = _tree(
        tmp_path,
        {
            "conduit/providers/__init__.py": "",
            "conduit/providers/ollama/__init__.py": "import httpx\n",
            "conduit/mcp/__init__.py": "import httpx\n",
            "conduit/stores/__init__.py": "import httpx\n",
            "conduit/providers/openai_compat/__init__.py": "import httpx\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {
        ("conduit.stores", "httpx"),
        ("conduit.providers.openai_compat", "httpx"),
    }


MANTLE = "conduit.providers.bedrock_mantle"


@pytest.mark.parametrize(
    ("library", "first_owner"),
    [("anthropic", "conduit.providers.anthropic"), ("botocore", "conduit.providers.bedrock")],
)
def test_the_bedrock_mantle_adapter_shares_its_sdks_and_no_other_layer_does(
    tmp_path: Path, library: str, first_owner: str
) -> None:
    # The bedrock-mantle workstream (Q0013) builds the anthropic SDK's mantle client and
    # presigns short-term keys with botocore, so it owns both alongside their first
    # adapter.
    assert OPTIONAL_LIBRARIES[library] == (first_owner, MANTLE)
    first = first_owner.replace(".", "/")
    src = _tree(
        tmp_path,
        {
            "conduit/providers/__init__.py": "",
            f"{first}/__init__.py": f"import {library}\n",
            "conduit/providers/bedrock_mantle/__init__.py": "",
            "conduit/providers/bedrock_mantle/factory.py": (
                f"import importlib\nimportlib.import_module('{library}.session')\n"
            ),
            "conduit/providers/bedrock_mantle_extra.py": f"import {library}\n",
            "conduit/providers/openai_compat/__init__.py": f"import {library}\n",
            "conduit/agent/__init__.py": f"from {library} import x\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {
        ("conduit.providers.bedrock_mantle_extra", library),
        ("conduit.providers.openai_compat", library),
        ("conduit.agent", library),
        ("conduit.agent", f"{library}.x"),
    }


def test_boto3_stays_with_the_bedrock_adapter_alone(tmp_path: Path) -> None:
    # The mantle adapter needs botocore only; boto3 is not one of its imports.
    assert OPTIONAL_LIBRARIES["boto3"] == ("conduit.providers.bedrock",)
    src = _tree(
        tmp_path,
        {
            "conduit/providers/__init__.py": "",
            "conduit/providers/bedrock_mantle/__init__.py": "import boto3\n",
        },
    )
    found = {(e.importer, e.imported) for e in optional_library_violations(scan_tree(src))}
    assert found == {(MANTLE, "boto3")}


@pytest.mark.parametrize(
    "owners",
    [
        (),
        "conduit.mcp",
        ("conduit.mcp", "conduit.mcp"),
        ("conduit.mcp", ""),
        ["conduit.mcp"],
    ],
)
def test_an_owner_entry_that_is_not_a_tuple_of_distinct_names_is_refused(owners: object) -> None:
    with pytest.raises(ValueError, match="httpx"):
        check_owners({"httpx": owners})  # type: ignore[dict-item]


def test_the_real_table_is_well_formed() -> None:
    check_owners(OPTIONAL_LIBRARIES)


def test_the_rule_refuses_a_malformed_table(tmp_path: Path) -> None:
    # A bare string would be read as one owner per character; the rule refuses it.
    src = _tree(tmp_path, {"conduit/mcp/__init__.py": "import httpx\n"})
    with pytest.raises(ValueError, match="httpx"):
        optional_library_violations(scan_tree(src), {"httpx": "conduit.mcp"})  # type: ignore[dict-item]


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
