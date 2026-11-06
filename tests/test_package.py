"""The package layout of plan section 4, and the single version source."""

from __future__ import annotations

import importlib
import subprocess
import sys
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
    "bedrock-mantle",
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


#: The core's main types, re-exported from the top-level package (Wave 1 ruling Q0003),
#: and every other core name an integration, a guide or the terminal interface imports
#: (Wave 6 ruling Q0044).
REEXPORTS: dict[str, tuple[str, ...]] = {
    "conduit.core.types": (
        "CacheBreakpoint",
        "ImageBlock",
        "Message",
        "ModelRequest",
        "ModelResponse",
        "ProviderStateBlock",
        "StopReason",
        "TextBlock",
        "ToolResultBlock",
        "ToolSpec",
        "ToolUseBlock",
        "Usage",
    ),
    "conduit.core.streaming": (
        "ProviderStateReported",
        "StreamStopped",
        "TextDelta",
        "ToolCallFinished",
        "ToolCallStarted",
        "UsageReported",
        "collect",
    ),
    "conduit.core.provider": ("BaseProvider", "Provider"),
    "conduit.core.errors": (
        "AssignmentRefused",
        "CapabilityRefused",
        "ConduitError",
        "ConfigurationError",
        "ModelNotAssigned",
        "ProviderError",
        "ProviderFailure",
        "ResidencyRefused",
        "SettingsRefused",
        "UndeclaredModel",
        "UnknownTaskClass",
        "SecretUnavailable",
        "UnknownTool",
        "ToolRegistrationError",
        "FacadeError",
        "ReplayMismatch",
    ),
    "conduit.core.registry": ("MeteredProvider", "ProviderFactory", "ProviderRegistry"),
    "conduit.core.runtime": ("ConduitSettings", "Runtime", "TaskBinding", "start"),
    "conduit.core.settings": ("AssignmentSettings", "ProviderSettings", "refusals"),
    "conduit.core.secrets": (
        "EnvironmentSecretResolver",
        "LiteralSecretResolver",
        "SecretRef",
        "SecretResolver",
    ),
    "conduit.core.catalogue": (
        "Catalogue",
        "CatalogueOverlay",
        "ModelEntry",
        "ModelKey",
        "default_catalogue",
    ),
    "conduit.core.tasks": ("CapabilityFloor", "TaskClass", "resolve_assignments"),
    "conduit.core.residency": ("ResidencyPolicy", "ResidencyScope"),
    "conduit.core.events": (
        "CaptureAs",
        "CapturePolicy",
        "Event",
        "EventSink",
        "FanOutSink",
        "ToolContent",
        "ContentKind",
        "CapturedContent",
        "ModelCallStartedEvent",
        "ModelCallFinishedEvent",
        "ToolCallEvent",
        "MemoryEventSink",
        "NullEventSink",
    ),
    "conduit.core.transcript": (
        "Transcript",
        "TranscriptStore",
        "TranscriptEntry",
        "MessageEntry",
        "CompactionPoint",
        "SkillUse",
    ),
    "conduit.core.backends": (
        "ApprovalHandler",
        "SkillSource",
        "TokenStore",
        "TokenSet",
        "Skill",
        "SkillSummary",
        "ApprovalRequest",
        "ApprovalDecision",
    ),
    "conduit.core.tools": (
        "Tool",
        "ToolCatalogue",
        "ToolRegistry",
        "TaskTools",
        "ToolCallContext",
        "ToolOutcome",
        "ToolExecutor",
    ),
    "conduit.core.facade": (
        "LoopThread",
        "SyncFacade",
        "SyncProvider",
        "SyncRuntime",
        "SyncTaskBinding",
        "start_sync",
        "sync",
    ),
}


@pytest.mark.parametrize(
    ("module", "name"), [(module, name) for module, names in REEXPORTS.items() for name in names]
)
def test_the_core_s_main_types_are_re_exported(module: str, name: str) -> None:
    assert name in conduit.__all__
    assert getattr(conduit, name) is getattr(importlib.import_module(module), name)


def test_the_top_level_exports_nothing_unlisted() -> None:
    listed = {name for names in REEXPORTS.values() for name in names}
    assert set(conduit.__all__) == listed | {"__version__"}


#: The reference stores, exported from their layer package rather than the top level,
#: which imports the core alone (Wave 6 ruling Q0044).
STORE_EXPORTS: dict[str, tuple[str, ...]] = {
    "conduit.stores.memory": ("MemoryTokenStore", "MemoryTranscriptStore"),
    "conduit.stores.json": ("JsonTokenStore", "JsonTranscriptStore"),
    "conduit.stores.errors": ("StoreError",),
}


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in STORE_EXPORTS.items() for name in names],
)
def test_the_reference_stores_are_exported_from_their_layer(module: str, name: str) -> None:
    stores = importlib.import_module("conduit.stores")
    assert name in stores.__all__
    assert getattr(stores, name) is getattr(importlib.import_module(module), name)
    assert name not in conduit.__all__


def test_importing_conduit_loads_the_core_alone() -> None:
    # The top level re-exports core names only, so no layer loads with it (C2).
    code = (
        "import sys, conduit; print(sorted(m for m in sys.modules if m.startswith('conduit.') "
        "and m != 'conduit.core' and not m.startswith('conduit.core.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "[]"


def test_the_stores_layer_exports_nothing_unlisted() -> None:
    stores = importlib.import_module("conduit.stores")
    listed = {name for names in STORE_EXPORTS.values() for name in names}
    assert set(stores.__all__) == listed


PROVIDER_EXTRAS = ("bedrock", "bedrock-mantle", "anthropic", "openai", "gemini", "ollama")


def _extra_requirements(extra: str) -> set[str]:
    found: set[str] = set()
    for spec in metadata.requires("cognisn-conduit") or []:
        requirement = Requirement(spec)
        if requirement.marker is not None and requirement.marker.evaluate({"extra": extra}):
            requirement.marker = None
            found.add(str(requirement))
    return found


def test_all_providers_installs_every_provider_extra_and_nothing_else() -> None:
    every_provider: set[str] = set()
    for extra in PROVIDER_EXTRAS:
        requirements = _extra_requirements(extra)
        assert requirements, f"the {extra} extra declares nothing"
        every_provider |= requirements
    assert _extra_requirements("all-providers") == every_provider
