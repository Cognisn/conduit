"""Static import scanner and the import-direction rules of the development plan.

The rules (plan section 4, decisions C2 and C10) are checked against the source
tree, never by importing it, so an import an extra would satisfy is still seen
when the extra is not installed, and an import inside a function, a
``TYPE_CHECKING`` block or an ``importlib.import_module`` call with a literal
name is seen too.

- ``conduit.core`` imports nothing from any other conduit package, nor the
  top-level ``conduit`` package itself.
- Each optional third-party library (a provider SDK or a layer's own
  dependency) is imported only inside the layer, or layers, of the extras that
  install it. Most libraries have one owner; a library two layers need (httpx,
  for the Ollama adapter and the MCP client) lists each owner by name.
- Nothing outside ``conduit.cli`` imports ``conduit.cli``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE = "conduit"
CORE = "conduit.core"
CLI = "conduit.cli"

#: Optional third-party libraries and the only layers allowed to import each.
#: Each value is a tuple of one or more distinct owning layers (Q0006). A later
#: workstream that needs another library behind an extra, or another owner for
#: one, adds it here (this file is a shared module: the change goes through the
#: integration session).
OPTIONAL_LIBRARIES: Mapping[str, tuple[str, ...]] = {
    # Provider SDKs (C6), each behind its own extra.
    "boto3": ("conduit.providers.bedrock",),
    # botocore also presigns the bedrock-mantle adapter's short-term keys.
    "botocore": ("conduit.providers.bedrock", "conduit.providers.bedrock_mantle"),
    "aioboto3": ("conduit.providers.bedrock",),
    "aiobotocore": ("conduit.providers.bedrock",),
    # The anthropic SDK also serves Claude on bedrock-mantle (its mantle client).
    "anthropic": ("conduit.providers.anthropic", "conduit.providers.bedrock_mantle"),
    "openai": ("conduit.providers.openai_compat",),
    "azure.identity": ("conduit.providers.openai_compat",),
    "google.genai": ("conduit.providers.gemini",),
    "google.generativeai": ("conduit.providers.gemini",),
    "ollama": ("conduit.providers.ollama",),
    # The HTTP client behind the ollama and mcp extras.
    "httpx": ("conduit.providers.ollama", "conduit.mcp"),
    # Layer dependencies behind their extras.
    "sentence_transformers": ("conduit.embeddings.local",),
    "torch": ("conduit.embeddings.local",),
    "transformers": ("conduit.embeddings.local",),
    "mcp": ("conduit.mcp",),
    "sqlalchemy": ("conduit.stores.sql",),
    "alembic": ("conduit.stores.sql",),
    # The sql extra's database drivers.
    "asyncpg": ("conduit.stores.sql",),
    "aiosqlite": ("conduit.stores.sql",),
    "konfig": ("conduit.integrations.konfig",),
    "warden": ("conduit.integrations.warden",),
    "opentelemetry": ("conduit.integrations.otel",),
}


@dataclass(frozen=True)
class ImportEdge:
    """One module importing one dotted name."""

    importer: str
    imported: str
    path: Path
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.importer} imports {self.imported}"


def within(name: str, prefix: str) -> bool:
    """True if ``name`` is ``prefix`` or a submodule of it."""
    return name == prefix or name.startswith(prefix + ".")


def module_name(path: Path, src_root: Path) -> str:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(relative: str, level: int, module: str, is_package: bool) -> str:
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".")
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    base = ".".join(parts)
    return f"{base}.{relative}" if relative else base


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dynamic_target(call: ast.Call) -> str | None:
    func = call.func
    named = (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}) or (
        isinstance(func, ast.Attribute) and func.attr == "import_module"
    )
    if not named or not call.args:
        return None
    return _literal(call.args[0])


def scan_source(source: str, module: str, *, is_package: bool, path: Path) -> list[ImportEdge]:
    """Every import in ``source``, as absolute dotted names."""
    edges: list[ImportEdge] = []

    def add(name: str, node: ast.AST) -> None:
        edges.append(ImportEdge(module, name, path, getattr(node, "lineno", 0)))

    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, node)
        elif isinstance(node, ast.ImportFrom):
            base = (
                _resolve(node.module or "", node.level, module, is_package)
                if node.level
                else node.module or ""
            )
            add(base, node)
            for alias in node.names:
                if alias.name != "*":
                    add(f"{base}.{alias.name}", node)
        elif isinstance(node, ast.Call):
            target = _dynamic_target(node)
            if target is None:
                continue
            if target.startswith("."):
                level = len(target) - len(target.lstrip("."))
                target = _resolve(target.lstrip("."), level, module, is_package)
            add(target, node)
    return edges


def source_files(src_root: Path = SRC_ROOT, package: str = PACKAGE) -> list[Path]:
    return sorted((src_root / package).rglob("*.py"))


def scan_tree(src_root: Path = SRC_ROOT, package: str = PACKAGE) -> list[ImportEdge]:
    edges: list[ImportEdge] = []
    for path in source_files(src_root, package):
        edges += scan_source(
            path.read_text(encoding="utf-8"),
            module_name(path, src_root),
            is_package=path.name == "__init__.py",
            path=path,
        )
    return edges


def core_violations(edges: Iterable[ImportEdge]) -> list[ImportEdge]:
    """``conduit.core`` importing any other conduit package, or ``conduit`` itself."""
    return [
        edge
        for edge in edges
        if within(edge.importer, CORE)
        and within(edge.imported, PACKAGE)
        and not within(edge.imported, CORE)
    ]


def check_owners(libraries: Mapping[str, tuple[str, ...]]) -> None:
    """Raise ``ValueError`` unless every library has a tuple of distinct, named owners.

    A bare string would otherwise be read as one owner per character, and an empty
    tuple would ban a library everywhere without saying why.
    """
    for library, owners in libraries.items():
        if (
            not isinstance(owners, tuple)
            or not owners
            or not all(isinstance(owner, str) and owner for owner in owners)
            or len(set(owners)) != len(owners)
        ):
            raise ValueError(f"{library}: owners must be a tuple of distinct layer names")


def optional_library_violations(
    edges: Iterable[ImportEdge],
    libraries: Mapping[str, tuple[str, ...]] = OPTIONAL_LIBRARIES,
) -> list[ImportEdge]:
    """An optional library imported outside every layer that owns it."""
    check_owners(libraries)
    found: list[ImportEdge] = []
    for edge in edges:
        for library, owners in libraries.items():
            if within(edge.imported, library) and not any(
                within(edge.importer, owner) for owner in owners
            ):
                found.append(edge)
                break
    return found


def cli_violations(edges: Iterable[ImportEdge]) -> list[ImportEdge]:
    """Anything outside ``conduit.cli`` importing it."""
    return [edge for edge in edges if within(edge.imported, CLI) and not within(edge.importer, CLI)]
