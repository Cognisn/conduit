"""Static scanner for the single provider construction point (plan section 4, C19).

The rule: no module constructs a provider client outside the provider registry. In
practice, an SDK client constructor may be called only inside a function named
``create_client`` (a provider adapter's factory method) in ``conduit.providers`` or
``conduit.embeddings``, and only ``conduit.core.registry`` may call ``create_client``.
So every client conduit makes goes through ``ProviderRegistry.build``, the one place
settings are validated, secrets resolved and calls metered.

The scan reads source, never imports it, and resolves the import forms adapters use:
``import x``, ``import x as y``, ``from x import Y as Z``, and
``importlib.import_module("x")`` (called directly or assigned to a name).
Session-based Bedrock clients (``session.client("bedrock-runtime")``) are caught by
their service name.

The scan also records every construction of ``MeteredProvider``: only the registry may
wrap a provider, since the registry is where a provider is bound to its resolved
assignments or to the residency policy.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.tripwires.importscan import (
    SRC_ROOT,
    module_name,
    source_files,
    within,
)

#: Fully qualified SDK client constructors.
CLIENT_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "anthropic.Anthropic",
        "anthropic.AsyncAnthropic",
        "anthropic.AnthropicBedrock",
        "anthropic.AsyncAnthropicBedrock",
        "anthropic.AnthropicVertex",
        "anthropic.AsyncAnthropicVertex",
        "anthropic.AnthropicFoundry",
        "anthropic.AsyncAnthropicFoundry",
        "openai.OpenAI",
        "openai.AsyncOpenAI",
        "openai.AzureOpenAI",
        "openai.AsyncAzureOpenAI",
        "boto3.client",
        "boto3.resource",
        "boto3.Session",
        "boto3.session.Session",
        "aioboto3.Session",
        "botocore.session.Session",
        "botocore.session.get_session",
        "google.genai.Client",
        "google.genai.client.Client",
        "ollama.Client",
        "ollama.AsyncClient",
    }
)

#: Generic HTTP clients count as provider clients only inside the provider layers
#: (the MCP client, for example, has its own HTTP stack).
HTTP_CLIENTS: frozenset[str] = frozenset({"httpx.Client", "httpx.AsyncClient"})

#: Layers where a ``create_client`` may construct a client.
CLIENT_LAYERS: tuple[str, ...] = ("conduit.providers", "conduit.embeddings")
FACTORY_METHOD = "create_client"
REGISTRY = "conduit.core.registry"
METERED = "MeteredProvider"
METERED_QUALIFIED = f"{REGISTRY}.{METERED}"
BEDROCK_SERVICE_PREFIX = "bedrock"


@dataclass(frozen=True)
class Finding:
    module: str
    path: Path
    line: int
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.module} {self.what}"


def _literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _import_module_target(node: ast.AST) -> str | None:
    """The literal module of an ``import_module("x")`` / ``__import__("x")`` call."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    func = node.func
    named = (isinstance(func, ast.Name) and func.id in {"import_module", "__import__"}) or (
        isinstance(func, ast.Attribute) and func.attr == "import_module"
    )
    return _literal(node.args[0]) if named else None


def _aliases(tree: ast.AST) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    found[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            module = _import_module_target(node.value)
            if isinstance(target, ast.Name) and module:
                found[target.id] = module
    return found


def _dotted(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        base = aliases.get(node.id)
        if base is None:
            return None
    else:
        base = _import_module_target(node)
        if base is None:
            return None
    return ".".join([base, *reversed(parts)])


class _Visitor(ast.NodeVisitor):
    def __init__(self, module: str, path: Path, aliases: Mapping[str, str]) -> None:
        self.module = module
        self.path = path
        self.aliases = aliases
        self.functions: list[str] = []
        self.constructions: list[Finding] = []
        self.factory_calls: list[Finding] = []
        self.metered: list[Finding] = []

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted(node.func, self.aliases)
        in_layer = any(within(self.module, layer) for layer in CLIENT_LAYERS)
        constructs = target in CLIENT_CONSTRUCTORS or (target in HTTP_CLIENTS and in_layer)
        if not constructs and isinstance(node.func, ast.Attribute) and node.func.attr == "client":
            service = _literal(node.args[0]) if node.args else None
            constructs = bool(service and service.startswith(BEDROCK_SERVICE_PREFIX))
            target = target or f"<session>.client({service!r})"
        if constructs:
            self.constructions.append(
                Finding(
                    self.module,
                    self.path,
                    node.lineno,
                    f"constructs {target}"
                    + (f" in {'.'.join(self.functions)}" if self.functions else " at module level"),
                )
            )
            allowed = in_layer and FACTORY_METHOD in self.functions
            if allowed:
                self.constructions.pop()
        called = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        bare = isinstance(node.func, ast.Name) and node.func.id == METERED and target is None
        if target == METERED_QUALIFIED or bare:
            self.metered.append(
                Finding(self.module, self.path, node.lineno, f"constructs {METERED}")
            )
        if called == FACTORY_METHOD:
            self.factory_calls.append(
                Finding(self.module, self.path, node.lineno, f"calls {FACTORY_METHOD}")
            )
        self.generic_visit(node)


@dataclass(frozen=True)
class Scan:
    constructions: list[Finding]  # constructions outside an allowed create_client
    factory_calls: list[Finding]  # every call of create_client
    metered: list[Finding]  # every construction of MeteredProvider


def scan_source(source: str, module: str, *, path: Path) -> Scan:
    tree = ast.parse(source, filename=str(path))
    visitor = _Visitor(module, path, _aliases(tree))
    visitor.visit(tree)
    return Scan(visitor.constructions, visitor.factory_calls, visitor.metered)


def scan_tree(src_root: Path = SRC_ROOT) -> Scan:
    constructions: list[Finding] = []
    factory_calls: list[Finding] = []
    metered: list[Finding] = []
    for path in source_files(src_root):
        scan = scan_source(path.read_text(encoding="utf-8"), module_name(path, src_root), path=path)
        constructions += scan.constructions
        factory_calls += scan.factory_calls
        metered += scan.metered
    return Scan(constructions, factory_calls, metered)


def construction_violations(scan: Scan) -> list[Finding]:
    return list(scan.constructions)


def factory_call_violations(calls: Iterable[Finding]) -> list[Finding]:
    return [c for c in calls if not within(c.module, REGISTRY)]


def metered_violations(constructions: Iterable[Finding]) -> list[Finding]:
    return [c for c in constructions if not within(c.module, REGISTRY)]
