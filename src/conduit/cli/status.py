"""``providers``, ``models`` and ``check``: what conduit offers and whether it is ready.

``check`` builds every configured provider through conduit's provider registry, so each
one's secrets are resolved and its client is made exactly as a product's would be, and
then resolves every assignment against the catalogue, the capability floors and the
residency policy, one at a time, so each refusal is shown on its own line. No model is
called: a check never reaches a model, and for the SigV4 modes of Amazon Bedrock the
credentials themselves are checked by AWS on the first call.
"""

from __future__ import annotations

import importlib.util

from conduit import (
    AssignmentSettings,
    Catalogue,
    ConduitError,
    ProviderRegistry,
    ProviderSettings,
    ResidencyPolicy,
    TaskClass,
    default_catalogue,
    resolve_assignments,
)
from conduit.cli.config import CliConfig
from conduit.cli.output import describe_error
from conduit.cli.setup import Context, task_classes

#: Each adapter's extra and the libraries it needs.
EXTRAS: dict[str, tuple[str, tuple[str, ...]]] = {
    "stub": ("(none)", ()),
    "bedrock": ("bedrock", ("boto3",)),
    "bedrock_mantle": ("bedrock-mantle", ("anthropic", "botocore")),
    "anthropic": ("anthropic", ("anthropic",)),
    "openai_compat": ("openai", ("openai",)),
    "gemini": ("gemini", ("google.genai",)),
    "ollama": ("ollama", ("httpx",)),
}

#: What building a Bedrock provider proves, by authentication mode.
BEDROCK_CHECKS = {
    "credential_chain": "SigV4 client built; AWS checks the credentials on the first call",
    "access_keys": "keys resolved and SigV4 client built; AWS checks them on the first call",
    "short_term_api_key": "short-term API key generated from AWS credentials (never shown)",
    "long_term_api_key": "long-term API key resolved (never shown)",
}


def installed(library: str) -> bool:
    """Whether ``library`` can be imported, found without importing it."""
    try:
        return importlib.util.find_spec(library) is not None
    except (ImportError, ValueError):
        return False


def auth_mode(settings: ProviderSettings) -> str:
    """A provider's authentication mode, as its settings name it, or ``-``."""
    auth = getattr(settings, "auth", None)
    mode = getattr(auth, "mode", None) or getattr(auth, "type", None)
    if not isinstance(mode, str):
        return "-"
    source = getattr(auth, "source", None)
    source_mode = getattr(source, "mode", None)
    if isinstance(source_mode, str):
        return f"{mode} (signed with {source_mode})"
    return mode


def what_was_checked(settings: ProviderSettings) -> str:
    if settings.kind == "stub":
        return "no credentials needed"
    if settings.kind == "bedrock":
        auth = getattr(settings, "auth", None)
        return BEDROCK_CHECKS.get(str(getattr(auth, "mode", "")), "client built")
    return "credentials resolved and client built (never shown)"


def catalogue_for(config: CliConfig) -> Catalogue:
    return default_catalogue().apply(*config.conduit.catalogue)


def providers(context: Context, config: CliConfig) -> int:
    screen = context.screen
    rows = []
    for factory in context.factories:
        extra, libraries = EXTRAS.get(factory.kind, ("?", ()))
        ready = all(installed(library) for library in libraries)
        rows.append([factory.kind, extra, "yes" if ready else f"no: install the {extra} extra"])
    screen.table("Provider adapters", ["kind", "extra", "installed"], rows)
    registry = ProviderRegistry(secrets=context.secrets(), factories=context.factories)
    configured = []
    for name, settings in config.conduit.providers.items():
        try:
            validated = registry.validate(name, settings)
        except ConduitError as error:
            configured.append([name, settings.kind, "-", "-", describe_error(error)])
            continue
        configured.append(
            [name, validated.kind, validated.region or "-", auth_mode(validated), "valid"]
        )
    screen.table(
        "Configured providers", ["name", "kind", "region", "authentication", "settings"], configured
    )
    return 0


def models(context: Context, config: CliConfig, provider: str | None) -> int:
    rows = []
    for entry in catalogue_for(config):
        if provider is not None and entry.provider != provider:
            continue
        caps = entry.capabilities
        flags = [
            name
            for name, on in (
                ("tools", caps.tools),
                ("vision", caps.vision),
                ("caching", caps.prompt_caching),
                ("structured", caps.structured_output),
            )
            if on
        ]
        rows.append(
            [
                entry.provider,
                entry.model_id,
                entry.kind,
                entry.residency_scope.value,
                ", ".join(sorted(entry.regions)) or "-",
                ", ".join(flags) or "-",
            ]
        )
    context.screen.table(
        "Model catalogue",
        ["provider", "model", "kind", "residency", "regions", "capabilities"],
        rows,
    )
    return 0


async def check(context: Context, config: CliConfig) -> int:
    """Build every provider and resolve every assignment; 0 when all are ready."""
    screen = context.screen
    failed = False
    catalogue = catalogue_for(config)
    settings = config.conduit
    registry = ProviderRegistry(
        secrets=context.secrets(),
        catalogue=catalogue,
        residency=settings.residency,
        factories=context.factories,
    )
    validated: dict[str, ProviderSettings] = {}
    rows = []
    try:
        for name, given in settings.providers.items():
            outcome = ""
            built: ProviderSettings | None = None
            try:
                built = registry.validate(name, given)
                await registry.build(name, built)
            except Exception as error:  # shown by class (and a conduit error's message)
                outcome = f"refused: {describe_error(error)}"
            if built is not None and not outcome:
                validated[name] = built
                outcome = f"ready: {what_was_checked(built)}"
            failed = failed or outcome.startswith("refused")
            mode = auth_mode(built) if built is not None else "-"
            region = (built.region if built is not None else None) or "-"
            rows.append([name, given.kind, region, mode, outcome])
    finally:
        await registry.aclose()
    screen.table("Providers", ["name", "kind", "region", "authentication", "check"], rows)

    declared = {task.name: task for task in task_classes(config)}
    rows = []
    for task_name, assignment in sorted(settings.assignments.items()):
        rows.append(
            _assignment_row(
                task_name, assignment, declared, validated, catalogue, settings.residency
            )
        )
        failed = failed or rows[-1][-1].startswith("refused")
    policy = settings.residency
    regions = ", ".join(sorted(policy.regions)) if policy.regions is not None else "any"
    scopes = ", ".join(sorted(scope.value for scope in policy.scopes)) or "none"
    screen.table(
        f"Assignments (residency: scopes {scopes}; regions {regions})",
        ["task class", "provider", "model", "residency", "check"],
        rows,
    )
    screen.say("All providers and assignments are ready." if not failed else "Some were refused.")
    return 1 if failed else 0


def _assignment_row(
    task_name: str,
    assignment: AssignmentSettings,
    declared: dict[str, TaskClass],
    validated: dict[str, ProviderSettings],
    catalogue: Catalogue,
    residency: ResidencyPolicy,
) -> list[str]:
    """One assignment resolved on its own, so each refusal is shown by itself."""
    row = [task_name, assignment.provider, assignment.model]
    task = declared.get(task_name)
    if task is None:
        return [*row, "-", "refused: not a task class this interface declares"]
    provider = validated.get(assignment.provider)
    if provider is None:
        return [*row, "-", "refused: its provider was refused or is not configured"]
    entry = catalogue.get(provider.catalogue_provider, assignment.model)
    scope = entry.residency_scope.value if entry is not None else "-"
    try:
        resolve_assignments(
            [task],
            {task_name: assignment},
            {assignment.provider: provider},
            catalogue,
            residency,
            require_all=False,
        )
    except ConduitError as error:
        return [*row, scope, f"refused: {describe_error(error)}"]
    return [*row, scope, "ready"]
