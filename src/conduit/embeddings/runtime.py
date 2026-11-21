"""Start-up for embeddings: embedding task classes resolved and their providers built.

``start_embeddings`` does for embedding task classes what ``conduit.core.runtime.start``
does for chat ones, from the same ``ConduitSettings``: it applies the catalogue
overlays, validates each configured provider an embedding task class is assigned to
against its embedding adapter, lists each one whose adapter can list (C18, as ``start``
does: every built-in embedding adapter inherits its chat adapter's discovery, and
``discovery`` in the settings may put a provider in catalogue mode), resolves every
assignment through the core's ``resolve_assignments`` (a listed or declared model,
capability floor with its ``kind``, residency on the listed Regions) and only then
builds the embedding providers, each on its listed catalogue. A chat model cannot satisfy an
embedding task class, because the floor's ``kind`` is ``embedding``; an embedding model
cannot satisfy a chat task class for the same reason.

Each task class's vector length is fixed here too: the model's own
(``embedding_dimensions`` in the catalogue), or a shorter one the product gives in
``dimensions``, which must meet the floor's ``min_embedding_dimensions`` and be offered
by the model. Every refusal happens at start-up, before any call; the registry is
closed on the way out. ``EmbeddingRuntime.discovery`` holds each listing made.

Only the task classes passed in are resolved. Assignments in the settings for other task
classes (the product's chat task classes) are left for ``start``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.discovery import DiscoveryResult
from conduit.core.errors import AssignmentRefused, CapabilityRefused, ConfigurationError
from conduit.core.events import EventSink
from conduit.core.provider import EmbeddingRequest, EmbeddingResponse
from conduit.core.registry import EmbeddingProviderFactory, MeteredEmbedder, ProviderRegistry
from conduit.core.runtime import ConduitSettings, _list_providers
from conduit.core.secrets import SecretResolver
from conduit.core.tasks import Assignments, ResolvedAssignment, TaskClass, resolve_assignments


@dataclass(frozen=True, slots=True)
class EmbeddingBinding:
    """An embedding task class's provider, resolved assignment and vector length."""

    embedder: MeteredEmbedder
    assignment: ResolvedAssignment
    dimensions: int

    def request(self, texts: Iterable[str], **fields: Any) -> EmbeddingRequest:
        """A request on this task's model, at its resolved dimensions."""
        return EmbeddingRequest(
            model=self.assignment.model,
            texts=tuple(texts),
            dimensions=self.dimensions,
            task_class=self.assignment.task_class.name,
            **fields,
        )

    async def embed(self, texts: Iterable[str], **fields: Any) -> EmbeddingResponse:
        """One vector per text, through the task's provider."""
        return await self.embedder.embed(self.request(texts, **fields))


class EmbeddingRuntime:
    """Resolved embedding assignments and built embedding providers."""

    def __init__(
        self,
        *,
        catalogue: Catalogue,
        assignments: Assignments,
        dimensions: Mapping[str, int],
        embedders: Mapping[str, MeteredEmbedder],
        registry: ProviderRegistry,
        discovery: Mapping[str, DiscoveryResult] | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.assignments = assignments
        self.dimensions = dict(dimensions)
        self.embedders = dict(embedders)
        self.registry = registry
        #: The listing start-up made for each provider it listed, by provider name.
        self.discovery: Mapping[str, DiscoveryResult] = dict(discovery or {})

    def for_task(self, task_class: str) -> EmbeddingBinding:
        assignment = self.assignments.for_task(task_class)
        return EmbeddingBinding(
            self.embedders[assignment.provider], assignment, self.dimensions[task_class]
        )

    async def aclose(self) -> None:
        await self.registry.aclose()


def _embedding_classes(task_classes: Iterable[TaskClass]) -> list[TaskClass]:
    found = list(task_classes)
    for task in found:
        if task.floor.kind != "embedding":
            raise ConfigurationError(
                f"task class {task.name!r} is not an embedding task class (its floor's kind "
                f"is {task.floor.kind})"
            )
    return found


async def start_embeddings(
    settings: ConduitSettings,
    *,
    task_classes: Iterable[TaskClass],
    factories: Iterable[EmbeddingProviderFactory],
    secrets: SecretResolver,
    events: EventSink | None = None,
    catalogue: Catalogue | None = None,
    dimensions: Mapping[str, int] | None = None,
) -> EmbeddingRuntime:
    """List, resolve and build. Raises a ``ConfigurationError`` before any call is made,
    or the ``ProviderError`` of a listing that failed, before any provider is built."""
    declared = _embedding_classes(task_classes)
    names = {task.name for task in declared}
    wanted = dict(dimensions or {})
    for task_name in sorted(wanted):
        if task_name not in names:
            raise AssignmentRefused(
                f"{task_name}: dimensions are given for a task class that is not declared"
            )
    base = catalogue if catalogue is not None else default_catalogue()
    applied = base.apply(*settings.catalogue)
    registry = ProviderRegistry(
        secrets=secrets,
        catalogue=applied,
        events=events,
        capture=settings.capture,
        residency=settings.residency,
        embedding_factories=factories,
    )
    chosen = {name: a for name, a in settings.assignments.items() if name in names}
    validated = {
        name: registry.validate_embedding(name, settings.providers[name])
        for name in sorted({a.provider for a in chosen.values()})
        if name in settings.providers
    }
    listings = await _list_providers(registry, settings, validated)
    assignments = resolve_assignments(
        declared,
        chosen,
        validated,
        applied,
        settings.residency,
        require_all=settings.require_all_assignments,
        listings=listings,
    )
    resolved_dimensions: dict[str, int] = {}
    for task_name, resolved in sorted(assignments.items()):
        own = resolved.entry.embedding_dimensions or 0
        length = wanted.get(task_name, own)
        floor = resolved.task_class.floor.min_embedding_dimensions
        if length < 1:
            raise CapabilityRefused(f"{task_name}: {length} dimensions are fewer than 1")
        if length < floor:
            raise CapabilityRefused(
                f"{task_name}: {length} dimensions are below the task's floor of {floor}"
            )
        resolved_dimensions[task_name] = length
    embedders: dict[str, MeteredEmbedder] = {}
    try:
        for provider in sorted(assignments.providers):
            mine: Sequence[str] = [t for t, a in assignments.items() if a.provider == provider]
            embedders[provider] = await registry.build_embedder(
                provider,
                validated[provider],
                assigned={task: assignments[task].model for task in mine},
                dimensions={task: wanted[task] for task in mine if task in wanted},
                catalogue=(
                    listings[provider].listed_catalogue(applied) if provider in listings else None
                ),
            )
    except BaseException:
        await registry.aclose()
        raise
    return EmbeddingRuntime(
        catalogue=applied,
        assignments=assignments,
        dimensions=resolved_dimensions,
        embedders=embedders,
        registry=registry,
        discovery=listings,
    )
