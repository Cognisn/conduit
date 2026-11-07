"""Start-up: conduit's settings, resolved and built in one step.

``start`` applies the settings' catalogue overlays, validates every configured
provider against its adapter, resolves every task-class assignment (capability floors,
residency, undeclared models) and only then builds the providers some task class is
assigned to. A refusal therefore happens at start-up, before any client exists, and
never part-way through a task.

Each provider is built for the assignments resolved to it, and serves only those: a
call through ``Runtime.providers`` or ``TaskBinding.provider`` that names another
model, or another task class's model, is refused with ``ModelNotAssigned`` before any
client call. ``TaskBinding.request`` builds requests that stay within the assignment.

Embedding task classes may be passed to ``start`` with the chat ones; it leaves them, and
any provider only they use, to ``conduit.embeddings.start_embeddings``. The runtime
exposes its event sink and capture policy (``Runtime.events``, ``Runtime.capture``), read
only, so a layer built on it can default to them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator

from conduit.core.catalogue import Catalogue, CatalogueOverlay, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.events import CapturePolicy, EventSink, FanOutSink
from conduit.core.identifiers import IDENTIFIER_PATTERN
from conduit.core.registry import MeteredProvider, ProviderFactory, ProviderRegistry
from conduit.core.residency import ResidencyPolicy
from conduit.core.secrets import SecretResolver
from conduit.core.settings import AssignmentSettings, ProviderSettings, SettingsModel
from conduit.core.tasks import Assignments, ResolvedAssignment, TaskClass, resolve_assignments
from conduit.core.types import Message, ModelRequest, SystemBlock, ToolSpec


class ConduitSettings(SettingsModel):
    """Everything conduit is configured with. Build it in code or from a mapping."""

    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    assignments: dict[str, AssignmentSettings] = Field(default_factory=dict)
    residency: ResidencyPolicy = ResidencyPolicy()
    capture: CapturePolicy = CapturePolicy()
    catalogue: tuple[CatalogueOverlay, ...] = ()
    require_all_assignments: bool = True

    @field_validator("providers")
    @classmethod
    def _names_are_identifiers(
        cls, value: dict[str, ProviderSettings]
    ) -> dict[str, ProviderSettings]:
        for name in value:
            if not re.fullmatch(IDENTIFIER_PATTERN, name):
                raise ValueError("provider names must be identifiers (no spaces)")
        return value


@dataclass(frozen=True, slots=True)
class TaskBinding:
    """A task class's provider and resolved assignment."""

    provider: MeteredProvider
    assignment: ResolvedAssignment

    def request(
        self,
        *,
        messages: Iterable[Message],
        system: Iterable[SystemBlock] = (),
        tools: Iterable[ToolSpec] = (),
        **fields: Any,
    ) -> ModelRequest:
        """A request on this task's model, within its output budget (``task_request``)."""
        return task_request(
            self.assignment, messages=messages, system=system, tools=tools, **fields
        )


def task_request(
    assignment: ResolvedAssignment,
    *,
    messages: Iterable[Message],
    system: Iterable[SystemBlock] = (),
    tools: Iterable[ToolSpec] = (),
    **fields: Any,
) -> ModelRequest:
    """A request on ``assignment``'s model, within its output budget.

    ``max_output_tokens`` defaults to the assignment's budget. One given explicitly is
    checked again against the model's output limit, as the budget was at start-up, and
    a ``ConfigurationError`` is raised above it.
    """
    fields.setdefault("max_output_tokens", assignment.max_output_tokens)
    request = ModelRequest(
        model=assignment.model,
        system=tuple(system),
        messages=tuple(messages),
        tools=tuple(tools),
        task_class=assignment.task_class.name,
        **fields,
    )
    limit = assignment.entry.output_limit
    if limit is not None and request.max_output_tokens > limit:
        raise ConfigurationError(
            f"{assignment.task_class.name}: max_output_tokens "
            f"{request.max_output_tokens} exceeds model {assignment.model!r}'s "
            f"output limit of {limit}"
        )
    return request


class Runtime:
    """Resolved assignments and built providers, from ``start``.

    ``events`` and ``capture`` are read only, so a layer (the agent runner, say) can
    default to the runtime's own: ``capture`` is the settings' capture policy, and
    ``events`` delivers to the product's sink (the one given to ``start``; none means
    events go nowhere), holding every event to ``capture`` whatever its emitter did.
    """

    def __init__(
        self,
        *,
        catalogue: Catalogue,
        assignments: Assignments,
        providers: Mapping[str, MeteredProvider],
        registry: ProviderRegistry,
        events: EventSink | None = None,
        capture: CapturePolicy | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.assignments = assignments
        self.providers = dict(providers)
        self.registry = registry
        self._capture = capture if capture is not None else CapturePolicy()
        self._events = FanOutSink([events] if events is not None else [], policy=self._capture)

    @property
    def events(self) -> EventSink:
        """The runtime's event sink: the product's, under the runtime's capture policy."""
        return self._events

    @property
    def capture(self) -> CapturePolicy:
        """The runtime's content-capture policy (the settings' ``capture``)."""
        return self._capture

    def for_task(self, task_class: str) -> TaskBinding:
        assignment = self.assignments.for_task(task_class)
        return TaskBinding(self.providers[assignment.provider], assignment)

    async def aclose(self) -> None:
        await self.registry.aclose()


async def start(
    settings: ConduitSettings,
    *,
    task_classes: Iterable[TaskClass],
    factories: Iterable[ProviderFactory],
    secrets: SecretResolver,
    events: EventSink | None = None,
    catalogue: Catalogue | None = None,
) -> Runtime:
    """Resolve and build. Raises a ``ConfigurationError`` before any client is made.

    Embedding task classes (a floor whose ``kind`` is ``embedding``) may be given with
    the chat ones, so one list and one ``ConduitSettings`` serve both start-ups: they,
    their assignments and any provider only they are assigned to are left for
    ``conduit.embeddings.start_embeddings``, which resolves and builds them through the
    registry's embedding adapters. Every other configured provider is validated here.
    """
    declared = list(task_classes)
    seen: set[str] = set()
    for task in declared:
        if task.name in seen:
            raise ConfigurationError(f"task class {task.name!r} is declared twice")
        seen.add(task.name)
    embedding = {task.name for task in declared if task.floor.kind == "embedding"}
    chosen = {name: a for name, a in settings.assignments.items() if name not in embedding}
    embedding_only = {
        a.provider for name, a in settings.assignments.items() if name in embedding
    } - {a.provider for a in chosen.values()}
    base = catalogue if catalogue is not None else default_catalogue()
    applied = base.apply(*settings.catalogue)
    registry = ProviderRegistry(
        secrets=secrets,
        catalogue=applied,
        events=events,
        capture=settings.capture,
        residency=settings.residency,
        factories=factories,
    )
    validated = {
        name: registry.validate(name, provider)
        for name, provider in settings.providers.items()
        if name not in embedding_only
    }
    assignments = resolve_assignments(
        [task for task in declared if task.name not in embedding],
        chosen,
        validated,
        applied,
        settings.residency,
        require_all=settings.require_all_assignments,
    )
    providers: dict[str, MeteredProvider] = {}
    try:
        for name in sorted(assignments.providers):
            assigned = {
                task: resolved.model
                for task, resolved in assignments.items()
                if resolved.provider == name
            }
            providers[name] = await registry.build(name, validated[name], assigned=assigned)
    except BaseException:
        await registry.aclose()
        raise
    return Runtime(
        catalogue=applied,
        assignments=assignments,
        providers=providers,
        registry=registry,
        events=events,
        capture=settings.capture,
    )
