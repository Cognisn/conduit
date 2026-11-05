"""Task classes, capability floors and the resolution of assignments at start-up.

Task classes are the product's: it declares each one with a capability floor and
assigns a configured provider and a model to it (C18). ``resolve_assignments`` checks
every assignment once, at start-up: the model must be declared in the catalogue,
offered in the provider's region, meet the floor and satisfy the residency policy,
and the output budget must fit the model. The result is immutable and holds its own
copy of each entry, so nothing about a running task changes part-way through, and no
floor or residency check can fail mid-task. conduit does no learned routing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, NonNegativeInt

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import (
    AssignmentRefused,
    CapabilityRefused,
    ConfigurationError,
    UnknownTaskClass,
)
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.settings import AssignmentSettings, ProviderSettings, SettingsModel

TASK_CLASS_PATTERN = r"^[a-z][a-z0-9_.-]{0,63}$"


class CapabilityFloor(SettingsModel):
    """The least a model must offer to serve a task class."""

    kind: Literal["chat", "embedding"] = "chat"
    tools: bool = False
    vision: bool = False
    prompt_caching: bool = False
    structured_output: bool = False
    min_context_window: NonNegativeInt = 0
    min_output_limit: NonNegativeInt = 0
    min_embedding_dimensions: NonNegativeInt = 0
    required_labels: frozenset[str] = frozenset()

    def shortfalls(self, entry: ModelEntry) -> list[str]:
        """What ``entry`` lacks, in words; empty when it meets the floor."""
        found: list[str] = []
        if entry.kind != self.kind:
            found.append(f"is a {entry.kind} model, not {self.kind}")
        caps = entry.capabilities
        for wanted, have, words in (
            (self.tools, caps.tools, "tools"),
            (self.vision, caps.vision, "vision"),
            (self.prompt_caching, caps.prompt_caching, "prompt caching"),
            (self.structured_output, caps.structured_output, "structured output"),
        ):
            if wanted and not have:
                found.append(f"lacks {words}")
        if entry.context_window < self.min_context_window:
            found.append(
                f"has a {entry.context_window}-token context window, below "
                f"{self.min_context_window}"
            )
        if (entry.output_limit or 0) < self.min_output_limit:
            found.append(
                f"has an output limit of {entry.output_limit}, below {self.min_output_limit}"
            )
        if (entry.embedding_dimensions or 0) < self.min_embedding_dimensions:
            found.append(
                f"has {entry.embedding_dimensions} embedding dimensions, below "
                f"{self.min_embedding_dimensions}"
            )
        missing = sorted(self.required_labels - entry.labels)
        if missing:
            found.append(f"lacks labels {', '.join(missing)}")
        return found


class TaskClass(SettingsModel):
    """A product-declared class of work, with its capability floor."""

    name: str = Field(pattern=TASK_CLASS_PATTERN)
    floor: CapabilityFloor = CapabilityFloor()
    description: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedAssignment:
    """A checked assignment: what a task class runs on, fixed at start-up."""

    task_class: TaskClass
    provider: str  # the configured provider's name
    kind: str  # the provider kind (bedrock, anthropic, stub)
    model: str
    entry: ModelEntry
    max_output_tokens: int
    provider_region: str | None

    @property
    def model_id(self) -> str:
        """``<kind>:<model>``, as recorded on events."""
        return f"{self.kind}:{self.model}"


class Assignments(Mapping[str, ResolvedAssignment]):
    """Every resolved assignment, by task-class name. Read only."""

    def __init__(self, resolved: Mapping[str, ResolvedAssignment]) -> None:
        self._resolved = dict(resolved)

    def for_task(self, task_class: str) -> ResolvedAssignment:
        if task_class not in self._resolved:
            raise UnknownTaskClass(f"no assignment for task class {task_class!r}")
        return self._resolved[task_class]

    def __getitem__(self, key: str) -> ResolvedAssignment:
        return self._resolved[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._resolved)

    def __len__(self) -> int:
        return len(self._resolved)

    @property
    def providers(self) -> frozenset[str]:
        """The configured providers some task class is assigned to."""
        return frozenset(a.provider for a in self._resolved.values())


def _offered_in(entry: ModelEntry, region: str | None) -> bool:
    if entry.residency_scope not in (ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION):
        return True
    return region is None or region in entry.regions


def resolve_assignments(
    task_classes: Iterable[TaskClass],
    assignments: Mapping[str, AssignmentSettings],
    providers: Mapping[str, ProviderSettings],
    catalogue: Catalogue,
    residency: ResidencyPolicy | None = None,
    *,
    require_all: bool = True,
) -> Assignments:
    """Every assignment, checked. Raises on the first defect, in task-class order.

    Raises ``UndeclaredModel``, ``CapabilityRefused``, ``ResidencyRefused`` or
    ``AssignmentRefused``; all are ``ConfigurationError``.
    """
    policy = residency or ResidencyPolicy()
    declared: dict[str, TaskClass] = {}
    for task in task_classes:
        if task.name in declared:
            raise ConfigurationError(f"task class {task.name!r} is declared twice")
        declared[task.name] = task
    for name in sorted(assignments):
        if name not in declared:
            raise AssignmentRefused(f"{name}: task class is not declared by the product")
    if require_all:
        for name in sorted(declared):
            if name not in assignments:
                raise AssignmentRefused(f"{name}: task class has no assignment")

    resolved: dict[str, ResolvedAssignment] = {}
    for name in sorted(assignments):
        task = declared[name]
        assignment = assignments[name]
        provider = providers.get(assignment.provider)
        if provider is None:
            raise AssignmentRefused(f"{name}: no provider named {assignment.provider!r}")
        entry = catalogue.require(provider.catalogue_provider, assignment.model)
        if not _offered_in(entry, provider.region):
            raise AssignmentRefused(
                f"{name}: model {assignment.model!r} is not offered in region {provider.region}"
            )
        shortfalls = task.floor.shortfalls(entry)
        if shortfalls:
            raise CapabilityRefused(
                f"{name}: model {assignment.model!r} is below the task's floor: "
                + "; ".join(shortfalls)
            )
        policy.check(entry, provider.region, task=name)
        limit = entry.output_limit or 0
        max_output = assignment.max_output_tokens or limit
        if max_output > limit:
            raise CapabilityRefused(
                f"{name}: max_output_tokens {max_output} exceeds model "
                f"{assignment.model!r}'s output limit of {limit}"
            )
        resolved[name] = ResolvedAssignment(
            task_class=task,
            provider=assignment.provider,
            kind=provider.kind,
            model=assignment.model,
            entry=entry,
            max_output_tokens=max_output,
            provider_region=provider.region,
        )
    return Assignments(resolved)
