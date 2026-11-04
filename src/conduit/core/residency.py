"""Residency policy: where a deployment allows its requests to be processed.

Each catalogue entry states its residency scope:

- ``local``: runs on the deployment's own hosts (Ollama, the stub).
- ``in_region``: processed in the region the provider is called in, one of the
  entry's ``regions``.
- ``cross_region``: may be processed in any of the entry's ``regions`` (a Bedrock
  geographic inference profile, for example).
- ``global``: may be processed anywhere the provider chooses.

A ``ResidencyPolicy`` allows a set of scopes and, optionally, a set of regions. An
assignment that would breach it is refused when assignments are resolved at
start-up (``conduit.core.tasks.resolve_assignments``), never part-way through a task.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from conduit.core.errors import ResidencyRefused
from conduit.core.settings import SettingsModel

if TYPE_CHECKING:
    from conduit.core.catalogue import ModelEntry


class ResidencyScope(StrEnum):
    LOCAL = "local"
    IN_REGION = "in_region"
    CROSS_REGION = "cross_region"
    GLOBAL = "global"


class ResidencyPolicy(SettingsModel):
    """The scopes a deployment allows, and the regions requests may be processed in.

    ``regions`` of ``None`` means no region restriction. With regions set, a global
    entry is always refused, since its processing regions are not bounded.
    """

    scopes: frozenset[ResidencyScope] = Field(default_factory=lambda: frozenset(ResidencyScope))
    regions: frozenset[str] | None = None

    def check(self, entry: ModelEntry, provider_region: str | None, *, task: str) -> None:
        """Raise ``ResidencyRefused`` if calling ``entry`` from ``provider_region`` breaches."""
        scope = entry.residency_scope
        where = f"{task}: model {entry.model_id!r} ({scope.value})"
        if scope not in self.scopes:
            allowed = ", ".join(sorted(s.value for s in self.scopes)) or "none"
            raise ResidencyRefused(f"{where}: scope {scope.value} is not allowed ({allowed})")
        if self.regions is None or scope is ResidencyScope.LOCAL:
            return
        if scope is ResidencyScope.GLOBAL:
            raise ResidencyRefused(
                f"{where}: global processing is refused when regions are restricted"
            )
        if provider_region is None:
            raise ResidencyRefused(f"{where}: the provider has no region set")
        if provider_region not in self.regions:
            raise ResidencyRefused(f"{where}: region {provider_region} is not allowed")
        if scope is ResidencyScope.CROSS_REGION:
            outside = sorted(entry.regions - self.regions)
            if outside:
                raise ResidencyRefused(
                    f"{where}: may be processed in {', '.join(outside)}, outside the allowed "
                    "regions"
                )
