"""Residency policy: where a deployment allows its requests to be processed.

Each model has a residency scope, taken from the provider's own listing where conduit
can list the provider (``conduit.core.discovery``), and otherwise from the catalogue
entry a product declares; a model whose scope nobody states is ``global``, the least
bounded, so it is never let through a policy that restricts Regions:

- ``local``: runs on the deployment's own hosts (Ollama, the stub).
- ``in_region``: processed in the region the provider is called in, one of the
  entry's ``regions``.
- ``cross_region``: may be processed in any of the entry's ``regions`` (a Bedrock
  geographic inference profile, for example).
- ``global``: may be processed anywhere the provider chooses.

For a listed model, ``regions`` are the Regions the provider reports: a Bedrock
inference profile's destination Regions, or the Region an in-Region model is called
in. A ``ResidencyPolicy`` allows a set of scopes and, optionally, a set of regions. An
assignment that would breach it is refused when assignments are resolved at
start-up (``conduit.core.tasks.resolve_assignments``), never part-way through a task.

A connection may narrow this further. Its adapter's settings build a ``RegionChoice``
from the person's choice of where requests may be processed (``region_choice``), and
the choice is checked after the deployment's policy, never instead of it: it can only
narrow. A connection may also declare the residency its endpoint guarantees for a model
(``declared_residency``: an OpenAI regional host, X.AI's US host, Anthropic's
``inference_geo``, a Foundry deployment type), which replaces the residency a listing or
an entry states. AWS Regions are bare codes (``ap-southeast-2``); a geography another
provider offers carries that provider's prefix (``anthropic:us``, ``openai:eu``,
``xai:us``, ``azure:swedencentral``, ``azure:zone-eu``), so a policy never matches a
code it did not mean. A connection that chooses and declares nothing is held to the
deployment's policy alone, as before.

``OfferedRegion`` and ``RegionListing`` describe the Regions an adapter offers, for a
product to show as choices (``ProviderRegistry.regions``): static, documented lists,
each with its source.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, StringConstraints

from conduit.core.errors import ResidencyRefused
from conduit.core.settings import ProviderSettings, SettingsModel

if TYPE_CHECKING:
    from conduit.core.catalogue import ModelEntry

#: A Region or geography code: an AWS Region, or a provider-prefixed geography.
REGION_CODE_PATTERN = r"^[a-z0-9][a-z0-9:-]{0,63}$"
#: The most Regions one connection may choose.
MAX_CHOSEN_REGIONS = 64

RegionCode = Annotated[str, StringConstraints(pattern=REGION_CODE_PATTERN)]


class ResidencyScope(StrEnum):
    LOCAL = "local"
    IN_REGION = "in_region"
    CROSS_REGION = "cross_region"
    GLOBAL = "global"


_REGION_BOUND = frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION})


@dataclass(frozen=True, slots=True)
class Residency:
    """Where requests to a model may be processed: a scope and, for ``in_region`` and
    ``cross_region``, the Regions."""

    scope: ResidencyScope
    regions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ResidencyScope):
            raise ValueError("a residency scope is a ResidencyScope")
        if not isinstance(self.regions, frozenset):
            raise ValueError("a residency's regions are a frozenset")
        if self.regions and self.scope not in _REGION_BOUND:
            raise ValueError("only an in_region or cross_region residency names Regions")


def effective_residency(entry: ModelEntry, provider: ProviderSettings | None = None) -> Residency:
    """Where ``entry`` is processed on ``provider``: the connection's declaration for the
    model if it makes one, else the entry's own scope (unstated is ``global``)."""
    declared = provider.declared_residency(entry.model_id) if provider is not None else None
    if declared is not None:
        return declared
    scope = entry.residency_scope or ResidencyScope.GLOBAL  # unstated is unbounded
    return Residency(scope, entry.regions if scope in _REGION_BOUND else frozenset())


class RegionChoice(SettingsModel):
    """A connection's choice of where its requests may be processed.

    ``regions`` are the Regions (or provider-prefixed geographies) chosen, which must
    include the Region the connection is called in; ``allow_global`` also allows models
    the provider may process anywhere. A ``local`` model always passes.
    """

    regions: frozenset[RegionCode] = Field(min_length=1, max_length=MAX_CHOSEN_REGIONS)
    allow_global: bool = False

    def check(self, residency: Residency, provider_region: str | None, *, where: str) -> None:
        """Raise ``ResidencyRefused`` if ``residency``, called from ``provider_region``,
        is outside this choice. ``where`` begins the message."""
        scope = residency.scope
        if scope is ResidencyScope.LOCAL:
            return
        if scope is ResidencyScope.GLOBAL:
            if not self.allow_global:
                raise ResidencyRefused(
                    f"{where}: global processing is outside the connection's Region choice"
                )
            return
        if provider_region is None or provider_region not in self.regions:
            raise ResidencyRefused(
                f"{where}: the connection is called in {provider_region or 'no Region'}, "
                "outside the connection's Region choice"
            )
        if scope is ResidencyScope.CROSS_REGION and not residency.regions:
            # Unbounded in practice: never let it through a choice.
            raise ResidencyRefused(
                f"{where}: cross-Region processing that names no Region is outside the "
                "connection's Region choice"
            )
        outside = sorted(residency.regions - self.regions)
        if outside:
            raise ResidencyRefused(
                f"{where}: may be processed in {', '.join(outside)}, outside the "
                "connection's Region choice"
            )


def _named(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


@dataclass(frozen=True, slots=True)
class OfferedRegion:
    """One Region or geography a provider offers a connection.

    ``invocation`` says a connection may be called there; ``scope`` says what choosing
    it means for processing; ``members`` are the Regions a geography covers, where the
    provider states them; ``stores_only`` marks a choice that keeps stored data there
    but not processing (processing may happen anywhere).
    """

    code: str
    name: str
    _: KW_ONLY
    invocation: bool
    scope: ResidencyScope
    members: frozenset[str] = field(default_factory=frozenset)
    stores_only: bool = False

    def __post_init__(self) -> None:
        if not _named(self.code) or not _named(self.name):
            raise ValueError("an offered Region has a code and a name")
        if not isinstance(self.scope, ResidencyScope):
            raise ValueError("an offered Region's scope is a ResidencyScope")
        if not isinstance(self.members, frozenset) or not all(map(_named, self.members)):
            raise ValueError("an offered Region's members are Region codes")


@dataclass(frozen=True, slots=True)
class RegionListing:
    """The Regions one adapter (or one service of it) offers, from a documented list.

    ``global_option`` says what "global" means for this provider, or ``None`` when it
    offers no global processing; ``documentation`` is the list's source, read on
    ``as_of``.
    """

    kind: str
    service: str | None
    regions: tuple[OfferedRegion, ...]
    global_option: str | None
    documentation: str
    as_of: str
    source: Literal["static"] = "static"


class ResidencyPolicy(SettingsModel):
    """The scopes a deployment allows, and the regions requests may be processed in.

    ``regions`` of ``None`` means no region restriction. With regions set, a global
    entry is always refused, since its processing regions are not bounded.
    """

    scopes: frozenset[ResidencyScope] = Field(default_factory=lambda: frozenset(ResidencyScope))
    regions: frozenset[str] | None = None

    def check(
        self,
        entry: ModelEntry,
        provider_region: str | None,
        *,
        task: str,
        provider: ProviderSettings | None = None,
    ) -> None:
        """Raise ``ResidencyRefused`` if calling ``entry`` from ``provider_region`` breaches
        this policy, or the Region choice of the connection ``provider`` (whose declared
        residency for the model, if any, replaces the entry's)."""
        residency = effective_residency(entry, provider)
        scope = residency.scope
        where = f"{task}: model {entry.model_id!r} ({scope.value})"
        if scope not in self.scopes:
            allowed = ", ".join(sorted(s.value for s in self.scopes)) or "none"
            raise ResidencyRefused(f"{where}: scope {scope.value} is not allowed ({allowed})")
        self._check_regions(residency, provider_region, where)
        choice = provider.region_choice() if provider is not None else None
        if choice is not None:
            choice.check(residency, provider_region, where=where)

    def _check_regions(self, residency: Residency, provider_region: str | None, where: str) -> None:
        scope = residency.scope
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
            outside = sorted(residency.regions - self.regions)
            if outside:
                raise ResidencyRefused(
                    f"{where}: may be processed in {', '.join(outside)}, outside the allowed "
                    "regions"
                )
