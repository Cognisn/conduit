"""Model discovery: the provider's own listing decides which models exist and where (C18).

A provider's listing is the source of truth for which models a configured provider can
use and where each request is processed. The catalogue keeps only what listings do not
report (capabilities and limits), and the difference between the two is reported as
drift, never used to hide a model.

A product asks the provider registry (``ProviderRegistry.discover``) what a configured
provider lists, and ``start`` does the same for every provider a task class is assigned
to, once, before anything is built. The registry validates the settings, resolves the
secrets and builds the client through the adapter's ``create_client``, exactly as
``build`` does, asks the adapter's discovery for the models it lists, and closes the
client again. Each listed model is joined to the catalogue and the residency policy:

- its residency scope and Regions are the listing's (a Bedrock profile's destination
  Regions, the Region an in-Region model is called in; ``global`` for an API with no
  Region notion, ``local`` for a model on the deployment's own host). An adapter that
  states no scope gets ``global``, the least bounded;
- its catalogue entry is matched by the listed id, else by the base model id the
  listing gives (the foundation model behind a Bedrock inference profile);
- its capabilities and limits are, field by field: a product overlay's entry outright;
  else limits the listing reports, then the default catalogue entry; a capability flag
  is set only if some source asserts it and none denies it (the default entry also says
  what conduit's adapter route supports); else the conservative floor (no capabilities,
  ``FLOOR_CONTEXT_WINDOW`` and ``FLOOR_OUTPUT_LIMIT``). An uncatalogued embedding model
  has no known dimensions and is refused until an overlay declares it;
- it is ``allowed``, or ``refused`` with the reason start-up gives: outside the
  residency policy, withdrawn by an overlay, or an embedding model with no entry.

Drift is reported per model (``CatalogueDrift``): a listed model with no entry
(``uncatalogued``), an entry the provider does not list (``unlisted``), limits or
capabilities the listing reports differently (``limits_differ``,
``capabilities_differ``), and an overlay's residency for a listed model
(``regions_declared``, the listing's is used). A detail is a fixed phrase with the two
values, never provider text.

An adapter's discovery returns ``DiscoveredModel`` values: the model id as the provider
names it, the provider's own label for the entry where the listing gives one, and what
the listing says about residency, the base model, the kind, limits and capabilities.
Nothing else from the provider's answer is kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from typing import Literal

from conduit.core.catalogue import Capabilities, Catalogue, ModelEntry, ModelKey
from conduit.core.errors import ResidencyRefused
from conduit.core.residency import RegionChoice, ResidencyPolicy, ResidencyScope
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import _offered_in  # start-up's own Region rule, not a copy

#: The longest model id or label kept from a listing.
MAX_ID_LENGTH = 256
#: The context window of a listed chat model nothing else describes.
FLOOR_CONTEXT_WINDOW = 8_192
#: The output limit of a listed chat model nothing else describes.
FLOOR_OUTPUT_LIMIT = 4_096
#: The most Regions kept for one listed model.
MAX_REGIONS = 64
#: A listed id naming embeddings, for a listing that states no kind: such a model is
#: taken as an embedding model (refused until an overlay declares it), never as chat.
EMBEDDING_ID_HINT = "embed"

_REGION_BOUND = frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION})
_FLAGS: tuple[tuple[str, str], ...] = (
    ("tools", "tools"),
    ("vision", "vision"),
    ("prompt_caching", "prompt caching"),
    ("structured_output", "structured output"),
)


def _text(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_ID_LENGTH


def _limit(value: object) -> bool:
    return value is None or (type(value) is int and value > 0)


@dataclass(frozen=True, slots=True)
class ReportedCapabilities:
    """The capability flags a listing reports: ``None`` where it says nothing."""

    tools: bool | None = None
    vision: bool | None = None
    prompt_caching: bool | None = None
    structured_output: bool | None = None

    def __post_init__(self) -> None:
        for name, _ in _FLAGS:
            if type(getattr(self, name)) not in (type(None), bool):
                raise ValueError("a reported capability is true, false or None")


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """One model a provider lists, as its adapter reads the listing.

    ``scope`` and ``regions`` are where the provider says a request is processed;
    ``None`` means the adapter cannot say, and the model is treated as ``global``.
    ``base_model_id`` is the model behind the listed id (a Bedrock inference profile's
    foundation model), used to find its catalogue entry. ``kind``, ``context_window``,
    ``output_limit`` and ``capabilities`` are reported only where the listing gives them.
    """

    model_id: str
    label: str | None = None
    _: KW_ONLY
    kind: Literal["chat", "embedding"] | None = None
    scope: ResidencyScope | None = None
    regions: frozenset[str] = field(default_factory=frozenset)
    base_model_id: str | None = None
    context_window: int | None = None
    output_limit: int | None = None
    capabilities: ReportedCapabilities | None = None

    def __post_init__(self) -> None:
        if not _text(self.model_id):
            raise ValueError("a discovered model id must be a non-empty string")
        if self.label is not None and not _text(self.label):
            raise ValueError("a discovered model's label must be a non-empty string or None")
        if self.base_model_id is not None and not _text(self.base_model_id):
            raise ValueError("a base model id must be a non-empty string or None")
        if self.kind not in (None, "chat", "embedding"):
            raise ValueError("a discovered model's kind is chat, embedding or None")
        if self.scope is not None and not isinstance(self.scope, ResidencyScope):
            raise ValueError("a discovered model's scope is a ResidencyScope or None")
        if not isinstance(self.regions, frozenset) or len(self.regions) > MAX_REGIONS:
            raise ValueError("a discovered model's regions are a frozenset of Region codes")
        if not all(_text(region) for region in self.regions):
            raise ValueError("a Region is a non-empty string")
        if (self.scope in _REGION_BOUND) != bool(self.regions):
            raise ValueError("only an in_region or cross_region model lists Regions")
        if self.scope is ResidencyScope.IN_REGION and len(self.regions) != 1:
            raise ValueError("an in_region model is processed in one Region")
        if not _limit(self.context_window) or not _limit(self.output_limit):
            raise ValueError("a reported limit is a positive integer or None")
        if self.capabilities is not None and not isinstance(
            self.capabilities, ReportedCapabilities
        ):
            raise ValueError("reported capabilities are ReportedCapabilities or None")


class DiscoveryOutcome(StrEnum):
    ALLOWED = "allowed"
    REFUSED = "refused"


class DriftReason(StrEnum):
    UNCATALOGUED = "uncatalogued"
    UNLISTED = "unlisted"
    LIMITS_DIFFER = "limits_differ"
    CAPABILITIES_DIFFER = "capabilities_differ"
    REGIONS_DECLARED = "regions_declared"


@dataclass(frozen=True, slots=True)
class CatalogueDrift:
    """One difference between the catalogue and the provider's listing."""

    model_id: str
    reason: DriftReason
    detail: str


@dataclass(frozen=True, slots=True)
class ModelAvailability:
    """A listed model, joined to the catalogue and the residency policy.

    ``entry`` is the model's effective entry (its listed scope and Regions, with the
    capabilities and limits the join settled on), ``None`` only for a model that cannot
    be served (withdrawn, or an embedding model with no entry). ``reason`` is why a model
    is refused. ``catalogued`` says whether a catalogue entry matched it.
    """

    model_id: str
    label: str | None
    outcome: DiscoveryOutcome
    entry: ModelEntry | None = None
    reason: str | None = None
    _: KW_ONLY
    catalogued: bool = False
    scope: ResidencyScope = ResidencyScope.GLOBAL
    regions: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """What one configured provider lists, joined to the catalogue, with the drift.

    ``supported`` is ``False`` when the adapter has no listing for these settings, with
    ``reason`` saying why; ``models`` is then empty. Unsupported is reported, never raised.
    ``choice`` is the connection's Region choice the listing was held to, or ``None``.
    """

    provider: str
    kind: str
    catalogue_provider: str
    region: str | None
    supported: bool
    reason: str | None = None
    models: tuple[ModelAvailability, ...] = field(default=())
    drift: tuple[CatalogueDrift, ...] = field(default=())
    choice: RegionChoice | None = None

    def _with(self, outcome: DiscoveryOutcome) -> tuple[ModelAvailability, ...]:
        return tuple(model for model in self.models if model.outcome is outcome)

    @property
    def allowed(self) -> tuple[ModelAvailability, ...]:
        return self._with(DiscoveryOutcome.ALLOWED)

    @property
    def refused(self) -> tuple[ModelAvailability, ...]:
        return self._with(DiscoveryOutcome.REFUSED)

    @property
    def uncatalogued(self) -> tuple[ModelAvailability, ...]:
        """The listed models no catalogue entry matched."""
        return tuple(model for model in self.models if not model.catalogued)

    def get(self, model_id: str) -> ModelAvailability | None:
        """The listed model ``model_id``, or ``None`` if the provider does not list it."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def listed_catalogue(self, base: Catalogue) -> Catalogue:
        """A catalogue of every listed model's effective entry: what a provider built on
        this listing looks its models up in. ``base`` gives the version and overlays."""
        return Catalogue(
            [model.entry for model in self.models if model.entry is not None],
            version=base.version,
            overlays=base.overlays,
        )


def _refusal(
    entry: ModelEntry,
    region: str | None,
    residency: ResidencyPolicy,
    provider: str,
    connection: ProviderSettings | None,
) -> str | None:
    """Why start-up would refuse an assignment to ``entry``, or ``None``."""
    if not _offered_in(entry, region):
        return f"{provider}: model {entry.model_id!r} is not offered in region {region}"
    reason: str | None = None
    try:
        residency.check(entry, region, task=provider, provider=connection)
    except ResidencyRefused as refused:
        reason = str(refused)  # only the message is kept, never the exception
    return reason


def _words(value: bool) -> str:
    return "supported" if value else "not supported"


def _where(scope: ResidencyScope | None, regions: frozenset[str]) -> str:
    named = ", ".join(sorted(regions)) or "no Regions"
    return f"{scope.value if scope else 'unscoped'} in {named}"


@dataclass(slots=True)
class _Join:
    """The effective entry of one listed model, and the drift found on the way."""

    listed: DiscoveredModel
    scope: ResidencyScope
    regions: frozenset[str]
    drift: list[CatalogueDrift]

    def note(self, reason: DriftReason, detail: str) -> None:
        self.drift.append(CatalogueDrift(self.listed.model_id, reason, detail))

    def limit(self, words: str, reported: int | None, declared: int | None) -> None:
        if reported is not None and declared is not None and reported != declared:
            self.note(
                DriftReason.LIMITS_DIFFER,
                f"{words} {reported} listed, {declared} in the catalogue",
            )

    def capabilities(self, declared: Capabilities, *, overlay: bool) -> Capabilities:
        reported = self.listed.capabilities or ReportedCapabilities()
        flags: dict[str, bool] = {}
        for name, words in _FLAGS:
            have: bool = getattr(declared, name)
            said: bool | None = getattr(reported, name)
            if said is not None and said != have:
                self.note(
                    DriftReason.CAPABILITIES_DIFFER,
                    f"{words} {_words(said)} listed, {_words(have)} in the catalogue",
                )
            # Asserted by some source and denied by none; an overlay's word is final.
            flags[name] = have if overlay else have and said is not False
        return Capabilities(**flags)

    def entry(self, catalogue_provider: str, matched: ModelEntry, *, overlay: bool) -> ModelEntry:
        listed = self.listed
        self.limit("context window", listed.context_window, matched.context_window)
        self.limit("output limit", listed.output_limit, matched.output_limit)
        if overlay and matched.residency_scope is not None:
            declared = _where(matched.residency_scope, matched.regions)
            found = _where(self.scope, self.regions)
            if declared != found:
                self.note(DriftReason.REGIONS_DECLARED, f"listed as {found}, declared {declared}")
        capabilities = self.capabilities(matched.capabilities, overlay=overlay)
        context = matched.context_window
        output = matched.output_limit
        if not overlay:
            context = listed.context_window or context
            if matched.kind == "chat":
                output = listed.output_limit or output
        return ModelEntry(
            model_id=listed.model_id,
            provider=catalogue_provider,
            kind=matched.kind,
            residency_scope=self.scope,
            regions=self.regions,
            capabilities=capabilities,
            context_window=context,
            output_limit=output,
            embedding_dimensions=matched.embedding_dimensions,
            labels=matched.labels,
        )

    def floor(self, catalogue_provider: str) -> ModelEntry:
        listed = self.listed
        reported = listed.capabilities or ReportedCapabilities()
        return ModelEntry(
            model_id=listed.model_id,
            provider=catalogue_provider,
            kind="chat",
            residency_scope=self.scope,
            regions=self.regions,
            capabilities=Capabilities(
                **{name: getattr(reported, name) is True for name, _ in _FLAGS}
            ),
            context_window=listed.context_window or FLOOR_CONTEXT_WINDOW,
            output_limit=listed.output_limit or FLOOR_OUTPUT_LIMIT,
        )


def _kind(listed: DiscoveredModel) -> Literal["chat", "embedding"]:
    """The listed model's kind: as the listing states it, else ``embedding`` for an id
    naming embeddings (the conservative reading), else ``chat``."""
    if listed.kind is not None:
        return listed.kind
    return "embedding" if EMBEDDING_ID_HINT in listed.model_id.lower() else "chat"


def _keys(catalogue_provider: str, listed: DiscoveredModel) -> list[ModelKey]:
    """The catalogue keys a listed model may match, most specific first."""
    ids = [listed.model_id]
    if listed.base_model_id is not None and listed.base_model_id != listed.model_id:
        ids.append(listed.base_model_id)
    return [ModelKey(provider=catalogue_provider, model_id=model_id) for model_id in ids]


def join_discovered(
    discovered: Iterable[DiscoveredModel],
    catalogue: Catalogue,
    *,
    catalogue_provider: str,
    region: str | None,
    residency: ResidencyPolicy,
    provider: str,
    connection: ProviderSettings | None = None,
) -> tuple[tuple[ModelAvailability, ...], tuple[CatalogueDrift, ...]]:
    """Each listed model, in listing order, joined to ``catalogue`` and ``residency``,
    and the drift between the two: each listed model's first, then the unlisted entries.

    A repeated id is reported once, as first listed. ``provider`` (the configured name)
    begins each refusal reason, as a task class begins it at start-up. ``connection``
    is the provider's validated settings: a residency it declares for a listed model
    replaces the listing's, and each model is held to its Region choice too.
    """
    joined: list[ModelAvailability] = []
    drift: list[CatalogueDrift] = []
    matched: set[ModelKey] = set()
    seen: set[str] = set()
    for model in discovered:
        if model.model_id in seen:
            continue
        seen.add(model.model_id)
        scope = model.scope or ResidencyScope.GLOBAL
        regions = model.regions if scope in _REGION_BOUND else frozenset()
        declaration = (
            connection.declared_residency(model.model_id) if connection is not None else None
        )
        if declaration is not None:  # the endpoint's guarantee replaces the listing's
            scope, regions = declaration.scope, declaration.regions
        join = _Join(model, scope, regions, drift)
        keys = _keys(catalogue_provider, model)
        matched.update(keys)  # an entry under the listed or the base id is listed
        found: ModelEntry | None = None
        for key in keys:
            found = catalogue.get(key.provider, key.model_id)
            if found is not None:
                break
        reason: str | None = None
        entry: ModelEntry | None = None
        if found is None:
            join.note(DriftReason.UNCATALOGUED, "listed by the provider, with no catalogue entry")
            if _kind(model) == "embedding":
                reason = (
                    f"{provider}: model {model.model_id!r} is an embedding model with no "
                    "catalogue entry, so its dimensions are not known"
                )
            else:
                entry = join.floor(catalogue_provider)
        else:
            entry = join.entry(catalogue_provider, found, overlay=found.key in catalogue.overlaid)
        if any(key in catalogue.withdrawn for key in keys):
            entry = None
            reason = f"{provider}: model {model.model_id!r} is withdrawn by a catalogue overlay"
        if entry is not None:
            reason = _refusal(entry, region, residency, provider, connection)
        outcome = DiscoveryOutcome.ALLOWED if reason is None else DiscoveryOutcome.REFUSED
        joined.append(
            ModelAvailability(
                model.model_id,
                model.label,
                outcome,
                entry,
                reason,
                catalogued=found is not None,
                scope=scope,
                regions=regions,
            )
        )
    for declared in catalogue:
        if declared.provider == catalogue_provider and declared.key not in matched:
            drift.append(
                CatalogueDrift(
                    declared.model_id,
                    DriftReason.UNLISTED,
                    "declared in the catalogue, not listed by the provider",
                )
            )
    # A withdrawal nothing listed matches may be a mistyped id: report it, never drop it.
    for withdrawn in sorted(catalogue.withdrawn, key=str):
        if withdrawn.provider == catalogue_provider and withdrawn not in matched:
            drift.append(
                CatalogueDrift(
                    withdrawn.model_id,
                    DriftReason.UNLISTED,
                    "withdrawn by an overlay, not listed by the provider",
                )
            )
    return tuple(joined), tuple(drift)


__all__ = [
    "FLOOR_CONTEXT_WINDOW",
    "FLOOR_OUTPUT_LIMIT",
    "CatalogueDrift",
    "DiscoveredModel",
    "DiscoveryOutcome",
    "DiscoveryResult",
    "DriftReason",
    "ModelAvailability",
    "ReportedCapabilities",
    "join_discovered",
]
