"""Model discovery: which models a configured provider can reach, set against the catalogue.

A product asks the provider registry (``ProviderRegistry.discover``) what a configured
provider lists. The registry validates the settings, resolves the secrets and builds
the client through the adapter's ``create_client``, exactly as ``build`` does, asks the
adapter's discovery for the models it lists, and closes the client again. Each listed
model is then joined to the catalogue and to the registry's residency policy and the
provider's Region:

- ``allowed``: declared, offered in the provider's Region and within the policy;
- ``refused``: declared, but not offered in the Region or outside the policy, with the
  reason start-up would give for an assignment to it;
- ``undeclared``: the catalogue does not declare it for the provider.

Discovery never makes a model usable (C18). The catalogue is unchanged by it, and a
built provider still refuses an undeclared model before any client call. A product that
wants an undeclared model declares it in a catalogue overlay of its own, reviewed by a
person, and starts conduit again.

An adapter's discovery returns ``DiscoveredModel`` values: the model id as the provider
names it, and the provider's own label for the entry where the listing gives one (a
Bedrock inference profile's type, for example). Nothing else from the provider's answer
is kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import ResidencyRefused
from conduit.core.residency import ResidencyPolicy
from conduit.core.tasks import _offered_in  # start-up's own Region rule, not a copy

#: The longest model id or label kept from a listing.
MAX_ID_LENGTH = 256


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """One model a provider lists: its id, and the provider's own label for it, if any."""

    model_id: str
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not 0 < len(self.model_id) <= MAX_ID_LENGTH:
            raise ValueError("a discovered model id must be a non-empty string")
        if self.label is not None and (
            not isinstance(self.label, str) or not 0 < len(self.label) <= MAX_ID_LENGTH
        ):
            raise ValueError("a discovered model's label must be a non-empty string or None")


class DiscoveryOutcome(StrEnum):
    ALLOWED = "allowed"
    REFUSED = "refused"
    UNDECLARED = "undeclared"


@dataclass(frozen=True, slots=True)
class ModelAvailability:
    """A listed model, joined to the catalogue and the residency policy.

    ``entry`` is the catalogue entry for a declared model (``None`` when undeclared);
    ``reason`` is why a declared model is refused (``None`` otherwise).
    """

    model_id: str
    label: str | None
    outcome: DiscoveryOutcome
    entry: ModelEntry | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """What one configured provider lists, joined to the catalogue.

    ``supported`` is ``False`` when the adapter has no listing for these settings, with
    ``reason`` saying why; ``models`` is then empty. Unsupported is reported, never raised.
    """

    provider: str
    kind: str
    catalogue_provider: str
    region: str | None
    supported: bool
    reason: str | None = None
    models: tuple[ModelAvailability, ...] = field(default=())

    def _with(self, outcome: DiscoveryOutcome) -> tuple[ModelAvailability, ...]:
        return tuple(model for model in self.models if model.outcome is outcome)

    @property
    def allowed(self) -> tuple[ModelAvailability, ...]:
        return self._with(DiscoveryOutcome.ALLOWED)

    @property
    def refused(self) -> tuple[ModelAvailability, ...]:
        return self._with(DiscoveryOutcome.REFUSED)

    @property
    def undeclared(self) -> tuple[ModelAvailability, ...]:
        return self._with(DiscoveryOutcome.UNDECLARED)


def _refusal(
    entry: ModelEntry, region: str | None, residency: ResidencyPolicy, provider: str
) -> str | None:
    """Why start-up would refuse an assignment to ``entry``, or ``None``."""
    if not _offered_in(entry, region):
        return f"{provider}: model {entry.model_id!r} is not offered in region {region}"
    reason: str | None = None
    try:
        residency.check(entry, region, task=provider)
    except ResidencyRefused as refused:
        reason = str(refused)  # only the message is kept, never the exception
    return reason


def join_discovered(
    discovered: Iterable[DiscoveredModel],
    catalogue: Catalogue,
    *,
    catalogue_provider: str,
    region: str | None,
    residency: ResidencyPolicy,
    provider: str,
) -> tuple[ModelAvailability, ...]:
    """Each listed model, in listing order, joined to ``catalogue`` and ``residency``.

    A repeated id is reported once, as first listed. ``provider`` (the configured name)
    begins each refusal reason, as a task class begins it at start-up.
    """
    joined: list[ModelAvailability] = []
    seen: set[str] = set()
    for model in discovered:
        if model.model_id in seen:
            continue
        seen.add(model.model_id)
        entry = catalogue.get(catalogue_provider, model.model_id)
        if entry is None:
            joined.append(
                ModelAvailability(model.model_id, model.label, DiscoveryOutcome.UNDECLARED)
            )
            continue
        reason = _refusal(entry, region, residency, provider)
        outcome = DiscoveryOutcome.ALLOWED if reason is None else DiscoveryOutcome.REFUSED
        joined.append(ModelAvailability(model.model_id, model.label, outcome, entry, reason))
    return tuple(joined)


__all__ = [
    "DiscoveredModel",
    "DiscoveryOutcome",
    "DiscoveryResult",
    "ModelAvailability",
    "join_discovered",
]
