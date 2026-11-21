"""The model catalogue: what providers do not report about a model, and product overlays.

The provider's own listing decides which models a configured provider can use and
where each request is processed (C18, ``conduit.core.discovery``). The catalogue holds
what listings do not give: a model's kind, capability flags, context window, output
limit, embedding dimensions and the product's labels. conduit ships a versioned default
catalogue as package data (``data/catalogue.json``), keyed by the provider's base model
id, with no Region data; a product overlays it to add, override or withdraw entries.
Catalogues are immutable; ``apply`` returns a new one, and remembers which entries an
overlay added or overrode (``overlaid``) and which models it withdrew (``withdrawn``),
since an overlay is the deployment's own reviewed statement and wins over a listing.

An entry may declare a residency scope and Regions. A listing replaces them for every
model it lists; they are used only for a provider conduit cannot list, and an entry
that declares no scope is then treated as ``global``.

Entries are keyed by ``(provider, model_id)``: the same model identifier under two
providers is two entries. ``provider`` is the catalogue provider name
(``ProviderSettings.catalogue_provider``), which is the provider kind unless the
adapter says otherwise (the OpenAI-compatible adapter serves ``openai`` and ``xai``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, PositiveInt, ValidationError, model_validator

from conduit.core.errors import ConfigurationError, UndeclaredModel
from conduit.core.residency import ResidencyScope
from conduit.core.settings import SettingsModel, failed_fields

_REGION_BOUND = frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION})


class Capabilities(SettingsModel):
    tools: bool = False
    vision: bool = False
    prompt_caching: bool = False
    structured_output: bool = False


class ModelKey(SettingsModel):
    provider: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model_id}"


class ModelEntry(SettingsModel):
    """One declared model.

    ``labels`` are product-defined tags (``tier:advanced``, for example) that a task
    class's capability floor can require; conduit gives them no meaning.
    """

    model_id: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=64)
    kind: Literal["chat", "embedding"] = "chat"
    residency_scope: ResidencyScope | None = None
    regions: frozenset[str] = frozenset()
    capabilities: Capabilities = Capabilities()
    context_window: PositiveInt
    output_limit: PositiveInt | None = None
    embedding_dimensions: PositiveInt | None = None
    labels: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.kind == "chat":
            if self.output_limit is None:
                raise ValueError("a chat model needs an output_limit")
            if self.embedding_dimensions is not None:
                raise ValueError("a chat model has no embedding_dimensions")
        else:
            if self.embedding_dimensions is None:
                raise ValueError("an embedding model needs embedding_dimensions")
            if self.output_limit is not None:
                raise ValueError("an embedding model has no output_limit")
        if self.residency_scope in _REGION_BOUND and not self.regions:
            raise ValueError(f"a {self.residency_scope.value} model needs regions")
        if self.residency_scope not in _REGION_BOUND and self.regions:
            scope = self.residency_scope.value if self.residency_scope else "unscoped"
            raise ValueError(f"a {scope} model lists no regions")
        return self

    @property
    def key(self) -> ModelKey:
        return ModelKey(provider=self.provider, model_id=self.model_id)


class CatalogueOverlay(SettingsModel):
    """A product's changes to a catalogue: withdrawn, then overridden, then added."""

    name: str = Field(default="overlay", min_length=1, max_length=128)
    add: tuple[ModelEntry, ...] = ()
    override: tuple[ModelEntry, ...] = ()
    withdraw: tuple[ModelKey, ...] = ()


class Catalogue:
    """An immutable set of model entries, with the overlays applied to it."""

    def __init__(
        self,
        entries: Iterable[ModelEntry],
        *,
        version: str,
        overlays: tuple[str, ...] = (),
        overlaid: Iterable[ModelKey] = (),
        withdrawn: Iterable[ModelKey] = (),
    ) -> None:
        found: dict[ModelKey, ModelEntry] = {}
        for entry in entries:
            if entry.key in found:
                raise ConfigurationError(f"catalogue declares {entry.key} twice")
            found[entry.key] = entry
        self._entries = dict(sorted(found.items(), key=lambda item: str(item[0])))
        self.version = version
        self.overlays = overlays
        #: The entries an overlay added or overrode: the deployment's own statements.
        self.overlaid = frozenset(key for key in overlaid if key in self._entries)
        #: The models an overlay withdrew: refused even when a provider lists them.
        self.withdrawn = frozenset(withdrawn) - self._entries.keys()

    def get(self, provider: str, model_id: str) -> ModelEntry | None:
        return self._entries.get(ModelKey(provider=provider, model_id=model_id))

    def require(self, provider: str, model_id: str) -> ModelEntry:
        """The declared entry. Raises ``UndeclaredModel`` for anything else."""
        found = self.get(provider, model_id)
        if found is None:
            raise UndeclaredModel(provider, model_id)
        return found

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[ModelEntry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Catalogue(version={self.version!r}, entries={len(self)}, overlays={self.overlays})"

    def apply(self, *overlays: CatalogueOverlay) -> Catalogue:
        entries = dict(self._entries)
        names = list(self.overlays)
        overlaid = set(self.overlaid)
        withdrawn = set(self.withdrawn)
        for overlay in overlays:
            keys = [e.key for e in overlay.add + overlay.override] + list(overlay.withdraw)
            if len(set(keys)) != len(keys):
                raise ConfigurationError(f"overlay {overlay.name!r} names a model twice")
            for key in overlay.withdraw:
                # A model the catalogue lacks may be withdrawn: a provider may list it.
                entries.pop(key, None)
                overlaid.discard(key)
                withdrawn.add(key)
            for entry in overlay.override:
                if entry.key not in entries:
                    raise ConfigurationError(
                        f"overlay {overlay.name!r} overrides absent {entry.key}"
                    )
                entries[entry.key] = entry
                overlaid.add(entry.key)
            for entry in overlay.add:
                if entry.key in entries:
                    raise ConfigurationError(
                        f"overlay {overlay.name!r} adds {entry.key}, which is already declared"
                    )
                entries[entry.key] = entry
                overlaid.add(entry.key)
                withdrawn.discard(entry.key)
            names.append(overlay.name)
        return Catalogue(
            entries.values(),
            version=self.version,
            overlays=tuple(names),
            overlaid=overlaid,
            withdrawn=withdrawn,
        )


class _CatalogueFile(SettingsModel):
    version: str = Field(min_length=1)
    models: tuple[ModelEntry, ...]


def load_catalogue(source: Mapping[str, Any] | str | Path) -> Catalogue:
    """A catalogue from a mapping or a JSON file of ``{"version", "models"}``."""
    problem = ""
    try:
        if isinstance(source, Mapping):
            data = _CatalogueFile.model_validate(source)
        else:
            text = Path(source).read_text(encoding="utf-8")
            data = _CatalogueFile.model_validate(json.loads(text))
    except ValidationError as error:
        problem = f"invalid at {', '.join(failed_fields(error))}"
    except json.JSONDecodeError as error:
        problem = f"not JSON (line {error.lineno}, column {error.colno})"
    if problem:
        # Raised after the handler: the error keeps neither pydantic's text nor the input.
        raise ConfigurationError(f"catalogue cannot be loaded: {problem}")
    return Catalogue(data.models, version=data.version)


@cache
def default_catalogue() -> Catalogue:
    """The catalogue shipped with conduit, reviewed at each conduit point release."""
    text = files("conduit.core").joinpath("data", "catalogue.json").read_text(encoding="utf-8")
    return load_catalogue(json.loads(text))
