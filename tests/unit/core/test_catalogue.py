"""The model catalogue: schema, loader, overlays, default data and undeclared models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conduit.core.catalogue import (
    Capabilities,
    Catalogue,
    CatalogueOverlay,
    ModelEntry,
    ModelKey,
    default_catalogue,
    load_catalogue,
)
from conduit.core.errors import ConfigurationError, UndeclaredModel
from conduit.core.residency import ResidencyScope


def entry(**overrides: Any) -> ModelEntry:
    values: dict[str, Any] = {
        "model_id": "example-large",
        "provider": "example",
        "residency_scope": "global",
        "capabilities": {"tools": True},
        "context_window": 200_000,
        "output_limit": 32_000,
    }
    values.update(overrides)
    return ModelEntry.model_validate(values)


class TestSchema:
    def test_a_chat_entry(self) -> None:
        model = entry()
        assert model.key == ModelKey(provider="example", model_id="example-large")
        assert model.capabilities == Capabilities(tools=True)
        assert model.kind == "chat"
        assert model.embedding_dimensions is None

    def test_an_embedding_entry_needs_dimensions_and_no_output_limit(self) -> None:
        model = entry(kind="embedding", output_limit=None, embedding_dimensions=1024)
        assert model.embedding_dimensions == 1024
        with pytest.raises(ValidationError):
            entry(kind="embedding", output_limit=None)
        with pytest.raises(ValidationError):
            entry(kind="embedding", embedding_dimensions=1024)

    def test_a_chat_entry_needs_an_output_limit(self) -> None:
        with pytest.raises(ValidationError):
            entry(output_limit=None)

    def test_region_bound_scopes_need_regions_and_others_have_none(self) -> None:
        entry(residency_scope="in_region", regions=["ap-southeast-2"])
        with pytest.raises(ValidationError):
            entry(residency_scope="in_region")
        with pytest.raises(ValidationError):
            entry(residency_scope="cross_region", regions=[])
        with pytest.raises(ValidationError):
            entry(residency_scope="global", regions=["us-east-1"])
        with pytest.raises(ValidationError):
            entry(residency_scope="local", regions=["us-east-1"])

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            entry(tier="advanced")


class TestCatalogue:
    def test_lookup_and_refusal_of_an_undeclared_model(self) -> None:
        catalogue = Catalogue([entry()], version="test-1")
        assert catalogue.require("example", "example-large").context_window == 200_000
        assert catalogue.get("example", "example-small") is None
        with pytest.raises(UndeclaredModel) as raised:
            catalogue.require("example", "example-small")
        assert raised.value.model == "example-small"
        # The same model id under another provider is a different entry.
        with pytest.raises(UndeclaredModel):
            catalogue.require("other", "example-large")

    def test_duplicate_entries_are_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            Catalogue([entry(), entry()], version="test-1")

    def test_iteration_is_sorted_by_key(self) -> None:
        catalogue = Catalogue([entry(model_id="b"), entry(model_id="a")], version="v")
        assert [e.model_id for e in catalogue] == ["a", "b"]
        assert len(catalogue) == 2
        assert ModelKey(provider="example", model_id="a") in catalogue


class TestOverlays:
    def base(self) -> Catalogue:
        return Catalogue([entry(model_id="a"), entry(model_id="b")], version="base-1")

    def test_add_override_and_withdraw(self) -> None:
        overlay = CatalogueOverlay(
            name="product",
            add=(entry(model_id="c"),),
            override=(entry(model_id="a", context_window=100_000, labels={"tier:advanced"}),),
            withdraw=(ModelKey(provider="example", model_id="b"),),
        )
        result = self.base().apply(overlay)
        assert [e.model_id for e in result] == ["a", "c"]
        assert result.require("example", "a").context_window == 100_000
        assert result.require("example", "a").labels == frozenset({"tier:advanced"})
        assert result.overlays == ("product",)
        assert result.version == "base-1"

    def test_the_original_catalogue_is_unchanged(self) -> None:
        base = self.base()
        base.apply(CatalogueOverlay(withdraw=(ModelKey(provider="example", model_id="a"),)))
        assert base.get("example", "a") is not None

    @pytest.mark.parametrize(
        "overlay",
        [
            pytest.param(CatalogueOverlay(add=(entry(model_id="a"),)), id="add-existing"),
            pytest.param(CatalogueOverlay(override=(entry(model_id="z"),)), id="override-missing"),
            pytest.param(
                CatalogueOverlay(withdraw=(ModelKey(provider="example", model_id="z"),)),
                id="withdraw-missing",
            ),
            pytest.param(
                CatalogueOverlay(
                    override=(entry(model_id="a"),),
                    withdraw=(ModelKey(provider="example", model_id="a"),),
                ),
                id="same-key-twice",
            ),
        ],
    )
    def test_a_contradictory_overlay_is_refused(self, overlay: CatalogueOverlay) -> None:
        with pytest.raises(ConfigurationError):
            self.base().apply(overlay)

    def test_overlay_from_a_mapping(self) -> None:
        overlay = CatalogueOverlay.model_validate(
            {
                "name": "from-config",
                "withdraw": [{"provider": "example", "model_id": "a"}],
                "add": [entry(model_id="d").model_dump(mode="json")],
            }
        )
        assert [e.model_id for e in self.base().apply(overlay)] == ["b", "d"]


class TestLoading:
    def test_from_a_mapping_and_a_file(self, tmp_path: Path) -> None:
        data = {"version": "file-1", "models": [entry().model_dump(mode="json")]}
        from_mapping = load_catalogue(data)
        path = tmp_path / "catalogue.json"
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        from_file = load_catalogue(path)
        assert list(from_mapping) == list(from_file)
        assert from_file.version == "file-1"

    def test_a_malformed_file_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            load_catalogue({"version": "x", "models": [{"model_id": "only"}]})


class TestDefaultData:
    def test_it_loads_and_covers_the_wave_2_providers(self) -> None:
        catalogue = default_catalogue()
        providers = {e.provider for e in catalogue}
        assert {"anthropic", "bedrock", "openai", "xai", "gemini", "ollama", "stub"} <= providers
        assert catalogue.version

    def test_it_has_embedding_and_cross_region_entries(self) -> None:
        catalogue = default_catalogue()
        assert any(e.kind == "embedding" for e in catalogue)
        assert any(e.residency_scope is ResidencyScope.CROSS_REGION for e in catalogue)
        assert any(e.residency_scope is ResidencyScope.IN_REGION for e in catalogue)

    def test_australian_residency_against_the_default_bedrock_entries(self) -> None:
        from conduit.core.errors import ResidencyRefused
        from conduit.core.residency import ResidencyPolicy

        catalogue = default_catalogue()
        au_profile = catalogue.require("bedrock", "au.anthropic.claude-sonnet-5-5")
        global_profile = catalogue.require("bedrock", "global.anthropic.claude-sonnet-5-5")
        au = frozenset({"ap-southeast-2", "ap-southeast-4"})
        in_region_only = ResidencyPolicy(scopes=frozenset({ResidencyScope.IN_REGION}), regions=au)
        within_au = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION}), regions=au
        )
        with pytest.raises(ResidencyRefused):
            in_region_only.check(au_profile, "ap-southeast-2", task="t")
        within_au.check(au_profile, "ap-southeast-2", task="t")
        with pytest.raises(ResidencyRefused):
            within_au.check(global_profile, "ap-southeast-2", task="t")
        # Sonnet 4.6's Australian profile also routes to ap-southeast-6.
        wider = catalogue.require("bedrock", "au.anthropic.claude-sonnet-4-6")
        with pytest.raises(ResidencyRefused, match="ap-southeast-6"):
            within_au.check(wider, "ap-southeast-2", task="t")

    def test_the_stub_model_is_local(self) -> None:
        stub = default_catalogue().require("stub", "stub-echo")
        assert stub.residency_scope is ResidencyScope.LOCAL
