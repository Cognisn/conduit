"""The konfig loader: ConduitSettings from a konfig configuration."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from conduit.core.catalogue import CatalogueOverlay, ModelKey
from conduit.core.errors import ConfigurationError
from conduit.core.residency import ResidencyScope
from conduit.core.runtime import ConduitSettings
from conduit.core.secrets import SecretRef
from conduit.integrations.konfig import DEFAULT_SECTION, load_settings
from tests.unit.integrations.konfig.chain import assert_carries_nothing

#: A plaintext value standing in for a credential written into a configuration by
#: mistake. Synthetic; it must never appear in an error.
PLAINTEXT = "sk-synthetic-plaintext-0000"

SECTION: dict[str, Any] = {
    "providers": {
        "primary": {"kind": "fake", "api_key": "secret://fake/key", "region": "ap-southeast-2"},
    },
    "assignments": {
        "chat": {"provider": "primary", "model": "fake-large", "max_output_tokens": 500},
    },
    "residency": {"scopes": ["local", "in_region"], "regions": ["ap-southeast-2"]},
    "capture": {"prompts": True},
    "catalogue": [
        {
            "name": "product",
            "withdraw": [{"provider": "fake", "model_id": "fake-small"}],
        }
    ],
    "require_all_assignments": False,
}


def _settings(konfig: ModuleType, section: Any, **kwargs: Any) -> Any:
    return konfig.Settings(defaults={DEFAULT_SECTION: section}, **kwargs)


def test_the_default_section_is_conduit() -> None:
    assert DEFAULT_SECTION == "conduit"


def test_builds_every_part_of_the_settings(konfig_module: ModuleType) -> None:
    settings = load_settings(_settings(konfig_module, SECTION))
    assert isinstance(settings, ConduitSettings)
    assert settings.providers["primary"].kind == "fake"
    assert settings.providers["primary"].region == "ap-southeast-2"
    assert settings.assignments["chat"].model == "fake-large"
    assert settings.assignments["chat"].max_output_tokens == 500
    assert settings.residency.scopes == {ResidencyScope.LOCAL, ResidencyScope.IN_REGION}
    assert settings.residency.regions == frozenset({"ap-southeast-2"})
    assert settings.capture.prompts is True
    assert settings.capture.tool_content is False
    assert settings.catalogue == (
        CatalogueOverlay(
            name="product", withdraw=(ModelKey(provider="fake", model_id="fake-small"),)
        ),
    )
    assert settings.require_all_assignments is False


def test_credentials_stay_secret_references(konfig_module: ModuleType) -> None:
    settings = load_settings(_settings(konfig_module, SECTION))
    extra = settings.providers["primary"].model_extra or {}
    # The generic provider settings keep the reference as written; the adapter's own
    # model turns it into a SecretRef when the registry validates it.
    assert SecretRef.parse(extra["api_key"]) == SecretRef("fake/key")
    dumped = settings.model_dump(mode="json")
    assert dumped["providers"]["primary"]["api_key"] == "secret://fake/key"


def test_reads_a_named_section(konfig_module: ModuleType) -> None:
    source = konfig_module.Settings(defaults={"ai": {"conduit": SECTION}})
    settings = load_settings(source, section="ai.conduit")
    assert set(settings.providers) == {"primary"}


def test_konfig_layers_apply_before_validation(konfig_module: ModuleType) -> None:
    source = _settings(konfig_module, SECTION)
    source.set("conduit.capture.tool_content", True)
    source.set("conduit.assignments.chat.model", "fake-small")
    settings = load_settings(source)
    assert settings.capture.tool_content is True
    assert settings.assignments["chat"].model == "fake-small"


def test_environment_overrides_arrive_as_text_and_validate(
    konfig_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CDTTEST__CONDUIT__CAPTURE__PROMPTS", "false")
    monkeypatch.setenv("CDTTEST__CONDUIT__ASSIGNMENTS__CHAT__MAX_OUTPUT_TOKENS", "250")
    source = _settings(konfig_module, SECTION, env_prefix="CDTTEST")
    settings = load_settings(source)
    assert settings.capture.prompts is False
    assert settings.assignments["chat"].max_output_tokens == 250


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_reads_a_configuration_file(konfig_module: ModuleType, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"config{suffix}"
    # JSON is a subset of YAML, so one rendering serves both formats.
    path.write_text(json.dumps({"conduit": SECTION}), encoding="utf-8", newline="\n")
    settings = load_settings(konfig_module.Settings(config_file=path))
    assert settings.assignments["chat"].provider == "primary"


@pytest.mark.parametrize("section", [None, {}, "text"])
def test_a_missing_empty_or_scalar_section_is_refused(
    konfig_module: ModuleType, section: Any
) -> None:
    defaults = {} if section is None else {DEFAULT_SECTION: section}
    with pytest.raises(ConfigurationError, match="'conduit'"):
        load_settings(konfig_module.Settings(defaults=defaults))


def test_invalid_settings_are_a_configuration_error_naming_the_field(
    konfig_module: ModuleType,
) -> None:
    section = {**SECTION, "capture": {"prompts": "perhaps"}}
    with pytest.raises(ConfigurationError, match=r"capture\.prompts") as caught:
        load_settings(_settings(konfig_module, section))
    assert_carries_nothing(caught.value, "perhaps")


def test_a_plaintext_credential_is_refused_without_echoing_it(
    konfig_module: ModuleType,
) -> None:
    section = {**SECTION, "providers": {"primary": {"kind": "fake", "api_key": PLAINTEXT}}}
    with pytest.raises(ConfigurationError, match=r"providers\.primary") as caught:
        load_settings(_settings(konfig_module, section))
    assert_carries_nothing(caught.value, PLAINTEXT)


def test_accepts_any_source_with_konfigs_section_interface() -> None:
    class Source:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def get_section(self, prefix: str) -> dict[str, Any]:
            self.asked.append(prefix)
            return {"providers": {}, "require_all_assignments": False}

    source = Source()
    settings = load_settings(source)
    assert source.asked == ["conduit"]
    assert settings.providers == {}
