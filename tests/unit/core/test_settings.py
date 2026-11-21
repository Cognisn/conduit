"""Typed provider settings: buildable in code or from a mapping, credentials only by reference."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from conduit.core.errors import ConfigurationError
from conduit.core.secrets import SecretRef
from conduit.core.settings import AssignmentSettings, ProviderSettings


class ExampleSettings(ProviderSettings):
    kind: Literal["example"] = "example"
    api_key: SecretRef
    session_token: SecretRef | None = None
    base_url: str = "https://api.example.com"


class TestCredentialFields:
    def test_a_credential_field_typed_as_plain_text_is_refused_at_definition(self) -> None:
        with pytest.raises(ConfigurationError, match="api_key"):

            class Leaky(ProviderSettings):
                kind: Literal["leaky"] = "leaky"
                api_key: str

    @pytest.mark.parametrize(
        "field", ["password", "client_secret", "access_token", "secret_access_key", "credentials"]
    )
    def test_every_credential_like_name_is_checked(self, field: str) -> None:
        with pytest.raises(ConfigurationError):
            type(
                "Leaky",
                (ProviderSettings,),
                {"__annotations__": {field: str}, "__module__": __name__},
            )

    def test_a_literal_credential_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(ValidationError) as raised:
            ExampleSettings.model_validate({"api_key": "sk-literal-0000"})
        assert "sk-literal-0000" not in str(raised.value)

    def test_an_unknown_provider_mapping_may_not_carry_a_literal_credential(self) -> None:
        # The generic settings keep provider-specific fields for the provider's own model
        # to validate later, but a credential-like extra must already be a reference.
        with pytest.raises(ValidationError) as raised:
            ProviderSettings.model_validate({"kind": "other", "api_key": "sk-literal-0000"})
        assert "sk-literal-0000" not in str(raised.value)
        accepted = ProviderSettings.model_validate({"kind": "other", "api_key": "secret://k"})
        assert accepted.model_extra == {"api_key": "secret://k"}


class TestBuilding:
    def test_built_in_code_or_from_a_mapping_alike(self) -> None:
        in_code = ExampleSettings(api_key=SecretRef("example/key"), region="ap-southeast-2")
        from_mapping = ExampleSettings.model_validate(
            {"api_key": "secret://example/key", "region": "ap-southeast-2"}
        )
        assert in_code == from_mapping

    def test_provider_specific_models_forbid_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExampleSettings.model_validate({"api_key": "secret://k", "colour": "blue"})

    def test_kind_is_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            ProviderSettings.model_validate({"kind": "Not A Kind"})

    def test_the_catalogue_provider_defaults_to_the_kind(self) -> None:
        settings = ExampleSettings(api_key=SecretRef("k"))
        assert settings.catalogue_provider == "example"

    def test_settings_are_immutable(self) -> None:
        settings = ExampleSettings(api_key=SecretRef("k"))
        with pytest.raises(ValidationError):
            settings.region = "us-east-1"


class TestAssignmentSettings:
    def test_from_mapping(self) -> None:
        assignment = AssignmentSettings.model_validate(
            {"provider": "primary", "model": "stub-echo", "max_output_tokens": 1024}
        )
        assert assignment.max_output_tokens == 1024

    def test_output_budget_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            AssignmentSettings(provider="p", model="m", max_output_tokens=0)
