"""Settings, Wave 3: no plaintext credential in an unknown field, and refusals that name
fields and give a validator's reason only when it quotes no input (Q0022)."""

from __future__ import annotations

from typing import Any, Literal, Self

import pytest
from pydantic import ValidationError, model_validator

from conduit.core.errors import SettingsRefused
from conduit.core.registry import ProviderRegistry
from conduit.core.runtime import ConduitSettings
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import WHOLE, ProviderSettings, refusals

PLAINTEXT = "zz-plaintext-credential-4417zz"


def _refused(fields: dict[str, Any]) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        ProviderSettings.model_validate({"kind": "custom", **fields})
    error = caught.value
    assert PLAINTEXT not in str(error)
    assert PLAINTEXT not in repr(error.errors(include_url=False, include_input=False))
    return error


class TestPlaintextInExtraFields:
    @pytest.mark.parametrize(
        ("fields", "path"),
        [
            ({"auth": {"token": PLAINTEXT}}, "auth.token"),
            ({"bearer": PLAINTEXT}, "bearer"),
            ({"auth": PLAINTEXT}, "auth"),
            ({"headers": {"Authorization": PLAINTEXT}}, "headers.Authorization"),
            ({"accessToken": PLAINTEXT}, "accessToken"),
            ({"client-secret": PLAINTEXT}, "client-secret"),
            ({"apikey": PLAINTEXT}, "apikey"),
            ({"service": {"auth": {"api_key": PLAINTEXT}}}, "service.auth.api_key"),
            ({"pool": [{"password": PLAINTEXT}]}, "pool.0.password"),
            ({"note": f"Bearer {PLAINTEXT}"}, "note"),
            ({"note": f"basic {PLAINTEXT}"}, "note"),
            ({"api_key": PLAINTEXT}, "api_key"),
            # A credential-like name carries into what it holds.
            ({"api_key": {"value": PLAINTEXT}}, "api_key.value"),
            ({"credentials": {"client": PLAINTEXT}}, "credentials.client"),
            ({"secret": [PLAINTEXT]}, "secret.0"),
            ({"token": {"inner": {"deeper": PLAINTEXT}}}, "token.inner.deeper"),
            # "auth" does not carry, but a password under any short name is still refused.
            ({"auth": {"user": "u", "pass": PLAINTEXT}}, "auth.pass"),
            # Plural names.
            ({"api_keys": [PLAINTEXT]}, "api_keys.0"),
            ({"tokens": {"primary": PLAINTEXT}}, "tokens.primary"),
            ({"secrets": PLAINTEXT}, "secrets"),
            ({"passwords": [PLAINTEXT]}, "passwords.0"),
            ({"apikeys": PLAINTEXT}, "apikeys"),
            ({"note": f"Token {PLAINTEXT}"}, "note"),
            ({"credentials": [{"client": PLAINTEXT}]}, "credentials.0.client"),
        ],
    )
    def test_a_plaintext_credential_is_refused_naming_the_field(
        self, fields: dict[str, Any], path: str
    ) -> None:
        error = _refused(fields)
        assert refusals(error)[0] == [path]

    @pytest.mark.parametrize(
        "name",
        [
            "bearer",
            "auth",
            "authorization",
            "authorisation",
            "pass",
            "pwd",
            "passwd",
            "jwt",
            "cookie",
        ],
    )
    def test_every_further_name_is_credential_like(self, name: str) -> None:
        assert refusals(_refused({name: PLAINTEXT}))[0] == [name]

    def test_every_offending_field_is_named(self) -> None:
        error = _refused({"bearer": PLAINTEXT, "auth": {"token": PLAINTEXT}})
        assert refusals(error)[0] == ["bearer", "auth.token"]

    @pytest.mark.parametrize(
        "fields",
        [
            {"auth": {"token": "secret://tenant/token"}},
            {"bearer": "secret://tenant/bearer"},
            {"service": {"auth": {"mode": "api_key", "api_key": "secret://openai/key"}}},
            {"allow_long_term_api_keys": False, "endpoint": "direct", "max_retries": 2},
            {"note": "a bearer of good news"},
            {"auth": {"mode": "credential_chain", "profile": "default"}},
            {"notes": "Token budget is high"},
            {"notes": "Bearer of the news"},
            {"notes": "basic setup"},
            {"api_key": {"ref": "secret://tenant/key"}},
            {"allow_long_term_api_keys": True, "rotate_keys": None},
            # Numbers are never credentials, whatever they are called.
            {"max_output_tokens": 1000},
            {"limits": {"max_tokens": 4096}},
            {"password_policy": {"min_length": 12}},
            {"secret_ttl": 30.5},
        ],
    )
    def test_references_and_ordinary_fields_are_kept(self, fields: dict[str, Any]) -> None:
        settings = ProviderSettings.model_validate({"kind": "custom", **fields})
        assert settings.model_extra == fields

    def test_conduit_settings_refuses_it_too(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ConduitSettings.model_validate(
                {"providers": {"main": {"kind": "custom", "auth": {"token": PLAINTEXT}}}}
            )
        assert PLAINTEXT not in str(caught.value)
        assert refusals(caught.value)[0] == ["providers.main.auth.token"]


class _Echoing(ProviderSettings):
    kind: Literal["echoing"] = "echoing"
    endpoint: str = "https://api.example.com"

    @model_validator(mode="after")
    def _https(self) -> Self:
        if not self.endpoint.startswith("https://"):
            raise ValueError(f"endpoint {self.endpoint} must be https")  # quotes its input
        return self


class _Plain(ProviderSettings):
    kind: Literal["plain"] = "plain"
    endpoint: str = "https://api.example.com"

    @model_validator(mode="after")
    def _https(self) -> Self:
        if not self.endpoint.startswith("https://"):
            raise ValueError("endpoint must be an https URL")
        return self


class TestReasons:
    def test_a_reason_that_quotes_no_input_is_kept(self) -> None:
        with pytest.raises(ValidationError) as caught:
            _Plain(endpoint=f"http://{PLAINTEXT}.example.com")
        assert refusals(caught.value, {"endpoint": f"http://{PLAINTEXT}.example.com"}) == (
            [WHOLE],
            ["endpoint must be an https URL"],
        )

    def test_a_reason_that_quotes_its_input_is_dropped(self) -> None:
        given = {"endpoint": f"http://{PLAINTEXT}.example.com"}
        with pytest.raises(ValidationError) as caught:
            _Echoing.model_validate(given)
        assert refusals(caught.value, given) == ([WHOLE], [])

    def test_pydantic_messages_are_never_reasons(self) -> None:
        with pytest.raises(ValidationError) as caught:
            _Plain.model_validate({"endpoint": 5})
        fields, reasons = refusals(caught.value, {"endpoint": 5})
        assert fields == ["endpoint"] and reasons == []

    async def test_the_registry_gives_the_kept_reasons(self) -> None:
        class Factory:
            kind = "echoing"
            settings_model = _Echoing

            def create_client(self, settings: Any, secrets: Any) -> Any:
                return None

            def create_provider(
                self, name: str, settings: Any, client: Any, *, catalogue: Any = None
            ) -> Any:
                raise AssertionError("not reached")

        registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[Factory()])
        with pytest.raises(SettingsRefused) as caught:
            await registry.build("p", {"kind": "echoing", "endpoint": f"http://{PLAINTEXT}"})
        assert PLAINTEXT not in str(caught.value)
        assert caught.value.reasons == ()

    def test_a_value_inside_a_longer_name_is_not_an_echo(self) -> None:
        class Gated(ProviderSettings):
            kind: Literal["gated"] = "gated"
            mode: Literal["long_term_key", "short_term_key"] = "short_term_key"
            allow_long_term_keys: bool = False

            @model_validator(mode="after")
            def _gated(self) -> Self:
                if self.mode == "long_term_key" and not self.allow_long_term_keys:
                    raise ValueError("long-term keys need allow_long_term_keys")
                return self

        given = {"mode": "long_term_key"}
        with pytest.raises(ValidationError) as caught:
            Gated.model_validate(given)
        assert refusals(caught.value, given)[1] == ["long-term keys need allow_long_term_keys"]
