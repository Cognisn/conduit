"""Secret references and the environment and literal resolvers."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from conduit.core.errors import SecretUnavailable
from conduit.core.secrets import (
    EnvironmentSecretResolver,
    LiteralSecretResolver,
    SecretRef,
    SecretResolver,
    resolve_secret_fields,
)


class TestSecretRef:
    def test_parses_and_prints_its_reference(self) -> None:
        ref = SecretRef.parse("secret://bedrock/access-key")
        assert ref.name == "bedrock/access-key"
        assert str(ref) == "secret://bedrock/access-key"
        assert ref == SecretRef("bedrock/access-key")

    @pytest.mark.parametrize(
        "text",
        ["sk-live-0000", "bedrock-key", "secret://", "secret://has space", "SECRET://x"],
    )
    def test_anything_but_a_reference_is_refused(self, text: str) -> None:
        with pytest.raises(ValueError):
            SecretRef.parse(text)

    def test_validates_as_a_pydantic_field_and_serialises_back(self) -> None:
        class Holder(BaseModel):
            key: SecretRef

        holder = Holder.model_validate({"key": "secret://anthropic"})
        assert holder.key == SecretRef("anthropic")
        assert holder.model_dump(mode="json") == {"key": "secret://anthropic"}
        with pytest.raises(ValidationError):
            Holder.model_validate({"key": "sk-ant-literal"})


class TestEnvironmentResolver:
    async def test_reads_the_named_variable_with_its_prefix(self) -> None:
        resolver = EnvironmentSecretResolver(
            prefix="CONDUIT_", environ={"CONDUIT_ANTHROPIC_KEY": "value-1"}
        )
        secret = await resolver.resolve(SecretRef("ANTHROPIC_KEY"))
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "value-1"
        assert isinstance(resolver, SecretResolver)

    async def test_a_missing_or_empty_variable_names_only_the_reference(self) -> None:
        resolver = EnvironmentSecretResolver(environ={"EMPTY": ""})
        for name in ("MISSING", "EMPTY"):
            with pytest.raises(SecretUnavailable) as raised:
                await resolver.resolve(SecretRef(name))
            assert name in str(raised.value)

    async def test_a_reference_that_is_not_a_variable_name_is_refused(self) -> None:
        resolver = EnvironmentSecretResolver(environ={})
        with pytest.raises(SecretUnavailable):
            await resolver.resolve(SecretRef("bedrock/access-key"))


class TestLiteralResolver:
    async def test_returns_the_value_held_for_the_reference(self) -> None:
        resolver = LiteralSecretResolver({"stub/key": "not-a-real-key"})
        secret = await resolver.resolve(SecretRef("stub/key"))
        assert secret.get_secret_value() == "not-a-real-key"
        with pytest.raises(SecretUnavailable):
            await resolver.resolve(SecretRef("other"))

    def test_repr_never_shows_values(self) -> None:
        assert "not-a-real-key" not in repr(LiteralSecretResolver({"k": "not-a-real-key"}))


async def test_every_reference_field_is_resolved_by_dotted_path() -> None:
    class Inner(BaseModel):
        token: SecretRef

    class Outer(BaseModel):
        key: SecretRef
        optional: SecretRef | None = None
        inner: Inner
        region: str = "ap-southeast-2"

    settings = Outer.model_validate({"key": "secret://a", "inner": {"token": "secret://b"}})
    resolved = await resolve_secret_fields(settings, LiteralSecretResolver({"a": "1", "b": "2"}))
    assert {k: v.get_secret_value() for k, v in resolved.items()} == {
        "key": "1",
        "inner.token": "2",
    }
