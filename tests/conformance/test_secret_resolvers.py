"""The secret resolver conformance suite, against conduit's resolvers, and the suite's own
checks: it must fail a resolver that breaks the contract."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from pydantic import SecretStr

from conduit.core.errors import SecretUnavailable
from conduit.core.secrets import (
    EnvironmentSecretResolver,
    LiteralSecretResolver,
    SecretRef,
    SecretResolver,
)
from conduit.integrations.konfig import KonfigSecretResolver
from conduit.testing.conformance import SecretResolverSuite

#: A master key for konfig's encrypted file in a temporary directory; never a real key.
MASTER_KEY = "conduit-suite-master-key-dev-only"


class TestLiteralSecretResolver(SecretResolverSuite):
    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        return LiteralSecretResolver(secrets)


class TestEnvironmentSecretResolver(SecretResolverSuite):
    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        return EnvironmentSecretResolver(environ=dict(secrets))


class TestPrefixedEnvironmentSecretResolver(SecretResolverSuite):
    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        environ = {f"EXAMPLE_APP_{name}": value for name, value in secrets.items()}
        # Unprefixed copies must never be read when a prefix is set.
        environ.update({name: "unprefixed-must-not-be-read" for name in secrets})
        return EnvironmentSecretResolver(prefix="EXAMPLE_APP_", environ=environ)


class _MappingSecrets:
    """konfig's ``Secrets.get`` interface over a mapping."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    def get(self, key: str) -> str | None:
        return self._values.get(key)


class TestKonfigSecretResolverOverAMapping(SecretResolverSuite):
    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        return KonfigSecretResolver(_MappingSecrets(secrets))


#: konfig variables that would point it at AWS or a real key; cleared for each scenario.
_KONFIG_ENVIRONMENT = (
    "KONFIG_AWS_SECRETS_MANAGER",
    "KONFIG_AWS_SETTINGS",
    "KONFIG_AWS_SEED",
    "KONFIG_CONFIG_FORMAT",
    "KONFIG_MASTER_KEY",
)


class TestKonfigSecretResolverOverItsEncryptedFile(SecretResolverSuite):
    """konfig's own encrypted file store, in a temporary directory (the konfig extra)."""

    @pytest.fixture(autouse=True)
    def _konfig(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        for name in _KONFIG_ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
        for name in list(os.environ):
            if name.upper().startswith(("SECRETS__", "CONDUIT__")):
                monkeypatch.delenv(name, raising=False)
        self.konfig: ModuleType = pytest.importorskip("konfig")
        self.directory = tmp_path
        self.made = 0
        yield

    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        self.made += 1
        settings = self.konfig.Settings(
            defaults={
                "secrets": {
                    "backend": "encrypted_file",
                    "file_path": str(self.directory / f"secrets-{self.made}.enc"),
                    "master_key": MASTER_KEY,
                }
            }
        )
        store = self.konfig.Secrets("conduit-suite", settings=settings)
        for name, value in secrets.items():
            await asyncio.to_thread(store.set, name, value)
        return KonfigSecretResolver.from_settings(settings, service_name="conduit-suite")


# --- The suite's own checks: each broken resolver fails the scenario it breaks. ----------


class _Base:
    def __init__(self, secrets: Mapping[str, str]) -> None:
        self.values = dict(secrets)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<{len(self.values)} values>)"


class _ReturnsPlainText(_Base):
    async def resolve(self, ref: SecretRef) -> Any:
        if ref.name not in self.values:
            raise SecretUnavailable(f"{ref} is unknown")
        return self.values[ref.name]


class _StripsValues(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        if ref.name not in self.values:
            raise SecretUnavailable(f"{ref} is unknown")
        return SecretStr(self.values[ref.name].strip())


class _EchoesHeldValues(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        if ref.name not in self.values:
            raise SecretUnavailable(f"{ref} is not one of {sorted(self.values.values())}")
        return SecretStr(self.values[ref.name])


class _ChainsTheStoreError(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        try:
            return SecretStr(self.values[ref.name])
        except KeyError:
            raise SecretUnavailable(f"{ref} is unknown") from RuntimeError(str(self.values))


class _KeepsTheStoreErrorAsContext(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        try:
            return SecretStr(self.values[ref.name])
        except KeyError:
            try:
                raise RuntimeError(f"store holds {sorted(self.values.values())}") from None
            except RuntimeError:
                raise SecretUnavailable(f"{ref} is unknown")  # noqa: B904 - the defect


class _RaisesKeyError(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        return SecretStr(self.values[ref.name])


class _ReturnsEmptyForUnknown(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        return SecretStr(self.values.get(ref.name, ""))


class _UnnamedError(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        if ref.name not in self.values:
            raise SecretUnavailable("a secret is unavailable")
        return SecretStr(self.values[ref.name])


class _ReprShowsValues(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        if ref.name not in self.values:
            raise SecretUnavailable(f"{ref} is unknown")
        return SecretStr(self.values[ref.name])

    def __repr__(self) -> str:
        return f"_ReprShowsValues({self.values!r})"


class _SharedSlot(_Base):
    """Remembers the reference being resolved on the instance: concurrent calls collide."""

    async def resolve(self, ref: SecretRef) -> SecretStr:
        self.current = ref.name
        await asyncio.sleep(0)
        if self.current not in self.values:
            raise SecretUnavailable(f"{ref} is unknown")
        return SecretStr(self.values[self.current])


class _LogsValues(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        if ref.name not in self.values:
            raise SecretUnavailable(f"{ref} is unknown")
        logging.getLogger("example.vault").debug("resolved %s as %s", ref, self.values[ref.name])
        return SecretStr(self.values[ref.name])


def _suite(resolver: type[_Base]) -> SecretResolverSuite:
    class Suite(SecretResolverSuite):
        async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
            return cast(SecretResolver, resolver(secrets))  # broken on purpose

    return Suite()


@pytest.mark.parametrize(
    ("broken", "scenario"),
    [
        (_ReturnsPlainText, "test_a_known_reference_resolves_to_its_value"),
        (_StripsValues, "test_every_character_of_a_value_is_kept"),
        (_EchoesHeldValues, "test_an_unknown_reference_is_secret_unavailable"),
        (_ChainsTheStoreError, "test_an_unknown_reference_is_secret_unavailable"),
        (_KeepsTheStoreErrorAsContext, "test_an_unknown_reference_is_secret_unavailable"),
        (_RaisesKeyError, "test_an_unknown_reference_is_secret_unavailable"),
        (_ReturnsEmptyForUnknown, "test_an_unknown_reference_is_secret_unavailable"),
        (_UnnamedError, "test_an_unknown_reference_is_secret_unavailable"),
        (_ReprShowsValues, "test_no_value_shows_in_a_repr"),
        (_SharedSlot, "test_concurrent_references_resolve_to_their_own_values"),
        (_LogsValues, "test_resolving_logs_no_value"),
    ],
)
def test_the_suite_fails_a_broken_resolver(broken: type[_Base], scenario: str) -> None:
    with pytest.raises(AssertionError):
        getattr(_suite(broken), scenario)()


@pytest.mark.parametrize(
    "scenario",
    [name for name in dir(SecretResolverSuite) if name.startswith("test_")],
)
def test_every_scenario_passes_a_sound_resolver(scenario: str) -> None:
    getattr(_suite(_ReprSafeLiteral), scenario)()


class _ReprSafeLiteral(_Base):
    async def resolve(self, ref: SecretRef) -> SecretStr:
        if ref.name not in self.values:
            raise SecretUnavailable(f"{ref} is unknown")
        return SecretStr(self.values[ref.name])


def test_an_unconfigured_suite_says_what_to_override() -> None:
    with pytest.raises(NotImplementedError, match="make_resolver"):
        SecretResolverSuite().test_a_known_reference_resolves_to_its_value()
