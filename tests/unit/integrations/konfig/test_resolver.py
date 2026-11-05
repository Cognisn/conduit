"""The konfig secret resolver: secret:// references through konfig's secrets store."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.errors import SecretUnavailable
from conduit.core.secrets import SecretRef, SecretResolver
from conduit.integrations.konfig import KONFIG_PLACEHOLDER, KonfigSecretResolver

#: Synthetic values only; none of these is a real credential.
VALUE = "synthetic-secret-value-0001"
MASTER_KEY = "conduit-tests-master-key-dev-only"


class FakeSecrets:
    """konfig's ``Secrets.get`` interface, held in memory."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.asked: list[str] = []
        self.threads: list[int] = []

    def get(self, key: str) -> Any:
        self.asked.append(key)
        self.threads.append(threading.get_ident())
        return self.values.get(key)


class FailingSecrets:
    def get(self, key: str) -> str | None:
        raise RuntimeError(f"store unreadable while holding {VALUE}")


def _encrypted_settings(konfig: ModuleType, path: Path) -> Any:
    return konfig.Settings(
        defaults={
            "secrets": {
                "backend": "encrypted_file",
                "file_path": str(path),
                "master_key": MASTER_KEY,
            }
        }
    )


def test_meets_the_secret_resolver_protocol() -> None:
    assert isinstance(KonfigSecretResolver(FakeSecrets({})), SecretResolver)


async def test_resolves_a_reference_by_its_name() -> None:
    secrets = FakeSecrets({"anthropic/api-key": VALUE})
    value = await KonfigSecretResolver(secrets).resolve(SecretRef("anthropic/api-key"))
    assert isinstance(value, SecretStr)
    assert value.get_secret_value() == VALUE
    assert secrets.asked == ["anthropic/api-key"]


async def test_reads_konfig_in_a_worker_thread() -> None:
    secrets = FakeSecrets({"k": VALUE})
    await KonfigSecretResolver(secrets).resolve(SecretRef("k"))
    assert secrets.threads != [threading.get_ident()]


async def test_a_slow_store_does_not_block_the_event_loop() -> None:
    # konfig's backends are synchronous and may wait on a file lock or AWS. The
    # store below answers only once the loop has run another task, which the loop
    # cannot do while a read blocks it.
    released = threading.Event()

    class SlowSecrets:
        def get(self, key: str) -> str | None:
            return VALUE if released.wait(timeout=5) else None

    async def release() -> None:
        released.set()

    resolver = KonfigSecretResolver(SlowSecrets())
    value, _ = await asyncio.wait_for(
        asyncio.gather(resolver.resolve(SecretRef("k")), release()), timeout=10
    )
    assert value.get_secret_value() == VALUE


async def test_concurrent_resolutions_each_get_their_own_value() -> None:
    names = [f"provider-{n}/key" for n in range(8)]
    secrets = FakeSecrets({name: f"synthetic-{name}" for name in names})
    resolver = KonfigSecretResolver(secrets)
    values = await asyncio.gather(*(resolver.resolve(SecretRef(n)) for n in names))
    assert [v.get_secret_value() for v in values] == [f"synthetic-{n}" for n in names]


@pytest.mark.parametrize("held", [None, ""])
async def test_an_absent_secret_is_unavailable_and_named(held: str | None) -> None:
    resolver = KonfigSecretResolver(FakeSecrets({} if held is None else {"k": held}))
    with pytest.raises(SecretUnavailable, match="secret://k"):
        await resolver.resolve(SecretRef("k"))


async def test_an_unpopulated_konfig_placeholder_is_refused(konfig_module: ModuleType) -> None:
    placeholder = konfig_module.secrets.secrets.PLACEHOLDER_SECRET_VALUE
    assert placeholder == KONFIG_PLACEHOLDER
    resolver = KonfigSecretResolver(FakeSecrets({"k": placeholder}))
    with pytest.raises(SecretUnavailable, match=r"secret://k.*placeholder"):
        await resolver.resolve(SecretRef("k"))


async def test_a_value_that_is_not_text_is_refused_without_echo() -> None:
    resolver = KonfigSecretResolver(FakeSecrets({"k": 4242}))
    with pytest.raises(SecretUnavailable, match="secret://k") as caught:
        await resolver.resolve(SecretRef("k"))
    assert "4242" not in str(caught.value)


async def test_a_failing_store_names_the_reference_never_the_value() -> None:
    resolver = KonfigSecretResolver(FailingSecrets())
    with pytest.raises(SecretUnavailable, match="secret://k") as caught:
        await resolver.resolve(SecretRef("k"))
    assert VALUE not in str(caught.value)
    assert "RuntimeError" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_repr_shows_no_values() -> None:
    resolver = KonfigSecretResolver(FakeSecrets({"k": VALUE}))
    assert VALUE not in repr(resolver)
    assert "FakeSecrets" in repr(resolver)


async def test_resolves_from_konfigs_encrypted_store(
    konfig_module: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "secrets.enc"
    settings = _encrypted_settings(konfig_module, path)
    konfig_module.Secrets("conduit-tests", settings=settings).set("openai/api-key", VALUE)
    assert VALUE.encode() not in path.read_bytes()

    resolver = KonfigSecretResolver(konfig_module.Secrets("conduit-tests", settings=settings))
    value = await resolver.resolve(SecretRef("openai/api-key"))
    assert value.get_secret_value() == VALUE
    with pytest.raises(SecretUnavailable, match="secret://openai/other"):
        await resolver.resolve(SecretRef("openai/other"))


async def test_from_settings_uses_the_store_konfigs_configuration_selects(
    konfig_module: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "secrets.enc"
    settings = _encrypted_settings(konfig_module, path)
    konfig_module.Secrets("conduit-tests", settings=settings).set("gemini/api-key", VALUE)

    resolver = KonfigSecretResolver.from_settings(settings, service_name="conduit-tests")
    value = await resolver.resolve(SecretRef("gemini/api-key"))
    assert value.get_secret_value() == VALUE
    assert "EncryptedFileBackend" in repr(resolver)
