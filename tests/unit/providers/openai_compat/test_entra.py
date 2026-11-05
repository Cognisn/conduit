"""The shared Entra ID helper: the credential is closed on failure, quietly."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from conduit.providers.openai_compat import entra
from conduit.providers.openai_compat.settings import FoundryEntraId


class _Credential:
    def __init__(self, *, fail_close: BaseException | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self._fail_close = fail_close

    async def close(self) -> None:
        self.closed = True
        if self._fail_close is not None:
            raise self._fail_close


@pytest.fixture
def identity(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = pytest.importorskip("azure.identity.aio")
    made: list[_Credential] = []

    def make(**kwargs: Any) -> _Credential:
        credential = _Credential(**kwargs)
        made.append(credential)
        return credential

    monkeypatch.setattr(module, "DefaultAzureCredential", make)
    module.made = made
    return module


def test_the_credential_and_token_provider_are_built_for_the_scope(identity: Any) -> None:
    seen: list[tuple[Any, str]] = []

    def provider(credential: Any, scope: str) -> Any:
        seen.append((credential, scope))
        return lambda: None

    identity.get_bearer_token_provider = provider
    auth = FoundryEntraId(managed_identity_client_id="example-client")
    credential, _ = entra.entra_id_token_provider(auth)
    assert credential.kwargs == {"managed_identity_client_id": "example-client"}
    assert seen == [(credential, "https://ai.azure.com/.default")]


def test_the_credential_is_closed_when_the_token_provider_cannot_be_made(
    identity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(credential: Any, scope: str) -> Any:
        raise ValueError("cannot make the token provider")

    monkeypatch.setattr(identity, "get_bearer_token_provider", refuse)
    with pytest.raises(ValueError):
        entra.entra_id_token_provider(FoundryEntraId())
    assert [c.closed for c in identity.made] == [True]


async def test_a_failing_close_on_the_loop_is_logged_by_class_and_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    credential = _Credential(fail_close=RuntimeError("zz-close-marker-zz"))
    with caplog.at_level(logging.WARNING, logger="conduit.providers.openai_compat"):
        entra.close_soon(credential)
        for _ in range(3):
            await asyncio.sleep(0)
    assert credential.closed
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    for record in caplog.records:
        assert "zz-close-marker-zz" not in record.getMessage()
        assert record.exc_info is None


def test_a_failing_close_without_a_loop_is_logged_and_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    credential = _Credential(fail_close=RuntimeError("zz-close-marker-zz"))
    with caplog.at_level(logging.WARNING, logger="conduit.providers.openai_compat"):
        entra.close_soon(credential)
    assert credential.closed
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


async def test_a_cancelled_close_is_not_swallowed() -> None:
    credential = _Credential(fail_close=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await entra.close_quietly(credential)
