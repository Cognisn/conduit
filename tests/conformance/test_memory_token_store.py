"""The token store conformance suite, against the in-memory store, and the suite's own
checks: it must fail a store that breaks the contract."""

from __future__ import annotations

import asyncio

import pytest

from conduit.core.backends import TokenSet, TokenStore
from conduit.stores.memory import MemoryTokenStore
from conduit.testing.conformance import TokenStoreSuite
from tests.conformance.meta import fails


class TestMemoryTokenStore(TokenStoreSuite):
    async def make_store(self) -> TokenStore:
        return MemoryTokenStore()


class _ForgetsOverwrites(MemoryTokenStore):
    """Keeps the first token set put under a key."""

    async def put(self, key: str, tokens: TokenSet) -> None:
        if await self.get(key) is None:
            await super().put(key, tokens)


class _DeleteIgnored(MemoryTokenStore):
    async def delete(self, key: str) -> None:
        return None


class _DropsRefresh(MemoryTokenStore):
    async def put(self, key: str, tokens: TokenSet) -> None:
        await super().put(key, tokens.model_copy(update={"refresh_token": None}))


class _CutsTokenType(MemoryTokenStore):
    """Keeps at most 255 characters of the token type, as a narrow column would."""

    async def put(self, key: str, tokens: TokenSet) -> None:
        await super().put(key, tokens.model_copy(update={"token_type": tokens.token_type[:255]}))


class _OneSlot(MemoryTokenStore):
    """Holds a single key: a store that ignores the key entirely."""

    async def get(self, key: str) -> TokenSet | None:
        return await super().get("only")

    async def put(self, key: str, tokens: TokenSet) -> None:
        await asyncio.sleep(0)
        await super().put("only", tokens)

    async def delete(self, key: str) -> None:
        await super().delete("only")


class _RaisesOnUnknown(MemoryTokenStore):
    async def get(self, key: str) -> TokenSet | None:
        found = await super().get(key)
        if found is None:
            raise KeyError(key)
        return found


def _suite(store: type[MemoryTokenStore]) -> TokenStoreSuite:
    class Suite(TokenStoreSuite):
        async def make_store(self) -> TokenStore:
            return store()

    return Suite()


@pytest.mark.parametrize(
    ("broken", "scenario"),
    [
        (_ForgetsOverwrites, "test_a_put_overwrites_the_earlier_tokens"),
        (_DeleteIgnored, "test_a_deleted_key_is_gone"),
        (_DropsRefresh, "test_every_field_round_trips"),
        (_CutsTokenType, "test_every_field_round_trips"),
        (_OneSlot, "test_concurrent_keys_do_not_interfere"),
        (_RaisesOnUnknown, "test_an_unknown_key_is_none"),
    ],
)
def test_the_suite_fails_a_broken_store(broken: type[MemoryTokenStore], scenario: str) -> None:
    with fails(AssertionError, KeyError):
        getattr(_suite(broken), scenario)()
