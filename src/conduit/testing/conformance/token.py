"""The token store conformance suite."""

# Suites are test code shipped for products to run under pytest: assertions are the point.
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import ClassVar

from pydantic import SecretStr

from conduit.core.backends import TokenSet, TokenStore
from conduit.testing.conformance.base import ConformanceSuite


def _tokens(n: int = 0, *, full: bool = True) -> TokenSet:
    if not full:
        return TokenSet(access_token=SecretStr(f"access-{n:04d}-example"))
    return TokenSet(
        access_token=SecretStr(f"access-{n:04d}-example"),
        refresh_token=SecretStr(f"refresh-{n:04d}-example"),
        token_type="Bearer",  # noqa: S106 - the OAuth token type, not a secret
        expires_at=datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC),
        scope="tools:read tools:call",
    )


#: A token type as long as conduit's MCP client may store: a registration bound to an
#: issuer with a long address (``registration:<method>;iss=<issuer>``), 2,100 characters.
LONG_TOKEN_TYPE = "registration:client_secret_basic;iss=https://auth.example.com/tenants/" + "t" * (
    2_100 - 70
)


def _long(n: int) -> TokenSet:
    return _tokens(n).model_copy(update={"token_type": LONG_TOKEN_TYPE})


def _revealed(tokens: TokenSet | None) -> dict[str, object] | None:
    # TokenSet compares its secrets by value, but compare the revealed form too so a
    # failure shows which field the store lost.
    return None if tokens is None else tokens.to_storage()


class TokenStoreSuite(ConformanceSuite):
    """Round trip (a long token type included), overwrite, delete, unknown keys and
    concurrent keys."""

    interface: ClassVar[str] = "token_store"
    #: Keys written at once in the concurrency scenario.
    concurrent_keys: ClassVar[int] = 20

    async def make_store(self) -> TokenStore:
        """A new, empty store for one scenario. Products override this."""
        raise NotImplementedError("override make_store to return your token store")

    async def reopen_store(self, store: TokenStore) -> TokenStore:
        """The same backend reached afresh (a new connection or instance). Durable stores
        override this; the default returns ``store`` itself."""
        return store

    async def close_store(self, store: TokenStore) -> None:
        """Release a store after a scenario. Override if the store needs closing."""
        closer = getattr(store, "aclose", None)
        if closer is not None:
            await closer()

    def _scenario(self, body: Callable[[TokenStore], Awaitable[None]]) -> None:
        async def scenario() -> None:
            store = await self.make_store()
            try:
                await body(store)
            finally:
                await self.close_store(store)

        self.run(scenario)

    def test_every_field_round_trips(self) -> None:
        async def body(store: TokenStore) -> None:
            await store.put("mcp:full", _tokens(1))
            await store.put("mcp:minimal", _tokens(2, full=False))
            await store.put("mcp:long:client", _long(3))
            again = await self.reopen_store(store)
            full, minimal = await again.get("mcp:full"), await again.get("mcp:minimal")
            assert _revealed(full) == _revealed(_tokens(1))
            assert full == _tokens(1)
            assert _revealed(minimal) == _revealed(_tokens(2, full=False))
            assert minimal is not None and minimal.refresh_token is None
            long = await again.get("mcp:long:client")
            assert long is not None and long.token_type == LONG_TOKEN_TYPE
            assert _revealed(long) == _revealed(_long(3))

        self._scenario(body)

    def test_an_unknown_key_is_none(self) -> None:
        async def body(store: TokenStore) -> None:
            assert await store.get("mcp:never-written") is None

        self._scenario(body)

    def test_a_put_overwrites_the_earlier_tokens(self) -> None:
        async def body(store: TokenStore) -> None:
            await store.put("mcp:rotated", _tokens(1))
            await store.put("mcp:rotated", _tokens(2, full=False))
            kept = await (await self.reopen_store(store)).get("mcp:rotated")
            assert _revealed(kept) == _revealed(_tokens(2, full=False))

        self._scenario(body)

    def test_a_deleted_key_is_gone(self) -> None:
        async def body(store: TokenStore) -> None:
            await store.put("mcp:keep", _tokens(1))
            await store.put("mcp:drop", _tokens(2))
            await store.delete("mcp:drop")
            again = await self.reopen_store(store)
            assert await again.get("mcp:drop") is None
            assert _revealed(await again.get("mcp:keep")) == _revealed(_tokens(1))
            await store.delete("mcp:drop")  # deleting what is absent is not an error
            await store.delete("mcp:never-written")

        self._scenario(body)

    def test_concurrent_keys_do_not_interfere(self) -> None:
        keys = self.concurrent_keys

        async def body(store: TokenStore) -> None:
            await asyncio.gather(*(store.put(f"mcp:k{n}", _tokens(n)) for n in range(keys)))
            found = await asyncio.gather(*(store.get(f"mcp:k{n}") for n in range(keys)))
            assert [_revealed(t) for t in found] == [_revealed(_tokens(n)) for n in range(keys)]
            await asyncio.gather(*(store.delete(f"mcp:k{n}") for n in range(0, keys, 2)))
            found = await asyncio.gather(*(store.get(f"mcp:k{n}") for n in range(keys)))
            assert [t is None for t in found] == [n % 2 == 0 for n in range(keys)]

        self._scenario(body)
