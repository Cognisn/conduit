"""Tokens and registrations stay with the issuer that granted them; refresh is shared.

- Stored tokens and the client registration record their issuer. Neither is ever
  sent to another one: if the resource metadata names a different authorisation
  server, both are dropped and the call fails with ``ISSUER_CHANGED``.
- ``OAuthAuth.issuer`` pins the authorisation server, and a client secret needs it.
- A refresh is shared by every client of one store in the process, and a refused
  refresh drops the tokens only if the store still holds the refresh token sent.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import weakref
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from conduit.core.backends import TokenSet
from conduit.core.secrets import LiteralSecretResolver
from conduit.mcp import oauth
from conduit.mcp.errors import McpAuthError, OAuthFailure
from conduit.mcp.oauth import OAuthClient, _Client, _shared_lock
from conduit.mcp.settings import McpServerSettings
from conduit.stores.memory import MemoryTokenStore
from tests.unit.mcp.fake_auth import ISSUER, RESOURCE, FakeAuthServer
from tests.unit.mcp.servers import port, serving
from tests.unit.mcp.test_oauth import Clock, assert_refused, settings

EVIL = "https://evil.example.net"


def evil_server() -> FakeAuthServer:
    """Another authorisation server, whose issued values never repeat the first one's."""
    return FakeAuthServer(issuer=EVIL, _counter=itertools.count(9000))


def routed(*servers: FakeAuthServer, delay: float = 0.0) -> httpx.MockTransport:
    """One transport for several fake servers, each answering for its own issuer's host."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if delay:
            await asyncio.sleep(delay)  # every request yields, so callers interleave
        host = request.url.host
        for server in servers:
            if httpx.URL(server.issuer).host == host and server is not servers[0]:
                return server.handle(request)
        return servers[0].handle(request)

    return httpx.MockTransport(handle)


def make(
    transport: httpx.AsyncBaseTransport,
    store: Any,
    clock: Clock,
    hook: Any = None,
    secrets: dict[str, str] | None = None,
    **auth: Any,
) -> OAuthClient:
    return OAuthClient(
        settings(**auth),
        resource=RESOURCE,
        store=store,
        sign_in=hook,
        secrets=LiteralSecretResolver(secrets or {}),
        http_transport=transport,
        clock=clock,
    )


def sent_to(server: FakeAuthServer) -> str:
    return "\n".join(
        str(r.url) + r.content.decode("latin-1") + str(dict(r.headers)) for r in server.requests
    )


class TestIssuerBinding:
    async def test_tokens_are_never_refreshed_at_another_issuer(self) -> None:
        legit = FakeAuthServer(secret_client=True)
        evil = evil_server()
        store = MemoryTokenStore()
        clock = Clock()
        transport = routed(legit, evil)
        await make(transport, store, clock, hook=legit.sign_in_hook()).access_token()
        held = await store.get("mcp:docs")
        registration = await store.get("mcp:docs:client")
        assert held is not None and held.refresh_token is not None and registration is not None

        # The resource metadata now names another, self-consistent, authorisation server.
        legit.resource_metadata = {"authorization_servers": [EVIL]}
        clock.advance(3600)
        later = make(transport, store, clock, hook=None)
        with pytest.raises(McpAuthError) as caught:
            await later.access_token(interactive=False)
        assert_refused(caught.value, OAuthFailure.ISSUER_CHANGED, EVIL)
        leaked = sent_to(evil)
        assert held.refresh_token.get_secret_value() not in leaked
        assert registration.access_token.get_secret_value() not in leaked
        assert registration.refresh_token is not None
        assert registration.refresh_token.get_secret_value() not in leaked
        assert not [r for r in evil.requests if r.method == "POST"]
        assert await store.get("mcp:docs") is None
        assert await store.get("mcp:docs:client") is None

    async def test_a_registration_from_another_issuer_is_not_reused(self) -> None:
        legit = FakeAuthServer()
        evil = evil_server()
        store = MemoryTokenStore()
        clock = Clock()
        transport = routed(legit, evil)
        await make(transport, store, clock, hook=legit.sign_in_hook()).access_token()
        registration = await store.get("mcp:docs:client")
        assert registration is not None
        await store.delete("mcp:docs")
        legit.resource_metadata = {"authorization_servers": [EVIL]}
        await make(transport, store, clock, hook=evil.sign_in_hook()).access_token()
        assert registration.access_token.get_secret_value() not in sent_to(evil)
        assert len(evil.clients) == 1

    async def test_an_entry_without_an_issuer_is_not_trusted(self) -> None:
        legit = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        transport = routed(legit)
        await make(transport, store, clock, hook=legit.sign_in_hook()).access_token()
        held = await store.get("mcp:docs")
        assert held is not None
        await store.put("mcp:docs", held.model_copy(update={"token_type": "Bearer"}))
        clock.advance(3600)
        with pytest.raises(McpAuthError) as caught:
            await make(transport, store, clock).access_token(interactive=False)
        assert caught.value.reason is OAuthFailure.ISSUER_CHANGED
        assert not [f for f in legit.token_requests if f["grant_type"] == "refresh_token"]

    async def test_a_pinned_issuer_refuses_another_before_any_request_to_it(self) -> None:
        legit = FakeAuthServer(resource_metadata={"authorization_servers": [EVIL]})
        evil = evil_server()
        evil.clients["pre"] = {"redirect_uris": ["http://127.0.0.1:22420/callback"]}
        oauth = make(
            routed(legit, evil),
            MemoryTokenStore(),
            Clock(),
            hook=evil.sign_in_hook(),
            secrets={"docs_client": "zz-pinned-client-secret-zz"},
            client_id="pre",
            client_secret="secret://docs_client",
            issuer=ISSUER,
        )
        with pytest.raises(McpAuthError) as caught:
            await oauth.access_token()
        assert_refused(caught.value, OAuthFailure.ISSUER_MISMATCH, EVIL)
        assert evil.requests == []

    async def test_a_pinned_issuer_is_accepted(self) -> None:
        legit = FakeAuthServer()
        await make(
            routed(legit), MemoryTokenStore(), Clock(), hook=legit.sign_in_hook(), issuer=ISSUER
        ).access_token()

    def test_a_client_secret_needs_a_pinned_issuer(self) -> None:
        with pytest.raises(ValidationError):
            settings(client_id="pre", client_secret="secret://docs_client")
        pinned = settings(client_id="pre", client_secret="secret://docs_client", issuer=ISSUER)
        assert pinned.auth.type == "oauth"

    @pytest.mark.parametrize("issuer", ["http://auth.example.com", "not a url"])
    def test_a_pinned_issuer_must_be_secure(self, issuer: str) -> None:
        with pytest.raises(ValidationError):
            settings(issuer=issuer)


class SharedBacking:
    """A store over another store's data: a second process sharing one database."""

    def __init__(self, inner: MemoryTokenStore) -> None:
        self._inner = inner

    async def get(self, key: str) -> TokenSet | None:
        return await self._inner.get(key)

    async def put(self, key: str, tokens: TokenSet) -> None:
        await self._inner.put(key, tokens)

    async def delete(self, key: str) -> None:
        await self._inner.delete(key)


class TestSharedRefresh:
    async def _signed_in(
        self, server: FakeAuthServer, store: Any, clock: Clock
    ) -> tuple[httpx.MockTransport, Any]:
        transport = routed(server, delay=0.05)
        stale = await make(transport, store, clock, hook=server.sign_in_hook()).access_token()
        return transport, stale

    async def test_two_clients_of_one_store_share_one_refresh(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        transport, stale = await self._signed_in(server, store, clock)
        before = len(server.token_requests)
        first, second = make(transport, store, clock), make(transport, store, clock)
        fresh = await asyncio.gather(first.refreshed(stale), second.refreshed(stale))
        assert len(server.token_requests) - before == 1
        assert fresh[0].get_secret_value() == fresh[1].get_secret_value()

    async def test_a_refused_refresh_keeps_tokens_another_process_rotated(self) -> None:
        server = FakeAuthServer()
        backing = MemoryTokenStore()
        clock = Clock()
        transport, stale = await self._signed_in(server, backing, clock)
        here = make(transport, backing, clock)
        there = make(transport, SharedBacking(backing), clock)
        fresh = await asyncio.gather(here.refreshed(stale), there.refreshed(stale))
        refreshes = [f for f in server.token_requests if f["grant_type"] == "refresh_token"]
        assert len(refreshes) == 2  # the second used the rotated-out token and was refused
        held = await backing.get("mcp:docs")
        assert held is not None  # the refused one did not delete the fresh tokens
        assert held.access_token.get_secret_value() in server.access_tokens
        assert {t.get_secret_value() for t in fresh} == {held.access_token.get_secret_value()}

    async def test_a_dead_grant_still_held_is_dropped(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        transport, stale = await self._signed_in(server, store, clock)
        server.revoked.update(server.refresh_tokens)
        with pytest.raises(McpAuthError) as caught:
            await make(transport, store, clock).refreshed(stale)
        assert caught.value.reason is OAuthFailure.SIGN_IN_REQUIRED
        assert await store.get("mcp:docs") is None


class SlottedStore:
    """A store that cannot be weakly referenced (``__slots__`` without ``__weakref__``)."""

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        self._inner = MemoryTokenStore()

    async def get(self, key: str) -> TokenSet | None:
        return await self._inner.get(key)

    async def put(self, key: str, tokens: TokenSet) -> None:
        await self._inner.put(key, tokens)

    async def delete(self, key: str) -> None:
        await self._inner.delete(key)


class UnhashableStore(SharedBacking):
    """A store that cannot be a dictionary key at all."""

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnhashableStore) and other._inner is self._inner


@dataclass
class DataclassStore:
    """A plain dataclass store: weakly referenceable, but unhashable (``eq`` without
    ``frozen``)."""

    inner: MemoryTokenStore = field(default_factory=MemoryTokenStore)

    async def get(self, key: str) -> TokenSet | None:
        return await self.inner.get(key)

    async def put(self, key: str, tokens: TokenSet) -> None:
        await self.inner.put(key, tokens)

    async def delete(self, key: str) -> None:
        await self.inner.delete(key)


class TestSharedRefreshFallback:
    """A store that cannot be weakly referenced still shares one refresh across clients."""

    async def _shared_by_two_clients(self, store: Any) -> None:
        server = FakeAuthServer()
        clock = Clock()
        transport = routed(server, delay=0.05)
        stale = await make(transport, store, clock, hook=server.sign_in_hook()).access_token()
        before = len(server.token_requests)
        first, second = make(transport, store, clock), make(transport, store, clock)
        fresh = await asyncio.gather(first.refreshed(stale), second.refreshed(stale))
        assert len(server.token_requests) - before == 1
        assert fresh[0].get_secret_value() == fresh[1].get_secret_value()

    async def test_a_store_without_weak_references(self) -> None:
        store = SlottedStore()
        with pytest.raises(TypeError):
            weakref.ref(store)
        await self._shared_by_two_clients(store)

    async def test_an_unhashable_store(self) -> None:
        store = UnhashableStore(MemoryTokenStore())
        with pytest.raises(TypeError):
            hash(store)
        await self._shared_by_two_clients(store)

    async def test_a_dataclass_store(self) -> None:
        store = DataclassStore()
        weakref.ref(store)
        with pytest.raises(TypeError):
            hash(store)
        await self._shared_by_two_clients(store)

    async def test_a_weakly_referenceable_store_is_not_kept_alive(self) -> None:
        # Review M1: an unhashable store that can be weakly referenced goes, with its
        # locks, when the product drops it; only stores without weak references stay.
        gc.collect()
        pinned, by_id = len(oauth._PINNED_LOCKS), len(oauth._ID_LOCKS)
        stores = [DataclassStore() for _ in range(100)]
        held = [_shared_lock(store, "mcp:docs") for store in stores]
        assert len(oauth._ID_LOCKS) == by_id + 100
        assert len(oauth._PINNED_LOCKS) == pinned
        probe = weakref.ref(stores[0])
        del stores, held
        gc.collect()
        assert probe() is None
        assert len(oauth._ID_LOCKS) == by_id

    async def test_two_such_stores_do_not_share_a_lock(self) -> None:
        first, second = SlottedStore(), SlottedStore()
        assert _shared_lock(first, "mcp:docs") is _shared_lock(first, "mcp:docs")
        assert _shared_lock(first, "mcp:docs") is not _shared_lock(second, "mcp:docs")
        assert _shared_lock(first, "mcp:docs") is not _shared_lock(first, "mcp:other")


class TestRefreshedLeavesNothingUnusable:
    async def test_rejected_tokens_without_a_refresh_token_are_dropped(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        oauth = make(routed(server), store, clock, hook=server.sign_in_hook())
        stale = await oauth.access_token()
        held = await store.get("mcp:docs")
        assert held is not None
        await store.put("mcp:docs", held.model_copy(update={"refresh_token": None}))
        with pytest.raises(McpAuthError) as caught:
            await oauth.refreshed(stale)
        assert caught.value.reason is OAuthFailure.SIGN_IN_REQUIRED
        assert await store.get("mcp:docs") is None


class TestOAuthRequests:
    async def test_a_redirect_from_the_token_endpoint_is_not_followed(self) -> None:
        server = FakeAuthServer()
        elsewhere: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.host == "elsewhere.example.org":
                elsewhere.append(request)
                return httpx.Response(200, json={"access_token": "x", "token_type": "Bearer"})
            if request.url.path == "/token":
                server.requests.append(request)
                return httpx.Response(
                    307, headers={"Location": "https://elsewhere.example.org/token"}
                )
            return server.handle(request)

        oauth = make(
            httpx.MockTransport(handle), MemoryTokenStore(), Clock(), hook=server.sign_in_hook()
        )
        with pytest.raises(McpAuthError) as caught:
            await oauth.access_token()
        assert caught.value.reason is OAuthFailure.TOKEN_EXCHANGE
        assert elsewhere == []

    async def test_the_pkce_verifier_is_86_characters(self) -> None:
        server = FakeAuthServer()
        await make(
            routed(server), MemoryTokenStore(), Clock(), hook=server.sign_in_hook()
        ).access_token()
        exchange = next(f for f in server.token_requests if f["grant_type"] == "authorization_code")
        assert len(exchange["code_verifier"]) == 86

    def test_a_client_never_shows_its_id_or_secret(self) -> None:
        client = _Client("zz-client-id-zz", "zz-client-secret-zz", "client_secret_post")
        assert "zz-client-id-zz" not in repr(client)
        assert "zz-client-secret-zz" not in repr(client)
        assert "client_secret_post" in repr(client)


class TestOAuthEnvironment:
    async def test_environment_proxies_are_ignored_on_the_oauth_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # httpx 0.28 reads no .netrc on its own (only an explicit NetRCAuth does), so
        # the proxy variables are what trust_env would let the environment change.
        origin = f"http://127.0.0.1:{port('auth')}"
        server = FakeAuthServer(resource=f"{origin}/mcp", issuer=origin)
        dead_proxy = f"http://127.0.0.1:{port('callback2')}"
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, dead_proxy)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)

        def oauth(trust: bool) -> OAuthClient:
            data = {
                "name": "docs",
                "transport": {"type": "streamable_http", "url": server.resource},
                "auth": {"type": "oauth"},
                "task_classes": ["research"],
                "connect_timeout": 5,
                "use_environment_proxies": trust,
            }
            return OAuthClient(
                McpServerSettings.model_validate(data),
                resource=server.resource,
                store=MemoryTokenStore(),
                sign_in=server.sign_in_hook(),
                secrets=LiteralSecretResolver({}),
                clock=Clock(),
            )

        with serving(server.asgi(), "auth"):
            token = await oauth(False).access_token()
            assert token.get_secret_value() in server.access_tokens
            with pytest.raises(McpAuthError) as caught:
                await oauth(True).access_token()
        assert caught.value.reason is OAuthFailure.DISCOVERY
