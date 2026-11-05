"""The OAuth 2.1 client: discovery, registration, PKCE, the callback checks and refresh.

Run in-process against the synthetic authorisation server in ``fake_auth`` (an
``httpx.MockTransport``): no network.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
import pytest
from pydantic import SecretStr

from conduit.core.backends import TokenSet
from conduit.core.secrets import LiteralSecretResolver
from conduit.mcp.errors import McpAuthError, OAuthFailure
from conduit.mcp.oauth import OAuthClient
from conduit.mcp.settings import McpServerSettings
from conduit.mcp.signin import SignInRequest, SignInResult
from conduit.stores.errors import StoreError
from conduit.stores.memory import MemoryTokenStore
from tests.unit.mcp.fake_auth import ISSUER, RESOURCE, FakeAuthServer, FakeSignIn, s256

PKCE_CHARSET = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")


def settings(**auth: Any) -> McpServerSettings:
    return McpServerSettings.model_validate(
        {
            "name": "docs",
            "transport": {"type": "streamable_http", "url": RESOURCE},
            "auth": {"type": "oauth", **auth},
            "task_classes": ["research"],
        }
    )


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def client(
    server: FakeAuthServer,
    store: Any = None,
    hook: Any = "default",
    clock: Clock | None = None,
    secrets: dict[str, str] | None = None,
    **auth: Any,
) -> OAuthClient:
    return OAuthClient(
        settings(**auth),
        resource=RESOURCE,
        store=store if store is not None else MemoryTokenStore(),
        sign_in=server.sign_in_hook() if hook == "default" else hook,
        secrets=LiteralSecretResolver(secrets or {}),
        http_transport=server.transport(),
        clock=clock or Clock(),
    )


def replace_param(url: str, name: str, value: str | None) -> str:
    parts = urlsplit(url)
    params = {k: v[0] for k, v in parse_qs(parts.query).items()}
    if value is None:
        params.pop(name, None)
    else:
        params[name] = value
    return urlunsplit(parts._replace(query=urlencode(params)))


def assert_refused(error: McpAuthError, reason: OAuthFailure, *hidden: str) -> None:
    assert error.reason is reason
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), repr(error.args)):
        for value in hidden:
            assert value not in text


class TestSignIn:
    async def test_the_whole_flow_stores_tokens_and_registration(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        oauth = client(server, store)
        token = await oauth.access_token()
        assert token.get_secret_value() in server.access_tokens
        held = await store.get("mcp:docs")
        assert held is not None
        assert held.access_token.get_secret_value() == token.get_secret_value()
        assert held.refresh_token is not None
        assert held.expires_at == datetime(2026, 11, 5, 10, 0, tzinfo=UTC)
        registration = await store.get("mcp:docs:client")
        assert registration is not None
        assert registration.access_token.get_secret_value() in server.clients

    async def test_stored_tokens_are_used_without_any_request(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        first = await client(server, store).access_token()
        before = len(server.requests)
        again = await client(server, store, hook=None).access_token()
        assert again.get_secret_value() == first.get_secret_value()
        assert len(server.requests) == before

    async def test_the_registration_request(self) -> None:
        server = FakeAuthServer()
        hook = server.sign_in_hook()
        await client(server, hook=hook, client_name="Example Product").access_token()
        (record,) = server.clients.values()
        body = record["request"]
        assert body["redirect_uris"] == [hook.redirect_uri]
        assert body["client_name"] == "Example Product"
        assert set(body["grant_types"]) == {"authorization_code", "refresh_token"}
        assert body["token_endpoint_auth_method"] == "none"

    async def test_the_authorisation_request(self) -> None:
        server = FakeAuthServer()
        hook = server.sign_in_hook()
        await client(server, hook=hook, scopes=["read"]).access_token()
        (request,) = hook.requests
        assert isinstance(request, SignInRequest)
        assert request.server == "docs"
        assert request.redirect_uri == hook.redirect_uri
        assert request.authorization_url.startswith(f"{ISSUER}/authorize?")
        (params,) = server.authorise_requests
        assert params["response_type"] == "code"
        assert params["code_challenge_method"] == "S256"
        assert params["resource"] == RESOURCE
        assert params["scope"] == "read"
        assert len(params["state"]) >= 32

    async def test_pkce_verifier_and_challenge(self) -> None:
        server = FakeAuthServer()
        await client(server).access_token()
        (params,) = server.authorise_requests
        exchange = next(f for f in server.token_requests if f["grant_type"] == "authorization_code")
        verifier = exchange["code_verifier"]
        assert PKCE_CHARSET.fullmatch(verifier)
        assert s256(verifier) == params["code_challenge"]
        assert verifier != params["code_challenge"]
        assert exchange["resource"] == RESOURCE

    async def test_every_sign_in_has_a_fresh_state_and_verifier(self) -> None:
        server = FakeAuthServer()
        await client(server).access_token()
        await client(server).access_token()
        first, second = server.authorise_requests
        assert first["state"] != second["state"]
        assert first["code_challenge"] != second["code_challenge"]

    async def test_a_pre_registered_client_skips_registration(self) -> None:
        server = FakeAuthServer()
        server.clients["pre-registered"] = {
            "redirect_uris": ["http://127.0.0.1:22420/callback"],
            "client_secret": "zz-pre-registered-secret-zz",
        }
        oauth = client(
            server,
            client_id="pre-registered",
            client_secret="secret://docs_client",
            issuer=ISSUER,
            secrets={"docs_client": "zz-pre-registered-secret-zz"},
        )
        await oauth.access_token()
        assert not any(r.url.path.endswith("/register") for r in server.requests)
        exchange = server.token_requests[-1]
        assert exchange["client_id"] == "pre-registered"
        assert exchange["client_secret"] == "zz-pre-registered-secret-zz"

    async def test_a_secret_issued_at_registration_is_used(self) -> None:
        server = FakeAuthServer(secret_client=True)
        store = MemoryTokenStore()
        await client(server, store).access_token()
        (record,) = server.clients.values()
        assert server.token_requests[-1]["client_secret"] == record["client_secret"]
        registration = await store.get("mcp:docs:client")
        assert registration is not None
        assert registration.refresh_token is not None
        assert registration.refresh_token.get_secret_value() == record["client_secret"]

    async def test_a_registration_for_another_redirect_is_replaced(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        await client(server, store).access_token()
        await store.delete("mcp:docs")
        other = server.sign_in_hook(redirect_uri="http://127.0.0.1:22421/callback")
        await client(server, store, hook=other).access_token()
        assert len(server.clients) == 2

    async def test_metadata_at_the_root_is_found(self) -> None:
        server = FakeAuthServer(metadata_at="root")
        await client(server).access_token()

    async def test_without_a_hook_sign_in_is_required(self) -> None:
        with pytest.raises(McpAuthError) as caught:
            await client(FakeAuthServer(), hook=None).access_token()
        assert_refused(caught.value, OAuthFailure.SIGN_IN_REQUIRED)

    async def test_without_interaction_sign_in_is_required(self) -> None:
        server = FakeAuthServer()
        hook = server.sign_in_hook()
        with pytest.raises(McpAuthError) as caught:
            await client(server, hook=hook).access_token(interactive=False)
        assert_refused(caught.value, OAuthFailure.SIGN_IN_REQUIRED)
        assert not hook.requests


class TestCallbackChecks:
    async def _refused(self, tamper: Any, server: FakeAuthServer | None = None) -> McpAuthError:
        server = server or FakeAuthServer()
        store = MemoryTokenStore()
        oauth = client(server, store, hook=server.sign_in_hook(tamper=tamper))
        with pytest.raises(McpAuthError) as caught:
            await oauth.access_token()
        # Nothing was exchanged and nothing stored.
        assert not [f for f in server.token_requests if f["grant_type"] == "authorization_code"]
        assert await store.get("mcp:docs") is None
        return caught.value

    async def test_a_changed_state_is_refused(self) -> None:
        error = await self._refused(lambda url: replace_param(url, "state", "forged-state-value"))
        assert_refused(error, OAuthFailure.STATE_MISMATCH, "forged-state-value")

    async def test_a_missing_state_is_refused(self) -> None:
        error = await self._refused(lambda url: replace_param(url, "state", None))
        assert_refused(error, OAuthFailure.STATE_MISMATCH)

    async def test_a_repeated_state_is_refused(self) -> None:
        error = await self._refused(lambda url: url + "&state=forged-state-value")
        assert_refused(error, OAuthFailure.STATE_MISMATCH)

    @pytest.mark.parametrize(
        "elsewhere",
        [
            "http://127.0.0.1:22421/callback",  # another port
            "http://127.0.0.1:22420/other",  # another path
            "http://192.0.2.10:22420/callback",  # another host
            "https://127.0.0.1:22420/callback",  # another scheme
        ],
    )
    async def test_a_callback_to_another_address_is_refused(self, elsewhere: str) -> None:
        def move(url: str) -> str:
            return elsewhere + "?" + urlsplit(url).query

        error = await self._refused(move)
        assert_refused(error, OAuthFailure.REDIRECT_MISMATCH)

    async def test_an_error_callback_is_refused(self) -> None:
        def deny(url: str) -> str:
            url = replace_param(url, "code", None)
            return replace_param(url, "error", "access_denied")

        error = await self._refused(deny)
        assert_refused(error, OAuthFailure.AUTHORISATION_DENIED, "access_denied")

    async def test_another_issuer_in_the_callback_is_refused(self) -> None:
        error = await self._refused(lambda url: replace_param(url, "iss", "https://evil.example"))
        assert_refused(error, OAuthFailure.ISSUER_MISMATCH, "evil.example")

    async def test_a_missing_issuer_is_refused_when_the_server_promises_one(self) -> None:
        error = await self._refused(lambda url: replace_param(url, "iss", None))
        assert_refused(error, OAuthFailure.ISSUER_MISMATCH)

    async def test_a_missing_issuer_is_accepted_when_the_server_does_not_send_one(self) -> None:
        await client(FakeAuthServer(iss_parameter=False)).access_token()

    async def test_a_missing_code_is_refused(self) -> None:
        error = await self._refused(lambda url: replace_param(url, "code", None))
        assert_refused(error, OAuthFailure.TOKEN_EXCHANGE)

    @pytest.mark.parametrize(
        "redirect_uri",
        [
            "http://203.0.113.5:22420/callback",  # plain http off loopback
            "https://app.example.com/callback#fragment",
            "app.example.com/callback",
        ],
    )
    async def test_an_unsafe_redirect_uri_is_refused_before_anything(
        self, redirect_uri: str
    ) -> None:
        server = FakeAuthServer()
        hook = server.sign_in_hook(redirect_uri=redirect_uri)
        with pytest.raises(McpAuthError) as caught:
            await client(server, hook=hook).access_token()
        assert_refused(caught.value, OAuthFailure.REDIRECT_MISMATCH)
        assert not server.clients
        assert not hook.requests

    async def test_a_hook_that_fails_is_a_clean_sign_in_failure(self) -> None:
        class Broken:
            redirect_uri = "http://127.0.0.1:22420/callback"

            async def sign_in(self, request: SignInRequest) -> SignInResult:
                raise RuntimeError("zz-hook-detail-zz")

        with pytest.raises(McpAuthError) as caught:
            await client(FakeAuthServer(), hook=Broken()).access_token()
        assert_refused(caught.value, OAuthFailure.SIGN_IN_FAILED, "zz-hook-detail-zz")

    async def test_a_cancelled_hook_propagates_the_cancellation(self) -> None:
        class Waiting:
            redirect_uri = "http://127.0.0.1:22420/callback"

            async def sign_in(self, request: SignInRequest) -> SignInResult:
                await asyncio.sleep(30)
                raise AssertionError("unreachable")

        task = asyncio.create_task(client(FakeAuthServer(), hook=Waiting()).access_token())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestDiscoveryChecks:
    async def _refused(self, server: FakeAuthServer) -> McpAuthError:
        with pytest.raises(McpAuthError) as caught:
            await client(server).access_token()
        return caught.value

    @pytest.mark.parametrize("iss_parameter", [True, False])
    async def test_another_issuer_in_the_metadata_is_refused(self, iss_parameter: bool) -> None:
        # Refused at discovery, before registering: not left to the callback's check,
        # which a server that sends no issuer parameter would never reach.
        server = FakeAuthServer(
            server_metadata={"issuer": "https://evil.example"}, iss_parameter=iss_parameter
        )
        error = await self._refused(server)
        assert_refused(error, OAuthFailure.ISSUER_MISMATCH, "evil.example")
        assert not server.clients
        assert not server.authorise_requests

    @pytest.mark.parametrize(
        "resource",
        [
            "https://evil.example/mcp",  # another origin
            "https://mcp.example.com/other",  # another path
            "https://mcp.example.com/mcp/deeper",  # narrower than the server
            "http://mcp.example.com/mcp",  # another scheme
            "https://mcp.example.com:8443/mcp",  # another port
        ],
    )
    async def test_metadata_for_another_resource_is_refused(self, resource: str) -> None:
        error = await self._refused(FakeAuthServer(resource_metadata={"resource": resource}))
        assert_refused(error, OAuthFailure.AUDIENCE_MISMATCH)

    @pytest.mark.parametrize(
        "resource", ["https://mcp.example.com/", "https://MCP.example.com/mcp/"]
    )
    async def test_metadata_for_the_server_or_its_parent_is_accepted(self, resource: str) -> None:
        await client(FakeAuthServer(resource_metadata={"resource": resource})).access_token()

    async def test_a_resource_with_no_metadata_is_refused(self) -> None:
        error = await self._refused(FakeAuthServer(metadata_at="nowhere"))
        assert_refused(error, OAuthFailure.DISCOVERY)

    async def test_metadata_naming_no_authorisation_server_is_refused(self) -> None:
        error = await self._refused(FakeAuthServer(resource_metadata={"authorization_servers": []}))
        assert_refused(error, OAuthFailure.DISCOVERY)

    @pytest.mark.parametrize("methods", [["plain"], None])
    async def test_a_server_without_s256_is_refused(self, methods: Any) -> None:
        server = FakeAuthServer(server_metadata={"code_challenge_methods_supported": methods})
        error = await self._refused(server)
        assert_refused(error, OAuthFailure.PKCE_UNSUPPORTED)
        assert not server.clients

    @pytest.mark.parametrize("endpoint", ["authorization_endpoint", "token_endpoint"])
    async def test_a_plain_http_endpoint_is_refused(self, endpoint: str) -> None:
        error = await self._refused(
            FakeAuthServer(server_metadata={endpoint: "http://auth.example.com/x"})
        )
        assert_refused(error, OAuthFailure.INSECURE_ENDPOINT)

    async def test_a_plain_http_authorisation_server_is_refused(self) -> None:
        error = await self._refused(
            FakeAuthServer(resource_metadata={"authorization_servers": ["http://auth.example.com"]})
        )
        assert_refused(error, OAuthFailure.INSECURE_ENDPOINT)

    async def test_no_registration_endpoint_without_a_client_id_is_refused(self) -> None:
        error = await self._refused(FakeAuthServer(server_metadata={"registration_endpoint": None}))
        assert_refused(error, OAuthFailure.REGISTRATION)

    async def test_a_token_type_other_than_bearer_is_refused(self) -> None:
        error = await self._refused(FakeAuthServer(token_type="mac"))
        assert_refused(error, OAuthFailure.TOKEN_EXCHANGE)

    async def test_an_unreachable_server_is_a_clean_discovery_failure(self) -> None:
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("zz-network-detail-zz", request=request)

        oauth = OAuthClient(
            settings(),
            resource=RESOURCE,
            store=MemoryTokenStore(),
            sign_in=FakeAuthServer().sign_in_hook(),
            secrets=LiteralSecretResolver({}),
            http_transport=httpx.MockTransport(unreachable),
        )
        with pytest.raises(McpAuthError) as caught:
            await oauth.access_token()
        assert_refused(caught.value, OAuthFailure.DISCOVERY, "zz-network-detail-zz")


class TestRefresh:
    async def test_an_expired_token_is_refreshed_and_the_refresh_token_rotates(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        oauth = client(server, store, clock=clock)
        first = await oauth.access_token()
        held = await store.get("mcp:docs")
        assert held is not None and held.refresh_token is not None
        old_refresh = held.refresh_token.get_secret_value()

        clock.advance(3600)  # inside the refresh margin
        second = await oauth.access_token(interactive=False)
        assert second.get_secret_value() != first.get_secret_value()
        rotated = await store.get("mcp:docs")
        assert rotated is not None and rotated.refresh_token is not None
        new_refresh = rotated.refresh_token.get_secret_value()
        assert new_refresh != old_refresh
        assert old_refresh in server.revoked
        assert server.token_requests[-1]["refresh_token"] == old_refresh
        assert server.token_requests[-1]["resource"] == RESOURCE

        # The rotated token is the one used next; the revoked one would be refused.
        clock.advance(3600)
        third = await oauth.access_token(interactive=False)
        assert third.get_secret_value() not in {first.get_secret_value(), second.get_secret_value()}
        assert server.token_requests[-1]["refresh_token"] == new_refresh

    async def test_a_token_inside_the_refresh_margin_is_refreshed_early(self) -> None:
        server = FakeAuthServer()
        clock = Clock()
        oauth = client(server, clock=clock, refresh_margin=60)
        first = await oauth.access_token()
        clock.advance(3570)  # 30 seconds left: still valid, but inside the margin
        second = await oauth.access_token(interactive=False)
        assert second.get_secret_value() != first.get_secret_value()
        assert server.token_requests[-1]["grant_type"] == "refresh_token"

    async def test_a_token_well_inside_its_life_is_not_refreshed(self) -> None:
        server = FakeAuthServer()
        clock = Clock()
        oauth = client(server, clock=clock)
        first = await oauth.access_token()
        clock.advance(3000)  # 600 seconds left, the margin is 60
        assert (await oauth.access_token()).get_secret_value() == first.get_secret_value()
        assert len([f for f in server.token_requests if f["grant_type"] == "refresh_token"]) == 0

    async def test_a_server_that_does_not_rotate_keeps_the_refresh_token(self) -> None:
        server = FakeAuthServer(rotate=False)
        store = MemoryTokenStore()
        clock = Clock()
        oauth = client(server, store, clock=clock)
        await oauth.access_token()
        held = await store.get("mcp:docs")
        assert held is not None and held.refresh_token is not None
        clock.advance(3600)
        await oauth.access_token(interactive=False)
        kept = await store.get("mcp:docs")
        assert kept is not None and kept.refresh_token is not None
        assert kept.refresh_token.get_secret_value() == held.refresh_token.get_secret_value()

    async def test_a_revoked_refresh_token_falls_back_to_sign_in(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        hook = server.sign_in_hook()
        oauth = client(server, store, hook=hook, clock=clock)
        await oauth.access_token()
        server.revoked.update(server.refresh_tokens)
        clock.advance(3600)
        with pytest.raises(McpAuthError) as caught:
            await oauth.access_token(interactive=False)
        assert_refused(caught.value, OAuthFailure.SIGN_IN_REQUIRED)
        assert await store.get("mcp:docs") is None  # the dead tokens are dropped
        await oauth.access_token(interactive=True)
        assert len(hook.requests) == 2

    async def test_refresh_after_a_restart_uses_the_stored_registration(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        clock = Clock()
        await client(server, store, clock=clock).access_token()
        clock.advance(3600)
        restarted = client(server, store, hook=None, clock=clock)
        await restarted.access_token(interactive=False)
        assert server.token_requests[-1]["grant_type"] == "refresh_token"

    async def test_a_rejected_token_is_refreshed_once_under_concurrency(self) -> None:
        server = FakeAuthServer()

        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)  # every request yields, so the callers interleave
            return server.handle(request)

        oauth = OAuthClient(
            settings(),
            resource=RESOURCE,
            store=MemoryTokenStore(),
            sign_in=server.sign_in_hook(),
            secrets=LiteralSecretResolver({}),
            http_transport=httpx.MockTransport(slow),
            clock=Clock(),
        )
        stale = await oauth.access_token()
        before = len(server.token_requests)
        fresh = await asyncio.gather(*(oauth.refreshed(stale) for _ in range(8)))
        refreshes = server.token_requests[before:]
        assert len(refreshes) == 1
        assert len({t.get_secret_value() for t in fresh}) == 1
        assert fresh[0].get_secret_value() != stale.get_secret_value()

    async def test_a_token_without_expiry_is_kept_until_rejected(self) -> None:
        server = FakeAuthServer(expires_in=None)
        clock = Clock()
        oauth = client(server, clock=clock)
        first = await oauth.access_token()
        clock.advance(10**6)
        assert (await oauth.access_token()).get_secret_value() == first.get_secret_value()

    async def test_sign_out_drops_the_tokens(self) -> None:
        server = FakeAuthServer()
        store = MemoryTokenStore()
        oauth = client(server, store)
        await oauth.access_token()
        await oauth.sign_out()
        assert await store.get("mcp:docs") is None


class TestHttpxAuth:
    async def test_a_401_refreshes_once_and_retries(self) -> None:
        server = FakeAuthServer()
        oauth = client(server)
        stale = await oauth.access_token()
        seen: list[str] = []

        def resource(request: httpx.Request) -> httpx.Response:
            sent = request.headers.get("authorization", "")
            seen.append(sent)
            if sent == f"Bearer {stale.get_secret_value()}":
                return httpx.Response(401)
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(
            auth=oauth.httpx_auth(), transport=httpx.MockTransport(resource)
        ) as http:
            response = await http.get(RESOURCE)
        assert response.status_code == 200
        assert len(seen) == 2
        assert seen[1].startswith("Bearer ")
        assert seen[1] != seen[0]

    async def test_a_second_401_is_returned_not_looped(self) -> None:
        server = FakeAuthServer()
        oauth = client(server)
        await oauth.access_token()
        calls = 0

        def resource(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401)

        async with httpx.AsyncClient(
            auth=oauth.httpx_auth(), transport=httpx.MockTransport(resource)
        ) as http:
            response = await http.get(RESOURCE)
        assert response.status_code == 401
        assert calls == 2


class FailingStore:
    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.inner = MemoryTokenStore()

    async def get(self, key: str) -> TokenSet | None:
        if self.operation == "get":
            raise StoreError("token store", "get a token set", "OperationalError")
        return await self.inner.get(key)

    async def put(self, key: str, tokens: TokenSet) -> None:
        if self.operation == "put":
            raise RuntimeError(f"driver text with {tokens.access_token.get_secret_value()}")
        await self.inner.put(key, tokens)

    async def delete(self, key: str) -> None:
        await self.inner.delete(key)


class TestStoreFailures:
    @pytest.mark.parametrize("operation", ["get", "put"])
    async def test_a_failing_store_is_a_clean_store_failure(self, operation: str) -> None:
        server = FakeAuthServer()
        with pytest.raises(McpAuthError) as caught:
            await client(server, FailingStore(operation)).access_token()
        assert_refused(caught.value, OAuthFailure.STORE, "mcp:docs", *server.issued())


class TestSecretsStayOut:
    async def test_no_token_code_or_secret_reaches_a_log_or_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        server = FakeAuthServer(secret_client=True)
        store = MemoryTokenStore()
        clock = Clock()
        oauth = client(server, store, clock=clock)
        await oauth.access_token()
        clock.advance(3600)
        await oauth.access_token()
        errors: list[str] = []
        for tamper in (
            lambda url: replace_param(url, "state", "x" * 43),
            lambda url: replace_param(url, "iss", "https://evil.example"),
        ):
            await store.delete("mcp:docs")
            with pytest.raises(McpAuthError) as caught:
                await client(server, store, hook=server.sign_in_hook(tamper=tamper)).access_token()
            errors += [str(caught.value), repr(caught.value)]
        verifiers = {f["code_verifier"] for f in server.token_requests if "code_verifier" in f}
        hidden = server.issued() | verifiers
        assert hidden
        logged = [r.getMessage() + repr(r.args) for r in caplog.records]
        for text in logged + errors + [repr(oauth)]:
            for value in hidden:
                assert value not in text

    async def test_the_token_set_never_shows_its_values(self) -> None:
        tokens = TokenSet(access_token=SecretStr("zz-a-zz"), refresh_token=SecretStr("zz-r-zz"))
        assert "zz-a-zz" not in repr(tokens)
        assert "zz-r-zz" not in str(tokens)


def test_the_fake_server_really_checks_the_verifier() -> None:
    # Guards against a vacuous PKCE test: the fake must refuse a wrong verifier.
    server = FakeAuthServer()
    hook = server.sign_in_hook()
    assert isinstance(hook, FakeSignIn)
    server.clients["c"] = {"redirect_uris": [hook.redirect_uri]}
    callback = server.authorise(
        f"{ISSUER}/authorize?"
        + urlencode(
            {
                "client_id": "c",
                "redirect_uri": hook.redirect_uri,
                "response_type": "code",
                "code_challenge": s256("right-verifier" * 4),
                "code_challenge_method": "S256",
                "resource": RESOURCE,
            }
        )
    )
    code = parse_qs(urlsplit(callback).query)["code"][0]
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": "c",
        "redirect_uri": hook.redirect_uri,
        "resource": RESOURCE,
        "code_verifier": "wrong-verifier" * 4,
    }
    response = server.handle(
        httpx.Request("POST", f"{ISSUER}/token", content=urlencode(form).encode("ascii"))
    )
    assert response.status_code == 400
