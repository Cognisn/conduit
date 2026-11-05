"""The loopback sign-in hook, and the whole OAuth sign-in over loopback HTTP.

The browser is simulated by a synchronous httpx request that follows redirects
(the hook runs its opener in a worker thread). Every server binds 127.0.0.1 inside
this worktree's slot.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.responses import JSONResponse

from conduit.core.errors import ConfigurationError
from conduit.core.secrets import LiteralSecretResolver
from conduit.mcp.client import McpConnection
from conduit.mcp.errors import McpAuthError, OAuthFailure
from conduit.mcp.settings import McpServerSettings
from conduit.mcp.signin import LoopbackSignIn, SignInHook, SignInRequest
from conduit.stores.memory import MemoryTokenStore
from tests.unit.mcp.fake_auth import FakeAuthServer
from tests.unit.mcp.servers import LoopbackServer, build_server, port, serving


def browser(url: str) -> bool:
    """The person approves: the browser follows every redirect to the callback."""
    with httpx.Client(follow_redirects=True, timeout=10, trust_env=False) as client:
        client.get(url)
    return True


def visit(path: str) -> Any:
    def opener(url: str) -> bool:
        with httpx.Client(timeout=10, trust_env=False) as client:
            client.get(f"http://127.0.0.1:{port('callback')}{path}")
        return True

    return opener


def request(url: str = "https://auth.example.com/authorize?x=1&state=s1") -> SignInRequest:
    return SignInRequest(
        server="docs",
        authorization_url=url,
        redirect_uri=f"http://127.0.0.1:{port('callback')}/callback",
    )


def port_is_free(number: int) -> bool:
    """Nothing listens on ``number`` any more (a refused connection).

    Binding would be the wrong probe: on POSIX a just-closed port stays unbindable
    while its connections sit in TIME_WAIT.
    """
    try:
        with socket.create_connection(("127.0.0.1", number), timeout=1):
            return False
    except OSError:
        return True


class TestLoopbackSignIn:
    def test_it_is_a_sign_in_hook(self) -> None:
        hook = LoopbackSignIn(port("callback"))
        assert isinstance(hook, SignInHook)
        assert hook.redirect_uri == f"http://127.0.0.1:{port('callback')}/callback"

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "example.com", ""])  # noqa: S104
    def test_it_listens_on_loopback_only(self, host: str) -> None:
        with pytest.raises(ConfigurationError):
            LoopbackSignIn(port("callback"), host=host)

    @pytest.mark.parametrize("path", ["callback", "/callback?x=1", "/cb#frag"])
    def test_a_bad_path_is_refused(self, path: str) -> None:
        with pytest.raises(ConfigurationError):
            LoopbackSignIn(port("callback"), path=path)

    async def test_the_redirect_completes_the_sign_in(self) -> None:
        opened: list[str] = []

        def opener(url: str) -> bool:
            opened.append(url)
            return bool(visit("/callback?code=zz-code-zz&state=s1")(url))

        hook = LoopbackSignIn(port("callback"), open_url=opener, timeout=10)
        result = await hook.sign_in(request())
        assert opened == ["https://auth.example.com/authorize?x=1&state=s1"]
        assert result.callback_url == (
            f"http://127.0.0.1:{port('callback')}/callback?code=zz-code-zz&state=s1"
        )
        assert "zz-code-zz" not in repr(result)
        assert port_is_free(port("callback"))

    async def test_another_path_does_not_complete_it(self) -> None:
        def opener(url: str) -> bool:
            with httpx.Client(timeout=10, trust_env=False) as client:
                base = f"http://127.0.0.1:{port('callback')}"
                assert client.get(f"{base}/favicon.ico").status_code == 404
                assert client.get(f"{base}/callback?code=c&state=s1").status_code == 200
            return True

        result = await LoopbackSignIn(port("callback"), open_url=opener, timeout=10).sign_in(
            request()
        )
        assert urlsplit(result.callback_url).path == "/callback"

    async def test_no_redirect_times_out_and_frees_the_port(self) -> None:
        hook = LoopbackSignIn(port("callback"), open_url=lambda url: True, timeout=0.3)
        with pytest.raises(TimeoutError):
            await hook.sign_in(request())
        assert port_is_free(port("callback"))

    async def test_without_a_browser_the_address_is_shown(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async def person_later() -> None:
            await asyncio.sleep(0.2)
            await asyncio.to_thread(visit("/callback?code=c&state=s1"), "")

        hook = LoopbackSignIn(port("callback"), open_url=lambda url: False, timeout=10)
        visiting = asyncio.create_task(person_later())
        await hook.sign_in(request("https://auth.example.com/authorize?shown=1&state=s1"))
        await visiting
        assert "https://auth.example.com/authorize?shown=1&state=s1" in capsys.readouterr().err

    async def test_a_callback_without_the_state_sent_does_not_end_it(self) -> None:
        answers: list[int] = []

        def opener(url: str) -> bool:
            with httpx.Client(timeout=10, trust_env=False) as client:
                base = f"http://127.0.0.1:{port('callback')}/callback"
                answers.append(client.get(f"{base}?code=forged&state=forged").status_code)
                answers.append(client.get(f"{base}?code=forged").status_code)
                answers.append(client.get(f"{base}?code=real&state=s1").status_code)
            return True

        hook = LoopbackSignIn(port("callback"), open_url=opener, timeout=10)
        result = await hook.sign_in(request())
        assert answers == [404, 404, 200]
        assert "code=real" in result.callback_url

    async def test_an_error_callback_does_not_report_success(self) -> None:
        pages: list[str] = []

        def opener(url: str) -> bool:
            with httpx.Client(timeout=10, trust_env=False) as client:
                base = f"http://127.0.0.1:{port('callback')}/callback"
                pages.append(client.get(f"{base}?error=access_denied&state=s1").text)
            return True

        result = await LoopbackSignIn(port("callback"), open_url=opener, timeout=10).sign_in(
            request()
        )
        assert "error=access_denied" in result.callback_url
        assert pages and "complete" not in pages[0].lower()
        assert "did not" in pages[0].lower()

    async def test_an_idle_connection_does_not_hold_the_sign_in_past_its_timeout(
        self,
    ) -> None:
        def idle(url: str) -> bool:
            sockets.append(socket.create_connection(("127.0.0.1", port("callback"))))
            return True

        sockets: list[socket.socket] = []
        hook = LoopbackSignIn(port("callback"), open_url=idle, timeout=0.5)
        started = asyncio.get_running_loop().time()
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(hook.sign_in(request()), timeout=20)
            assert asyncio.get_running_loop().time() - started < 5
        finally:
            for held in sockets:
                held.close()
        assert port_is_free(port("callback"))

    async def test_an_idle_connection_does_not_block_the_callback(self) -> None:
        sockets: list[socket.socket] = []

        def idle_then_visit(url: str) -> bool:
            sockets.append(socket.create_connection(("127.0.0.1", port("callback"))))
            return bool(visit("/callback?code=c&state=s1")(url))

        hook = LoopbackSignIn(port("callback"), open_url=idle_then_visit, timeout=10)
        started = asyncio.get_running_loop().time()
        try:
            result = await asyncio.wait_for(hook.sign_in(request()), timeout=20)
            assert asyncio.get_running_loop().time() - started < 5
        finally:
            for held in sockets:
                held.close()
        assert "code=c" in result.callback_url

    async def test_cancellation_closes_the_listener(self) -> None:
        hook = LoopbackSignIn(port("callback"), open_url=lambda url: True, timeout=30)
        task = asyncio.create_task(hook.sign_in(request()))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert port_is_free(port("callback"))


def protect(app: Any, fake: FakeAuthServer) -> Any:
    """``app`` accepting only the fake server's live access tokens; serves its metadata."""

    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            path = scope["path"]
            if path.startswith("/.well-known/oauth-protected-resource"):
                origin = fake.resource.rsplit("/mcp", 1)[0]
                answer = fake.handle(httpx.Request("GET", f"{origin}{path}"))
                await JSONResponse(answer.json(), status_code=answer.status_code)(
                    scope, receive, send
                )
                return
            sent = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
            token = sent.removeprefix("Bearer ")
            if not sent.startswith("Bearer ") or token not in fake.access_tokens:
                await JSONResponse({"error": "invalid_token"}, status_code=401)(
                    scope, receive, send
                )
                return
        await app(scope, receive, send)

    return guarded


@pytest.fixture(scope="module")
def fake() -> FakeAuthServer:
    return FakeAuthServer(
        resource=f"http://127.0.0.1:{port('protected')}/mcp",
        issuer=f"http://127.0.0.1:{port('auth')}",
    )


@pytest.fixture(scope="module")
def servers(fake: FakeAuthServer) -> Iterator[tuple[LoopbackServer, LoopbackServer]]:
    app = build_server("protected", host="127.0.0.1").streamable_http_app()
    with serving(fake.asgi(), "auth") as auth, serving(protect(app, fake), "protected") as mcp:
        yield auth, mcp


def oauth_settings(fake: FakeAuthServer) -> McpServerSettings:
    return McpServerSettings.model_validate(
        {
            "name": "protected",
            "transport": {"type": "streamable_http", "url": fake.resource},
            "auth": {"type": "oauth", "scopes": ["read"]},
            "task_classes": ["research"],
            "connect_timeout": 10,
            "close_timeout": 5,
        }
    )


class TestOAuthOverLoopback:
    async def test_sign_in_connect_call_and_refresh_after_a_rejection(
        self, fake: FakeAuthServer, servers: Any
    ) -> None:
        store = MemoryTokenStore()
        hook = LoopbackSignIn(port("callback"), open_url=browser, timeout=20)
        conn = McpConnection(
            oauth_settings(fake), secrets=LiteralSecretResolver({}), tokens=store, sign_in=hook
        )
        async with conn:
            assert (await conn.call_tool("echo", {"text": "signed in"})).text == "signed in"
            held = await store.get("mcp:protected")
            assert held is not None and held.refresh_token is not None
            first_refresh = held.refresh_token.get_secret_value()
            # The server stops accepting the access token: the next request is refused
            # with 401, refreshed once, and retried.
            fake.access_tokens.discard(held.access_token.get_secret_value())
            assert (await conn.call_tool("echo", {"text": "refreshed"})).text == "refreshed"
        rotated = await store.get("mcp:protected")
        assert rotated is not None and rotated.refresh_token is not None
        assert rotated.refresh_token.get_secret_value() != first_refresh
        assert first_refresh in fake.revoked
        (authorise,) = [
            r for r in fake.authorise_requests if r["redirect_uri"] == hook.redirect_uri
        ]
        assert authorise["resource"] == fake.resource
        assert port_is_free(port("callback"))

        # A second process with the same store signs in without the person.
        quiet = McpConnection(
            oauth_settings(fake), secrets=LiteralSecretResolver({}), tokens=store, sign_in=None
        )
        async with quiet:
            assert (await quiet.call_tool("add", {"a": 1, "b": 1})).text == "2"

    async def test_no_tokens_and_no_hook_is_refused_before_connecting(
        self, fake: FakeAuthServer, servers: Any
    ) -> None:
        conn = McpConnection(
            oauth_settings(fake), secrets=LiteralSecretResolver({}), tokens=MemoryTokenStore()
        )
        with pytest.raises(McpAuthError) as caught:
            await conn.connect()
        assert caught.value.reason is OAuthFailure.SIGN_IN_REQUIRED
        assert caught.value.__context__ is None
        assert not conn.connected

    async def test_an_oauth_server_needs_a_token_store(self, fake: FakeAuthServer) -> None:
        with pytest.raises(ConfigurationError):
            McpConnection(oauth_settings(fake), secrets=LiteralSecretResolver({}))

    async def test_the_callback_carries_the_state_sent(
        self, fake: FakeAuthServer, servers: Any
    ) -> None:
        seen: list[str] = []

        def watching(url: str) -> bool:
            seen.append(parse_qs(urlsplit(url).query)["state"][0])
            return browser(url)

        store = MemoryTokenStore()
        hook = LoopbackSignIn(port("callback"), open_url=watching, timeout=20)
        conn = McpConnection(
            oauth_settings(fake), secrets=LiteralSecretResolver({}), tokens=store, sign_in=hook
        )
        async with conn:
            pass
        assert len(seen) == 1 and len(seen[0]) >= 43
