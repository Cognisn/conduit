"""A synthetic OAuth 2.1 authorisation server and protected-resource metadata, for tests.

It enforces what a real one would, so the client's guards are tested against a
server that refuses what it should: registered redirect URIs match exactly, the PKCE
verifier must hash to the challenge, codes are single use and bound to the client,
redirect URI and resource, and refresh tokens rotate (a used one is revoked).

It answers httpx requests in-process (``transport()``, an ``httpx.MockTransport``),
or over loopback as an ASGI app (``asgi()``). Every value it issues is synthetic.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from conduit.mcp.signin import SignInRequest, SignInResult

RESOURCE = "https://mcp.example.com/mcp"
ISSUER = "https://auth.example.com"


def s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@dataclass
class Grant:
    client_id: str
    redirect_uri: str
    challenge: str
    resource: str | None
    scope: str | None


@dataclass
class FakeAuthServer:
    resource: str = RESOURCE
    issuer: str = ISSUER
    rotate: bool = True
    iss_parameter: bool = True
    expires_in: int | None = 3600
    token_type: str = "Bearer"
    #: Overrides merged into the protected-resource and authorisation-server metadata.
    resource_metadata: dict[str, Any] = field(default_factory=dict)
    server_metadata: dict[str, Any] = field(default_factory=dict)
    #: Where the protected-resource metadata is served: ``path``, ``root`` or nowhere.
    metadata_at: str = "path"
    secret_client: bool = False
    requests: list[httpx.Request] = field(default_factory=list)
    clients: dict[str, dict[str, Any]] = field(default_factory=dict)
    grants: dict[str, Grant] = field(default_factory=dict)
    access_tokens: set[str] = field(default_factory=set)
    refresh_tokens: dict[str, str] = field(default_factory=dict)  # token -> client id
    revoked: set[str] = field(default_factory=set)
    token_requests: list[dict[str, str]] = field(default_factory=list)
    authorise_requests: list[dict[str, str]] = field(default_factory=list)
    _counter: Any = field(default_factory=lambda: itertools.count(1))

    # -- Issued values ---------------------------------------------------------------

    def _new(self, kind: str) -> str:
        return f"zz-{kind}-{next(self._counter):04d}-synthetic-zz"

    def issued(self) -> set[str]:
        """Every secret value this server has issued (tokens, codes, client secrets)."""
        values = set(self.access_tokens) | set(self.refresh_tokens) | set(self.grants)
        values |= self.revoked
        values |= {c["client_secret"] for c in self.clients.values() if c.get("client_secret")}
        return values

    # -- Metadata --------------------------------------------------------------------

    def _prm_paths(self) -> set[str]:
        path = urlsplit(self.resource).path.rstrip("/")
        well_known = "/.well-known/oauth-protected-resource"
        return {"path": {f"{well_known}{path}"}, "root": {well_known}}.get(self.metadata_at, set())

    def _prm(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "resource": self.resource,
            "authorization_servers": [self.issuer],
            "scopes_supported": ["read", "write"],
        }
        data.update(self.resource_metadata)
        return {k: v for k, v in data.items() if v is not None}

    def _asm(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "registration_endpoint": f"{self.issuer}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
            "authorization_response_iss_parameter_supported": self.iss_parameter,
        }
        data.update(self.server_metadata)
        return {k: v for k, v in data.items() if v is not None}

    # -- The browser's half: the user approves -------------------------------------

    def authorise(self, url: str) -> str:
        """What the browser is redirected to after the user approves ``url``."""
        parts = urlsplit(url)
        params = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.authorise_requests.append(params)
        client = self.clients.get(params.get("client_id", ""))
        redirect_uri = params.get("redirect_uri", "")
        if client is None or redirect_uri not in client["redirect_uris"]:
            raise AssertionError("the fake server refuses an unregistered client or redirect")
        if params.get("response_type") != "code" or params.get("code_challenge_method") != "S256":
            raise AssertionError("the fake server needs the code flow with S256")
        code = self._new("code")
        self.grants[code] = Grant(
            client_id=params["client_id"],
            redirect_uri=redirect_uri,
            challenge=params.get("code_challenge", ""),
            resource=params.get("resource"),
            scope=params.get("scope"),
        )
        answer: dict[str, str] = {"code": code}
        if "state" in params:
            answer["state"] = params["state"]
        if self.iss_parameter:
            answer["iss"] = self.issuer
        separator = "&" if "?" in redirect_uri else "?"
        return f"{redirect_uri}{separator}{urlencode(answer)}"

    def sign_in_hook(
        self,
        redirect_uri: str = "http://127.0.0.1:22420/callback",
        tamper: Callable[[str], str] | None = None,
    ) -> FakeSignIn:
        return FakeSignIn(self, redirect_uri, tamper)

    # -- Requests ----------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = request.url
        origin = f"{url.scheme}://{url.netloc.decode('ascii')}"
        path = url.path
        resource_origin = "{0.scheme}://{0.netloc}".format(urlsplit(self.resource))
        if origin == resource_origin and path in self._prm_paths() and request.method == "GET":
            return httpx.Response(200, json=self._prm())
        issuer = urlsplit(self.issuer)
        if origin == f"{issuer.scheme}://{issuer.netloc}":
            issuer_path = issuer.path.rstrip("/")
            if request.method == "GET" and path == (
                f"/.well-known/oauth-authorization-server{issuer_path}"
            ):
                return httpx.Response(200, json=self._asm())
            if path == f"{issuer_path}/register" and request.method == "POST":
                return self._register(request)
            if path == f"{issuer_path}/token" and request.method == "POST":
                return self._token(request)
        return httpx.Response(404, json={"error": "not_found"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _register(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        uris = body.get("redirect_uris")
        if not isinstance(uris, list) or not uris:
            return httpx.Response(400, json={"error": "invalid_redirect_uri"})
        client_id = self._new("client")
        record: dict[str, Any] = {"redirect_uris": uris, "request": body}
        answer: dict[str, Any] = {"client_id": client_id, "redirect_uris": uris}
        if self.secret_client:
            record["client_secret"] = self._new("client-secret")
            answer["client_secret"] = record["client_secret"]
            answer["token_endpoint_auth_method"] = "client_secret_post"
        else:
            answer["token_endpoint_auth_method"] = "none"
        self.clients[client_id] = record
        return httpx.Response(201, json=answer)

    def _client_ok(self, form: dict[str, str]) -> bool:
        client = self.clients.get(form.get("client_id", ""))
        if client is None:
            return False
        secret = client.get("client_secret")
        return secret is None or form.get("client_secret") == secret

    def _tokens(self, client_id: str) -> dict[str, Any]:
        access = self._new("access")
        refresh = self._new("refresh")
        self.access_tokens.add(access)
        self.refresh_tokens[refresh] = client_id
        answer: dict[str, Any] = {
            "access_token": access,
            "token_type": self.token_type,
            "refresh_token": refresh,
        }
        if self.expires_in is not None:
            answer["expires_in"] = self.expires_in
        return answer

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode("ascii")).items()}
        self.token_requests.append(form)
        if not self._client_ok(form):
            return httpx.Response(401, json={"error": "invalid_client"})
        if form.get("grant_type") == "authorization_code":
            grant = self.grants.pop(form.get("code", ""), None)
            if (
                grant is None
                or grant.client_id != form.get("client_id")
                or grant.redirect_uri != form.get("redirect_uri")
                or grant.resource != form.get("resource")
                or s256(form.get("code_verifier", "")) != grant.challenge
            ):
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(200, json=self._tokens(grant.client_id))
        if form.get("grant_type") == "refresh_token":
            refresh = form.get("refresh_token", "")
            owner = self.refresh_tokens.get(refresh)
            if owner is None or owner != form.get("client_id") or refresh in self.revoked:
                return httpx.Response(400, json={"error": "invalid_grant"})
            answer = self._tokens(owner)
            if self.rotate:
                self.revoked.add(refresh)
                del self.refresh_tokens[refresh]
            else:
                del self.refresh_tokens[answer.pop("refresh_token")]
            return httpx.Response(200, json=answer)
        return httpx.Response(400, json={"error": "unsupported_grant_type"})

    def asgi(self) -> Any:
        """The same server as an ASGI app, for loopback tests (the browser's half too)."""
        from starlette.requests import Request
        from starlette.responses import RedirectResponse, Response

        async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            if scope["type"] != "http":
                return
            request = Request(scope, receive)
            body = await request.body()
            if request.url.path.endswith("/authorize"):
                target = self.authorise(str(request.url))
                await RedirectResponse(target, status_code=302)(scope, receive, send)
                return
            answer = self.handle(
                httpx.Request(
                    request.method,
                    str(request.url),
                    headers=list(request.headers.items()),
                    content=body,
                )
            )
            await Response(
                answer.content,
                status_code=answer.status_code,
                media_type=answer.headers.get("content-type"),
            )(scope, receive, send)

        return app


@dataclass
class FakeSignIn:
    """A sign-in hook that approves at once, optionally tampering with the callback."""

    server: FakeAuthServer
    redirect_uri: str
    tamper: Callable[[str], str] | None = None
    requests: list[SignInRequest] = field(default_factory=list)

    async def sign_in(self, request: SignInRequest) -> SignInResult:
        self.requests.append(request)
        callback = self.server.authorise(request.authorization_url)
        if self.tamper is not None:
            callback = self.tamper(callback)
        return SignInResult(callback_url=callback)
