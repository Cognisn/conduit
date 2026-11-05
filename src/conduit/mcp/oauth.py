"""The OAuth 2.1 client for MCP servers (C14).

What it does, in order, for a server with no usable tokens:

1. Discovery. The protected-resource metadata (RFC 9728) is read from the MCP
   server's origin, at the well-known path with the server's path appended, then
   at the root. Its ``resource`` must be the server's own URL, or a parent of it
   on the same origin (the audience check), and it must name an authorisation
   server. That server's metadata (RFC 8414, then OpenID Connect discovery) must
   name the same issuer exactly (the issuer check), must offer PKCE with ``S256``,
   and every endpoint must be https (or http on loopback).
2. Registration. A pre-registered ``client_id`` is used as given; otherwise the
   client registers itself dynamically (RFC 7591) for the hook's redirect URI, and
   keeps the registration in the token store under ``mcp:<server>:client``.
3. Authorisation code with PKCE. A fresh ``state`` (256 bits) and verifier (512
   bits, 86 characters) for every sign-in; the challenge is the verifier's SHA-256.
   The request names the resource (RFC 8707). The sign-in hook takes the person
   through it and returns the callback URL, which must be at the registered
   redirect URI (scheme, host, port and path), carry exactly the ``state`` sent
   (compared in constant time), and name the issuer when the server sends one or
   promises to (RFC 9207). Only then is the code exchanged, with the verifier.
4. Tokens are held only in the product's ``TokenStore``, under ``mcp:<server>``.
   An access token near its expiry (``refresh_margin``) is refreshed with the
   refresh grant; a new refresh token replaces the old one (rotation), and a
   refused one (``invalid_grant``) drops the tokens so the person signs in again.
   Concurrent callers share one refresh.

Failures raise ``McpAuthError`` with the step's ``OAuthFailure``, unchained, and
never carry a token, a code, a verifier, a state, a client secret, a URL or the
server's text. Nothing here logs any of them. A store failure (the reference
stores raise ``StoreError``; a product's may raise anything) is ``STORE``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import secrets
from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import SecretStr

from conduit.core.backends import TokenSet, TokenStore
from conduit.core.secrets import SecretResolver
from conduit.mcp.errors import McpAuthError, OAuthFailure
from conduit.mcp.settings import McpServerSettings, OAuthAuth, check_endpoint_url
from conduit.mcp.signin import SignInHook, SignInRequest

__all__ = ["OAuthClient", "pkce_pair"]

logger = logging.getLogger("conduit.mcp")

_PRM_PATH = "/.well-known/oauth-protected-resource"
_ASM_PATH = "/.well-known/oauth-authorization-server"
_OIDC_PATH = "/.well-known/openid-configuration"
_REGISTRATION_TYPE = "registration:"
_DEAD_GRANT_ERRORS = frozenset({"invalid_grant", "invalid_client", "unauthorized_client"})
_SECRET_METHODS = frozenset({"client_secret_post", "client_secret_basic"})


def pkce_pair() -> tuple[str, str]:
    """A fresh PKCE verifier and its S256 challenge (RFC 7636)."""
    verifier = secrets.token_urlsafe(64)  # 86 characters from the unreserved set
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _address(url: str) -> tuple[str, str, int, str] | None:
    """Scheme, host, port and path, normalised; ``None`` if ``url`` cannot be read."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if not scheme or not host:
        return None
    return scheme, host, port or _default_port(scheme), parts.path


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _serves(metadata_resource: str, server_url: str) -> bool:
    """Whether metadata for ``metadata_resource`` covers ``server_url`` (the audience).

    The same origin, and the same path or a parent of it, ignoring a trailing slash.
    """
    named = _address(metadata_resource)
    ours = _address(server_url)
    if named is None or ours is None or named[:3] != ours[:3]:
        return False
    named_path = named[3].rstrip("/")
    our_path = ours[3].rstrip("/")
    return named_path == our_path or our_path.startswith(named_path + "/")


@dataclass(frozen=True, slots=True)
class _Metadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    iss_promised: bool


@dataclass(frozen=True, slots=True)
class _Client:
    client_id: str = field(repr=False)
    secret: str | None = field(repr=False)
    method: str


class OAuthClient:
    """OAuth tokens for one MCP server, held in the product's ``TokenStore``."""

    def __init__(
        self,
        settings: McpServerSettings,
        *,
        resource: str,
        store: TokenStore,
        sign_in: SignInHook | None,
        secrets: SecretResolver,
        http_transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(settings.auth, OAuthAuth):
            raise TypeError("OAuthClient needs settings with OAuth authentication")
        self._settings = settings
        self._auth: OAuthAuth = settings.auth
        self._server = settings.name
        self._resource = resource
        self._store = store
        self._hook = sign_in
        self._secrets = secrets
        self._http_transport = http_transport
        self._clock = clock or (lambda: datetime.now(UTC))
        self._metadata_cache: _Metadata | None = None
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"OAuthClient({self._server!r})"

    def _fail(self, reason: OAuthFailure) -> McpAuthError:
        logger.warning("MCP server %s: OAuth failed: %s", self._server, reason.value)
        return McpAuthError(self._server, reason)

    # -- The token store -------------------------------------------------------------

    async def _get(self, key: str) -> TokenSet | None:
        failed = False
        try:
            return await self._store.get(key)
        except Exception:
            failed = True
        if failed:
            raise self._fail(OAuthFailure.STORE)
        return None  # pragma: no cover - unreachable

    async def _put(self, key: str, tokens: TokenSet) -> None:
        failed = False
        try:
            await self._store.put(key, tokens)
        except Exception:
            failed = True
        if failed:
            raise self._fail(OAuthFailure.STORE)

    async def _delete(self, key: str) -> None:
        failed = False
        try:
            await self._store.delete(key)
        except Exception:
            failed = True
        if failed:
            raise self._fail(OAuthFailure.STORE)

    # -- HTTP ------------------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.connect_timeout),
            follow_redirects=False,
            trust_env=self._settings.use_environment_proxies,
            transport=self._http_transport,
        )

    async def _send(
        self,
        method: str,
        url: str,
        step: OAuthFailure,
        *,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, dict[str, Any] | None]:
        failed = False
        response: httpx.Response | None = None
        sent = {"Accept": "application/json", **(headers or {})}
        try:
            async with self._http() as http:
                response = await http.request(method, url, headers=sent, **kwargs)
        except Exception:
            failed = True
        if failed or response is None:
            raise self._fail(step)
        data: Any = None
        try:
            data = response.json()
        except ValueError:
            data = None
        return response.status_code, data if isinstance(data, dict) else None

    async def _first_document(self, urls: list[str]) -> dict[str, Any] | None:
        for url in urls:
            status, data = await self._send("GET", url, OAuthFailure.DISCOVERY)
            if status == 200 and data is not None:
                return data
        return None

    # -- Discovery -------------------------------------------------------------------

    async def _metadata(self) -> _Metadata:
        if self._metadata_cache is not None:
            return self._metadata_cache
        origin = _origin(self._resource)
        path = urlsplit(self._resource).path.rstrip("/")
        candidates = [f"{origin}{_PRM_PATH}{path}"] if path else []
        candidates.append(f"{origin}{_PRM_PATH}")
        resource_metadata = await self._first_document(candidates)
        if resource_metadata is None:
            raise self._fail(OAuthFailure.DISCOVERY)
        named = resource_metadata.get("resource")
        if not isinstance(named, str) or not _serves(named, self._resource):
            raise self._fail(OAuthFailure.AUDIENCE_MISMATCH)
        servers = resource_metadata.get("authorization_servers")
        if not isinstance(servers, list) or not servers or not isinstance(servers[0], str):
            raise self._fail(OAuthFailure.DISCOVERY)
        issuer: str = servers[0]
        if check_endpoint_url(issuer) is not None:
            raise self._fail(OAuthFailure.INSECURE_ENDPOINT)

        issuer_origin = _origin(issuer)
        issuer_path = urlsplit(issuer).path.rstrip("/")
        urls = [
            f"{issuer_origin}{_ASM_PATH}{issuer_path}",
            f"{issuer_origin}{_OIDC_PATH}{issuer_path}",
        ]
        if issuer_path:
            urls.append(f"{issuer_origin}{issuer_path}{_OIDC_PATH}")
        document = await self._first_document(urls)
        if document is None:
            raise self._fail(OAuthFailure.DISCOVERY)
        stated = document.get("issuer")
        if not isinstance(stated, str) or stated.rstrip("/") != issuer.rstrip("/"):
            raise self._fail(OAuthFailure.ISSUER_MISMATCH)
        methods = document.get("code_challenge_methods_supported")
        if not isinstance(methods, list) or "S256" not in methods:
            raise self._fail(OAuthFailure.PKCE_UNSUPPORTED)
        authorise = document.get("authorization_endpoint")
        token = document.get("token_endpoint")
        registration = document.get("registration_endpoint")
        if not isinstance(authorise, str) or not isinstance(token, str):
            raise self._fail(OAuthFailure.DISCOVERY)
        if registration is not None and not isinstance(registration, str):
            raise self._fail(OAuthFailure.DISCOVERY)
        for endpoint in (authorise, token, registration):
            if endpoint is not None and check_endpoint_url(endpoint) is not None:
                raise self._fail(OAuthFailure.INSECURE_ENDPOINT)
        self._metadata_cache = _Metadata(
            issuer=stated,
            authorization_endpoint=authorise,
            token_endpoint=token,
            registration_endpoint=registration,
            iss_promised=document.get("authorization_response_iss_parameter_supported") is True,
        )
        return self._metadata_cache

    # -- Registration ----------------------------------------------------------------

    async def _configured_client(self) -> _Client | None:
        if self._auth.client_id is None:
            return None
        secret: str | None = None
        if self._auth.client_secret is not None:
            secret = (await self._secrets.resolve(self._auth.client_secret)).get_secret_value()
        return _Client(self._auth.client_id, secret, "client_secret_post" if secret else "none")

    @staticmethod
    def _from_registration(stored: TokenSet) -> _Client | None:
        if not stored.token_type.startswith(_REGISTRATION_TYPE):
            return None
        secret = stored.refresh_token.get_secret_value() if stored.refresh_token else None
        return _Client(
            stored.access_token.get_secret_value(),
            secret,
            stored.token_type[len(_REGISTRATION_TYPE) :],
        )

    async def _stored_client(self, redirect_uri: str | None = None) -> _Client | None:
        """The client to use: pre-registered, or the stored registration (for
        ``redirect_uri``, when one is named)."""
        configured = await self._configured_client()
        if configured is not None:
            return configured
        stored = await self._get(self._settings.registration_key)
        if stored is None or (redirect_uri is not None and stored.scope != redirect_uri):
            return None
        return self._from_registration(stored)

    async def _register(self, metadata: _Metadata, redirect_uri: str) -> _Client:
        if metadata.registration_endpoint is None:
            raise self._fail(OAuthFailure.REGISTRATION)
        body: dict[str, Any] = {
            "client_name": self._auth.client_name,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        if self._auth.scopes:
            body["scope"] = " ".join(self._auth.scopes)
        status, data = await self._send(
            "POST", metadata.registration_endpoint, OAuthFailure.REGISTRATION, json=body
        )
        client_id = data.get("client_id") if data else None
        if (
            status not in (200, 201)
            or data is None
            or not isinstance(client_id, str)
            or not client_id
        ):
            raise self._fail(OAuthFailure.REGISTRATION)
        secret = data.get("client_secret")
        secret = secret if isinstance(secret, str) and secret else None
        method = data.get("token_endpoint_auth_method")
        if secret is None:
            method = "none"
        elif method not in _SECRET_METHODS:
            method = "client_secret_post"
        registration = TokenSet(
            access_token=SecretStr(client_id),
            refresh_token=SecretStr(secret) if secret else None,
            token_type=f"{_REGISTRATION_TYPE}{method}",
            scope=redirect_uri,
        )
        await self._put(self._settings.registration_key, registration)
        return _Client(client_id, secret, str(method))

    # -- Tokens ----------------------------------------------------------------------

    def _fresh(self, tokens: TokenSet) -> bool:
        if tokens.expires_at is None:
            return True
        margin = timedelta(seconds=self._auth.refresh_margin)
        return tokens.expires_at - margin > self._clock()

    async def _token_request(
        self, metadata: _Metadata, client: _Client, form: dict[str, str], step: OAuthFailure
    ) -> tuple[int, dict[str, Any] | None]:
        headers: dict[str, str] = {}
        body = dict(form)
        if client.method == "client_secret_basic" and client.secret is not None:
            pair = f"{quote(client.client_id, safe='')}:{quote(client.secret, safe='')}"
            headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode("ascii")
        else:
            body["client_id"] = client.client_id
            if client.method == "client_secret_post" and client.secret is not None:
                body["client_secret"] = client.secret
        return await self._send(
            "POST", metadata.token_endpoint, step, data=body, headers=headers or None
        )

    def _token_set(
        self, data: Mapping[str, Any], step: OAuthFailure, previous: SecretStr | None = None
    ) -> TokenSet:
        access = data.get("access_token")
        token_type = data.get("token_type")
        if not isinstance(access, str) or not access:
            raise self._fail(step)
        if not isinstance(token_type, str) or token_type.lower() != "bearer":
            raise self._fail(step)
        expires_in = data.get("expires_in")
        expires_at: datetime | None = None
        lifetime = (
            float(expires_in)
            if isinstance(expires_in, int | float) and not isinstance(expires_in, bool)
            else 0.0
        )
        if lifetime > 0:
            expires_at = self._clock() + timedelta(seconds=lifetime)
        refresh = data.get("refresh_token")
        scope = data.get("scope")
        return TokenSet(
            access_token=SecretStr(access),
            # Rotation: a new refresh token replaces the old; without one, the old stays.
            refresh_token=SecretStr(refresh) if isinstance(refresh, str) and refresh else previous,
            token_type="Bearer",  # noqa: S106 - the OAuth token type, not a secret
            expires_at=expires_at,
            scope=scope if isinstance(scope, str) else None,
        )

    async def _refresh(self, tokens: TokenSet) -> TokenSet | None:
        """New tokens by the refresh grant, or ``None`` (tokens dropped) if refused."""
        if tokens.refresh_token is None:
            return None
        client = await self._stored_client()
        if client is None:
            await self._delete(self._settings.token_key)
            return None
        metadata = await self._metadata()
        form = {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token.get_secret_value(),
            "resource": self._resource,
        }
        status, data = await self._token_request(metadata, client, form, OAuthFailure.REFRESH)
        if status != 200 or data is None:
            error = data.get("error") if data else None
            if status in (400, 401) and error in _DEAD_GRANT_ERRORS:
                await self._delete(self._settings.token_key)
                if error == "invalid_client":
                    await self._delete(self._settings.registration_key)
                logger.info("MCP server %s: refresh refused; sign-in needed", self._server)
                return None
            raise self._fail(OAuthFailure.REFRESH)
        fresh = self._token_set(data, OAuthFailure.REFRESH, previous=tokens.refresh_token)
        await self._put(self._settings.token_key, fresh)
        logger.info("MCP server %s: access token refreshed", self._server)
        return fresh

    # -- Sign-in ---------------------------------------------------------------------

    def _callback_code(
        self, callback_url: str, redirect_uri: str, state: str, metadata: _Metadata
    ) -> str:
        """The code in ``callback_url``, once every check on it has passed."""
        expected = _address(redirect_uri)
        received = _address(callback_url)
        if expected is None or received != expected:
            raise self._fail(OAuthFailure.REDIRECT_MISMATCH)
        params = parse_qs(urlsplit(callback_url).query, keep_blank_values=True)
        states = params.get("state", [])
        if len(states) != 1 or not hmac.compare_digest(states[0].encode(), state.encode()):
            raise self._fail(OAuthFailure.STATE_MISMATCH)
        issuers = params.get("iss", [])
        if len(issuers) > 1 or (issuers and issuers[0] != metadata.issuer):
            raise self._fail(OAuthFailure.ISSUER_MISMATCH)
        if not issuers and metadata.iss_promised:
            raise self._fail(OAuthFailure.ISSUER_MISMATCH)
        if "error" in params:
            raise self._fail(OAuthFailure.AUTHORISATION_DENIED)
        codes = params.get("code", [])
        if len(codes) != 1 or not codes[0]:
            raise self._fail(OAuthFailure.TOKEN_EXCHANGE)
        return codes[0]

    def _authorisation_url(
        self, metadata: _Metadata, client: _Client, redirect_uri: str, state: str, challenge: str
    ) -> str:
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": client.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": self._resource,
        }
        if self._auth.scopes:
            params["scope"] = " ".join(self._auth.scopes)
        parts = urlsplit(metadata.authorization_endpoint)
        query = [*parse_qsl(parts.query), *params.items()]
        return urlunsplit(parts._replace(query=urlencode(query)))

    async def _sign_in(self) -> TokenSet:
        hook = self._hook
        if hook is None:
            raise self._fail(OAuthFailure.SIGN_IN_REQUIRED)
        redirect_uri = getattr(hook, "redirect_uri", None)
        if not isinstance(redirect_uri, str) or check_endpoint_url(redirect_uri) is not None:
            raise self._fail(OAuthFailure.REDIRECT_MISMATCH)
        metadata = await self._metadata()
        client = await self._stored_client(redirect_uri)
        if client is None:
            client = await self._register(metadata, redirect_uri)
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(32)
        url = self._authorisation_url(metadata, client, redirect_uri, state, challenge)
        failed = False
        callback_url: Any = None
        try:
            result = await hook.sign_in(
                SignInRequest(server=self._server, authorization_url=url, redirect_uri=redirect_uri)
            )
            callback_url = getattr(result, "callback_url", None)
        except Exception:
            failed = True
        if failed or not isinstance(callback_url, str):
            raise self._fail(OAuthFailure.SIGN_IN_FAILED)
        code = self._callback_code(callback_url, redirect_uri, state, metadata)
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
            "resource": self._resource,
        }
        status, data = await self._token_request(
            metadata, client, form, OAuthFailure.TOKEN_EXCHANGE
        )
        if status != 200 or data is None:
            raise self._fail(OAuthFailure.TOKEN_EXCHANGE)
        tokens = self._token_set(data, OAuthFailure.TOKEN_EXCHANGE)
        await self._put(self._settings.token_key, tokens)
        logger.info("MCP server %s: signed in", self._server)
        return tokens

    # -- Public ------------------------------------------------------------------------

    async def access_token(self, *, interactive: bool = True) -> SecretStr:
        """A usable access token: stored, refreshed, or (if ``interactive``) new."""
        async with self._lock:
            tokens = await self._get(self._settings.token_key)
            if tokens is not None and self._fresh(tokens):
                return tokens.access_token
            if tokens is not None:
                refreshed = await self._refresh(tokens)
                if refreshed is not None:
                    return refreshed.access_token
                if tokens.refresh_token is None:
                    await self._delete(self._settings.token_key)
            if not interactive:
                raise self._fail(OAuthFailure.SIGN_IN_REQUIRED)
            return (await self._sign_in()).access_token

    async def refreshed(self, stale: SecretStr) -> SecretStr:
        """A token to replace ``stale``, which the server refused. Never interactive.

        If another caller has already replaced it, that token is returned, so
        concurrent callers share one refresh.
        """
        async with self._lock:
            tokens = await self._get(self._settings.token_key)
            if tokens is not None:
                current = tokens.access_token.get_secret_value()
                if current != stale.get_secret_value() and self._fresh(tokens):
                    return tokens.access_token
                refreshed = await self._refresh(tokens)
                if refreshed is not None:
                    return refreshed.access_token
            raise self._fail(OAuthFailure.SIGN_IN_REQUIRED)

    async def sign_out(self) -> None:
        """Drop this server's tokens (the client registration is kept)."""
        async with self._lock:
            await self._delete(self._settings.token_key)

    def httpx_auth(self) -> httpx.Auth:
        """An ``httpx.Auth`` adding the bearer token, refreshing once on a 401."""
        return _BearerFlow(self)


class _BearerFlow(httpx.Auth):
    def __init__(self, oauth: OAuthClient) -> None:
        self._oauth = oauth

    def __repr__(self) -> str:
        return f"_BearerFlow({self._oauth!r})"

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        raise RuntimeError("the MCP OAuth flow is async only")

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self._oauth.access_token(interactive=False)
        request.headers["Authorization"] = f"Bearer {token.get_secret_value()}"
        response = yield request
        if response.status_code == 401:
            fresh = await self._oauth.refreshed(token)
            request.headers["Authorization"] = f"Bearer {fresh.get_secret_value()}"
            yield request
