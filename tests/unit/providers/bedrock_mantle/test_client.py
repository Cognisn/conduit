"""Real mantle clients from the factory, for each key mode, answered without a network.

These tests need the bedrock-mantle extra (anthropic and boto3) and are skipped without
it. Each client is given an HTTP transport that answers locally (the SDK's public
``with_options(http_client=...)``), so the request conduit's client really sends can be
read: its URL, its headers and its body.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

import pytest
from pydantic import SecretStr

from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamStopped, TextDelta
from conduit.core.types import Message, ModelRequest, Usage
from conduit.providers.bedrock.auth import SHORT_TERM_PREFIX, SIGNER_LOGGER, SignatureRecordFilter
from conduit.providers.bedrock_mantle import (
    BedrockMantleFactory,
    BedrockMantleProvider,
    BedrockMantleSettings,
)

from ..anthropic.fakes import message_end, message_start, text_block, usage

pytest.importorskip("anthropic.lib.bedrock", reason="needs the bedrock-mantle extra")
pytest.importorskip("botocore", reason="needs the bedrock-mantle extra")
httpx2 = pytest.importorskip("httpx2", reason="needs the bedrock-mantle extra")
ANTHROPIC_API_VERSION = pytest.importorskip("anthropic._constants").ANTHROPIC_API_VERSION

# The documentation example access key pair, never a real credential.
ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
ACCESS_KEY_SECRETS = {
    "auth.source.access_key_id": SecretStr(ACCESS_KEY_ID),
    "auth.source.secret_access_key": SecretStr(SECRET_ACCESS_KEY),
}
LONG_TERM_KEY = "synthetic-long-term-key-not-real"
MELBOURNE = "ap-southeast-4"
MELBOURNE_URL = "https://bedrock-mantle.ap-southeast-4.api.aws/anthropic/v1/messages"
MARKER = "SECRET-MARKER-91c2"
SONNET_5 = "anthropic.claude-sonnet-5"
REQUEST = ModelRequest(model=SONNET_5, messages=(Message.user("Hi"),), max_output_tokens=16)


def short_term(**auth: Any) -> BedrockMantleSettings:
    return BedrockMantleSettings.model_validate(
        {
            "kind": "bedrock_mantle",
            "region": MELBOURNE,
            "auth": {
                "mode": "short_term_api_key",
                "source": {
                    "mode": "access_keys",
                    "access_key_id": "secret://mantle/id",
                    "secret_access_key": "secret://mantle/secret",
                },
                **auth,
            },
            "max_retries": 0,
        }
    )


def long_term(**fields: Any) -> BedrockMantleSettings:
    return BedrockMantleSettings.model_validate(
        {
            "kind": "bedrock_mantle",
            "region": MELBOURNE,
            "auth": {"mode": "long_term_api_key", "api_key": "secret://mantle/key"},
            "allow_long_term_api_keys": True,
            "max_retries": 0,
            **fields,
        }
    )


def sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


HELLO = sse(
    [
        message_start(usage(input_tokens=6, output_tokens=1, cache_read=20, cache_write=3)),
        *text_block(0, "Hel", "lo."),
        *message_end("end_turn", usage(6, 4, cache_read=20, cache_write=3)),
    ]
)


class Server:
    """Answers each request from a list of (status, headers, body) and keeps what was sent."""

    def __init__(self, *answers: tuple[int, dict[str, str], bytes]) -> None:
        self.answers = list(answers) or [(200, {}, HELLO)]
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> Any:
        request.read()
        self.requests.append(request)
        status, headers, body = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        content_type = "text/event-stream" if status == 200 else "application/json"
        return httpx2.Response(
            status, headers={"content-type": content_type, **headers}, content=body
        )


def wired(
    settings: BedrockMantleSettings,
    secrets: Mapping[str, SecretStr],
    server: Server,
    *,
    clock: Callable[[], float] = time.time,
) -> tuple[Any, Any]:
    """The factory's client, re-pointed at ``server``, and a provider around it."""
    factory = BedrockMantleFactory(clock=clock)
    client = factory.create_client(settings, secrets)
    local = client.with_options(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    )
    return local, factory.create_provider("mantle_melbourne", settings, local)


def decoded(key: str) -> dict[str, list[str]]:
    assert key.startswith(SHORT_TERM_PREFIX)
    url = base64.b64decode(key.removeprefix(SHORT_TERM_PREFIX)).decode()
    host, _, query = url.partition("/?")
    assert host == "bedrock.amazonaws.com"
    return parse_qs(query)


def bearer(request: Any) -> str:
    value = request.headers["authorization"]
    assert value.startswith("Bearer ")
    return str(value.removeprefix("Bearer "))


class TestShortTermKeys:
    async def test_a_call_goes_to_the_regional_endpoint_with_a_bearer_key(self) -> None:
        server = Server()
        _, provider = wired(short_term(), ACCESS_KEY_SECRETS, server)
        events = [e async for e in provider.stream(REQUEST)]
        await provider.aclose()

        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello."
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.usage == Usage(
            input_tokens=6, output_tokens=4, cache_read_tokens=20, cache_write_tokens=3
        )

        (sent,) = server.requests
        assert str(sent.url) == MELBOURNE_URL
        assert sent.method == "POST"
        # The SDK sends its own API version; that mantle accepts it is a live-suite check.
        assert sent.headers["anthropic-version"] == ANTHROPIC_API_VERSION
        assert "x-api-key" not in sent.headers
        body = json.loads(sent.content)
        assert body["model"] == SONNET_5
        assert body["stream"] is True

        query = decoded(bearer(sent))
        assert query["Action"] == ["CallWithBearerToken"]
        assert query["X-Amz-Expires"] == ["43200"]
        assert query["Version"] == ["1"]
        credential = query["X-Amz-Credential"][0].split("/")
        assert credential[0] == ACCESS_KEY_ID
        assert credential[2:] == [MELBOURNE, "bedrock", "aws4_request"]
        assert SECRET_ACCESS_KEY not in bearer(sent)

    async def test_the_key_is_replaced_once_it_is_within_its_margin(self) -> None:
        now = [1_000_000.0]
        server = Server((200, {}, HELLO))
        settings = short_term(lifetime_s=900, refresh_before_s=60)
        _, provider = wired(settings, ACCESS_KEY_SECRETS, server, clock=lambda: now[0])
        await provider.complete(REQUEST)
        await provider.complete(REQUEST)
        time.sleep(1.1)  # the presigned URL carries its signing time to the second
        now[0] += 900 - 60  # inside the refresh margin
        await provider.complete(REQUEST)
        await provider.aclose()
        first, again, refreshed = (bearer(r) for r in server.requests)
        assert first == again
        assert refreshed != first
        assert decoded(refreshed)["X-Amz-Expires"] == ["900"]

    async def test_every_retry_carries_the_bearer_key(self) -> None:
        busy = (529, {"retry-after-ms": "5"}, b'{"type": "error"}')
        server = Server(busy, (200, {}, HELLO))
        settings = short_term().model_copy(update={"max_retries": 1})
        _, provider = wired(settings, ACCESS_KEY_SECRETS, server)
        response = await provider.complete(REQUEST)
        await provider.aclose()
        assert response.text == "Hello."
        assert len(server.requests) == 2
        assert all(bearer(r).startswith(SHORT_TERM_PREFIX) for r in server.requests)

    async def test_a_key_is_replaced_before_its_credentials_expire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Temporary credentials expiring in 700 s cap a 900 s key: it is replaced once
        # within its 60 s margin of the credentials' expiry, not of its own lifetime.
        botocore_session = pytest.importorskip("botocore.session")
        read_only = pytest.importorskip("botocore.credentials").ReadOnlyCredentials
        start = 1_000_000.0
        now = [start]

        class Expiring:
            """Refreshes itself when read again, as botocore's temporary credentials do."""

            def __init__(self) -> None:
                self.reads = 0
                self._expiry_time = datetime.fromtimestamp(start + 700, tz=UTC)

            def get_frozen_credentials(self) -> Any:
                self.reads += 1
                if self.reads > 1:
                    self._expiry_time = datetime.fromtimestamp(now[0] + 3_600, tz=UTC)
                return read_only(ACCESS_KEY_ID, SECRET_ACCESS_KEY, "synthetic-session")

        credentials = Expiring()
        monkeypatch.setattr(botocore_session.Session, "get_credentials", lambda self: credentials)
        settings = short_term(lifetime_s=900, refresh_before_s=60)
        _, provider = wired(settings, ACCESS_KEY_SECRETS, Server(), clock=lambda: now[0])
        assert credentials.reads == 1  # the first key, at creation
        now[0] = start + 630  # 70 s before the credentials expire: still fresh
        await provider.complete(REQUEST)
        assert credentials.reads == 1
        now[0] = start + 650  # inside the margin of the credentials' expiry
        await provider.complete(REQUEST)
        assert credentials.reads == 2  # replaced, though its own lifetime runs to 900 s
        await provider.aclose()

    def test_missing_credentials_are_refused_at_creation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        settings = BedrockMantleSettings.model_validate(
            {
                "kind": "bedrock_mantle",
                "region": MELBOURNE,
                "auth": {
                    "mode": "short_term_api_key",
                    "source": {"mode": "credential_chain", "profile": "conduit-missing"},
                },
            }
        )
        with pytest.raises(ConfigurationError, match="short-term") as caught:
            BedrockMantleFactory().create_client(settings, {})
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert "conduit-missing" not in str(caught.value)


class TestLongTermKeys:
    async def test_sent_as_a_bearer_token_when_allowed(self) -> None:
        server = Server()
        _, provider = wired(long_term(), {"auth.api_key": SecretStr(LONG_TERM_KEY)}, server)
        await provider.complete(REQUEST)
        await provider.aclose()
        (sent,) = server.requests
        assert bearer(sent) == LONG_TERM_KEY
        assert "x-api-key" not in sent.headers

    async def test_an_endpoint_override_is_used(self) -> None:
        server = Server()
        settings = long_term(
            endpoint_url="https://vpce-synthetic.bedrock-mantle.example.com/anthropic"
        )
        _, provider = wired(settings, {"auth.api_key": SecretStr(LONG_TERM_KEY)}, server)
        await provider.complete(REQUEST)
        await provider.aclose()
        assert str(server.requests[0].url) == (
            "https://vpce-synthetic.bedrock-mantle.example.com/anthropic/v1/messages"
        )


class TestEnvironment:
    async def test_the_sdk_takes_nothing_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        planted = {
            "AWS_BEARER_TOKEN_BEDROCK": f"bedrock-api-key-{MARKER}",
            "ANTHROPIC_AWS_API_KEY": MARKER,
            "ANTHROPIC_API_KEY": MARKER,
            "AWS_REGION": "eu-west-1",
            "AWS_DEFAULT_REGION": "eu-west-1",
            "ANTHROPIC_BEDROCK_MANTLE_BASE_URL": "https://planted.example.com/anthropic",
            "ANTHROPIC_BASE_URL": "https://planted.example.com",
            "ANTHROPIC_CUSTOM_HEADERS": f"x-api-key: {MARKER}",
        }
        for name, value in planted.items():
            monkeypatch.setenv(name, value)
        server = Server()
        client, provider = wired(long_term(), {"auth.api_key": SecretStr(LONG_TERM_KEY)}, server)
        assert client.api_key is None  # the client holds no key of its own
        await provider.complete(REQUEST)
        await provider.aclose()
        (sent,) = server.requests
        assert str(sent.url) == MELBOURNE_URL
        assert bearer(sent) == LONG_TERM_KEY
        for name, value in sent.headers.items():
            assert MARKER not in value, name
        assert {name: os.environ[name] for name in planted} == planted  # nothing written


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "failure"),
        [
            (400, ProviderFailure.REJECTED),
            (403, ProviderFailure.REJECTED),
            (429, ProviderFailure.UNAVAILABLE),
            (500, ProviderFailure.UNAVAILABLE),
            (529, ProviderFailure.UNAVAILABLE),
        ],
    )
    async def test_an_http_error_is_a_failure_without_its_text(
        self, status: int, failure: ProviderFailure
    ) -> None:
        body = json.dumps(
            {"type": "error", "error": {"type": "x", "message": f"refused: {MARKER}"}}
        ).encode()
        server = Server((status, {}, body))
        _, provider = wired(long_term(), {"auth.api_key": SecretStr(LONG_TERM_KEY)}, server)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(REQUEST)
        await provider.aclose()
        error = caught.value
        assert error.failure is failure
        assert error.__cause__ is None
        assert error.__context__ is None
        for text in (str(error), repr(error), *(repr(a) for a in error.args)):
            assert MARKER not in text
            assert LONG_TERM_KEY not in text


async def test_closing_the_provider_closes_the_sdk_client() -> None:
    client, provider = wired(long_term(), {"auth.api_key": SecretStr(LONG_TERM_KEY)}, Server())
    await provider.aclose()
    assert client.is_closed()


async def test_the_registry_builds_a_working_provider() -> None:
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver(
            {"mantle/id": ACCESS_KEY_ID, "mantle/secret": SECRET_ACCESS_KEY}
        ),
        factories=[BedrockMantleFactory()],
    )
    built = await registry.build("mantle_melbourne", short_term())
    try:
        assert built.kind == "bedrock_mantle"
        assert isinstance(built._inner, BedrockMantleProvider)
    finally:
        await registry.aclose()


class TestLogging:
    """No key reaches a log record, even with every logger at DEBUG.

    A product that runs ``logging.basicConfig(level=DEBUG)`` enables the anthropic
    SDK's own debug logging (its request options among it) without ANTHROPIC_LOG.
    """

    @pytest.mark.parametrize("mode", ["short_term", "long_term"])
    async def test_the_key_is_in_no_record_across_a_retry(
        self, mode: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Start without the signature filter, whichever test installed it first, so
        # only this client's creation can put it there.
        signer = logging.getLogger(SIGNER_LOGGER)
        monkeypatch.setattr(
            signer,
            "filters",
            [f for f in signer.filters if not isinstance(f, SignatureRecordFilter)],
        )
        caplog.set_level(logging.DEBUG)
        for name in ("anthropic", "httpx", "httpx2", "httpcore", "conduit", "botocore"):
            caplog.set_level(logging.DEBUG, logger=name)
        busy = (529, {"retry-after-ms": "5"}, b'{"type": "error"}')
        server = Server(busy, (200, {}, HELLO))
        if mode == "short_term":
            settings = short_term().model_copy(update={"max_retries": 1})
            secrets: Mapping[str, SecretStr] = ACCESS_KEY_SECRETS
        else:
            settings = long_term().model_copy(update={"max_retries": 1})
            secrets = {"auth.api_key": SecretStr(LONG_TERM_KEY)}
        _, provider = wired(settings, secrets, server)
        await provider.complete(REQUEST)
        await provider.aclose()
        keys = {bearer(r) for r in server.requests}
        assert len(server.requests) == 2  # the retry carried the key too
        assert caplog.records  # the SDK did log at DEBUG
        for record in caplog.records:
            text = record.getMessage() + (record.exc_text or "")
            for key in keys:
                assert key not in text, record.name
                # A presigned key's signature is enough to rebuild it.
                if key.startswith(SHORT_TERM_PREFIX):
                    signature = decoded(key)["X-Amz-Signature"][0]
                    assert signature not in text, record.name
