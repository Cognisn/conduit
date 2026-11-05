"""Real boto3 clients from the Bedrock factory, for each authentication mode.

These tests need the bedrock extra and are skipped without it. None reaches a network:
a ``before-send`` handler answers every request locally.
"""

from __future__ import annotations

import base64
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest
from pydantic import SecretStr

from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.types import Message, ModelRequest
from conduit.providers.bedrock import BedrockFactory, BedrockSettings
from conduit.providers.bedrock.auth import SHORT_TERM_PREFIX

# The documentation example access key pair, never a real credential.
ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
SECRETS = {"bedrock/id": ACCESS_KEY_ID, "bedrock/secret": SECRET_ACCESS_KEY}
ACCESS_KEYS = {
    "mode": "access_keys",
    "access_key_id": "secret://bedrock/id",
    "secret_access_key": "secret://bedrock/secret",
}
REQUEST = ModelRequest(
    model="anthropic.claude-sonnet-4-6", messages=(Message.user("Hi"),), max_output_tokens=16
)
CONVERSE_BODY = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Hello."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
    "metrics": {"latencyMs": 1},
}


boto3 = pytest.importorskip("boto3", reason="needs the bedrock extra")


class Sent:
    """Answers each request locally and keeps what was sent."""

    def __init__(self, status: int = 200, body: dict[str, Any] | None = None) -> None:
        self.requests: list[Any] = []
        self.status = status
        self.body = json.dumps(body if body is not None else CONVERSE_BODY).encode()

    def __call__(self, request: Any, **_: Any) -> Any:
        aws_response = importlib.import_module("botocore.awsrequest").AWSResponse
        self.requests.append(request)
        body = self.body

        class Raw:
            def stream(self, **_: Any) -> Iterator[bytes]:
                yield body

        return aws_response(request.url, self.status, {"Content-Type": "application/json"}, Raw())


def client_for(auth: dict[str, Any] | None = None, **fields: Any) -> tuple[Any, Sent]:
    settings = BedrockSettings.model_validate(
        {"region": "eu-west-2", **({"auth": auth} if auth else {}), **fields}
    )
    secrets = {
        f"auth.{prefix}{name}": SecretStr(value)
        for prefix in ("", "source.")
        for name, value in (
            ("access_key_id", ACCESS_KEY_ID),
            ("secret_access_key", SECRET_ACCESS_KEY),
            ("api_key", "long-term-example-key"),
        )
    }
    client = BedrockFactory().create_client(settings, secrets)
    sent = Sent()
    client.meta.events.register("before-send.bedrock-runtime", sent)
    return client, sent


def decode(key: str) -> dict[str, list[str]]:
    assert key.startswith(SHORT_TERM_PREFIX)
    url = base64.b64decode(key[len(SHORT_TERM_PREFIX) :]).decode()
    host, _, query = url.partition("/?")
    assert host == "bedrock.amazonaws.com"
    return parse_qs(query)


class TestClients:
    def test_a_runtime_client_for_the_region(self) -> None:
        client, _ = client_for(ACCESS_KEYS)
        assert client.meta.service_model.service_name == "bedrock-runtime"
        assert client.meta.region_name == "eu-west-2"
        assert client.meta.endpoint_url == "https://bedrock-runtime.eu-west-2.amazonaws.com"

    def test_access_keys_sign_with_sigv4(self) -> None:
        client, sent = client_for(ACCESS_KEYS)
        client.converse(modelId="anthropic.claude-sonnet-4-6", messages=[])
        authorisation = sent.requests[0].headers["Authorization"].decode()
        assert authorisation.startswith("AWS4-HMAC-SHA256 Credential=" + ACCESS_KEY_ID)
        assert "/eu-west-2/bedrock/aws4_request" in authorisation

    def test_a_named_profile_from_the_credential_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        credentials = tmp_path / "credentials"
        credentials.write_text(
            f"[example-profile]\naws_access_key_id = {ACCESS_KEY_ID}\n"
            f"aws_secret_access_key = {SECRET_ACCESS_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
        client, sent = client_for({"mode": "credential_chain", "profile": "example-profile"})
        client.converse(modelId="anthropic.claude-sonnet-4-6", messages=[])
        assert ACCESS_KEY_ID in sent.requests[0].headers["Authorization"].decode()

    def test_an_endpoint_override(self) -> None:
        client, _ = client_for(ACCESS_KEYS, endpoint_url="https://bedrock.vpce.example.com")
        assert client.meta.endpoint_url == "https://bedrock.vpce.example.com"


class TestApiKeys:
    def test_a_short_term_key_is_generated_and_sent_as_a_bearer_token(self) -> None:
        client, sent = client_for(
            {"mode": "short_term_api_key", "source": ACCESS_KEYS, "lifetime_s": 3_600}
        )
        client.converse(modelId="anthropic.claude-sonnet-4-6", messages=[])
        authorisation = sent.requests[0].headers["Authorization"]
        authorisation = (
            authorisation.decode() if isinstance(authorisation, bytes) else authorisation
        )
        assert authorisation.startswith("Bearer " + SHORT_TERM_PREFIX)
        query = decode(authorisation.removeprefix("Bearer "))
        assert query["Action"] == ["CallWithBearerToken"]
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert query["X-Amz-Expires"] == ["3600"]
        assert query["X-Amz-Credential"][0].startswith(ACCESS_KEY_ID + "/")
        assert query["X-Amz-Credential"][0].endswith("/eu-west-2/bedrock/aws4_request")
        assert query["Version"] == ["1"]
        assert "X-Amz-Signature" in query
        assert not any(h.lower().startswith("x-amz-date") for h in sent.requests[0].headers)

    def test_a_long_term_key_is_sent_as_given_when_allowed(self) -> None:
        client, sent = client_for(
            {"mode": "long_term_api_key", "api_key": "secret://bedrock/key"},
            allow_long_term_api_keys=True,
        )
        client.converse(modelId="anthropic.claude-sonnet-4-6", messages=[])
        authorisation = sent.requests[0].headers["Authorization"]
        authorisation = (
            authorisation.decode() if isinstance(authorisation, bytes) else authorisation
        )
        assert authorisation == "Bearer long-term-example-key"

    def test_the_environment_bearer_variable_is_not_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "environment-example-key")
        client, sent = client_for(
            {"mode": "long_term_api_key", "api_key": "secret://bedrock/key"},
            allow_long_term_api_keys=True,
        )
        client.converse(modelId="anthropic.claude-sonnet-4-6", messages=[])
        assert "environment-example-key" not in str(sent.requests[0].headers)

    def test_missing_credentials_for_a_short_term_key_are_refused_at_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing in the chain can answer, and nothing in it may reach a network.
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        with pytest.raises(ConfigurationError, match="NoCredentialsError"):
            client_for({"mode": "short_term_api_key"})


class TestThroughTheRegistry:
    async def test_a_converse_call_end_to_end_without_a_network(self) -> None:
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver(SECRETS), factories=[BedrockFactory()]
        )
        provider = await registry.build(
            "bedrock_london",
            {
                "kind": "bedrock",
                "region": "eu-west-2",
                "auth": ACCESS_KEYS,
                "use_converse_stream": False,
            },
        )
        sent = Sent()
        client = provider.inner._client  # type: ignore[attr-defined]
        client.meta.events.register("before-send.bedrock-runtime", sent)
        response = await provider.complete(REQUEST)
        assert response.text == "Hello."
        assert json.loads(sent.requests[0].body)["inferenceConfig"] == {"maxTokens": 16}
        await registry.aclose()

    async def test_a_service_error_becomes_a_provider_failure_without_its_text(self) -> None:
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver(SECRETS), factories=[BedrockFactory()]
        )
        provider = await registry.build(
            "bedrock_london",
            {
                "kind": "bedrock",
                "region": "eu-west-2",
                "auth": ACCESS_KEYS,
                "use_converse_stream": False,
                "max_attempts": 1,
            },
        )
        failing = Sent(status=400, body={"message": "echoed prompt text Hi"})
        provider.inner._client.meta.events.register(  # type: ignore[attr-defined]
            "before-send.bedrock-runtime", failing
        )
        with pytest.raises(ProviderError) as raised:
            await provider.complete(REQUEST)
        assert raised.value.failure is ProviderFailure.REJECTED
        assert "echoed" not in str(raised.value)
        await registry.aclose()
