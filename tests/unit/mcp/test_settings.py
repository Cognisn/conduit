"""MCP server settings: typed, buildable from a mapping, credentials only by reference."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from conduit.core.secrets import SecretRef
from conduit.mcp.settings import (
    ApiKeyAuth,
    BearerTokenAuth,
    McpServerSettings,
    NoAuth,
    OAuthAuth,
    SseTransport,
    StdioTransport,
    StreamableHttpTransport,
)

SYNTHETIC_SECRET = "zz-plaintext-credential-0000zz"


def _http(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "docs",
        "transport": {"type": "streamable_http", "url": "https://mcp.example.com/mcp"},
        "task_classes": ["research"],
    }
    data.update(overrides)
    return data


class TestBuilding:
    def test_built_from_a_mapping(self) -> None:
        settings = McpServerSettings.model_validate(
            _http(auth={"type": "bearer", "token": "secret://docs_token"})
        )
        assert isinstance(settings.transport, StreamableHttpTransport)
        assert isinstance(settings.auth, BearerTokenAuth)
        assert settings.auth.token == SecretRef("docs_token")
        assert settings.task_classes == frozenset({"research"})

    def test_built_in_code(self) -> None:
        settings = McpServerSettings(
            name="files",
            transport=StdioTransport(command="example-server", args=("--root", "/srv")),
            task_classes=frozenset({"chat"}),
        )
        assert isinstance(settings.auth, NoAuth)
        assert settings.connect_timeout > 0
        assert settings.call_timeout > 0

    @pytest.mark.parametrize(
        ("transport", "kind"),
        [
            ({"type": "stdio", "command": "srv"}, StdioTransport),
            ({"type": "sse", "url": "https://mcp.example.com/sse"}, SseTransport),
            (
                {"type": "streamable_http", "url": "https://mcp.example.com/"},
                StreamableHttpTransport,
            ),
        ],
    )
    def test_every_transport(self, transport: dict[str, Any], kind: type) -> None:
        settings = McpServerSettings.model_validate(_http(transport=transport))
        assert isinstance(settings.transport, kind)

    def test_api_key_and_oauth(self) -> None:
        key = McpServerSettings.model_validate(
            _http(auth={"type": "api_key", "key": "secret://k", "header": "X-Example-Key"})
        )
        assert isinstance(key.auth, ApiKeyAuth)
        oauth = McpServerSettings.model_validate(_http(auth={"type": "oauth", "scopes": ["read"]}))
        assert isinstance(oauth.auth, OAuthAuth)
        assert oauth.auth.scopes == ("read",)

    def test_the_token_key_is_the_server_name(self) -> None:
        settings = McpServerSettings.model_validate(_http(auth={"type": "oauth"}))
        assert settings.token_key == "mcp:docs"
        assert "/" not in settings.token_key


class TestRefusals:
    @pytest.mark.parametrize(
        "name",
        ["", "Docs", "docs-1", "1docs", "do__cs", "docs_", "d" * 33, "docs.example", "docs/x"],
    )
    def test_a_server_name_outside_the_pattern_is_refused(self, name: str) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(_http(name=name))

    @pytest.mark.parametrize(
        "url",
        [
            "http://mcp.example.com/mcp",  # plain http off loopback
            "ftp://mcp.example.com/mcp",
            "https://user:pw@mcp.example.com/mcp",  # userinfo
            "https://mcp.example.com/mcp#fragment",
            "https:///mcp",
            "mcp.example.com/mcp",
        ],
    )
    def test_an_insecure_or_malformed_url_is_refused(self, url: str) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(
                _http(transport={"type": "streamable_http", "url": url})
            )

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1:22410/mcp", "http://localhost:22410/mcp", "http://[::1]:22410/"]
    )
    def test_plain_http_is_accepted_on_loopback_only(self, url: str) -> None:
        settings = McpServerSettings.model_validate(_http(transport={"type": "sse", "url": url}))
        assert isinstance(settings.transport, SseTransport)

    def test_a_plaintext_bearer_token_is_refused_without_echo(self) -> None:
        with pytest.raises(ValidationError) as caught:
            McpServerSettings.model_validate(
                _http(auth={"type": "bearer", "token": SYNTHETIC_SECRET})
            )
        assert SYNTHETIC_SECRET not in str(caught.value)
        assert SYNTHETIC_SECRET not in repr(caught.value)

    @pytest.mark.parametrize("name", ["API_KEY", "GITHUB_TOKEN", "DB_PASSWORD", "ExampleSecret"])
    def test_a_credential_in_a_plain_environment_variable_is_refused(self, name: str) -> None:
        with pytest.raises(ValidationError) as caught:
            McpServerSettings.model_validate(
                _http(
                    transport={"type": "stdio", "command": "srv", "env": {name: SYNTHETIC_SECRET}}
                )
            )
        assert SYNTHETIC_SECRET not in str(caught.value)

    def test_a_credential_goes_in_secret_env(self) -> None:
        settings = McpServerSettings.model_validate(
            _http(
                transport={
                    "type": "stdio",
                    "command": "srv",
                    "env": {"LOG_LEVEL": "info"},
                    "secret_env": {"API_KEY": "secret://srv_key"},
                }
            )
        )
        assert isinstance(settings.transport, StdioTransport)
        assert settings.transport.secret_env == {"API_KEY": SecretRef("srv_key")}

    def test_an_authorisation_value_is_refused_in_a_plain_environment_variable(self) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(
                _http(
                    transport={
                        "type": "stdio",
                        "command": "srv",
                        "env": {"HEADER": f"Bearer {SYNTHETIC_SECRET}"},
                    }
                )
            )

    def test_http_auth_on_a_stdio_server_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(
                _http(
                    transport={"type": "stdio", "command": "srv"},
                    auth={"type": "bearer", "token": "secret://t"},
                )
            )

    @pytest.mark.parametrize("header", ["", "Bad Header", "X-Key:", "Host", "Content-Length"])
    def test_a_bad_api_key_header_is_refused(self, header: str) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(
                _http(auth={"type": "api_key", "key": "secret://k", "header": header})
            )

    @pytest.mark.parametrize("task_classes", [[], [""], ["Bad Class"]])
    def test_task_classes_must_be_granted(self, task_classes: list[str]) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(_http(task_classes=task_classes))

    def test_a_tool_grant_may_only_narrow_the_server_grant(self) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(
                _http(task_classes=["research"], tool_task_classes={"search": ["admin"]})
            )
        narrowed = McpServerSettings.model_validate(
            _http(task_classes=["research", "chat"], tool_task_classes={"search": ["chat"]})
        )
        assert narrowed.tool_task_classes == {"search": frozenset({"chat"})}

    @pytest.mark.parametrize("field", ["connect_timeout", "call_timeout", "list_timeout"])
    @pytest.mark.parametrize("value", [0, -1, 100000])
    def test_timeouts_are_bounded(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(_http(**{field: value}))

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            McpServerSettings.model_validate(_http(api_key=SYNTHETIC_SECRET))

    def test_a_pre_registered_client_secret_is_a_reference(self) -> None:
        with pytest.raises(ValidationError) as caught:
            McpServerSettings.model_validate(
                _http(auth={"type": "oauth", "client_id": "cid", "client_secret": SYNTHETIC_SECRET})
            )
        assert SYNTHETIC_SECRET not in str(caught.value)
        ok = McpServerSettings.model_validate(
            _http(
                auth={
                    "type": "oauth",
                    "client_id": "cid",
                    "client_secret": "secret://cs",
                    "issuer": "https://auth.example.com",
                }
            )
        )
        assert isinstance(ok.auth, OAuthAuth)
        assert ok.auth.client_secret == SecretRef("cs")
