"""``providers``, ``models`` and ``check``: adapters, credentials and residency refusals.

Every credential here is synthetic, and no check reaches AWS: building a Bedrock client
and presigning a short-term key are local, and the instance metadata lookup is off.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from conduit import (
    Catalogue,
    ConfigurationError,
    Provider,
    ProviderFactory,
    ProviderSettings,
    SecretRef,
)
from conduit.cli.setup import interface_factories
from tests.unit.cli.helpers import cli, stub_conduit, write_config

ACCESS_KEY_ID = "AKIAZZEXAMPLE0000000"  # synthetic, never a real key
SECRET_KEY = "zz-synthetic-secret-access-key-0000000000zz"
LONG_TERM_KEY = "zz-synthetic-long-term-bedrock-key-0000zz"
ENVIRON = {
    "EXAMPLE_KEY_ID": ACCESS_KEY_ID,
    "EXAMPLE_SECRET_KEY": SECRET_KEY,
    "EXAMPLE_LONG_TERM_KEY": LONG_TERM_KEY,
}
SECRETS = (ACCESS_KEY_ID, SECRET_KEY, LONG_TERM_KEY, "bedrock-api-key-")

ACCESS_KEYS = {
    "mode": "access_keys",
    "access_key_id": "secret://EXAMPLE_KEY_ID",
    "secret_access_key": "secret://EXAMPLE_SECRET_KEY",
}


@pytest.fixture
def no_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No AWS configuration, profile or instance metadata from this machine."""
    for name in list(__import__("os").environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def bedrock_config(**assignments: Any) -> dict[str, Any]:
    return {
        "conduit": {
            "providers": {
                "chain": {"kind": "bedrock", "region": "ap-southeast-2"},
                "keys": {"kind": "bedrock", "region": "ap-southeast-2", "auth": ACCESS_KEYS},
                "short": {
                    "kind": "bedrock",
                    "region": "ap-southeast-2",
                    "auth": {"mode": "short_term_api_key", "source": ACCESS_KEYS},
                },
                "long": {
                    "kind": "bedrock",
                    "region": "ap-southeast-2",
                    "auth": {
                        "mode": "long_term_api_key",
                        "api_key": "secret://EXAMPLE_LONG_TERM_KEY",
                    },
                    "allow_long_term_api_keys": True,
                },
            },
            "assignments": assignments,
            "residency": {
                "scopes": ["in_region", "cross_region", "local"],
                "regions": ["ap-southeast-2", "ap-southeast-4"],
            },
        }
    }


def row(output: str, name: str) -> str:
    """The providers table's row for the provider ``name``."""
    (found,) = [
        line for line in output.splitlines() if re.match(rf"\s*{re.escape(name)}\s+bedrock\s", line)
    ]
    return found


class TestProviders:
    async def test_every_adapter_is_listed_with_its_extra(self) -> None:
        result = await cli("providers")
        assert result.code == 0
        for kind, extra in (
            ("stub", "(none)"),
            ("bedrock", "bedrock"),
            ("bedrock_mantle", "bedrock-mantle"),
            ("anthropic", "anthropic"),
            ("openai_compat", "openai"),
            ("gemini", "gemini"),
            ("ollama", "ollama"),
        ):
            assert re.search(rf"{kind}\s+\S*\s*{re.escape(extra)}", result.output), kind

    async def test_a_missing_sdk_names_its_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "boto3" else real(name, *a),
        )
        result = await cli("providers")
        assert "no: install the bedrock extra" in result.output

    async def test_configured_providers_show_their_authentication_mode(
        self, tmp_path: Path
    ) -> None:
        path = write_config(tmp_path, bedrock_config())
        result = await cli("providers", "--config", path)
        assert re.search(
            r"short\s+bedrock\s+ap-southeast-2\s+short_term_api_key \(signed with "
            r"access_keys\)",
            result.output,
        )
        assert re.search(r"chain\s+bedrock\s+ap-southeast-2\s+credential_chain", result.output)
        assert re.search(r"long\s+bedrock\s+ap-southeast-2\s+long_term_api_key", result.output)


class TestModels:
    async def test_the_catalogue_is_listed_with_residency(self) -> None:
        result = await cli("models", "--provider", "bedrock")
        assert re.search(
            r"bedrock\s+global\.anthropic\.claude-sonnet-5\s+chat\s+global", result.output
        )
        assert "stub-echo" not in result.output

    async def test_overlays_are_applied(self, tmp_path: Path) -> None:
        overlay = {
            "name": "example",
            "add": [
                {
                    "model_id": "stub-extra",
                    "provider": "stub",
                    "residency_scope": "local",
                    "context_window": 4096,
                    "output_limit": 1024,
                }
            ],
        }
        path = write_config(tmp_path, {"conduit": stub_conduit(catalogue=[overlay])})
        result = await cli("models", "--config", path, "--provider", "stub")
        assert "stub-extra" in result.output and "stub-echo" in result.output

    async def test_table_cells_are_shown_as_written(self, tmp_path: Path) -> None:
        overlay = {
            "name": "example",
            "add": [
                {
                    "model_id": "stub-[bold]x[/bold]",
                    "provider": "stub",
                    "residency_scope": "local",
                    "context_window": 4096,
                    "output_limit": 1024,
                }
            ],
        }
        path = write_config(tmp_path, {"conduit": stub_conduit(catalogue=[overlay])})
        result = await cli("models", "--config", path, "--provider", "stub")
        assert "stub-[bold]x[/bold]" in result.output


class TestCheck:
    async def test_the_stub_demonstration_is_ready(self) -> None:
        result = await cli("check")
        assert result.code == 0
        assert "ready: no credentials needed" in result.output
        assert re.search(r"chat\s+stub\s+stub-echo\s+local\s+ready", result.output)
        assert "All providers and assignments are ready." in result.output

    @pytest.mark.usefixtures("no_aws")
    async def test_every_bedrock_mode_is_built_and_no_key_is_shown(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            bedrock_config(chat={"provider": "keys", "model": "au.anthropic.claude-sonnet-5"}),
        )
        result = await cli("check", "--config", path, environ=ENVIRON)
        assert result.code == 0, result.output
        assert "ready: SigV4 client built; AWS checks the credentials" in row(
            result.output, "chain"
        )
        assert "ready: keys resolved and SigV4 client built" in row(result.output, "keys")
        assert "ready: short-term API key generated" in row(result.output, "short")
        assert "ready: long-term API key resolved (never shown)" in row(result.output, "long")
        for secret in SECRETS:
            assert secret not in result.output

    @pytest.mark.usefixtures("no_aws")
    async def test_residency_refusals_are_shown_one_by_one(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            bedrock_config(
                chat={"provider": "keys", "model": "global.anthropic.claude-sonnet-5"},
                summary={"provider": "keys", "model": "au.anthropic.claude-sonnet-5"},
            ),
        )
        result = await cli("check", "--config", path, environ=ENVIRON)
        assert result.code == 1
        assert re.search(
            r"chat\s+keys\s+global\.anthropic\.claude-sonnet-5\s+global\s+refused: "
            r"ResidencyRefused: chat: model 'global\.anthropic\.claude-sonnet-5' \(global\): "
            r"scope global is not allowed",
            result.output,
        )
        assert re.search(
            r"summary\s+keys\s+au\.anthropic\.claude-sonnet-5\s+cross_region\s+"
            r"ready",
            result.output,
        )
        assert (
            "residency: scopes cross_region, in_region, local; regions ap-southeast-2, "
            "ap-southeast-4" in result.output
        )

    @pytest.mark.usefixtures("no_aws")
    async def test_a_region_outside_the_policy_is_refused(self, tmp_path: Path) -> None:
        config = bedrock_config(chat={"provider": "eu", "model": "eu.anthropic.claude-sonnet-5"})
        config["conduit"]["providers"]["eu"] = {"kind": "bedrock", "region": "eu-central-1"}
        result = await cli("check", "--config", write_config(tmp_path, config), environ=ENVIRON)
        assert "ResidencyRefused" in result.output
        assert "region eu-central-1 is not allowed" in result.output

    @pytest.mark.usefixtures("no_aws")
    async def test_a_missing_secret_is_named_by_its_reference(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, bedrock_config())
        result = await cli("check", "--config", path, environ={})
        assert result.code == 1
        assert "refused: SecretUnavailable: secret://EXAMPLE_KEY_ID is not set" in row(
            result.output, "keys"
        )

    @pytest.mark.usefixtures("no_aws")
    async def test_long_term_keys_are_refused_unless_allowed(self, tmp_path: Path) -> None:
        config = bedrock_config()
        config["conduit"]["providers"]["long"]["allow_long_term_api_keys"] = False
        result = await cli("check", "--config", write_config(tmp_path, config), environ=ENVIRON)
        assert result.code == 1
        assert "long-term Bedrock API keys are refused" in result.output

    async def test_an_undeclared_model_is_refused(self, tmp_path: Path) -> None:
        conduit = stub_conduit()
        conduit["assignments"]["chat"]["model"] = "stub-nothing"
        result = await cli("check", "--config", write_config(tmp_path, {"conduit": conduit}))
        assert result.code == 1
        assert "UndeclaredModel: model 'stub-nothing' is not declared" in result.output

    async def test_a_task_class_the_interface_does_not_declare_is_refused(
        self, tmp_path: Path
    ) -> None:
        conduit = stub_conduit()
        conduit["assignments"]["review"] = {"provider": "stub", "model": "stub-echo"}
        conduit["require_all_assignments"] = False
        result = await cli("check", "--config", write_config(tmp_path, {"conduit": conduit}))
        assert "refused: not a task class this interface declares" in result.output


class LeakySettings(ProviderSettings):
    kind: str = "leaky"
    api_key: SecretRef


class LeakyFactory:
    """An adapter whose client construction fails with the key in its message."""

    kind = "leaky"
    settings_model = LeakySettings

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        raise RuntimeError(f"rejected key {secrets['api_key'].get_secret_value()}")

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        raise ConfigurationError("never reached")


LEAKY = {
    "conduit": {
        "providers": {"leaky": {"kind": "leaky", "api_key": "secret://EXAMPLE_LONG_TERM_KEY"}},
        "assignments": {"chat": {"provider": "leaky", "model": "leaky-model"}},
        "catalogue": [
            {
                "name": "leaky",
                "add": [
                    {
                        "model_id": "leaky-model",
                        "provider": "leaky",
                        "residency_scope": "local",
                        "capabilities": {"tools": True},
                        "context_window": 4096,
                        "output_limit": 1024,
                    }
                ],
            }
        ],
    }
}


class TestSecretsAreNeverPrinted:
    async def test_a_failing_client_is_shown_by_its_class(self, tmp_path: Path) -> None:
        factories: list[ProviderFactory] = [*interface_factories(), LeakyFactory()]
        path = write_config(tmp_path, LEAKY)
        for command in ("check", "chat"):
            result = await cli(
                command, "--config", path, environ=ENVIRON, factories=factories, stdin="hello\n"
            )
            assert result.code == 1
            assert "RuntimeError" in result.output
            assert LONG_TERM_KEY not in result.output

    async def test_a_plaintext_credential_is_refused_by_field_without_its_value(
        self, tmp_path: Path
    ) -> None:
        conduit = stub_conduit()
        conduit["providers"]["stub"]["api_key"] = LONG_TERM_KEY
        result = await cli("check", "--config", write_config(tmp_path, {"conduit": conduit}))
        assert result.code == 1
        assert "conduit.providers.stub" in result.output
        assert LONG_TERM_KEY not in result.output
