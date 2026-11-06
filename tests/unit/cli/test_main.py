"""The ``conduit`` entry point, its configuration errors, and ``python -m conduit.cli``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib import metadata
from io import StringIO
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import conduit
from conduit import ConfigurationError
from conduit.cli import main as entry
from conduit.cli.config import load_config
from tests.unit.cli.helpers import cli, console, write_config


class TestEntryPoint:
    def test_without_the_extra_it_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name, *a: None if name == "rich" else real(name, *a)
        )
        assert entry.main(["providers"]) == 2
        assert 'uvx --from "cognisn-conduit[cli]" conduit' in capsys.readouterr().err

    def test_it_runs_a_command(self) -> None:
        buffer = StringIO()
        from conduit.cli.app import main

        assert main(["check"], console=console(buffer)) == 0
        assert "All providers and assignments are ready." in buffer.getvalue()

    def test_it_runs_as_a_module(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "conduit.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == f"conduit {conduit.__version__}"


class TestArguments:
    async def test_no_command_shows_the_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = await cli()
        assert result.code == 2
        assert "providers" in capsys.readouterr().out

    async def test_a_bad_option_is_a_usage_error(self) -> None:
        assert (await cli("chat", "--max-tool-calls", "-1")).code == 2


class TestConfiguration:
    async def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        result = await cli("check", "--config", str(tmp_path / "absent.json"))
        assert result.code == 1
        assert "the configuration file cannot be read (FileNotFoundError)" in result.output

    async def test_broken_json_is_refused_by_position(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text('{"system": "zz-hidden-zz",', encoding="utf-8")
        result = await cli("check", "--config", str(path))
        assert "is not JSON (line 1" in result.output
        assert "zz-hidden-zz" not in result.output

    async def test_toml_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "conduit.toml"
        path.write_text("[budget]\nmax_model_calls = 1\n", encoding="utf-8", newline="\n")
        result = await cli("chat", "--config", str(path), stdin='call add {"a": 1, "b": 1}\n')
        assert "the model_calls budget is exhausted" in result.output

    async def test_an_unknown_field_is_named_without_its_value(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"sytem": "zz-misspelt-zz"})
        result = await cli("check", "--config", path)
        assert result.code == 1
        assert "the configuration was refused at sytem" in result.output
        assert "zz-misspelt-zz" not in result.output

    async def test_the_skills_directory_is_relative_to_the_file(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills" / "example-skill"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text(
            "---\nname: example-skill\ndescription: A synthetic skill for example.com.\n---\n"
            "Say hello.\n",
            encoding="utf-8",
            newline="\n",
        )
        path = write_config(tmp_path, {"skills": "skills"})
        result = await cli("chat", "--config", path, stdin="/skills\n")
        assert "example-skill" in result.output


class TestConfigurationErrors:
    """A refused file raises an error that keeps nothing of what it was raised for (D0040)."""

    @pytest.mark.parametrize(
        ("name", "text"),
        [
            ("broken.json", '{"system": "zz-hidden-zz",'),
            ("broken.toml", 'system = "zz-hidden-zz'),
            ("list.json", '["zz-hidden-zz"]'),
            ("unknown.json", '{"sytem": "zz-hidden-zz"}'),
            (
                "plaintext.json",
                '{"conduit": {"providers": {"p": {"kind": "stub", "api_key": "zz-hidden-zz"}}}}',
            ),
        ],
    )
    def test_the_error_has_no_cause_context_or_value(
        self, tmp_path: Path, name: str, text: str
    ) -> None:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigurationError) as caught:
            load_config(path)
        error = caught.value
        assert error.__cause__ is None and error.__context__ is None
        assert "zz-hidden-zz" not in str(error) and "zz-hidden-zz" not in repr(error)
        assert all("zz-hidden-zz" not in str(arg) for arg in error.args)

    def test_an_unreadable_file_has_no_cause_or_context(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError) as caught:
            load_config(tmp_path / "absent.json")
        assert caught.value.__cause__ is None and caught.value.__context__ is None


def _extra(extra: str) -> set[str]:
    found: set[str] = set()
    for spec in metadata.requires("cognisn-conduit") or []:
        requirement = Requirement(spec)
        if requirement.marker is not None and requirement.marker.evaluate({"extra": extra}):
            requirement.marker = None
            found.add(str(requirement))
    return found


def test_the_cli_extra_brings_rich_and_the_mcp_extra() -> None:
    # Q0039: the uvx form connects MCP servers as it is; provider SDKs stay in their extras.
    # The build writes the mcp extra's own requirements in place of the self-reference.
    rich = {r for r in _extra("cli") if Requirement(r).name == "rich"}
    assert len(rich) == 1
    assert _extra("mcp")
    assert _extra("cli") == rich | _extra("mcp")
