"""No live variable's value reaches pytest's failure output, at any verbosity.

pytest prints frame arguments in long tracebacks and every frame's locals under ``-l``.
A child pytest runs synthetic failing tests under ``-vvv -l --tb=long`` with a
placeholder in the live variables (never a real credential), each failing through a
path the live suite really takes: a helper holding the redacted environment, a probe
step that fails with the value in its message, a probe client constructor that fails
after its keys are resolved, and the gate failing a live test with no needs. The
placeholder must appear nowhere in the output. Nothing reaches a network.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.live.needs import PREFIX

REPO_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "conduit-placeholder-not-a-credential-9f2c"

SYNTHETIC = """
import os
import sys
import types

import pytest

from tests.live.needs import LiveEnv
from tests.live.probes import _guarded, anthropic_limits, bedrock_control_client
from tests.live.targets import deployment_overlay

ENV = LiveEnv(
    {
        "CONDUIT_LIVE_ANTHROPIC_API_KEY": os.environ["CONDUIT_TEST_PLACEHOLDER"],
        "CONDUIT_LIVE_AWS_ACCESS_KEY_ID": os.environ["CONDUIT_TEST_PLACEHOLDER"],
        "CONDUIT_LIVE_AWS_SECRET_ACCESS_KEY": os.environ["CONDUIT_TEST_PLACEHOLDER"],
        "CONDUIT_LIVE_FOUNDRY_CONTEXT_WINDOW": "not-a-number",
        "CONDUIT_LIVE_FOUNDRY_OPENAI_API_KEY": os.environ["CONDUIT_TEST_PLACEHOLDER"],
    }
)


def test_a_helper_holding_the_environment_fails() -> None:
    environ = ENV
    deployment_overlay(environ, "openai", "offline-deployment")


def test_an_assertion_with_the_environment_in_scope_fails() -> None:
    environ = ENV
    assert environ is None


async def test_a_probe_step_failing_with_the_value_in_its_message() -> None:
    environ = ENV

    async def step() -> int:
        raise ValueError(environ["CONDUIT_LIVE_AWS_SECRET_ACCESS_KEY"])

    await _guarded(step)


async def test_a_probe_client_constructor_failing_after_its_keys_resolve() -> None:
    pytest.importorskip("boto3")
    await bedrock_control_client("not a region!", ENV)


class _RefusingClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        raise ValueError("the constructor refused its arguments")


async def test_a_keyed_client_constructor_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = _RefusingClient
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    await anthropic_limits("offline-model", ENV)


@pytest.mark.integration
def test_a_live_test_without_needs() -> None:
    raise AssertionError("the body ran")
"""


def test_no_placeholder_reaches_the_failure_output(tmp_path: Path) -> None:
    pytest.importorskip("boto3")
    test_file = tmp_path / "test_synthetic_redaction.py"
    test_file.write_text(SYNTHETIC, encoding="utf-8", newline="\n")
    env = {name: found for name, found in os.environ.items() if not name.startswith(PREFIX)}
    env |= {
        "CONDUIT_TEST_PLACEHOLDER": PLACEHOLDER,
        # In the process environment too, as a live run's would be: the gate reads it.
        "CONDUIT_LIVE_ANTHROPIC_API_KEY": PLACEHOLDER,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.live.conftest",
            "-o",
            "addopts=",
            "-o",
            "asyncio_mode=auto",
            "-vvv",
            "-l",
            "--tb=long",
            f"--rootdir={tmp_path}",
            str(test_file),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    # Reduced to booleans first: the output itself never reaches this test's own report.
    output = result.stdout + result.stderr
    leaked = PLACEHOLDER in output
    five_failed = "5 failed" in output
    gate_error = "1 error" in output
    frames_shown = "values redacted" in output
    del output
    assert five_failed, "the five synthetic failures did not all fail"
    assert gate_error, "the gate did not fail the live test without needs"
    assert frames_shown, "no frame holding the environment was printed"
    assert not leaked, "a live variable's value reached pytest's failure output"
