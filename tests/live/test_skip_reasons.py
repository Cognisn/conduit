"""The live suite's skip guard. Runs in the default suite; never runs a live test.

It proves, with no credential in the environment:

- the live suite collects cleanly, and every live test is marked ``integration`` and
  states its needs, so the default run deselects it and the gate can check it;
- every collected live test skips with a stated reason naming its variables when they
  are absent, and is let through when they are set;
- the gate itself skips a live test before its body runs, and fails one that states no
  needs, in a child pytest running a synthetic test (never a live one).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from conduit.core.catalogue import default_catalogue
from tests.live import needs
from tests.live.needs import ALL_NEEDS, PREFIX, Need, check
from tests.live.test_catalogue import PROVIDER_NEEDS

LIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = LIVE_DIR.parents[1]
#: Modules that hold live tests; each must collect at least one.
LIVE_MODULES = frozenset(
    {
        "test_anthropic.py",
        "test_bedrock.py",
        "test_bedrock_mantle.py",
        "test_catalogue.py",
        "test_embeddings.py",
        "test_gemini_ollama.py",
        "test_openai_compat.py",
    }
)
#: Modules that run offline in the default suite: never integration, never gated.
OFFLINE_MODULES = frozenset({"test_skip_reasons.py", "test_harness.py"})


def without_live_variables() -> dict[str, str]:
    """This process's environment with every ``CONDUIT_LIVE_`` variable removed."""
    return {name: found for name, found in os.environ.items() if not name.startswith(PREFIX)}


def everything_set(needs_found: tuple[Need, ...]) -> dict[str, str]:
    """An environment meeting ``needs_found``, with a placeholder in every variable."""
    return {name: "set" for need in needs_found for name in need.variables()}


def run_pytest(args: list[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REPO_ROOT,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.fixture(scope="module")
def collected(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Every item under tests/live, collected (not run) with no live variable set."""
    output = tmp_path_factory.mktemp("collect") / "items.json"
    env = without_live_variables() | {"CONDUIT_TEST_COLLECT_OUTPUT": str(output)}
    result = run_pytest(
        [
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "tests.live.guard_plugin",
            str(LIVE_DIR),
        ],
        env,
    )
    assert result.returncode == 0, f"tests/live did not collect cleanly (exit {result.returncode})"
    items: list[dict[str, Any]] = json.loads(output.read_text(encoding="utf-8"))
    return items


def _needs(item: Mapping[str, Any]) -> tuple[Need, ...]:
    return tuple(
        Need(found["label"], tuple(tuple(group) for group in found["groups"]))
        for found in item["needs"]
    )


def _live(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if _module(item) not in OFFLINE_MODULES]


def _module(item: Mapping[str, Any]) -> str:
    return Path(str(item["nodeid"]).split("::")[0]).name


def test_the_live_suite_collects_from_every_module(collected: list[dict[str, Any]]) -> None:
    modules = {_module(item) for item in _live(collected)}
    assert modules == LIVE_MODULES
    present_offline = {_module(item) for item in collected} & OFFLINE_MODULES
    assert present_offline == OFFLINE_MODULES
    offline_integration = [
        item["nodeid"]
        for item in collected
        if _module(item) in OFFLINE_MODULES and item["integration"]
    ]
    assert offline_integration == []
    entries = sum(1 for e in default_catalogue() if e.provider != "stub")
    catalogue_items = sum(1 for i in collected if "test_default_catalogue_entry" in i["nodeid"])
    assert catalogue_items == entries, "the catalogue pass does not cover every declared model"


def test_every_live_test_is_integration_and_states_its_needs(
    collected: list[dict[str, Any]],
) -> None:
    live = _live(collected)
    unmarked = [item["nodeid"] for item in live if not item["integration"]]
    needless = [item["nodeid"] for item in live if not item["needs"]]
    assert len(live) > 0
    assert unmarked == [], "live tests not marked integration"
    assert needless == [], "live tests that state no needs"


def test_every_live_test_skips_with_its_variables_named_when_they_are_absent(
    collected: list[dict[str, Any]],
) -> None:
    for item in _live(collected):
        found = _needs(item)
        reason = ""
        with pytest.raises(pytest.skip.Exception) as skipped:
            check(found, {})
        reason = str(skipped.value.msg)
        unnamed = sorted(name for need in found for name in need.variables() if name not in reason)
        assert reason, f"{item['nodeid']} skips with no reason"
        assert unnamed == [], f"{item['nodeid']} skips without naming {unnamed}"


def test_every_live_test_is_let_through_when_its_variables_are_set(
    collected: list[dict[str, Any]],
) -> None:
    for item in _live(collected):
        found = _needs(item)
        check(found, everything_set(found))


def test_a_reason_names_variables_and_never_a_value() -> None:
    secretish = "do-not-echo-this-value"
    reason = ""
    with pytest.raises(pytest.skip.Exception) as skipped:
        check(
            (needs.BEDROCK_REGION, needs.ANTHROPIC_DIRECT),
            {"CONDUIT_LIVE_BEDROCK_REGION": secretish},
        )
    reason = str(skipped.value.msg)
    assert "CONDUIT_LIVE_ANTHROPIC_API_KEY" in reason
    assert "CONDUIT_LIVE_BEDROCK_REGION" not in reason  # that need is met
    assert secretish not in reason


def test_an_alternative_need_is_met_by_any_one_group() -> None:
    check((needs.BEDROCK_ANY_AUTH,), {"CONDUIT_LIVE_BEDROCK_API_KEY": "set"})
    check(
        (needs.BEDROCK_ANY_AUTH,),
        {"CONDUIT_LIVE_AWS_ACCESS_KEY_ID": "set", "CONDUIT_LIVE_AWS_SECRET_ACCESS_KEY": "set"},
    )
    with pytest.raises(pytest.skip.Exception):
        check((needs.BEDROCK_ANY_AUTH,), {"CONDUIT_LIVE_AWS_ACCESS_KEY_ID": "set"})
    with pytest.raises(pytest.skip.Exception):
        check((needs.BEDROCK_ANY_AUTH,), {"CONDUIT_LIVE_BEDROCK_API_KEY": "  "})


def test_a_live_test_with_no_needs_fails() -> None:
    with pytest.raises(pytest.fail.Exception):
        check((), {})


def test_every_need_is_listed_and_prefixed() -> None:
    names = {name for need in ALL_NEEDS for name in need.variables()}
    assert all(name.startswith(PREFIX) for name in names)
    defined = {value for value in vars(needs).values() if isinstance(value, Need)}
    assert defined == set(ALL_NEEDS)


def test_every_declared_provider_has_catalogue_needs() -> None:
    declared = {entry.provider for entry in default_catalogue()} - {"stub"}
    unknown = sorted(declared - set(PROVIDER_NEEDS))
    assert unknown == [], "catalogue providers without needs in the verification pass"


SYNTHETIC = """
import pytest

from tests.live import needs

pytestmark = pytest.mark.integration


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
def test_gated() -> None:
    raise AssertionError("the body ran")


def test_without_needs() -> None:
    raise AssertionError("the body ran")


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
@pytest.mark.parametrize("n", [1])
def test_gated_parametrised(n: int) -> None:
    raise AssertionError("the body ran")
"""


def _run_synthetic(tmp_path: Path, env: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """Each synthetic test's outcome and message, from a child pytest."""
    test_file = tmp_path / "test_synthetic_live.py"
    test_file.write_text(SYNTHETIC, encoding="utf-8", newline="\n")
    output = tmp_path / "outcomes.json"
    run_pytest(
        [
            "-p",
            "tests.live.conftest",
            "-p",
            "tests.live.guard_plugin",
            "-o",
            "addopts=",
            "-m",
            "integration",
            f"--rootdir={tmp_path}",
            str(test_file),
        ],
        dict(env) | {"CONDUIT_TEST_OUTCOME_OUTPUT": str(output)},
    )
    found: dict[str, dict[str, str]] = json.loads(output.read_text(encoding="utf-8"))
    return {name: (case["outcome"], case["message"]) for name, case in found.items()}


def test_the_gate_skips_before_the_body_and_fails_a_test_without_needs(tmp_path: Path) -> None:
    outcomes = _run_synthetic(tmp_path, without_live_variables())
    gated, gated_message = outcomes["test_gated"]
    parametrised, parametrised_message = outcomes["test_gated_parametrised[1]"]
    unneeded, unneeded_message = outcomes["test_without_needs"]
    assert gated == "skipped"
    assert "CONDUIT_LIVE_ANTHROPIC_API_KEY" in gated_message
    assert parametrised == "skipped"
    assert "CONDUIT_LIVE_ANTHROPIC_API_KEY" in parametrised_message
    assert unneeded == "error"
    assert "must state its needs" in unneeded_message


def test_the_gate_lets_a_test_through_when_its_needs_are_met(tmp_path: Path) -> None:
    env = without_live_variables() | {"CONDUIT_LIVE_ANTHROPIC_API_KEY": "placeholder"}
    outcomes = _run_synthetic(tmp_path, env)
    gated, _ = outcomes["test_gated"]
    assert gated == "failed"  # the body ran, and raised
