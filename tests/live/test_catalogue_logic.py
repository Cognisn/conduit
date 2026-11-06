"""The catalogue pass, the probes' guard and the report, proven offline on the stub.

Runs in the default suite and reaches no network: every target is the stub provider,
declared by the default catalogue (``stub-echo``) or by a test overlay.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from conduit.core.catalogue import Capabilities, CatalogueOverlay, ModelEntry, default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.residency import ResidencyScope
from conduit.core.types import ModelRequest
from conduit.providers.stub import (
    StubFactory,
    StubToolCall,
    StubTurn,
    answer_tool_results,
    scripted,
)
from tests.live.harness import (
    ask,
    check_residency_refusal,
    started,
    try_complete,
)
from tests.live.probes import ProbeFailed, _guarded
from tests.live.report import REPORT, Report, Verdict
from tests.live.targets import Target
from tests.live.test_catalogue import (
    compare,
    compare_regions,
    fail_on_mismatches,
    verify_chat,
    verify_output_limit,
)

STUB = {"kind": "stub", "not_for_production": True}
DECOY = "provider-text-that-must-not-travel"


@pytest.fixture(autouse=True)
def _report_left_as_found() -> Iterator[None]:
    kept = len(REPORT.lines)
    yield
    del REPORT.lines[kept:]


def _lines(subject: str) -> dict[str, tuple[Verdict, str]]:
    return {
        line.check: (line.verdict, line.detail) for line in REPORT.lines if line.subject == subject
    }


def _overlay(entry: ModelEntry) -> tuple[CatalogueOverlay, ...]:
    return (CatalogueOverlay(name="offline", add=(entry,)),)


def _quiet_entry() -> ModelEntry:
    return ModelEntry(
        model_id="stub-quiet",
        provider="stub",
        residency_scope=ResidencyScope.LOCAL,
        capabilities=Capabilities(tools=True),
        context_window=8192,
        output_limit=4096,
    )


# compare and compare_regions -------------------------------------------------------------


def test_compare_records_each_verdict() -> None:
    compare("s", "equal", 100, 100)
    compare("s", "differs", 100, 200)
    compare("s", "unstated", 100, None)
    found = _lines("s")
    assert found["equal"] == (Verdict.MATCH, "100")
    assert found["differs"] == (Verdict.MISMATCH, "catalogue 100, API 200")
    assert found["unstated"] == (Verdict.UNOBSERVED, "not stated by the API")


def test_compare_regions_records_a_match_and_names_each_difference() -> None:
    compare_regions("same", frozenset({"a", "b"}), frozenset({"b", "a"}))
    compare_regions("differ", frozenset({"a", "b"}), frozenset({"b", "c"}))
    assert _lines("same")["profile region list"][0] is Verdict.MATCH
    verdict, detail = _lines("differ")["profile region list"]
    assert verdict is Verdict.MISMATCH
    assert detail == "routed but not catalogued: c; catalogued but not routed: a"


# fail_on_mismatches ---------------------------------------------------------------------


def test_a_mismatch_fails_the_entry_and_names_the_check() -> None:
    REPORT.record("catalogue", "entry", "vision", Verdict.MISMATCH, "x")
    REPORT.record("catalogue", "other", "tools", Verdict.MISMATCH, "x")
    with pytest.raises(AssertionError) as failed:
        fail_on_mismatches(["entry"])
    assert "entry :: vision" in str(failed.value)
    assert "other" not in str(failed.value)


def test_unobserved_and_findings_do_not_fail() -> None:
    REPORT.record("catalogue", "entry", "limits", Verdict.UNOBSERVED)
    REPORT.record("catalogue", "entry", "regions", Verdict.FINDING)
    REPORT.record("catalogue", "entry", "invocable", Verdict.MATCH)
    fail_on_mismatches(["entry"])


# verify_chat on the stub ------------------------------------------------------------------


async def test_verify_chat_records_a_real_mismatch_that_fails_the_entry() -> None:
    """stub-echo is catalogued for structured output, but echoes the prompt."""
    target = Target("stub", dict(STUB), "stub-echo")
    await verify_chat(default_catalogue().require("stub", "stub-echo"), target)
    found = _lines(target.subject)
    assert found["invocable"][0] is Verdict.MATCH
    assert found["tools"][0] is Verdict.UNOBSERVED  # the echo stub never calls the tool
    assert found["structured output"][0] is Verdict.MISMATCH
    assert found["vision"][0] is Verdict.MATCH
    assert found["output limit 128000 accepted"][0] is Verdict.MATCH
    with pytest.raises(AssertionError) as failed:
        fail_on_mismatches([target.subject])
    assert "structured output" in str(failed.value)


async def test_verify_chat_with_only_unobserved_checks_passes() -> None:
    entry = _quiet_entry()
    target = Target("stub", dict(STUB), entry.model_id, _overlay(entry))
    await verify_chat(entry, target)
    found = _lines(target.subject)
    assert found["tools"][0] is Verdict.UNOBSERVED
    assert found["output limit 4096 accepted"][0] is Verdict.MATCH
    fail_on_mismatches([target.subject])


async def test_a_tool_loop_defect_is_a_mismatch_not_unobserved() -> None:
    other = StubTurn(tool_calls=(StubToolCall("other_tool", {"code": "R-1"}),))
    loop = answer_tool_results(other, lambda results: StubTurn(text="done"))

    def script(request: ModelRequest, index: int) -> StubTurn:
        # Plain chat answers; offered tools, it calls the wrong one.
        return loop(request, index) if request.tools else StubTurn(text="ready")

    entry = _quiet_entry()
    target = Target("stub", dict(STUB) | {"script": "other"}, entry.model_id, _overlay(entry))
    await verify_chat(entry, target, factories=[StubFactory(scripts={"other": script})])
    verdict, detail = _lines(target.subject)["tools"]
    assert verdict is Verdict.MISMATCH
    assert detail == "a tool call named another tool"


async def test_a_refused_invocation_is_a_mismatch_with_the_failure_class_only() -> None:
    script = scripted([ProviderError(ProviderFailure.REJECTED)])
    entry = _quiet_entry()
    target = Target("stub", dict(STUB) | {"script": "no"}, entry.model_id, _overlay(entry))
    await verify_chat(entry, target, factories=[StubFactory(scripts={"no": script})])
    assert _lines(target.subject)["invocable"] == (Verdict.MISMATCH, "ProviderError(rejected)")


async def test_an_entry_without_an_output_limit_is_unobserved() -> None:
    entry = ModelEntry.model_construct(
        model_id="stub-echo",
        provider="stub",
        kind="chat",
        residency_scope=ResidencyScope.LOCAL,
        regions=frozenset(),
        capabilities=Capabilities(),
        context_window=8192,
        output_limit=None,
        embedding_dimensions=None,
        labels=frozenset(),
    )
    target = Target("stub", dict(STUB), "stub-echo")
    await verify_output_limit(entry, target)
    assert _lines(target.subject) == {"output limit": (Verdict.UNOBSERVED, "no output limit")}


# try_complete and the probe guard --------------------------------------------------------


async def test_try_complete_reports_the_failure_class_only() -> None:
    script = scripted([ProviderError(ProviderFailure.REJECTED)])
    target = Target("stub", dict(STUB) | {"script": "no"}, "stub-echo")
    async with started(target, factories=[StubFactory(scripts={"no": script})]) as binding:
        response, failure = await try_complete(binding, binding.request(messages=ask("x")))
    assert response is None
    assert failure == "ProviderError(rejected)"


async def test_the_probe_guard_keeps_the_class_name_and_nothing_else() -> None:
    async def step() -> int:
        raise ValueError(DECOY)

    with pytest.raises(ProbeFailed) as failed:
        await _guarded(step)
    error = failed.value
    assert str(error) == "ValueError"
    assert error.args == ("ValueError",)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert DECOY not in repr(error)


async def test_the_probe_guard_passes_its_own_reasons_and_cancellation_on() -> None:
    async def own() -> int:
        raise ProbeFailed("model not pulled on the Ollama server")

    async def cancelled() -> int:
        raise asyncio.CancelledError

    async def fine() -> int:
        return 7

    with pytest.raises(ProbeFailed) as failed:
        await _guarded(own)
    assert str(failed.value) == "model not pulled on the Ollama server"
    with pytest.raises(asyncio.CancelledError):
        await _guarded(cancelled)
    assert await _guarded(fine) == 7


# Residency (the assertion the live Bedrock test relies on) -----------------------------


def _regional(model: str, scope: ResidencyScope, regions: frozenset[str]) -> Target:
    entry = ModelEntry(
        model_id=model,
        provider="stub",
        residency_scope=scope,
        regions=regions,
        capabilities=Capabilities(tools=True),
        context_window=8192,
        output_limit=4096,
    )
    return Target("stub", dict(STUB) | {"region": "eu-west-2"}, model, _overlay(entry))


async def test_an_in_region_policy_refuses_a_cross_region_entry_at_start() -> None:
    target = _regional(
        "stub-cross", ResidencyScope.CROSS_REGION, frozenset({"eu-west-1", "eu-west-2"})
    )
    await check_residency_refusal(target, "eu-west-2", "stub residency")
    assert _lines("stub residency")["cross-region refused under in-region policy"][0] is (
        Verdict.MATCH
    )


async def test_the_residency_check_fails_when_nothing_is_refused() -> None:
    target = _regional("stub-local-region", ResidencyScope.IN_REGION, frozenset({"eu-west-2"}))
    with pytest.raises(AssertionError) as failed:
        await check_residency_refusal(target, "eu-west-2", "stub residency")
    assert "not refused" in str(failed.value)


# The report -----------------------------------------------------------------------------


def test_render_groups_by_area_with_counts() -> None:
    report = Report()
    assert report.render() == ["no live check ran (every live test was skipped or deselected)"]
    report.record("tools", "s1", "tool loop", Verdict.MATCH, "1 call(s)")
    report.record("chat", "s1", "streaming chat", Verdict.MISMATCH)
    report.record("chat", "s2", "streaming chat", Verdict.UNOBSERVED, "d")
    assert report.render() == [
        "[chat]",
        "  mismatch   s1 :: streaming chat",
        "  unobserved s2 :: streaming chat (d)",
        "[tools]",
        "  match      s1 :: tool loop (1 call(s))",
        "match 1, mismatch 1, unobserved 1, finding 0",
    ]


def test_write_json_holds_every_line(tmp_path: Path) -> None:
    report = Report()
    report.record("usage", "xai grok-4.3", "Q0021", Verdict.FINDING, "inside")
    target = tmp_path / "report.json"
    report.write_json(target)
    assert json.loads(target.read_text(encoding="utf-8")) == [
        {
            "area": "usage",
            "subject": "xai grok-4.3",
            "check": "Q0021",
            "verdict": "finding",
            "detail": "inside",
        }
    ]
    assert b"\r\n" not in target.read_bytes()


def test_mismatches_are_found_by_subject() -> None:
    report = Report()
    report.record("a", "s1", "c", Verdict.MISMATCH)
    report.record("a", "s2", "c", Verdict.MATCH)
    assert [line.subject for line in report.mismatches()] == ["s1"]
    assert report.mismatches("s2") == []
