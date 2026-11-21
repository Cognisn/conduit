"""The warden sink in the cross-layer run: audit records hold metadata only, always.

The run from ``tests.unit.integrations.otel.test_capture_across_layers`` (a product tool
behind an approval, an MCP tool, a skill and a compaction, with a marker in every piece
of content) is sent to the warden sink as well, under each capture policy.
"""

from __future__ import annotations

import pytest

pytest.importorskip("warden.audit")
pytest.importorskip("mcp")
pytest.importorskip("opentelemetry.sdk.trace")

from warden.audit import ActionRegistry

from conduit.core.events import CapturePolicy
from conduit.integrations.warden import WardenAuditSink, register_conduit_actions
from tests.unit.integrations.otel.conftest import Telemetry, telemetry  # noqa: F401
from tests.unit.integrations.otel.test_capture_across_layers import MARKER, _observe
from tests.unit.integrations.warden.test_sink import Recorder


@pytest.mark.parametrize(
    "capture",
    [
        CapturePolicy(),
        CapturePolicy(prompts=True),
        CapturePolicy(prompts=True, tool_content=True),
    ],
    ids=["off", "prompts", "both"],
)
async def test_no_content_reaches_the_audit_trail_under_any_policy(
    capture: CapturePolicy,
    telemetry: Telemetry,  # noqa: F811 (the otel tests' fixture, imported above)
) -> None:
    registry = ActionRegistry()
    register_conduit_actions(registry)
    recorder = Recorder(registry)
    observed = await _observe(capture, telemetry, [WardenAuditSink(recorder)])
    if capture.prompts and capture.tool_content:
        assert any(MARKER in e.model_dump_json() for e in observed.events)  # not vacuous
    actions = {r.action for r in recorder.events}
    assert {
        "conduit.run_started",
        "conduit.model_call_finished",
        "conduit.tool_call",
        "conduit.mcp_call",
        "conduit.approval_requested",
        "conduit.approval_decided",
        "conduit.skill_loaded",
        "conduit.conversation_compacted",
        "conduit.compaction",
        "conduit.run_finished",
    } <= actions
    leaks = [r.action for r in recorder.events if MARKER in r.model_dump_json()]
    assert not leaks, leaks
