"""warden audit trail adapter for conduit events, behind the warden extra."""

# Register conduit's actions in the product's warden ActionRegistry with
# register_conduit_actions, then pass a WardenAuditSink wrapping the product's
# AuditRecorder to a FanOutSink. Each audited event becomes a conduit.<event type> action
# in warden's hash-chained trail, with metadata only, never content. See
# conduit.integrations.warden.sink.

from conduit.integrations.warden.sink import (
    ACTION_NAMESPACE,
    CONDUIT_ACTIONS,
    AuditRecordFailed,
    AuditRecording,
    ConduitAction,
    WardenAuditSink,
    action_for,
    register_conduit_actions,
)

__all__ = [
    "ACTION_NAMESPACE",
    "CONDUIT_ACTIONS",
    "AuditRecordFailed",
    "AuditRecording",
    "ConduitAction",
    "WardenAuditSink",
    "action_for",
    "register_conduit_actions",
]
