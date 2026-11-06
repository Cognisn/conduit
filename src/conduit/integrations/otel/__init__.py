"""OpenTelemetry spans and metrics for conduit events, behind the otel extra."""

# OtelEventSink is an event sink: pass it to a FanOutSink with the product's other sinks.
# It exports metadata only (identifiers, counts, durations, outcomes, and the hash and size
# of content), never content, whatever the capture policy allows. See
# conduit.integrations.otel.sink for its spans and metrics.

from conduit.integrations.otel.attributes import AttributeValue, metadata_attributes
from conduit.integrations.otel.sink import (
    DEFAULT_MAX_OPEN_SPANS,
    METRIC_NAMES,
    SPAN_NAMES,
    OtelEventSink,
)

__all__ = [
    "DEFAULT_MAX_OPEN_SPANS",
    "METRIC_NAMES",
    "SPAN_NAMES",
    "AttributeValue",
    "OtelEventSink",
    "metadata_attributes",
]
