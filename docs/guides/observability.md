# Audit and telemetry

conduit emits a structured event for every run, model call, tool call, MCP call, approval, skill load, compaction, hand-off and embedding call. A product receives them through its event sink (see [Integrating conduit](integration.md#events-and-the-capture-policy)). Two optional extras turn those events into an audit trail and into telemetry:

- `warden` records events in warden's hash-chained audit trail, as registered actions;
- `otel` produces OpenTelemetry spans and metrics.

**Both record metadata only, never content**, whatever the capture policy allows an event to carry: identifiers, counts, durations, outcomes, usage, and the hash and size of each piece of content. Prompt and tool text, where a deployment captures it at all, stays in the product's own sinks.

Both are event sinks. Combine them with the product's own in a `FanOutSink`, and pass that to `start`, the agent runner, the compactor, the MCP bridge and the skill session:

```python
import asyncio

from conduit import (
    CapturePolicy,
    ConduitSettings,
    FanOutSink,
    LiteralSecretResolver,
    Message,
    TaskClass,
    start,
)
from conduit.providers.stub import StubFactory

settings = ConduitSettings.model_validate(
    {
        "providers": {"local": {"kind": "stub", "not_for_production": True}},
        "assignments": {"triage": {"provider": "local", "model": "stub-echo"}},
        "capture": {"prompts": True},  # the product's own log may keep prompts
    }
)


async def one_call(events: FanOutSink) -> None:
    runtime = await start(
        settings,
        task_classes=[TaskClass(name="triage")],
        factories=[StubFactory()],
        secrets=LiteralSecretResolver({}),
        events=events,
    )
    try:
        binding = runtime.for_task("triage")
        await binding.provider.complete(
            binding.request(messages=[Message.user("Classify: disk full on 192.0.2.10")])
        )
    finally:
        await runtime.aclose()
```

## warden

```sh
pip install "cognisn-conduit[warden]"
```

The extra installs cognisn-warden, which is published as pre-releases only (the extra pins one, so resolvers accept it). warden's audit recorder writes to PostgreSQL through psycopg, which is licensed LGPL-3.0: conduit's licence policy accepts it as a named, reviewed exception for this extra. The extra brings in about 37 packages in all; install it only where the product records an audit trail.

Each audited event type is a warden action named `conduit.<event type>` (`conduit.model_call_finished`, `conduit.tool_call`, `conduit.run_finished` and so on), with a severity and a summary. **Register conduit's actions** in the product's warden `ActionRegistry` with `register_conduit_actions` before recording: the sink records an event only when its action is registered, so nothing unregistered ever reaches the trail.

```python
from warden.audit import ActionRegistry, AuditEvent

from conduit.integrations.warden import WardenAuditSink, register_conduit_actions


class ExampleRecorder:
    """Stands in for warden's AuditRecorder: its registry and record()."""

    def __init__(self) -> None:
        self.registry = ActionRegistry()
        self.records: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> int:
        self.records.append(event)
        return len(self.records)


recorder = ExampleRecorder()
register_conduit_actions(recorder.registry)
audit = WardenAuditSink(recorder, actor="service:example-app")

asyncio.run(one_call(FanOutSink([audit], policy=settings.capture)))

(record,) = recorder.records
assert record.action == "conduit.model_call_finished"
assert record.outcome == "success"
assert "192.0.2.10" not in str(record.details)  # metadata only, never the prompt
```

In a product, `recorder` is warden's own `AuditRecorder`, built the way warden documents. `actions=` limits the sink to some event types (and `register_conduit_actions(registry, event_types)` registers only those); `actor` names who the records are for. A record's outcome is `failure` for a failed call or run, a refusal, an exhausted budget or an invalid output, and `success` otherwise. The stream's own events (text deltas, and the started events whose finished event is audited) are not audit actions.

A recorder that fails raises `AuditRecordFailed` naming the action only; through a `FanOutSink` that is logged and the other sinks still receive the event. warden's trail is append-only and cannot forget, which is why the sink never writes content to it.

## OpenTelemetry

```sh
pip install "cognisn-conduit[otel]" opentelemetry-sdk
```

The `otel` extra holds the OpenTelemetry API only. The product supplies the SDK, its exporters and their configuration (including any `OTEL_*` variables): conduit configures no exporter and reads no environment variable. `OtelEventSink` uses the tracer and meter providers it is given, or OpenTelemetry's global ones.

**Export through `BatchSpanProcessor` and a periodic metric reader.** The sink runs on the event loop; a span processor that exports synchronously (`SimpleSpanProcessor` with a network exporter) blocks the loop while it exports, while the batch processor and the periodic reader export on their own threads.

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conduit.integrations.otel import OtelEventSink

spans = InMemorySpanExporter()  # a product exports to its collector instead
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(spans))
# A product uses PeriodicExportingMetricReader with its exporter; this reader is read here.
metric_reader = InMemoryMetricReader()
meter_provider = MeterProvider(metric_readers=[metric_reader])

telemetry = OtelEventSink(tracer_provider=tracer_provider, meter_provider=meter_provider)
asyncio.run(one_call(FanOutSink([telemetry], policy=settings.capture)))
tracer_provider.force_flush()

(span,) = spans.get_finished_spans()
assert span.name == "conduit.model_call"
assert not any("192.0.2.10" in str(value) for value in span.attributes.values())

metric_names = {
    metric.name
    for resource in metric_reader.get_metrics_data().resource_metrics
    for scope in resource.scope_metrics
    for metric in scope.metrics
}
assert "conduit.model.tokens" in metric_names

tracer_provider.shutdown()
meter_provider.shutdown()
```

### Spans

Each run is one trace. `conduit.run` spans a run; its children are `conduit.model_call`, `conduit.tool_call`, `conduit.skill_load`, `conduit.compaction` and `conduit.hand_off`. `conduit.approval` and `conduit.mcp_call` are children of the tool call they belong to, and `conduit.embedding_call` spans an embedding call. Output-contract failures, skill use, budget exhaustion and the runner's record of a compacted conversation are span events on the run span. A failure sets the span's status to error, described by an identifier only. Spans left open are bounded (`max_open_spans`); the oldest is ended and marked `conduit.incomplete` when the bound is reached.

### Metrics

| Metric | Records |
|---|---|
| `conduit.model.tokens` | tokens by kind (`conduit.token.type`: `input`, `output`, `cache_read`, `cache_write`), for chat and embedding calls |
| `conduit.model.calls`, `conduit.model.cache_hits` | model calls, and those that read from the prompt cache |
| `conduit.operation.duration` | latency of runs, model, embedding, tool and MCP calls |
| `conduit.errors` | failures by operation and error type |
| `conduit.budget.exhausted` | runs stopped by a budget |
| `conduit.run.tokens`, `conduit.run.model_calls`, `conduit.run.tool_calls` | what each run used of the budgets that limit it |
| `conduit.hand_offs`, `conduit.compactions` | hand-offs by outcome and task classes; compactions by strategy, trigger and outcome |

Metric attributes are low-cardinality identifiers only (provider, model, task class, outcome), never a run, call or event id. The sink never raises into the emitter: a failure in the OpenTelemetry providers is logged by exception class only.
