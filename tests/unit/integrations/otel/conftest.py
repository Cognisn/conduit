"""Fixtures for the otel integration tests: in-memory exporters only, never a collector.

The OpenTelemetry API is the otel extra's dependency and the SDK a dev dependency; these
tests skip when either is missing. Each test gets its own tracer and meter providers,
built here, so nothing is read from the process-wide providers or the ``OTEL_*``
environment, and nothing leaves the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("opentelemetry.trace")
pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@dataclass
class Telemetry:
    """A tracer provider and a meter provider exporting to memory."""

    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    exporter: InMemorySpanExporter
    reader: InMemoryMetricReader

    def spans(self) -> list[ReadableSpan]:
        return list(self.exporter.get_finished_spans())

    def span(self, name: str) -> ReadableSpan:
        found = [s for s in self.spans() if s.name == name]
        assert len(found) == 1, [s.name for s in self.spans()]
        return found[0]

    def points(self, name: str) -> list[Any]:
        """Every data point of the metric ``name``, across every scope."""
        data = self.reader.get_metrics_data()
        found: list[Any] = []
        if data is None:
            return found
        for resource in data.resource_metrics:
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    if metric.name == name:
                        found += list(metric.data.data_points)
        return found

    def metric_names(self) -> set[str]:
        data = self.reader.get_metrics_data()
        if data is None:
            return set()
        return {
            metric.name
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }


@pytest.fixture
def telemetry() -> Iterator[Telemetry]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    yield Telemetry(tracer_provider, meter_provider, exporter, reader)
    tracer_provider.shutdown()
    meter_provider.shutdown()
