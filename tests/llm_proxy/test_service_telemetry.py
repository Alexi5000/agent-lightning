# Copyright (c) Microsoft. All rights reserved.

from datetime import datetime

import pytest
from litellm.types.services import ServiceLoggerPayload, ServiceTypes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlightning.llm_proxy import LightningOpenTelemetry


@pytest.mark.asyncio
async def test_parentless_service_event_does_not_require_timestamps():
    telemetry = LightningOpenTelemetry.__new__(LightningOpenTelemetry)
    payload = ServiceLoggerPayload(
        is_error=False, service=ServiceTypes.LITELLM, duration=0.0, call_type="completion", event_metadata=None
    )
    await telemetry.async_service_success_hook(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_datetime", [False, True])
async def test_parented_service_event_retains_span_and_timestamps(as_datetime: bool):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = LightningOpenTelemetry.__new__(LightningOpenTelemetry)
    telemetry.tracer = provider.get_tracer("service-test")
    payload = ServiceLoggerPayload(
        is_error=False, service=ServiceTypes.LITELLM, duration=1.0, call_type="completion", event_metadata=None
    )
    start, end = (datetime.fromtimestamp(100), datetime.fromtimestamp(101)) if as_datetime else (100.0, 101.0)
    with telemetry.tracer.start_as_current_span("request") as parent:
        await telemetry.async_service_success_hook(payload, parent, start, end, {"purpose": "regression"})
    spans = exporter.get_finished_spans()
    service_span = next(span for span in spans if span.name == ServiceTypes.LITELLM)
    assert service_span.start_time == 100_000_000_000
    assert service_span.end_time == 101_000_000_000
    assert service_span.parent is not None
    assert service_span.parent.span_id == parent.get_span_context().span_id
    assert service_span.attributes["purpose"] == "regression"
    provider.shutdown()


@pytest.mark.asyncio
async def test_parented_missing_timestamps_still_reports_upstream_error():
    provider = TracerProvider()
    telemetry = LightningOpenTelemetry.__new__(LightningOpenTelemetry)
    telemetry.tracer = provider.get_tracer("service-test")
    payload = ServiceLoggerPayload(
        is_error=False, service=ServiceTypes.LITELLM, duration=0.0, call_type="completion", event_metadata=None
    )
    with telemetry.tracer.start_as_current_span("request") as parent:
        with pytest.raises(AttributeError, match="timestamp"):
            await telemetry.async_service_success_hook(payload, parent)
    provider.shutdown()
