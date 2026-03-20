# tests/test_vitals_service.py
import asyncio
import pytest
import numpy as np
from server.services.event_emitter import EventEmitter
from server.services.vitals_service import VitalsService


@pytest.fixture
def emitter():
    return EventEmitter()


@pytest.fixture
def service(emitter):
    return VitalsService(sample_rate=100.0, emitter=emitter)


def test_push_csi(service):
    amp = np.random.rand(56).astype(np.float32)
    service.push_csi(amp)
    assert len(service.extractor.csi_buffer) == 1


@pytest.mark.asyncio
async def test_emits_vitals_on_interval(emitter):
    results = []
    async def on_vitals(data): results.append(data)
    emitter.on("vitals", on_vitals)
    svc = VitalsService(sample_rate=100.0, emitter=emitter, emit_interval_sec=0.0)
    svc.push_csi(np.random.rand(56).astype(np.float32))
    await asyncio.sleep(0.1)
    assert len(results) >= 1
    assert "breathing_bpm" in results[0]
