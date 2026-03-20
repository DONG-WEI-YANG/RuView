# tests/test_pipeline_service.py
import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock
from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.pipeline_service import PipelineService
from server.csi_frame import CSIFrame


def _make_frame(node_id=1, n_sub=56):
    return CSIFrame(
        node_id=node_id, sequence=0, timestamp_ms=0,
        rssi=-50, noise_floor=-90, channel=6, bandwidth=20,
        num_subcarriers=n_sub,
        amplitude=np.random.rand(n_sub).astype(np.float32),
        phase=np.zeros(n_sub, dtype=np.float32),
        raw_complex=np.zeros(n_sub, dtype=np.complex64),
    )


@pytest.fixture
def emitter():
    return EventEmitter()


@pytest.fixture
def service(emitter):
    settings = Settings()
    return PipelineService(settings=settings, emitter=emitter)


def test_on_frame_increments_count(service):
    frame = _make_frame()
    service.on_frame(frame)
    assert service.csi_frames_received == 1


@pytest.mark.asyncio
async def test_on_frame_emits_csi(emitter, service):
    results = []
    async def on_csi(data): results.append(data)
    emitter.on("csi", on_csi)
    frame = _make_frame()
    service.on_frame(frame, trigger_pipeline=True)
    await asyncio.sleep(0.1)
    assert len(results) > 0
    assert "amplitudes" in results[0]
