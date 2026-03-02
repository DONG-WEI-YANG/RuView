import numpy as np
import pytest
from unittest.mock import MagicMock
from server.pipeline import PosePipeline
from server.config import Settings
from server.csi_frame import CSIFrame


def _make_frame(node_id, seq=0):
    return CSIFrame(
        node_id=node_id, sequence=seq, timestamp_ms=1000,
        rssi=-45, noise_floor=-90, channel=6, bandwidth=20,
        num_subcarriers=56,
        amplitude=np.random.randn(56).astype(np.float32),
        phase=np.random.randn(56).astype(np.float32),
        raw_complex=np.random.randn(56).astype(np.complex64),
    )


def test_pipeline_accumulates_frames():
    settings = Settings(max_nodes=3)
    pipeline = PosePipeline(settings, model=None)
    pipeline.on_csi_frame(_make_frame(1))
    pipeline.on_csi_frame(_make_frame(2))
    assert len(pipeline._current_frame_nodes) == 2


def test_pipeline_produces_joints_with_mock_model():
    settings = Settings(max_nodes=2)
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.detach.return_value.cpu.return_value.numpy.return_value = (
        np.random.randn(1, 24, 3).astype(np.float32)
    )
    pipeline = PosePipeline(settings, model=mock_model, window_size=5)

    for i in range(10):
        pipeline.on_csi_frame(_make_frame(1, seq=i))
        pipeline.on_csi_frame(_make_frame(2, seq=i))
        pipeline.flush_frame()

    assert pipeline.latest_joints is not None
    assert pipeline.latest_joints.shape == (24, 3)
