import numpy as np
import pytest
from server.signal_processor import SignalProcessor
from server.config import Settings


@pytest.fixture
def processor():
    return SignalProcessor(Settings())


def _fake_amplitudes(n_frames=100, n_sub=56):
    t = np.linspace(0, 5, n_frames)
    base = np.random.randn(n_frames, n_sub) * 0.1
    for s in range(10, 30):
        base[:, s] += np.sin(2 * np.pi * 1.0 * t) * 2.0
    return base


def test_bandpass_filter(processor):
    raw = _fake_amplitudes()
    filtered = processor.bandpass_filter(raw, low=0.5, high=3.0, fs=20)
    assert filtered.shape == raw.shape
    assert np.std(filtered[:, 15]) > 0.5


def test_normalize(processor):
    raw = np.random.randn(50, 56) * 100 + 500
    normed = processor.normalize(raw)
    assert abs(np.mean(normed)) < 0.5
    assert abs(np.std(normed) - 1.0) < 0.5


def test_fuse_nodes(processor):
    node_data = {
        1: np.random.randn(56),
        2: np.random.randn(56),
        3: np.random.randn(56),
    }
    fused = processor.fuse_nodes(node_data)
    assert fused.shape == (56 * 3,)
