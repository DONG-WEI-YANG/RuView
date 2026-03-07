import time
import numpy as np
import pytest
from dataclasses import dataclass
from server.calibration import CalibrationManager


@dataclass
class FakeCSIFrame:
    node_id: int
    amplitude: np.ndarray
    rssi: int


class TestCalibrationManager:
    def test_initial_state(self):
        mgr = CalibrationManager()
        assert not mgr.is_active
        assert mgr.get_result() is None

    def test_start_activates(self):
        mgr = CalibrationManager(duration=10.0)
        result = mgr.start()
        assert result["status"] == "calibrating"
        assert mgr.is_active

    def test_feed_frames_and_finish(self):
        mgr = CalibrationManager(duration=60.0)
        mgr.start()

        # Simulate CSI from 3 nodes
        for _ in range(30):
            for nid in range(3):
                frame = FakeCSIFrame(
                    node_id=nid,
                    amplitude=np.random.randn(56).astype(np.float32) * 10 + 50,
                    rssi=-45 - nid * 5,
                )
                mgr.on_csi_frame(frame)

        result = mgr.finish()
        assert result["status"] == "complete"
        assert result["node_count"] == 3
        assert result["total_samples"] == 90  # 30 * 3

        # Each node should have distance estimate
        for nid_str in ["0", "1", "2"]:
            node = result["nodes"][nid_str]
            assert node["estimated_distance_m"] > 0
            assert node["sample_count"] == 30

    def test_distance_increases_with_weaker_rssi(self):
        mgr = CalibrationManager(duration=60.0)
        mgr.start()

        # Node 0: strong signal (close)
        for _ in range(20):
            mgr.on_csi_frame(FakeCSIFrame(0, np.ones(56, dtype=np.float32), -30))

        # Node 1: weak signal (far)
        for _ in range(20):
            mgr.on_csi_frame(FakeCSIFrame(1, np.ones(56, dtype=np.float32), -70))

        result = mgr.finish()
        d_close = result["nodes"]["0"]["estimated_distance_m"]
        d_far = result["nodes"]["1"]["estimated_distance_m"]
        assert d_far > d_close

    def test_status_while_active(self):
        mgr = CalibrationManager(duration=60.0)
        mgr.start()
        status = mgr.get_status()
        assert status["status"] == "calibrating"
        assert status["progress"] >= 0

    def test_no_frames_produces_empty(self):
        mgr = CalibrationManager(duration=60.0)
        mgr.start()
        result = mgr.finish()
        assert result["status"] == "complete"
        assert result["node_count"] == 0

    def test_get_node_positions(self):
        mgr = CalibrationManager(duration=60.0)
        mgr.start()
        for _ in range(10):
            mgr.on_csi_frame(FakeCSIFrame(0, np.ones(56, dtype=np.float32), -40))
        mgr.finish()
        positions = mgr.get_node_positions()
        assert "0" in positions

    def test_finish_without_start(self):
        mgr = CalibrationManager()
        result = mgr.finish()
        assert result["status"] == "error"
