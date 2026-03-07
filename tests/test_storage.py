import numpy as np
import pytest
from server.storage import Storage


@pytest.fixture
def db(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


class TestPoseStorage:
    def test_save_and_retrieve(self, db):
        joints = np.random.randn(24, 3).astype(np.float32)
        db.save_pose(joints)
        poses = db.get_recent_poses(10)
        assert len(poses) == 1
        assert len(poses[0]["joints"]) == 24

    def test_multiple_poses(self, db):
        for _ in range(5):
            db.save_pose(np.random.randn(24, 3).astype(np.float32))
        poses = db.get_recent_poses(3)
        assert len(poses) == 3


class TestVitalsStorage:
    def test_save_and_retrieve(self, db):
        vitals = {
            "breathing_bpm": 15.0,
            "heart_bpm": 72.0,
            "hrv_rmssd": 45.0,
            "stress_index": 0.3,
            "motion_intensity": 0.1,
        }
        db.save_vitals(vitals)
        results = db.get_recent_vitals(10)
        assert len(results) == 1
        assert results[0]["breathing_bpm"] == 15.0


class TestFallAlerts:
    def test_save_alert(self, db):
        aid = db.save_fall_alert(0.95, 0.3, 1.2)
        assert aid > 0

    def test_unnotified(self, db):
        db.save_fall_alert(0.9, 0.4, 1.0)
        unnotified = db.get_unnotified_alerts()
        assert len(unnotified) == 1

    def test_mark_notified(self, db):
        aid = db.save_fall_alert(0.9, 0.4, 1.0)
        db.mark_notified(aid)
        unnotified = db.get_unnotified_alerts()
        assert len(unnotified) == 0


class TestCalibrationStorage:
    def test_save_and_retrieve(self, db):
        db.save_calibration(
            "esp32s3",
            {"0": {"distance": 2.5}},
            {"0": {"mean_amplitude": 0.5}},
        )
        cal = db.get_latest_calibration("esp32s3")
        assert cal is not None
        assert cal["profile_id"] == "esp32s3"

    def test_missing_profile(self, db):
        cal = db.get_latest_calibration("nonexistent")
        assert cal is None


class TestStats:
    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["poses"] == 0
        assert stats["vitals"] == 0
        assert stats["fall_alerts"] == 0
