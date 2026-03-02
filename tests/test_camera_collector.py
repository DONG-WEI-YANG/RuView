"""Tests for the camera-based data collector.

All tests mock cv2.VideoCapture and rtmlib Body so they run in CI
without a physical camera or GPU.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.camera_collector import (
    CameraCollector,
    coco17_to_24joint,
    COCO_LEFT_EAR,
    COCO_RIGHT_EAR,
    COCO_LEFT_EYE,
)


# ---------------------------------------------------------------------------
# Helper: create realistic fake COCO-17 keypoints
# ---------------------------------------------------------------------------

def _make_fake_keypoints(
    image_w: int = 640,
    image_h: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (keypoints, scores) mimicking rtmlib Body output.

    The pose roughly represents a standing person centred in the frame.
    keypoints shape: (1, 17, 2)  -- one person, 17 joints, (x, y)
    scores shape:    (1, 17)     -- confidence per joint
    """
    # Standing person: head near top-centre, feet near bottom
    cx = image_w / 2.0  # centre-x
    head_y = image_h * 0.15
    shoulder_y = image_h * 0.30
    elbow_y = image_h * 0.45
    wrist_y = image_h * 0.58
    hip_y = image_h * 0.55
    knee_y = image_h * 0.72
    ankle_y = image_h * 0.88

    shoulder_spread = image_w * 0.12
    hip_spread = image_w * 0.06

    kp = np.array([
        [cx, head_y],                             # 0: nose
        [cx - 10, head_y - 5],                    # 1: left_eye
        [cx + 10, head_y - 5],                    # 2: right_eye
        [cx - 25, head_y],                        # 3: left_ear
        [cx + 25, head_y],                        # 4: right_ear
        [cx - shoulder_spread, shoulder_y],        # 5: left_shoulder
        [cx + shoulder_spread, shoulder_y],        # 6: right_shoulder
        [cx - shoulder_spread * 1.1, elbow_y],    # 7: left_elbow
        [cx + shoulder_spread * 1.1, elbow_y],    # 8: right_elbow
        [cx - shoulder_spread * 1.0, wrist_y],    # 9: left_wrist
        [cx + shoulder_spread * 1.0, wrist_y],    # 10: right_wrist
        [cx - hip_spread, hip_y],                  # 11: left_hip
        [cx + hip_spread, hip_y],                  # 12: right_hip
        [cx - hip_spread, knee_y],                 # 13: left_knee
        [cx + hip_spread, knee_y],                 # 14: right_knee
        [cx - hip_spread, ankle_y],                # 15: left_ankle
        [cx + hip_spread, ankle_y],                # 16: right_ankle
    ], dtype=np.float32)

    # Wrap in batch dimension: (1, 17, 2)
    keypoints = kp[np.newaxis, :, :]

    # High confidence for all joints
    scores = np.ones((1, 17), dtype=np.float32) * 0.9

    return keypoints, scores


def _make_fake_frame(
    w: int = 640, h: int = 480
) -> np.ndarray:
    """Return a blank BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_camera_collector():
    """Create a CameraCollector with mocked camera and body detector."""
    fake_kp, fake_sc = _make_fake_keypoints()
    fake_frame = _make_fake_frame()

    with patch("server.camera_collector.cv2") as mock_cv2, \
         patch("rtmlib.Body") as MockBody:

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap

        # Mock Body detector
        mock_body_instance = MagicMock()
        mock_body_instance.return_value = (fake_kp, fake_sc)
        MockBody.return_value = mock_body_instance

        collector = CameraCollector(
            camera_index=0,
            model_mode="lightweight",
            standing_height=1.70,
        )
        # Override the body and cap that were set in __init__
        collector.body = mock_body_instance
        collector.cap = mock_cap

        yield collector, fake_kp, fake_sc


# ---------------------------------------------------------------------------
# Test 1: COCO 17 -> 24-joint mapping
# ---------------------------------------------------------------------------

class TestCoco17To24JointMapping:
    """Verify the 17 -> 24 mapping produces correct shape and values."""

    def test_output_shape(self):
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        assert joints.shape == (24, 3), f"Expected (24, 3), got {joints.shape}"
        assert joints.dtype == np.float32

    def test_reasonable_values(self):
        """Converted joints should be within physically plausible ranges."""
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)

        # All X coords should be within a few metres of origin
        assert np.all(np.abs(joints[:, 0]) < 3.0), "X coords out of range"
        # Y coords: relative to hip, head above, feet below
        assert np.all(np.abs(joints[:, 1]) < 3.0), "Y coords out of range"
        # Z coords (pseudo-depth) should be small
        assert np.all(np.abs(joints[:, 2]) < 2.0), "Z coords out of range"

    def test_no_nans(self):
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        assert np.all(np.isfinite(joints)), "Output contains NaN or Inf"

    def test_head_above_spine(self):
        """Head (joint 0) should have higher Y than spine (joint 3)."""
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        # Y-up: head.y > spine.y
        assert joints[0, 1] > joints[3, 1], (
            f"Head Y ({joints[0, 1]:.3f}) should be above spine Y ({joints[3, 1]:.3f})"
        )

    def test_left_right_symmetry(self):
        """Left and right joints should be roughly symmetric about X=0."""
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        # l_shoulder (4) and r_shoulder (7) should have opposite X signs
        assert joints[4, 0] * joints[7, 0] <= 0 or abs(joints[4, 0] - joints[7, 0]) < 0.05, (
            "Shoulders should be on opposite sides of X=0"
        )


# ---------------------------------------------------------------------------
# Test 2: Pixel to meter conversion
# ---------------------------------------------------------------------------

class TestPixelToMeterConversion:
    """Given known pixel coords and standing height, outputs correct metres."""

    def test_scale_factor(self):
        """A person 400px tall with 1.70m height gives scale ~0.00425 m/px."""
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        # The head-to-ankle distance in output should approximate standing height
        # Head (joint 0) to average ankle (joints 12, 15) in Y
        head_y = joints[0, 1]
        ankle_y = (joints[12, 1] + joints[15, 1]) / 2.0
        height_m = head_y - ankle_y
        # Should be close to standing_height (within 30% tolerance due to
        # head vs nose offset and neck/foot offsets)
        assert 0.5 < height_m < 2.5, (
            f"Estimated height {height_m:.2f}m should be near 1.70m"
        )

    def test_different_height(self):
        """Changing standing_height scales output proportionally."""
        kp, sc = _make_fake_keypoints()
        joints_170 = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        joints_180 = coco17_to_24joint(kp[0], sc[0], standing_height=1.80, image_height=480)

        # All positions should scale up by 1.80/1.70
        ratio = 1.80 / 1.70
        # Check a specific joint (l_shoulder, index 4)
        scale_actual = np.abs(joints_180[4, 0]) / max(np.abs(joints_170[4, 0]), 1e-8)
        assert abs(scale_actual - ratio) < 0.15, (
            f"Scale ratio {scale_actual:.3f} should be near {ratio:.3f}"
        )

    def test_origin_at_mid_hip(self):
        """Mid-hip should be approximately at the origin (0, 0)."""
        kp, sc = _make_fake_keypoints()
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        # Mid-hip = average of l_hip (10) and r_hip (13)
        mid_hip = (joints[10] + joints[13]) / 2.0
        assert np.abs(mid_hip[0]) < 0.05, f"Mid-hip X should be near 0, got {mid_hip[0]:.4f}"
        assert np.abs(mid_hip[1]) < 0.05, f"Mid-hip Y should be near 0, got {mid_hip[1]:.4f}"


# ---------------------------------------------------------------------------
# Test 3: Recording lifecycle
# ---------------------------------------------------------------------------

class TestRecordingLifecycle:
    """Mock camera, start recording, feed frames, stop, verify shapes."""

    def test_record_and_stop(self, mock_camera_collector):
        collector, fake_kp, fake_sc = mock_camera_collector

        collector.start_recording("walking", n_frames=10)
        assert collector._recording is True

        # Capture several frames
        for _ in range(10):
            collector.capture_frame()

        data = collector.stop_recording()
        assert collector._recording is False
        assert data is not None

        assert data["joints"].shape == (10, 24, 3)
        assert data["joints"].dtype == np.float32
        assert data["labels"].shape == (10,)
        assert all(label == "walking" for label in data["labels"])
        assert data["csi"].shape[0] == 10
        assert data["csi"].shape[1] == 4   # n_nodes
        assert data["csi"].shape[2] == 56  # n_sub

    def test_stop_without_frames_returns_none(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector
        collector.start_recording("standing", n_frames=10)
        # Stop immediately without capturing
        data = collector.stop_recording()
        assert data is None

    def test_max_frames_limit(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector
        collector.start_recording("walking", n_frames=5)

        # Capture more frames than the limit
        for _ in range(20):
            collector.capture_frame()

        data = collector.stop_recording()
        assert data is not None
        # Should have at most n_frames entries
        assert data["joints"].shape[0] == 5


# ---------------------------------------------------------------------------
# Test 4: Save NPZ format
# ---------------------------------------------------------------------------

class TestSaveNpzFormat:
    """Save a sequence, reload, verify keys and shapes match WiFiPoseDataset."""

    def test_save_and_reload(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector

        collector.start_recording("standing", n_frames=15)
        for _ in range(15):
            collector.capture_frame()
        data = collector.stop_recording()
        assert data is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = collector.save_sequence(data, tmpdir, prefix="test")
            assert os.path.exists(path)
            assert path.endswith(".npz")

            # Note: allow_pickle=True is required to load string label arrays
            # from our own generated .npz files (not untrusted data).
            loaded = np.load(path, allow_pickle=True)
            try:
                assert "csi" in loaded
                assert "joints" in loaded
                assert "labels" in loaded

                assert loaded["csi"].shape == (15, 4, 56)
                assert loaded["joints"].shape == (15, 24, 3)
                assert loaded["labels"].shape == (15,)
                assert loaded["csi"].dtype == np.float32
                assert loaded["joints"].dtype == np.float32
                assert all(label == "standing" for label in loaded["labels"])
            finally:
                loaded.close()

    def test_output_dir_created(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector

        collector.start_recording("sitting", n_frames=5)
        for _ in range(5):
            collector.capture_frame()
        data = collector.stop_recording()
        assert data is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            path = collector.save_sequence(data, nested)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Test 5: No person detected
# ---------------------------------------------------------------------------

class TestNoPersonDetected:
    """When model returns empty keypoints, capture_frame returns None."""

    def test_empty_keypoints(self):
        with patch("server.camera_collector.cv2") as mock_cv2, \
             patch("rtmlib.Body") as MockBody:

            mock_cap = MagicMock()
            mock_cap.read.return_value = (True, _make_fake_frame())
            mock_cv2.VideoCapture.return_value = mock_cap

            # Return empty arrays (no person detected)
            mock_body_instance = MagicMock()
            mock_body_instance.return_value = (np.empty((0, 17, 2)), np.empty((0, 17)))
            MockBody.return_value = mock_body_instance

            collector = CameraCollector(standing_height=1.70)
            collector.body = mock_body_instance
            collector.cap = mock_cap

            frame, joints, confidence = collector.capture_frame()
            assert frame is not None
            assert joints is None
            assert confidence is None

    def test_none_keypoints(self):
        with patch("server.camera_collector.cv2") as mock_cv2, \
             patch("rtmlib.Body") as MockBody:

            mock_cap = MagicMock()
            mock_cap.read.return_value = (True, _make_fake_frame())
            mock_cv2.VideoCapture.return_value = mock_cap

            mock_body_instance = MagicMock()
            mock_body_instance.return_value = (None, None)
            MockBody.return_value = mock_body_instance

            collector = CameraCollector(standing_height=1.70)
            collector.body = mock_body_instance
            collector.cap = mock_cap

            frame, joints, confidence = collector.capture_frame()
            assert joints is None
            assert confidence is None

    def test_camera_read_failure(self):
        with patch("server.camera_collector.cv2") as mock_cv2, \
             patch("rtmlib.Body") as MockBody:

            mock_cap = MagicMock()
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap

            mock_body_instance = MagicMock()
            MockBody.return_value = mock_body_instance

            collector = CameraCollector(standing_height=1.70)
            collector.body = mock_body_instance
            collector.cap = mock_cap

            frame, joints, confidence = collector.capture_frame()
            assert frame is not None  # returns blank frame
            assert joints is None
            assert confidence is None


# ---------------------------------------------------------------------------
# Test 6: Mixed mode CSI generation
# ---------------------------------------------------------------------------

class TestMixedModeCSIGeneration:
    """Verify simulated CSI is generated from detected poses."""

    def test_csi_shape(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector

        collector.start_recording("exercising", n_frames=8)
        for _ in range(8):
            collector.capture_frame()
        data = collector.stop_recording()
        assert data is not None

        # CSI shape: (n_frames, n_nodes, n_sub) = (8, 4, 56)
        assert data["csi"].shape == (8, 4, 56)
        assert data["csi"].dtype == np.float32

    def test_csi_is_not_zeros(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector

        collector.start_recording("walking", n_frames=5)
        for _ in range(5):
            collector.capture_frame()
        data = collector.stop_recording()
        assert data is not None

        # CSI should contain non-trivial values
        assert np.abs(data["csi"]).max() > 0.01, "CSI should not be all zeros"

    def test_csi_no_nan(self, mock_camera_collector):
        collector, _, _ = mock_camera_collector

        collector.start_recording("standing", n_frames=5)
        for _ in range(5):
            collector.capture_frame()
        data = collector.stop_recording()
        assert data is not None

        assert np.all(np.isfinite(data["csi"])), "CSI contains NaN or Inf"

    def test_single_frame_csi_simulation(self, mock_camera_collector):
        """Directly test the _simulate_csi_frame method."""
        collector, _, _ = mock_camera_collector
        # Create a known joint pose
        from server.data_generator import BASE_SKELETON
        csi_frame = collector._simulate_csi_frame(BASE_SKELETON)
        assert csi_frame.shape == (4, 56)
        assert csi_frame.dtype == np.float32
        assert np.all(np.isfinite(csi_frame))


# ---------------------------------------------------------------------------
# Test 7: Confidence filtering
# ---------------------------------------------------------------------------

class TestConfidenceFiltering:
    """Low-confidence keypoints should still produce valid output."""

    def test_mixed_confidence(self):
        """Some keypoints have low confidence; output should still be valid."""
        kp, sc = _make_fake_keypoints()

        # Set some keypoints to low confidence
        sc[0, COCO_LEFT_EAR] = 0.05
        sc[0, COCO_RIGHT_EAR] = 0.08
        sc[0, COCO_LEFT_EYE] = 0.10

        # coco17_to_24joint should still work with these inputs
        joints = coco17_to_24joint(kp[0], sc[0], standing_height=1.70, image_height=480)
        assert joints.shape == (24, 3)
        assert np.all(np.isfinite(joints))

    def test_low_confidence_capture(self):
        """capture_frame handles low-confidence keypoints gracefully."""
        kp, sc = _make_fake_keypoints()
        # Set some to low confidence
        sc[0, 3] = 0.1  # left_ear
        sc[0, 4] = 0.1  # right_ear

        with patch("server.camera_collector.cv2") as mock_cv2, \
             patch("rtmlib.Body") as MockBody:

            mock_cap = MagicMock()
            mock_cap.read.return_value = (True, _make_fake_frame())
            mock_cv2.VideoCapture.return_value = mock_cap

            mock_body_instance = MagicMock()
            mock_body_instance.return_value = (kp, sc)
            MockBody.return_value = mock_body_instance

            collector = CameraCollector(standing_height=1.70)
            collector.body = mock_body_instance
            collector.cap = mock_cap

            frame, joints, confidence = collector.capture_frame()
            assert joints is not None
            assert joints.shape == (24, 3)
            assert np.all(np.isfinite(joints))

    def test_too_few_confident_keypoints(self):
        """If too few keypoints are confident, return None."""
        kp, sc = _make_fake_keypoints()
        # Set most to very low confidence
        sc[0, :] = 0.05  # all below threshold
        sc[0, 0] = 0.9   # only nose is confident

        with patch("server.camera_collector.cv2") as mock_cv2, \
             patch("rtmlib.Body") as MockBody:

            mock_cap = MagicMock()
            mock_cap.read.return_value = (True, _make_fake_frame())
            mock_cv2.VideoCapture.return_value = mock_cap

            mock_body_instance = MagicMock()
            mock_body_instance.return_value = (kp, sc)
            MockBody.return_value = mock_body_instance

            collector = CameraCollector(standing_height=1.70)
            collector.body = mock_body_instance
            collector.cap = mock_cap

            frame, joints, confidence = collector.capture_frame()
            # Only 1 confident keypoint < 6 minimum
            assert joints is None
