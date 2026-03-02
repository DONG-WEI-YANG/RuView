import numpy as np
import pytest
from server.fall_detector import FallDetector


def _make_joints(num_joints=24):
    joints = np.zeros((num_joints, 3))
    for i in range(num_joints):
        joints[i, 1] = 1.7 - (i / num_joints) * 1.7
    return joints


def test_no_fall_standing():
    det = FallDetector(threshold=0.8)
    standing = _make_joints()
    for _ in range(10):
        det.update(standing)
    assert det.is_fallen is False


def test_fall_detected():
    det = FallDetector(threshold=0.8)
    standing = _make_joints()
    for _ in range(5):
        det.update(standing)
    fallen = _make_joints()
    fallen[:, 1] = np.random.uniform(0.0, 0.3, 24)
    for _ in range(5):
        det.update(fallen)
    assert det.is_fallen is True


def test_cooldown():
    det = FallDetector(threshold=0.8, cooldown_sec=0.5)
    standing = _make_joints()
    fallen = _make_joints()
    fallen[:, 1] = 0.1
    for _ in range(5):
        det.update(standing)
    for _ in range(5):
        det.update(fallen)
    alerts = det.get_alerts()
    assert len(alerts) >= 1
