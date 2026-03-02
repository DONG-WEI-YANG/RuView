import numpy as np
import pytest
from server.fitness_tracker import FitnessTracker, ActivityType


def _standing_pose():
    joints = np.zeros((24, 3))
    for i in range(24):
        joints[i, 1] = 1.7 - (i / 24) * 1.7
    return joints


def _sitting_pose():
    joints = _standing_pose()
    joints[:, 1] *= 0.6
    joints[12:, 1] *= 0.3
    return joints


def test_classify_standing():
    tracker = FitnessTracker()
    pose = _standing_pose()
    for _ in range(10):
        tracker.update(pose)
    assert tracker.current_activity == ActivityType.STANDING


def test_classify_sitting():
    tracker = FitnessTracker()
    pose = _sitting_pose()
    for _ in range(10):
        tracker.update(pose)
    assert tracker.current_activity == ActivityType.SITTING


def test_activity_duration():
    tracker = FitnessTracker()
    pose = _standing_pose()
    for _ in range(20):
        tracker.update(pose)
    stats = tracker.get_stats()
    assert ActivityType.STANDING in stats
    assert stats[ActivityType.STANDING] >= 1
