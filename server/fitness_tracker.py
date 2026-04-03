"""Skeleton-based fitness and activity tracking."""
from enum import Enum
from collections import defaultdict
import numpy as np


class ActivityType(Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    EXERCISING = "exercising"


class FitnessTracker:
    def __init__(self):
        self.current_activity = ActivityType.UNKNOWN
        self._history: list[np.ndarray] = []
        self._activity_frames: dict[ActivityType, int] = defaultdict(int)
        self._rep_count = 0

    def update(self, joints: np.ndarray):
        self._history.append(joints.copy())
        if len(self._history) > 60:
            self._history.pop(0)
        activity = self._classify_activity(joints)
        self.current_activity = activity
        self._activity_frames[activity] += 1

    def _classify_activity(self, joints: np.ndarray) -> ActivityType:
        head_y = joints[0, 1]
        hip_y = np.mean(joints[11:13, 1]) if joints.shape[0] > 12 else joints[len(joints) // 2, 1]
        feet_y = np.mean(joints[-2:, 1])
        total_height = head_y - feet_y
        torso_ratio = (head_y - hip_y) / max(total_height, 0.01)
        lower_body_height = hip_y - feet_y
        lower_body_ratio = lower_body_height / max(total_height, 0.01)
        if len(self._history) >= 5:
            recent_x = np.array([h[0, 0] for h in self._history[-5:]])
            x_movement = np.std(recent_x)
        else:
            x_movement = 0.0
        if total_height < 0.5:
            # Short stature: use torso_ratio to distinguish upright (wheelchair/
            # child) from genuinely seated.  An upright torso (>0.45) with lateral
            # movement is walking; upright and still is standing.
            if torso_ratio > 0.45:
                if x_movement > 0.1:
                    return ActivityType.WALKING
                return ActivityType.STANDING
            return ActivityType.SITTING
        if lower_body_ratio < 0.35:
            return ActivityType.SITTING
        if torso_ratio > 0.45 and x_movement < 0.05:
            return ActivityType.STANDING
        if x_movement > 0.1:
            return ActivityType.WALKING
        return ActivityType.STANDING

    def get_stats(self) -> dict[ActivityType, int]:
        return dict(self._activity_frames)

    def get_rep_count(self) -> int:
        return self._rep_count
