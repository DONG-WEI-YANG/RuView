"""Skeleton-based fall detection."""
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class FallAlert:
    timestamp: float
    confidence: float
    head_height: float
    velocity: float


class FallDetector:
    def __init__(self, threshold: float = 0.8, cooldown_sec: float = 30.0):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.is_fallen = False
        self._history: list[np.ndarray] = []
        self._alerts: list[FallAlert] = []
        self._last_alert_time = 0.0
        # Standing reference height; estimated from first few frames
        self._ref_height: float | None = None

    def update(self, joints: np.ndarray, vitals: dict | None = None):
        self._history.append(joints.copy())
        if len(self._history) > 30:
            self._history.pop(0)
        if len(self._history) < 3:
            # Use early frames to establish reference standing height
            head_h = float(joints[0, 1])
            if self._ref_height is None or head_h > self._ref_height:
                self._ref_height = head_h
            return
        head_h = self._head_height(joints)
        if self._ref_height is None or (not self.is_fallen and head_h > self._ref_height):
            self._ref_height = head_h
        confidence = self._compute_fall_confidence()
        
        # Dual Verification: Cross-check with vital signs if available
        if confidence >= self.threshold and vitals:
            # If breathing is normal and stable, reduce confidence (likely false positive)
            # e.g., person lying down on couch
            if vitals.get("breathing_bpm", 0) > 10 and vitals.get("stress_index", 0) < 50:
                # Reduce confidence slightly but don't ignore completely
                confidence *= 0.8
            # If vitals show distress (apnea, high stress), boost confidence
            elif vitals.get("stress_index", 0) > 80:
                confidence = min(1.0, confidence * 1.2)

        if confidence >= self.threshold:
            if not self.is_fallen:
                # Transition to fallen
                self.is_fallen = True
                now = time.time()
                if now - self._last_alert_time > self.cooldown_sec:
                    vel = self._vertical_velocity()
                    self._alerts.append(FallAlert(now, confidence, head_h, vel))
                    self._last_alert_time = now
        else:
            # Only recover if head is back above half reference height
            if self.is_fallen:
                ref = self._ref_height if self._ref_height else 1.0
                if head_h > ref * 0.5:
                    self.is_fallen = False
            # Not fallen yet and confidence below threshold: stay not fallen

    def _head_height(self, joints: np.ndarray) -> float:
        return float(joints[0, 1])

    def _vertical_velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        upper_now = np.mean(self._history[-1][:8, 1])
        upper_prev = np.mean(self._history[-2][:8, 1])
        return float(upper_prev - upper_now)

    def _compute_fall_confidence(self) -> float:
        current = self._history[-1]
        y_spread = np.std(current[:, 1])
        spread_score = max(0, 1.0 - y_spread / 0.5)
        head_h = current[0, 1]
        max_h = np.max(current[:, 1])
        height_range = max_h - np.min(current[:, 1])
        low_ratio = 1.0 - (head_h / max(height_range + 0.01, 0.5))
        # Absolute height score: head below 30% of reference is fully fallen
        ref = self._ref_height if self._ref_height else 1.0
        abs_height_score = float(np.clip(1.0 - head_h / max(ref * 0.3, 0.01), 0, 1))
        velocity = self._vertical_velocity()
        vel_score = min(1.0, max(0, velocity / 0.3))
        # Use max of weighted sum and velocity-based detection
        # A sudden large drop is itself strong evidence of a fall
        weighted = (
            0.10 * low_ratio
            + 0.15 * spread_score
            + 0.40 * abs_height_score
            + 0.35 * vel_score
        )
        confidence = max(weighted, vel_score * 0.85 + abs_height_score * 0.15)
        return float(np.clip(confidence, 0, 1))

    def get_alerts(self) -> list[FallAlert]:
        return list(self._alerts)

    def clear_alerts(self):
        self._alerts.clear()
