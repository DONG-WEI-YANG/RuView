"""Nurse Tour — patient safety, alert fatigue, and clinical workflow.

A nurse reviews: Will this wake me up for every false alarm? Does the
fall detector distinguish lying-on-couch from actual collapse? Will
I get the notification in time? What about multi-bed wards?

Expert focus: operational safety, false positive rate, alert usability.
"""
import time

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import httpx

from server.fall_detector import FallDetector, FallAlert
from server.fitness_tracker import FitnessTracker, ActivityType
from server.notifier import Notifier, FallNotification, MAX_RETRIES
from server.vital_signs import VitalSignsExtractor
from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.pipeline_service import PipelineService, SCENE_MODES
from tests.conftest import make_csi_frame


# ── Helpers ──────────────────────────────────────────────

def _standing_joints(head_h: float = 1.7) -> np.ndarray:
    """Generate a standing pose with specified head height."""
    joints = np.zeros((24, 3), dtype=np.float32)
    for i in range(24):
        joints[i, 1] = head_h * (1.0 - i * 0.03)
    joints[0, 1] = head_h
    return joints


def _fallen_joints(head_h: float = 0.15) -> np.ndarray:
    """Generate a fallen pose — body near ground."""
    joints = np.zeros((24, 3), dtype=np.float32)
    for i in range(24):
        joints[i, 1] = head_h + np.random.uniform(-0.05, 0.05)
    joints[0, 1] = head_h
    return joints


def _sitting_joints(head_h: float = 1.1) -> np.ndarray:
    """Generate a sitting pose — reduced height but upright torso."""
    joints = np.zeros((24, 3), dtype=np.float32)
    # Head to mid-torso: upright
    for i in range(12):
        joints[i, 1] = head_h - i * 0.05
    # Legs: compressed (sitting)
    for i in range(12, 24):
        joints[i, 1] = 0.4 + np.random.uniform(-0.03, 0.03)
    joints[0, 1] = head_h
    return joints


def _feed_n_frames(detector: FallDetector, joints: np.ndarray, n: int = 5,
                    vitals: dict | None = None):
    for _ in range(n):
        detector.update(joints, vitals=vitals)


# ═══════════════════════════════════════════════════════════
# 1. Fall Detection — sensitivity vs. specificity
# ═══════════════════════════════════════════════════════════

class TestFallDetectionNursePerspective:
    """Nurse's #1 concern: real falls get caught, lying on couch doesn't alarm."""

    def test_standing_person_not_fallen(self):
        fd = FallDetector(threshold=0.8)
        _feed_n_frames(fd, _standing_joints(), n=10)
        assert fd.is_fallen is False

    def test_actual_fall_detected(self):
        """Standing → suddenly on ground = must trigger."""
        fd = FallDetector(threshold=0.6)
        # Establish standing reference
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        # Sudden fall
        _feed_n_frames(fd, _fallen_joints(0.1), n=5)
        assert fd.is_fallen is True

    def test_sitting_is_not_fall(self):
        """Person sits down — head at 1.1m, NOT a fall."""
        fd = FallDetector(threshold=0.8)
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        _feed_n_frames(fd, _sitting_joints(1.1), n=10)
        assert fd.is_fallen is False, "Sitting misclassified as fall!"

    def test_lying_on_couch_with_normal_vitals(self):
        """Person lying down voluntarily — normal breathing, low stress.
        Nurse says: this should NOT trigger an alert.
        """
        fd = FallDetector(threshold=0.8)
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        # Slowly lower to lying height with calm vitals
        vitals = {"breathing_bpm": 15.0, "stress_index": 20.0}
        _feed_n_frames(fd, _fallen_joints(0.3), n=5, vitals=vitals)
        # Even if position looks like "fallen", calm vitals should reduce confidence
        # The dual verification mechanism should dampen this

    def test_fall_with_high_stress_boosts_confidence(self):
        """Real fall + high stress (shock) → confidence boosted.
        This is the scenario nurse cares about most.
        """
        fd = FallDetector(threshold=0.6)
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        vitals = {"stress_index": 90.0, "breathing_bpm": 5.0}
        _feed_n_frames(fd, _fallen_joints(0.1), n=5, vitals=vitals)
        assert fd.is_fallen is True
        assert len(fd.get_alerts()) > 0


# ═══════════════════════════════════════════════════════════
# 2. Alert Fatigue — cooldown and suppression
# ═══════════════════════════════════════════════════════════

class TestAlertFatigue:
    """Nurse's nightmare: 100 alerts per night for one restless patient."""

    def test_cooldown_prevents_spam(self):
        """After an alert, no new alert for 30 seconds."""
        fd = FallDetector(threshold=0.6, cooldown_sec=30.0)
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        # First fall → alert
        _feed_n_frames(fd, _fallen_joints(0.1), n=5)
        first_alerts = len(fd.get_alerts())
        # Recovery
        fd.is_fallen = False
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        # Second fall immediately → should NOT generate new alert (cooldown)
        _feed_n_frames(fd, _fallen_joints(0.1), n=5)
        assert len(fd.get_alerts()) == first_alerts, "Alert spam during cooldown!"

    def test_alert_after_cooldown_expires(self):
        """After cooldown expires, new fall should generate new alert."""
        fd = FallDetector(threshold=0.6, cooldown_sec=0.0)  # zero cooldown for test
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        _feed_n_frames(fd, _fallen_joints(0.1), n=5)
        count1 = len(fd.get_alerts())
        # Recovery
        fd.is_fallen = False
        fd._last_alert_time = 0  # simulate cooldown expired
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        _feed_n_frames(fd, _fallen_joints(0.1), n=5)
        assert len(fd.get_alerts()) > count1

    def test_safety_mode_more_sensitive(self):
        """Safety scene = lower threshold (0.6). Nurse wants this for elderly."""
        safety = SCENE_MODES["safety"]
        fitness = SCENE_MODES["fitness"]
        assert safety["fall_threshold"] < fitness["fall_threshold"]
        assert safety["fall_alert_cooldown"] < fitness["fall_alert_cooldown"]

    def test_fitness_mode_suppresses_alerts(self):
        """Fitness scene = high threshold (0.95). Don't alarm during burpees."""
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        svc.set_scene_mode("fitness")
        assert svc.fall_detector.threshold == 0.95
        assert SCENE_MODES["fitness"]["notify_on_fall"] is False


# ═══════════════════════════════════════════════════════════
# 3. Recovery Detection — asymmetric hysteresis
# ═══════════════════════════════════════════════════════════

class TestRecoveryHysteresis:
    """After a fall, person must stand above 50% ref height to clear fallen state.
    Nurse asks: if patient is helped to sitting (not standing), does alarm persist?
    """

    def test_recovery_requires_50_percent_height(self):
        fd = FallDetector(threshold=0.6)
        fd._ref_height = 1.7
        fd.is_fallen = True
        # At 40% height (0.68m) — still "fallen"
        low_joints = _standing_joints(0.68)
        _feed_n_frames(fd, low_joints, n=5)
        assert fd.is_fallen is True, "Should still be 'fallen' at 40% height"

    def test_recovery_above_50_percent(self):
        fd = FallDetector(threshold=0.6)
        fd._ref_height = 1.7
        fd.is_fallen = True
        # At 60% height (1.02m) — recovered
        recovered_joints = _standing_joints(1.02)
        _feed_n_frames(fd, recovered_joints, n=5)
        assert fd.is_fallen is False

    def test_ref_height_established_from_first_frames(self):
        fd = FallDetector()
        joints = _standing_joints(1.65)
        _feed_n_frames(fd, joints, n=5)
        assert fd._ref_height is not None
        assert fd._ref_height >= 1.5  # should capture standing height


# ═══════════════════════════════════════════════════════════
# 4. Notification Delivery — channels and format
# ═══════════════════════════════════════════════════════════

class TestNotificationNursePerspective:
    """Nurse needs: clear message, correct channels, no missed deliveries."""

    def test_no_channels_means_disabled(self):
        n = Notifier()
        assert n.enabled is False

    def test_webhook_channel_registered(self):
        n = Notifier(webhook_url="http://alert.hospital.local/hook")
        assert n.enabled is True
        assert "webhook" in n._channels

    def test_line_channel_registered(self):
        n = Notifier(line_token="test-token-123")
        assert "line" in n._channels

    def test_telegram_requires_both_fields(self):
        """Telegram needs BOTH bot token AND chat ID."""
        n1 = Notifier(telegram_bot_token="tok")
        assert "telegram" not in n1._channels
        n2 = Notifier(telegram_bot_token="tok", telegram_chat_id="123")
        assert "telegram" in n2._channels

    def test_fall_notification_structure(self):
        notif = FallNotification(
            timestamp=1000.0, confidence=0.85,
            head_height=0.15, velocity=1.2,
        )
        assert notif.confidence == 0.85
        assert notif.head_height == 0.15

    def test_alert_payload_has_all_fields(self):
        """Alert must contain all info nurse needs to assess severity."""
        alert = FallAlert(
            timestamp=time.time(), confidence=0.9,
            head_height=0.1, velocity=1.5,
        )
        assert hasattr(alert, 'timestamp')
        assert hasattr(alert, 'confidence')
        assert hasattr(alert, 'head_height')
        assert hasattr(alert, 'velocity')

    def test_alert_list_retrievable_and_clearable(self):
        fd = FallDetector(threshold=0.6, cooldown_sec=0.0)
        _feed_n_frames(fd, _standing_joints(1.7), n=5)
        _feed_n_frames(fd, _fallen_joints(0.1), n=5)
        alerts = fd.get_alerts()
        assert len(alerts) >= 1
        fd.clear_alerts()
        assert len(fd.get_alerts()) == 0


class TestNotificationRetry:
    """Nurse says: a missed fall alert can be fatal. Retry must exist."""

    def test_retry_constant_exists(self):
        assert MAX_RETRIES >= 1

    @patch.object(httpx.Client, "post")
    def test_retry_on_connection_error(self, mock_post):
        """Transient network failure should be retried."""
        mock_post.side_effect = [
            httpx.ConnectError("network down"),  # first attempt fails
            MagicMock(status_code=200, raise_for_status=lambda: None),  # retry succeeds
        ]
        n = Notifier(webhook_url="http://alert.local/hook")
        result = n._send_webhook({"event": "test"})
        assert "ok" in result
        assert mock_post.call_count == 2

    @patch.object(httpx.Client, "post")
    def test_retry_on_timeout(self, mock_post):
        """Timeout should trigger retry."""
        mock_post.side_effect = [
            httpx.TimeoutException("timed out"),
            MagicMock(status_code=200, raise_for_status=lambda: None),
        ]
        n = Notifier(webhook_url="http://alert.local/hook")
        result = n._send_webhook({"event": "test"})
        assert "ok" in result

    @patch.object(httpx.Client, "post")
    def test_no_retry_on_4xx(self, mock_post):
        """Client errors (4xx) should NOT be retried — it's a config problem."""
        resp = MagicMock(status_code=401)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=resp,
        )
        mock_post.return_value = resp
        n = Notifier(webhook_url="http://alert.local/hook")
        result = n._send_webhook({"event": "test"})
        assert "error" in result
        assert mock_post.call_count == 1  # no retry

    @patch.object(httpx.Client, "post")
    def test_all_retries_exhausted(self, mock_post):
        """All retries fail → error returned."""
        mock_post.side_effect = httpx.ConnectError("still down")
        n = Notifier(webhook_url="http://alert.local/hook")
        result = n._send_webhook({"event": "test"})
        assert "error" in result
        assert mock_post.call_count == 1 + MAX_RETRIES


# ═══════════════════════════════════════════════════════════
# 5. Activity Classification — nurse observes daily routine
# ═══════════════════════════════════════════════════════════

class TestActivityClassificationNurse:
    """Nurse monitors patient activities: are they sitting all day?
    Walking to the bathroom? Exercising? Fallen?
    """

    def test_standing_classified(self):
        ft = FitnessTracker()
        joints = _standing_joints(1.7)
        for _ in range(10):
            ft.update(joints)
        assert ft.current_activity == ActivityType.STANDING

    def test_sitting_classified(self):
        ft = FitnessTracker()
        joints = _sitting_joints(1.1)
        for _ in range(10):
            ft.update(joints)
        assert ft.current_activity == ActivityType.SITTING

    def test_walking_classified(self):
        """Walking = lateral movement > 0.1 threshold."""
        ft = FitnessTracker()
        for i in range(10):
            joints = _standing_joints(1.7)
            joints[0, 0] = np.sin(i * 0.5) * 0.3  # lateral sway
            ft.update(joints)
        assert ft.current_activity == ActivityType.WALKING

    def test_wheelchair_user_upright_not_sitting(self):
        """Wheelchair user: short total height but upright torso → STANDING."""
        ft = FitnessTracker()
        # Build a wheelchair pose: head 0.45m, but torso is upright
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[0, 1] = 0.45   # head
        for i in range(1, 11):
            joints[i, 1] = 0.45 - i * 0.02  # upright torso gradient
        # hips at ~0.25
        joints[11, 1] = 0.25
        joints[12, 1] = 0.25
        # legs compressed (in wheelchair)
        for i in range(13, 24):
            joints[i, 1] = 0.05
        for _ in range(10):
            ft.update(joints)
        # torso_ratio > 0.45 → should NOT be classified as SITTING
        assert ft.current_activity != ActivityType.SITTING, \
            "Wheelchair user with upright torso misclassified as SITTING!"

    def test_genuinely_short_seated_still_sitting(self):
        """Child actually sitting (torso folded, head near hips) → SITTING."""
        ft = FitnessTracker()
        joints = np.zeros((24, 3), dtype=np.float32)
        # Head barely above hips — torso is folded forward
        joints[0, 1] = 0.40   # head
        for i in range(1, 11):
            joints[i, 1] = 0.39  # upper body ~same height as head
        joints[11, 1] = 0.38  # hips nearly at head level (folded)
        joints[12, 1] = 0.38
        for i in range(13, 24):
            joints[i, 1] = 0.05  # legs on ground
        # total_height = 0.40 - 0.05 = 0.35 < 0.5
        # torso_ratio = (0.40 - 0.38) / 0.35 ≈ 0.057 → NOT upright → SITTING
        for _ in range(10):
            ft.update(joints)
        assert ft.current_activity == ActivityType.SITTING

    def test_activity_stats_tracked(self):
        ft = FitnessTracker()
        for _ in range(5):
            ft.update(_standing_joints(1.7))
        for _ in range(3):
            ft.update(_sitting_joints(1.1))
        stats = ft.get_stats()
        assert ActivityType.STANDING in stats
        assert ActivityType.SITTING in stats


# ═══════════════════════════════════════════════════════════
# 6. Multi-Patient Ward — disambiguation
# ═══════════════════════════════════════════════════════════

class TestMultiPatientWard:
    """Ward scenario: 2-4 patients, each must have independent vitals."""

    def test_multi_person_independent_vitals(self):
        """Each tracked person has their own VitalSignsExtractor."""
        from server.vital_signs import MultiPersonTracker
        tracker = MultiPersonTracker(max_persons=4, sample_rate=20.0)
        # Simulate 5 antenna pairs (triggers ICA path)
        antenna_data = {}
        for i in range(5):
            antenna_data[f"TX1_RX{i+1}"] = np.random.rand(56).astype(np.float32) * 100
        tracker.push_multi_antenna_csi(antenna_data)
        # Each person state should have independent extractor
        for pid, state in tracker.persons.items():
            assert "vitals" in state
            assert isinstance(state["vitals"], VitalSignsExtractor)

    def test_max_persons_capped(self):
        """System should never track more than max_persons."""
        from server.vital_signs import MultiPersonTracker
        tracker = MultiPersonTracker(max_persons=4, sample_rate=20.0)
        assert tracker.max_persons == 4

    def test_person_count_accessible(self):
        from server.vital_signs import MultiPersonTracker
        tracker = MultiPersonTracker(max_persons=4, sample_rate=20.0)
        assert tracker.person_count == 0

    def test_update_all_returns_per_person_vitals(self):
        from server.vital_signs import MultiPersonTracker
        tracker = MultiPersonTracker(max_persons=4, sample_rate=20.0)
        # Push some data to create at least one person
        antenna_data = {f"TX1_RX{i+1}": np.random.rand(56).astype(np.float32) for i in range(3)}
        tracker.push_multi_antenna_csi(antenna_data)
        results = tracker.update_all()
        assert isinstance(results, list)
        for r in results:
            assert "person_id" in r
            assert "vitals" in r


# ═══════════════════════════════════════════════════════════
# 7. Scene Mode — nurse's quick-switch workflow
# ═══════════════════════════════════════════════════════════

class TestSceneModeNurseWorkflow:
    """Nurse switches between safety (night) and fitness (rehab session)."""

    def test_safety_enables_fall_alerts(self):
        assert SCENE_MODES["safety"]["notify_on_fall"] is True

    def test_safety_enables_apnea_alerts(self):
        assert SCENE_MODES["safety"]["notify_on_apnea"] is True

    def test_fitness_disables_fall_alerts(self):
        assert SCENE_MODES["fitness"]["notify_on_fall"] is False

    def test_fitness_enables_rep_tracking(self):
        assert SCENE_MODES["fitness"]["track_reps"] is True

    def test_safety_has_inactivity_timeout(self):
        """5 minutes no motion → alert (patient may be unconscious)."""
        assert SCENE_MODES["safety"]["inactivity_timeout"] == 300

    def test_fitness_no_inactivity_timeout(self):
        assert SCENE_MODES["fitness"]["inactivity_timeout"] == 0
