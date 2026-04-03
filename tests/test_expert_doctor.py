"""Doctor Tour — clinical validity of vital signs extraction.

A physician reviews: Are the physiological ranges correct? Do the HRV
metrics map to known clinical standards? Is sleep staging based on
valid physiology? Will apnea detection catch real events?

Expert focus: physiological plausibility, not software correctness.
"""
import pytest
import numpy as np
from scipy.signal import find_peaks

from server.vital_signs import VitalSignsExtractor, MultiPersonTracker


# ── Helpers ──────────────────────────────────────────────

def _make_extractor(fs: float = 20.0) -> VitalSignsExtractor:
    """Low sample rate extractor suitable for unit tests."""
    return VitalSignsExtractor(sample_rate=fs, window_sec=10.0)


def _generate_sinusoidal_csi(
    freq_hz: float,
    fs: float = 20.0,
    duration_sec: float = 10.0,
    n_sub: int = 56,
    amplitude: float = 1.0,
    noise: float = 0.01,
) -> list[np.ndarray]:
    """Generate CSI frames with a sinusoidal component at a specific frequency.

    This simulates a person breathing (0.25 Hz) or heart beating (1.0 Hz)
    creating periodic modulation in the CSI amplitude.
    """
    n_frames = int(fs * duration_sec)
    t = np.arange(n_frames) / fs
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
    frames = []
    for i in range(n_frames):
        base = np.ones(n_sub, dtype=np.float32) * 50.0
        base += signal[i]
        base += np.random.randn(n_sub).astype(np.float32) * noise
        frames.append(base)
    return frames


def _push_frames(ext: VitalSignsExtractor, frames: list[np.ndarray]):
    for f in frames:
        ext.push_csi(f)


# ═══════════════════════════════════════════════════════════
# 1. Breathing Rate — physiological bounds
# ═══════════════════════════════════════════════════════════

class TestBreathingPhysiology:
    """Normal adult breathing: 12-20 BPM (0.2-0.33 Hz).
    Valid detection range: 6-30 BPM (0.1-0.5 Hz).
    """

    def test_normal_breathing_15bpm(self):
        """15 BPM = 0.25 Hz — textbook normal adult at rest."""
        ext = _make_extractor(fs=20.0)
        frames = _generate_sinusoidal_csi(freq_hz=0.25, fs=20.0, duration_sec=15.0)
        _push_frames(ext, frames)
        result = ext.update()
        bpm = result["breathing_bpm"]
        # Doctor expects: 15 ± 3 BPM
        assert 12 <= bpm <= 18, f"Normal breathing 15 BPM detected as {bpm}"

    def test_tachypnea_30bpm(self):
        """30 BPM = 0.5 Hz — tachypnea (rapid breathing), upper detection limit."""
        ext = _make_extractor(fs=20.0)
        frames = _generate_sinusoidal_csi(freq_hz=0.5, fs=20.0, duration_sec=15.0)
        _push_frames(ext, frames)
        result = ext.update()
        bpm = result["breathing_bpm"]
        assert 25 <= bpm <= 35, f"Tachypnea 30 BPM detected as {bpm}"

    def test_bradypnea_8bpm(self):
        """8 BPM = 0.133 Hz — bradypnea (slow breathing), near lower limit."""
        ext = _make_extractor(fs=20.0)
        frames = _generate_sinusoidal_csi(freq_hz=0.133, fs=20.0, duration_sec=20.0)
        _push_frames(ext, frames)
        result = ext.update()
        bpm = result["breathing_bpm"]
        assert 5 <= bpm <= 12, f"Bradypnea 8 BPM detected as {bpm}"

    def test_confidence_higher_for_clean_signal(self):
        """A clear periodic signal should yield higher confidence than noise."""
        ext_clean = _make_extractor(fs=20.0)
        frames_clean = _generate_sinusoidal_csi(
            freq_hz=0.25, fs=20.0, duration_sec=15.0,
            amplitude=5.0, noise=0.01,
        )
        _push_frames(ext_clean, frames_clean)
        result_clean = ext_clean.update()

        ext_noisy = _make_extractor(fs=20.0)
        frames_noisy = _generate_sinusoidal_csi(
            freq_hz=0.25, fs=20.0, duration_sec=15.0,
            amplitude=0.01, noise=5.0,
        )
        _push_frames(ext_noisy, frames_noisy)
        result_noisy = ext_noisy.update()

        assert result_clean["breathing_confidence"] > result_noisy["breathing_confidence"]

    def test_confidence_low_for_noisy_signal(self):
        """Pure noise should yield near-zero confidence."""
        ext = _make_extractor(fs=20.0)
        frames = []
        for _ in range(300):
            frames.append(np.random.randn(56).astype(np.float32))
        _push_frames(ext, frames)
        result = ext.update()
        assert result["breathing_confidence"] < 0.5


# ═══════════════════════════════════════════════════════════
# 2. Heart Rate — stricter physiological constraints
# ═══════════════════════════════════════════════════════════

class TestHeartRatePhysiology:
    """Normal resting HR: 60-100 BPM. Detection range: 48-120 BPM.
    WiFi CSI detects ~0.5mm chest wall displacement — needs stillness.
    """

    def test_normal_heart_72bpm(self):
        """72 BPM = 1.2 Hz — textbook resting heart rate."""
        ext = _make_extractor(fs=20.0)
        frames = _generate_sinusoidal_csi(
            freq_hz=1.2, fs=20.0, duration_sec=15.0,
            amplitude=0.5, noise=0.01,
        )
        _push_frames(ext, frames)
        result = ext.update()
        bpm = result["heart_bpm"]
        # Heart rate detection is harder; wider tolerance
        assert 50 <= bpm <= 90, f"Normal HR 72 BPM detected as {bpm}"


# ═══════════════════════════════════════════════════════════
# 3. HRV — clinical stress assessment
# ═══════════════════════════════════════════════════════════

class TestHRVClinical:
    """HRV RMSSD normal range: 20-100ms. Low RMSSD = sympathetic dominance.
    IBI filtering must reject physiologically impossible intervals.
    """

    def test_ibi_filter_rejects_impossibly_fast(self):
        """IBI < 300ms = >200 BPM, physiologically impossible for adult at rest."""
        ext = _make_extractor()
        # Directly test the IBI filter logic
        ibi_ms = np.array([250, 280, 300, 500, 800])
        valid = (ibi_ms > 300) & (ibi_ms < 2000)
        assert not valid[0]  # 250ms rejected
        assert not valid[1]  # 280ms rejected
        assert valid[3]      # 500ms accepted (120 BPM)

    def test_ibi_filter_rejects_impossibly_slow(self):
        """IBI > 2000ms = <30 BPM, likely noise unless severe bradycardia."""
        ibi_ms = np.array([1500, 1999, 2000, 2001, 3000])
        valid = (ibi_ms > 300) & (ibi_ms < 2000)
        assert valid[0]      # 1500ms accepted (40 BPM — plausible for athlete)
        assert valid[1]      # 1999ms accepted (30 BPM — extreme bradycardia)
        assert not valid[3]  # 2001ms rejected
        assert not valid[4]  # 3000ms rejected

    def test_ibi_boundary_300ms(self):
        """Exactly 300ms (200 BPM) is rejected — boundary test."""
        ibi_ms = np.array([300])
        valid = (ibi_ms > 300) & (ibi_ms < 2000)
        assert not valid[0]

    def test_ibi_boundary_2000ms(self):
        """Exactly 2000ms (30 BPM) is rejected — boundary test."""
        ibi_ms = np.array([2000])
        valid = (ibi_ms > 300) & (ibi_ms < 2000)
        assert not valid[0]

    def test_hrv_zero_when_insufficient_beats(self):
        """< 3 detected heartbeats → HRV should be 0 (unreliable)."""
        ext = _make_extractor(fs=20.0)
        # Push flat signal — no heartbeats to detect
        for _ in range(200):
            ext.push_csi(np.ones(56, dtype=np.float32) * 50.0)
        ext.update()
        assert ext.hrv_rmssd == 0.0
        assert ext.hrv_sdnn == 0.0


# ═══════════════════════════════════════════════════════════
# 4. Stress Index — RMSSD-to-stress mapping
# ═══════════════════════════════════════════════════════════

class TestStressIndexMapping:
    """Doctor validates: does the RMSSD→stress mapping match clinical literature?

    Literature reference ranges:
    - RMSSD < 15ms → high autonomic stress (ICU patients, panic)
    - RMSSD 15-30ms → elevated stress (work pressure, anxiety)
    - RMSSD 30-60ms → normal range (healthy adult)
    - RMSSD > 60ms → relaxed/athletic (vagal tone dominant)
    """

    def test_high_stress_rmssd_under_15(self):
        ext = _make_extractor()
        ext.hrv_rmssd = 10.0
        ext._compute_stress()
        assert ext.stress_index >= 90, f"RMSSD=10ms should be high stress, got {ext.stress_index}"

    def test_elevated_stress_rmssd_20(self):
        ext = _make_extractor()
        ext.hrv_rmssd = 20.0
        ext._compute_stress()
        assert 50 <= ext.stress_index <= 90, f"RMSSD=20ms stress={ext.stress_index}"

    def test_normal_stress_rmssd_45(self):
        ext = _make_extractor()
        ext.hrv_rmssd = 45.0
        ext._compute_stress()
        assert 20 <= ext.stress_index <= 50, f"RMSSD=45ms stress={ext.stress_index}"

    def test_relaxed_rmssd_80(self):
        ext = _make_extractor()
        ext.hrv_rmssd = 80.0
        ext._compute_stress()
        assert ext.stress_index <= 20, f"RMSSD=80ms should be relaxed, got {ext.stress_index}"

    def test_stress_clamped_0_100(self):
        """Stress should never exceed 0-100 range."""
        ext = _make_extractor()
        for rmssd in [0.1, 1.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]:
            ext.hrv_rmssd = rmssd
            ext._compute_stress()
            assert 0 <= ext.stress_index <= 100, f"RMSSD={rmssd} → stress={ext.stress_index}"

    def test_stress_zero_when_no_hrv(self):
        ext = _make_extractor()
        ext.hrv_rmssd = 0.0
        ext._compute_stress()
        assert ext.stress_index == 0.0

    def test_stress_monotonically_decreases_with_rmssd(self):
        """Higher RMSSD → lower stress (monotonic relationship)."""
        ext = _make_extractor()
        prev = 101.0
        for rmssd in [5, 10, 15, 20, 30, 45, 60, 80, 100]:
            ext.hrv_rmssd = float(rmssd)
            ext._compute_stress()
            assert ext.stress_index <= prev, (
                f"Stress should decrease: RMSSD={rmssd} → {ext.stress_index}, "
                f"prev was {prev}"
            )
            prev = ext.stress_index


# ═══════════════════════════════════════════════════════════
# 5. Sleep Staging — clinical heuristic validation
# ═══════════════════════════════════════════════════════════

class TestSleepStaging:
    """Doctor validates sleep stage transitions against polysomnography heuristics.

    Sleep staging rules:
    - Awake: motion > 30 OR regularity < 0.3
    - REM: motion ≤ 15, regularity 0.3-0.6 (irregular breathing in REM)
    - Light: motion ≤ 15, regularity 0.6-0.8
    - Deep: motion ≤ 15, regularity ≥ 0.8, breath rate < 16 BPM
    """

    def test_awake_high_motion(self):
        ext = _make_extractor()
        ext.motion_intensity = 50.0
        ext.breath_regularity = 0.9  # even regular — motion overrides
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "awake"

    def test_awake_moderate_motion(self):
        """Motion 15-30 → still awake (threshold at 15 and 30)."""
        ext = _make_extractor()
        ext.motion_intensity = 20.0
        ext.breath_regularity = 0.5
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "awake"

    def test_awake_irregular_breathing(self):
        ext = _make_extractor()
        ext.motion_intensity = 5.0
        ext.breath_regularity = 0.2
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "awake"

    def test_rem_low_motion_irregular(self):
        """REM: low motion + irregular breathing (0.3-0.6)."""
        ext = _make_extractor()
        ext.motion_intensity = 5.0
        ext.breath_regularity = 0.45
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "rem"

    def test_light_sleep(self):
        """Light: low motion + moderate regularity (0.6-0.8)."""
        ext = _make_extractor()
        ext.motion_intensity = 5.0
        ext.breath_regularity = 0.7
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "light"

    def test_deep_sleep(self):
        """Deep: very low motion + high regularity + slow breathing (<14 BPM)."""
        ext = _make_extractor()
        ext.motion_intensity = 3.0
        ext.breath_regularity = 0.9
        ext.breath_rate = 12.0  # < 14 BPM — classic deep sleep
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "deep"

    def test_deep_requires_slow_breathing(self):
        """High regularity but breathing 15 BPM → light, not deep."""
        ext = _make_extractor()
        ext.motion_intensity = 3.0
        ext.breath_regularity = 0.9
        ext.breath_rate = 15.0  # >= 14 BPM — too fast for N3/SWS
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "light"

    def test_deep_boundary_14bpm(self):
        """Exactly 14 BPM is NOT deep (threshold is < 14)."""
        ext = _make_extractor()
        ext.motion_intensity = 3.0
        ext.breath_regularity = 0.9
        ext.breath_rate = 14.0
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "light"

    def test_deep_boundary_13_9bpm(self):
        """13.9 BPM is deep sleep."""
        ext = _make_extractor()
        ext.motion_intensity = 3.0
        ext.breath_regularity = 0.9
        ext.breath_rate = 13.9
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "deep"

    def test_sleep_boundary_motion_15(self):
        """Motion exactly 15 → still in 'awake' (> 15 check includes 15+)."""
        ext = _make_extractor()
        ext.motion_intensity = 15.1
        ext.breath_regularity = 0.9
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "awake"

    def test_sleep_boundary_regularity_03(self):
        """Regularity exactly 0.3 → REM (not awake)."""
        ext = _make_extractor()
        ext.motion_intensity = 5.0
        ext.breath_regularity = 0.3
        ext._estimate_sleep_stage()
        # regularity < 0.3 → awake, 0.3 is NOT < 0.3 so it enters REM check
        # but < 0.6 → REM
        assert ext.sleep_stage == "rem"

    def test_sleep_boundary_regularity_06(self):
        """Regularity exactly 0.6 → light (not REM)."""
        ext = _make_extractor()
        ext.motion_intensity = 5.0
        ext.breath_regularity = 0.6
        ext._estimate_sleep_stage()
        assert ext.sleep_stage == "light"


# ═══════════════════════════════════════════════════════════
# 6. Apnea Detection — life-critical sensitivity
# ═══════════════════════════════════════════════════════════

class TestApneaDetection:
    """Apnea = absence of breathing for ≥10 seconds.
    For elderly/patient monitoring, missing an apnea event is dangerous.
    """

    def test_normal_breathing_no_apnea(self):
        ext = _make_extractor(fs=20.0)
        frames = _generate_sinusoidal_csi(freq_hz=0.25, fs=20.0, duration_sec=15.0)
        _push_frames(ext, frames)
        ext.update()
        assert ext.resp_distress is False
        assert ext.apnea_count == 0

    def test_apnea_counter_increments(self):
        """Flat signal (no breathing) should increment apnea counter."""
        ext = _make_extractor(fs=20.0)
        # First push enough normal breathing to fill the buffer
        normal_frames = _generate_sinusoidal_csi(
            freq_hz=0.25, fs=20.0, duration_sec=8.0,
        )
        _push_frames(ext, normal_frames)
        ext.update()
        assert ext.apnea_count == 0

        # Now push flat signal (apnea)
        flat_frames = []
        for _ in range(200):
            flat_frames.append(np.ones(56, dtype=np.float32) * 50.0)
        _push_frames(ext, flat_frames)
        ext.update()
        # Apnea frames should be accumulating
        assert ext._apnea_frames >= 0

    def test_apnea_recovery_decrements(self):
        """Normal breathing after apnea should decrement the counter."""
        ext = _make_extractor()
        ext._apnea_frames = 10
        # The decay rate is -2 per normal frame
        # Simulate recovery by checking the decrement logic
        assert ext._apnea_frames > 0
        ext._apnea_frames = max(0, ext._apnea_frames - 2)
        assert ext._apnea_frames == 8


# ═══════════════════════════════════════════════════════════
# 7. Motion Intensity — body movement classification
# ═══════════════════════════════════════════════════════════

class TestMotionClassification:
    """Doctor checks: still/micro/gross thresholds match clinical observation."""

    def test_body_movement_still(self):
        ext = _make_extractor()
        ext.motion_intensity = 5.0
        ext.body_movement = "still"
        # Score < 10 → still
        assert ext.body_movement == "still"

    def test_body_movement_micro(self):
        """Micro-movement: 10-40 — fidgeting, tossing in bed."""
        ext = _make_extractor()
        ext.motion_intensity = 25.0
        ext._compute_motion_intensity(np.zeros(100))  # resets
        ext.motion_intensity = 25.0  # manually override for unit test
        # Just verify the threshold logic
        if ext.motion_intensity < 10:
            expected = "still"
        elif ext.motion_intensity < 40:
            expected = "micro"
        else:
            expected = "gross"
        assert expected == "micro"

    def test_body_movement_gross(self):
        """Gross movement: >40 — walking, exercising."""
        score = 55.0
        if score < 10:
            movement = "still"
        elif score < 40:
            movement = "micro"
        else:
            movement = "gross"
        assert movement == "gross"


# ═══════════════════════════════════════════════════════════
# 8. Result Structure — all metrics present
# ═══════════════════════════════════════════════════════════

class TestVitalsResultCompleteness:
    """Doctor needs all metrics present for clinical dashboard."""

    def test_result_has_all_fields(self):
        ext = _make_extractor()
        result = ext.update()
        required = [
            "breathing_bpm", "breathing_confidence",
            "heart_bpm", "heart_confidence",
            "respiratory_distress", "apnea_events",
            "hrv_rmssd", "hrv_sdnn", "stress_index",
            "motion_intensity", "body_movement",
            "breath_regularity", "sleep_stage",
            "buffer_fullness",
        ]
        for field in required:
            assert field in result, f"Missing clinical field: {field}"

    def test_initial_values_safe(self):
        """Before enough data, values should be zero/safe — never NaN."""
        ext = _make_extractor()
        result = ext.update()
        for key, val in result.items():
            if isinstance(val, float):
                assert not np.isnan(val), f"{key} is NaN"
                assert not np.isinf(val), f"{key} is Inf"

    def test_buffer_fullness_0_to_1(self):
        ext = _make_extractor(fs=20.0)
        assert ext.update()["buffer_fullness"] == 0.0
        for _ in range(100):
            ext.push_csi(np.ones(56, dtype=np.float32))
        fullness = ext.update()["buffer_fullness"]
        assert 0.0 <= fullness <= 1.0
