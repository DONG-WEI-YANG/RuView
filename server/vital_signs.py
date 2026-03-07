"""Vital signs extraction from CSI data: breathing, heart rate, HRV, sleep, motion.

Enhanced with Wi-Mesh-inspired metrics:
- Heart Rate Variability (HRV) via inter-beat interval analysis
- Sleep stage estimation from breathing regularity + motion level
- Motion intensity score from broadband CSI variance
- Stress index derived from HRV (RMSSD)
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


class VitalSignsExtractor:
    """Extract vital signs and wellness metrics from CSI amplitude.

    Physics basis:
    - Breathing: chest displacement ~10mm at 0.15-0.5 Hz (9-30 BPM)
      → clear periodic signal in CSI amplitude variance
    - Heart rate: chest wall displacement ~0.5mm at 0.8-2.0 Hz (48-120 BPM)
      → requires higher SNR, person must be relatively still
    - HRV: beat-to-beat interval variance from heart signal peaks
      → autonomic nervous system / stress indicator
    - Sleep staging: breathing regularity + motion level classification
      → wake / light / deep / REM estimation
    - Motion intensity: broadband (1-8 Hz) CSI variance RMS
      → activity level 0-100
    - SpO2: NOT measurable from WiFi (requires optical sensing)
      → we detect breathing anomalies as a respiratory distress proxy
    """

    def __init__(self, sample_rate: float = 100.0, window_sec: float = 30.0):
        self.fs = sample_rate
        self.window_size = int(sample_rate * window_sec)
        self.csi_buffer = []  # rolling buffer of CSI amplitude vectors
        self.breath_rate = 0.0
        self.breath_confidence = 0.0
        self.heart_rate = 0.0
        self.heart_confidence = 0.0
        self.resp_distress = False
        self.apnea_count = 0
        self._apnea_frames = 0
        # New Wi-Mesh-inspired metrics
        self.hrv_rmssd = 0.0        # HRV: root mean square of successive differences (ms)
        self.hrv_sdnn = 0.0         # HRV: standard deviation of NN intervals (ms)
        self.stress_index = 0.0     # 0-100 stress score (high = stressed)
        self.motion_intensity = 0.0 # 0-100 motion activity score
        self.sleep_stage = "awake"  # awake / light / deep / rem
        self.body_movement = "still"  # still / micro / gross
        self.breath_regularity = 0.0  # 0-1 how regular breathing is

    def push_csi(self, amplitudes: np.ndarray):
        """Push one CSI frame (vector of subcarrier amplitudes)."""
        self.csi_buffer.append(amplitudes.copy())
        if len(self.csi_buffer) > self.window_size:
            self.csi_buffer.pop(0)

    def update(self) -> dict:
        """Compute vitals from current buffer. Call at ~1-2 Hz."""
        if len(self.csi_buffer) < int(self.fs * 5):
            return self._result()

        data = np.array(self.csi_buffer)  # (T, num_subcarriers)
        # Mean amplitude captures bulk motion (breathing, body movement)
        mean_amp = np.mean(data, axis=1)  # (T,)

        # ── Breathing rate (bandpass 0.1–0.5 Hz = 6–30 BPM) ──────
        breath_sig = self._bandpass(mean_amp, 0.1, 0.5)
        self.breath_rate, self.breath_confidence = self._estimate_rate(
            breath_sig, min_hz=0.1, max_hz=0.5, label="breath"
        )

        # ── Heart rate (bandpass 0.8–2.0 Hz = 48–120 BPM) ────────
        # Mean amplitude also carries heart signal (person must be still)
        heart_sig = self._bandpass(mean_amp, 0.8, 2.0)
        self.heart_rate, self.heart_confidence = self._estimate_rate(
            heart_sig, min_hz=0.8, max_hz=2.0, label="heart"
        )

        # ── HRV from heart signal peaks ──────────────────────────
        self._compute_hrv(heart_sig)

        # ── Stress index from HRV ────────────────────────────────
        self._compute_stress()

        # ── Motion intensity (broadband 1–8 Hz) ─────────────────
        self._compute_motion_intensity(mean_amp)

        # ── Breathing regularity ─────────────────────────────────
        self._compute_breath_regularity(breath_sig)

        # ── Sleep stage estimation ───────────────────────────────
        self._estimate_sleep_stage()

        # ── Respiratory distress detection ────────────────────────
        self._detect_resp_distress(breath_sig)

        return self._result()

    def _bandpass(self, sig: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
        """Bandpass filter a 1D signal using SOS for numerical stability."""
        nyq = self.fs / 2.0
        low_n = max(low / nyq, 0.001)
        high_n = min(high / nyq, 0.999)
        if low_n >= high_n:
            return sig
        sos = butter(order, [low_n, high_n], btype="band", output="sos")
        return sosfiltfilt(sos, sig)

    def _estimate_rate(self, sig: np.ndarray, min_hz: float, max_hz: float, label: str) -> tuple[float, float]:
        """Estimate rate (BPM) via FFT peak detection."""
        n = len(sig)
        if n < 32:
            return 0.0, 0.0

        # FFT
        fft = np.abs(np.fft.rfft(sig * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        # Mask to valid range
        mask = (freqs >= min_hz) & (freqs <= max_hz)
        if not np.any(mask):
            return 0.0, 0.0

        fft_masked = fft[mask]
        freqs_masked = freqs[mask]

        # Peak
        peak_idx = np.argmax(fft_masked)
        peak_freq = freqs_masked[peak_idx]
        peak_power = fft_masked[peak_idx]

        # Confidence: ratio of peak to mean (SNR proxy)
        mean_power = np.mean(fft_masked)
        if mean_power < 1e-10:
            return 0.0, 0.0
        snr = peak_power / mean_power
        confidence = min(1.0, max(0.0, (snr - 1.5) / 5.0))

        bpm = peak_freq * 60.0
        return round(bpm, 1), round(confidence, 2)

    def _compute_hrv(self, heart_sig: np.ndarray):
        """Compute Heart Rate Variability from inter-beat intervals.

        Detects peaks in the bandpassed heart signal, computes IBI
        (inter-beat intervals), then derives RMSSD and SDNN.
        """
        if len(heart_sig) < int(self.fs * 5):
            return

        # Find heartbeat peaks (positive peaks in bandpassed signal)
        min_distance = int(self.fs * 0.4)  # minimum 0.4s between beats (~150 BPM max)
        # Adaptive threshold: use signal statistics
        threshold = np.mean(heart_sig) + 0.3 * np.std(heart_sig)
        peaks, _ = find_peaks(heart_sig, distance=min_distance, height=threshold)

        if len(peaks) < 3:
            self.hrv_rmssd = 0.0
            self.hrv_sdnn = 0.0
            return

        # Inter-beat intervals in milliseconds
        ibi_samples = np.diff(peaks)
        ibi_ms = ibi_samples * (1000.0 / self.fs)

        # Filter out physiologically impossible intervals (<300ms or >2000ms)
        valid = (ibi_ms > 300) & (ibi_ms < 2000)
        ibi_ms = ibi_ms[valid]

        if len(ibi_ms) < 2:
            return

        # RMSSD: root mean square of successive differences
        successive_diffs = np.diff(ibi_ms)
        self.hrv_rmssd = round(float(np.sqrt(np.mean(successive_diffs ** 2))), 1)

        # SDNN: standard deviation of all NN intervals
        self.hrv_sdnn = round(float(np.std(ibi_ms)), 1)

    def _compute_stress(self):
        """Derive stress index from HRV metrics.

        Low HRV (RMSSD) correlates with high sympathetic activation (stress).
        Normal RMSSD range: 20-100ms. Below 20ms = high stress.
        """
        if self.hrv_rmssd <= 0:
            self.stress_index = 0.0
            return

        # Map RMSSD to stress: low RMSSD = high stress
        # RMSSD < 15ms → stress ~90-100
        # RMSSD 15-30ms → stress ~50-90
        # RMSSD 30-60ms → stress ~20-50
        # RMSSD > 60ms → stress ~0-20
        rmssd = self.hrv_rmssd
        if rmssd < 15:
            stress = 90 + (15 - rmssd) / 15 * 10
        elif rmssd < 30:
            stress = 50 + (30 - rmssd) / 15 * 40
        elif rmssd < 60:
            stress = 20 + (60 - rmssd) / 30 * 30
        else:
            stress = max(0, 20 - (rmssd - 60) / 40 * 20)

        self.stress_index = round(min(100.0, max(0.0, stress)), 1)

    def _compute_motion_intensity(self, mean_amp: np.ndarray):
        """Compute motion intensity from broadband CSI variance (1-8 Hz).

        Higher variance in the motion band = more body movement.
        Scaled to 0-100 score.
        """
        if len(mean_amp) < int(self.fs * 2):
            self.motion_intensity = 0.0
            self.body_movement = "still"
            return

        motion_sig = self._bandpass(mean_amp, 1.0, 8.0)
        rms = float(np.sqrt(np.mean(motion_sig ** 2)))

        # Scale: typical RMS values 0.001-0.1 → 0-100
        # Using log scale for better dynamic range
        if rms < 1e-6:
            score = 0.0
        else:
            score = min(100.0, max(0.0, (np.log10(rms) + 4) * 25))

        self.motion_intensity = round(score, 1)

        # Classify body movement level
        if score < 10:
            self.body_movement = "still"
        elif score < 40:
            self.body_movement = "micro"
        else:
            self.body_movement = "gross"

    def _compute_breath_regularity(self, breath_sig: np.ndarray):
        """Measure breathing regularity via coefficient of variation of breath intervals."""
        if len(breath_sig) < int(self.fs * 10):
            self.breath_regularity = 0.0
            return

        # Find breathing cycle peaks
        min_dist = int(self.fs * 1.5)  # min 1.5s between breaths (~40 BPM max)
        peaks, _ = find_peaks(breath_sig, distance=min_dist)

        if len(peaks) < 3:
            self.breath_regularity = 0.0
            return

        intervals = np.diff(peaks) / self.fs  # in seconds
        mean_interval = np.mean(intervals)
        if mean_interval < 0.01:
            self.breath_regularity = 0.0
            return

        # Coefficient of variation: lower = more regular
        cv = np.std(intervals) / mean_interval
        # Map CV to regularity: CV=0 → regularity=1.0, CV>0.5 → regularity≈0
        self.breath_regularity = round(float(np.clip(1.0 - cv * 2.0, 0, 1)), 2)

    def _estimate_sleep_stage(self):
        """Estimate sleep stage from breathing regularity and motion level.

        Sleep staging heuristic:
        - Awake: motion > 30 OR breathing regularity < 0.3
        - REM: low motion, irregular breathing (regularity 0.3-0.6)
        - Light sleep: low motion, moderate regularity (0.6-0.8)
        - Deep sleep: very low motion, high regularity (>0.8), slow breathing
        """
        motion = self.motion_intensity
        reg = self.breath_regularity
        bpm = self.breath_rate

        if motion > 30:
            self.sleep_stage = "awake"
        elif motion > 15 or reg < 0.3:
            self.sleep_stage = "awake"
        elif reg < 0.6:
            # Low motion + irregular breathing → REM
            self.sleep_stage = "rem"
        elif reg < 0.8:
            self.sleep_stage = "light"
        else:
            # Very regular breathing + very low motion
            # Deep sleep tends to have slower breathing (10-14 BPM)
            if bpm > 0 and bpm < 16:
                self.sleep_stage = "deep"
            else:
                self.sleep_stage = "light"

    def _detect_resp_distress(self, breath_sig: np.ndarray):
        """Detect apnea / abnormal breathing patterns as SpO2 proxy."""
        # Apnea: breathing signal amplitude drops below threshold
        recent = breath_sig[-int(self.fs * 5):]  # last 5 seconds
        amplitude = np.max(recent) - np.min(recent)

        if amplitude < 0.01:  # near-zero breathing signal
            self._apnea_frames += 1
        else:
            self._apnea_frames = max(0, self._apnea_frames - 2)

        # Apnea event if breathing absent for >10 seconds
        if self._apnea_frames > int(self.fs * 10 / (self.fs * 5)):
            if not self.resp_distress:
                self.apnea_count += 1
            self.resp_distress = True
        else:
            self.resp_distress = False

    def _result(self) -> dict:
        return {
            "breathing_bpm": self.breath_rate,
            "breathing_confidence": self.breath_confidence,
            "heart_bpm": self.heart_rate,
            "heart_confidence": self.heart_confidence,
            "respiratory_distress": self.resp_distress,
            "apnea_events": self.apnea_count,
            "buffer_fullness": min(1.0, len(self.csi_buffer) / self.window_size),
            # Wi-Mesh-inspired extended metrics
            "hrv_rmssd": self.hrv_rmssd,
            "hrv_sdnn": self.hrv_sdnn,
            "stress_index": self.stress_index,
            "motion_intensity": self.motion_intensity,
            "body_movement": self.body_movement,
            "breath_regularity": self.breath_regularity,
            "sleep_stage": self.sleep_stage,
        }

    def get_subcarrier_amplitudes(self) -> list[float] | None:
        """Return latest CSI frame for waterfall visualization."""
        if self.csi_buffer:
            return self.csi_buffer[-1].tolist()
        return None


class MultiPersonTracker:
    """Separate and track multiple persons from multi-antenna CSI.

    Uses spatial diversity from TX-RX pairs to identify distinct
    signal signatures. Each person creates unique multipath patterns.

    With 3 TX × 6 RX = 18 CSI streams, we can theoretically separate
    up to 3-4 persons using signal subspace decomposition.
    """

    def __init__(self, max_persons: int = 4, sample_rate: float = 100.0):
        self.max_persons = max_persons
        self.fs = sample_rate
        self.persons = {}  # person_id -> PersonState
        self._next_id = 0

    def push_multi_antenna_csi(self, antenna_data: dict[str, np.ndarray]):
        """Push CSI from multiple TX-RX pairs.

        Args:
            antenna_data: dict mapping "TX{i}_RX{j}" -> amplitude vector
        """
        if not antenna_data:
            return

        # Stack all antenna pair data into matrix
        keys = sorted(antenna_data.keys())
        matrix = np.array([antenna_data[k] for k in keys])  # (n_pairs, n_subcarriers)

        # Variance profile per antenna pair → spatial signature
        var_profile = np.var(matrix, axis=1)

        # Simple clustering: split by dominant variance peaks
        n_detected = self._count_persons(var_profile)

        # Update person states
        for pid in range(n_detected):
            if pid not in self.persons:
                self.persons[pid] = {
                    "id": self._next_id,
                    "vitals": VitalSignsExtractor(self.fs),
                    "position": [0.0, 0.0],
                    "confidence": 0.0,
                }
                self._next_id += 1

            # Feed person-specific CSI subset to their vital signs extractor
            start = pid * (len(keys) // max(n_detected, 1))
            end = (pid + 1) * (len(keys) // max(n_detected, 1))
            person_csi = np.mean(matrix[start:end], axis=0)
            self.persons[pid]["vitals"].push_csi(person_csi)

        # Remove stale persons
        stale = [pid for pid in self.persons if pid >= n_detected]
        for pid in stale:
            del self.persons[pid]

    def _count_persons(self, var_profile: np.ndarray) -> int:
        """Estimate number of persons from variance profile peaks."""
        if len(var_profile) < 3:
            return 1 if np.max(var_profile) > 0.01 else 0

        # Find peaks in spatial variance
        threshold = np.mean(var_profile) + np.std(var_profile)
        peaks, _ = find_peaks(var_profile, height=threshold, distance=2)

        n = max(1, len(peaks))  # at least 1 if any signal
        return min(n, self.max_persons)

    def update_all(self) -> list[dict]:
        """Update vitals for all tracked persons."""
        results = []
        for pid, state in self.persons.items():
            vitals = state["vitals"].update()
            results.append({
                "person_id": state["id"],
                "vitals": vitals,
                "position": state["position"],
            })
        return results

    @property
    def person_count(self) -> int:
        return len(self.persons)
