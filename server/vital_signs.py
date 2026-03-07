"""Vital signs extraction from CSI data: breathing, heart rate, respiratory distress."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


class VitalSignsExtractor:
    """Extract breathing rate and heart rate from CSI amplitude variance.

    Physics basis:
    - Breathing: chest displacement ~10mm at 0.15-0.5 Hz (9-30 BPM)
      → clear periodic signal in CSI amplitude variance
    - Heart rate: chest wall displacement ~0.5mm at 0.8-2.0 Hz (48-120 BPM)
      → requires higher SNR, person must be relatively still
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
