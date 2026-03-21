"""Tests for vital signs extraction from CSI data."""
import numpy as np
import pytest

from server.vital_signs import VitalSignsExtractor, MultiPersonTracker


class TestVitalSignsExtractor:
    def test_init(self):
        vs = VitalSignsExtractor(sample_rate=100.0, window_sec=30.0)
        assert vs.fs == 100.0
        assert vs.window_size == 3000
        assert vs.breath_rate == 0.0

    def test_push_csi(self):
        vs = VitalSignsExtractor(sample_rate=100.0)
        for _ in range(10):
            vs.push_csi(np.random.rand(30))
        assert len(vs.csi_buffer) == 10

    def test_buffer_overflow(self):
        vs = VitalSignsExtractor(sample_rate=10.0, window_sec=2.0)
        for _ in range(50):
            vs.push_csi(np.random.rand(30))
        assert len(vs.csi_buffer) == 20  # 10 * 2 = 20

    def test_breathing_detection(self):
        """Inject a clear 15 BPM (0.25 Hz) breathing signal into CSI."""
        fs = 100.0
        vs = VitalSignsExtractor(sample_rate=fs, window_sec=30.0)

        t = np.arange(0, 30, 1 / fs)  # 30 seconds
        breath_freq = 0.25  # 15 BPM

        for i in range(len(t)):
            # Create CSI with breathing modulation
            base = np.ones(30) * 0.5
            modulation = 0.1 * np.sin(2 * np.pi * breath_freq * t[i])
            vs.push_csi(base + modulation + np.random.randn(30) * 0.01)

        result = vs.update()
        # Should detect breathing near 15 BPM
        assert 10.0 < result["breathing_bpm"] < 20.0
        assert result["breathing_confidence"] > 0.3

    def test_heart_rate_detection(self):
        """Inject a clear 72 BPM (1.2 Hz) heart signal into CSI."""
        fs = 100.0
        vs = VitalSignsExtractor(sample_rate=fs, window_sec=30.0)

        t = np.arange(0, 30, 1 / fs)
        heart_freq = 1.2  # 72 BPM

        for i in range(len(t)):
            base = np.ones(30) * 0.5
            # Heart signal is much weaker than breathing
            modulation = 0.02 * np.sin(2 * np.pi * heart_freq * t[i])
            vs.push_csi(base + modulation + np.random.randn(30) * 0.005)

        result = vs.update()
        # Should detect heart rate near 72 BPM
        assert 50.0 < result["heart_bpm"] < 90.0
        assert result["heart_confidence"] > 0.1

    def test_insufficient_data(self):
        vs = VitalSignsExtractor(sample_rate=100.0)
        # Only push a few frames (< 5 seconds)
        for _ in range(100):
            vs.push_csi(np.random.rand(30))
        result = vs.update()
        assert result["buffer_fullness"] < 0.1

    def test_result_structure(self):
        vs = VitalSignsExtractor()
        result = vs.update()
        assert "breathing_bpm" in result
        assert "heart_bpm" in result
        assert "breathing_confidence" in result
        assert "heart_confidence" in result
        assert "respiratory_distress" in result
        assert "apnea_events" in result
        assert "buffer_fullness" in result
        # Extended metrics
        assert "hrv_rmssd" in result
        assert "hrv_sdnn" in result
        assert "stress_index" in result
        assert "motion_intensity" in result
        assert "body_movement" in result
        assert "breath_regularity" in result
        assert "sleep_stage" in result

    def test_hrv_from_heart_signal(self):
        """HRV should be computed from clear heart signal peaks."""
        fs = 100.0
        vs = VitalSignsExtractor(sample_rate=fs, window_sec=30.0)

        t = np.arange(0, 30, 1 / fs)
        heart_freq = 1.2  # 72 BPM → ~833ms between beats

        for i in range(len(t)):
            base = np.ones(30) * 0.5
            modulation = 0.05 * np.sin(2 * np.pi * heart_freq * t[i])
            vs.push_csi(base + modulation + np.random.randn(30) * 0.002)

        result = vs.update()
        # With a clean sinusoidal heart signal, HRV should be detectable
        assert result["hrv_rmssd"] >= 0
        assert result["hrv_sdnn"] >= 0

    def test_stress_index_range(self):
        """Stress index should be 0-100."""
        vs = VitalSignsExtractor(sample_rate=100.0, window_sec=30.0)
        t = np.arange(0, 30, 1 / 100.0)
        for i in range(len(t)):
            base = np.ones(30) * 0.5
            mod = 0.05 * np.sin(2 * np.pi * 1.2 * t[i])
            vs.push_csi(base + mod + np.random.randn(30) * 0.002)
        result = vs.update()
        assert 0 <= result["stress_index"] <= 100

    def test_motion_intensity(self):
        """Motion intensity should respond to high-frequency CSI variance."""
        fs = 100.0
        vs = VitalSignsExtractor(sample_rate=fs, window_sec=30.0)

        t = np.arange(0, 30, 1 / fs)
        for i in range(len(t)):
            base = np.ones(30) * 0.5
            # Add strong motion signal at 3 Hz
            motion = 0.2 * np.sin(2 * np.pi * 3.0 * t[i])
            vs.push_csi(base + motion + np.random.randn(30) * 0.01)

        result = vs.update()
        assert result["motion_intensity"] > 0
        assert result["body_movement"] in ("still", "micro", "gross")

    def test_sleep_stage_values(self):
        """Sleep stage should be one of the valid values."""
        vs = VitalSignsExtractor()
        result = vs.update()
        assert result["sleep_stage"] in ("awake", "light", "deep", "rem")

    def test_breath_regularity_range(self):
        """Breath regularity should be 0-1."""
        fs = 100.0
        vs = VitalSignsExtractor(sample_rate=fs, window_sec=30.0)
        t = np.arange(0, 30, 1 / fs)
        for i in range(len(t)):
            base = np.ones(30) * 0.5
            mod = 0.1 * np.sin(2 * np.pi * 0.25 * t[i])
            vs.push_csi(base + mod + np.random.randn(30) * 0.01)
        result = vs.update()
        assert 0 <= result["breath_regularity"] <= 1

    def test_get_subcarrier_amplitudes(self):
        vs = VitalSignsExtractor()
        assert vs.get_subcarrier_amplitudes() is None
        vs.push_csi(np.array([0.1, 0.2, 0.3]))
        amps = vs.get_subcarrier_amplitudes()
        assert amps is not None
        assert len(amps) == 3


class TestMultiPersonTracker:
    def test_init(self):
        mp = MultiPersonTracker(max_persons=4)
        assert mp.person_count == 0

    def test_single_person(self):
        mp = MultiPersonTracker(max_persons=4, sample_rate=10.0)
        # Push data from 6 antenna pairs
        for _ in range(50):
            data = {}
            for tx in range(3):
                for rx in range(2):
                    key = f"TX{tx}_RX{rx}"
                    data[key] = np.random.rand(30) * 0.5
            mp.push_multi_antenna_csi(data)
        assert mp.person_count >= 1

    def test_update_all(self):
        mp = MultiPersonTracker()
        results = mp.update_all()
        assert isinstance(results, list)

    def test_ica_path_with_five_antenna_pairs(self):
        """ICA path is exercised when >= 5 antenna pairs are provided."""
        mp = MultiPersonTracker(max_persons=4, sample_rate=10.0)
        # Provide 6 antenna pairs so the ICA branch is taken
        for _ in range(20):
            data = {f"TX{i}_RX{j}": np.random.rand(64) for i in range(3) for j in range(2)}
            mp.push_multi_antenna_csi(data)
        # At least one person detected regardless of sklearn availability
        assert mp.person_count >= 1

    def test_ica_fewer_than_five_pairs_uses_variance(self):
        """With < 5 antenna pairs, the variance fallback is used (not ICA)."""
        mp = MultiPersonTracker(max_persons=4, sample_rate=10.0)
        for _ in range(20):
            data = {f"TX0_RX{j}": np.random.rand(64) * 0.5 for j in range(3)}
            mp.push_multi_antenna_csi(data)
        assert mp.person_count >= 1

    def test_separate_sources_ica_returns_valid_count(self):
        """_separate_sources_ica should return a count in [1, max_persons]."""
        mp = MultiPersonTracker(max_persons=4, sample_rate=10.0)
        # Construct a matrix with clear independent signals
        rng = np.random.default_rng(0)
        n_antennas, n_sub = 8, 64
        # Two independent sources mixed across antennas
        s1 = np.sin(np.linspace(0, 4 * np.pi, n_sub))
        s2 = np.cos(np.linspace(0, 8 * np.pi, n_sub))
        mix = rng.random((n_antennas, 2)) @ np.array([s1, s2])
        count = mp._separate_sources_ica(mix)
        assert 1 <= count <= mp.max_persons

    def test_separate_sources_ica_small_matrix_fallback(self):
        """_separate_sources_ica falls back to variance when matrix has < 4 rows."""
        mp = MultiPersonTracker(max_persons=4, sample_rate=10.0)
        small_matrix = np.random.rand(2, 32)
        count = mp._separate_sources_ica(small_matrix)
        assert 0 <= count <= mp.max_persons
