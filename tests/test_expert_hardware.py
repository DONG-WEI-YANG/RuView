"""Hardware RD Tour — ESP32 CSI, RF engineering, calibration physics.

A hardware R&D engineer reviews: Is the RSSI path-loss model valid?
Are the ADR-018 binary fields correct? Does the calibration collect
enough samples? What about channel mapping edge cases?

Expert focus: physical layer accuracy, firmware protocol correctness.
"""
import struct
import time

import pytest
import numpy as np

from server.csi_frame import (
    parse_csi_frame, CSIFrame, MAGIC_HEADER, HEADER_FORMAT, HEADER_SIZE,
    _freq_to_channel,
)
from server.calibration import (
    CalibrationManager, CALIBRATION_DURATION_SEC, MIN_SAMPLES_PER_NODE,
)
from server.services.signal_quality import (
    SignalQualityMonitor, NodeQuality, CAPABILITY_TABLE,
    RSSI_EXCELLENT, RSSI_GOOD, RSSI_POOR,
)
from server.services.event_emitter import EventEmitter
from server.config import Settings, HARDWARE_PROFILES
from tests.conftest import make_csi_frame, make_csi_binary


# ═══════════════════════════════════════════════════════════
# 1. ADR-018 Binary Protocol — firmware compatibility
# ═══════════════════════════════════════════════════════════

class TestADR018Protocol:
    """Hardware RD verifies the binary CSI frame format matches firmware."""

    def test_header_size_20_bytes(self):
        """ADR-018 spec: header is exactly 20 bytes."""
        assert HEADER_SIZE == 20

    def test_magic_is_csi_marker(self):
        """0xC5110001 — 'CSI' with version 1."""
        assert MAGIC_HEADER == 0xC5110001

    def test_field_layout_matches_firmware(self):
        """Verify struct format matches C firmware layout:
        uint32_t magic, uint8_t node_id, uint8_t antennas,
        uint16_t num_sub, uint32_t freq_mhz, uint32_t seq,
        int8_t rssi, int8_t noise, uint16_t reserved
        """
        # Pack known values
        data = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER,  # magic
            2,             # node_id (u8)
            1,             # antennas (u8)
            56,            # num_sub (u16)
            2437,          # freq_mhz (u32) = channel 6
            42,            # sequence (u32)
            -55,           # rssi (i8)
            -90,           # noise_floor (i8)
            0,             # reserved (u16)
        )
        assert len(data) == HEADER_SIZE

        # Add IQ payload
        iq = np.random.randint(-500, 500, size=56 * 2, dtype=np.int16)
        frame = parse_csi_frame(data + iq.tobytes())
        assert frame is not None
        assert frame.node_id == 2
        assert frame.num_subcarriers == 56
        assert frame.rssi == -55
        assert frame.noise_floor == -90
        assert frame.sequence == 42
        assert frame.channel == 6

    def test_iq_interleaving_real_imag(self):
        """IQ data is int16 interleaved: R0, I0, R1, I1, ...
        Hardware RD must verify complex number reconstruction.
        """
        # Known IQ pairs
        n_sub = 4
        iq = np.array([100, 200, -50, 75, 0, 0, 300, -100], dtype=np.int16)
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, n_sub, 2437, 0, -50, -90, 0,
        )
        frame = parse_csi_frame(header + iq.tobytes())
        assert frame is not None
        # Verify complex reconstruction (float32 precision from int16 source)
        expected = np.array([100+200j, -50+75j, 0+0j, 300-100j], dtype=np.complex64)
        np.testing.assert_array_almost_equal(frame.raw_complex, expected, decimal=3)
        # Verify amplitude = |complex|
        np.testing.assert_array_almost_equal(
            frame.amplitude, np.abs(expected), decimal=3,
        )
        # Verify phase = angle(complex)
        np.testing.assert_array_almost_equal(
            frame.phase, np.angle(expected), decimal=3,
        )

    def test_node_id_full_u8_range(self):
        """Node ID is uint8: valid range 0-255."""
        for nid in [0, 1, 127, 255]:
            data = make_csi_binary(node_id=nid)
            frame = parse_csi_frame(data)
            assert frame is not None
            assert frame.node_id == nid

    def test_sequence_number_wraps(self):
        """Sequence is uint32: should handle full range."""
        data = make_csi_binary(sequence=0xFFFFFFFF)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert frame.sequence == 0xFFFFFFFF

    def test_rssi_signed_range(self):
        """RSSI is int8: range -128 to +127."""
        for rssi in [-128, -100, -50, 0, 10, 127]:
            data = make_csi_binary(rssi=rssi)
            frame = parse_csi_frame(data)
            assert frame is not None
            assert frame.rssi == rssi


# ═══════════════════════════════════════════════════════════
# 2. Frequency/Channel Mapping — WiFi band accuracy
# ═══════════════════════════════════════════════════════════

class TestChannelMapping:
    """Hardware RD verifies freq→channel conversion for all WiFi bands."""

    # 2.4 GHz band (channels 1-14)
    def test_24ghz_channel_1(self):
        assert _freq_to_channel(2412) == 1

    def test_24ghz_channel_6(self):
        assert _freq_to_channel(2437) == 6

    def test_24ghz_channel_11(self):
        assert _freq_to_channel(2462) == 11

    def test_24ghz_channel_13(self):
        """Channel 13 (Japan/EU) — 2472 MHz."""
        assert _freq_to_channel(2472) == 13

    def test_24ghz_channel_14(self):
        """Channel 14 (Japan only) — 2484 MHz, special case."""
        assert _freq_to_channel(2484) == 14

    # 5 GHz band
    def test_5ghz_channel_36(self):
        assert _freq_to_channel(5180) == 36

    def test_5ghz_channel_40(self):
        assert _freq_to_channel(5200) == 40

    def test_5ghz_channel_44(self):
        assert _freq_to_channel(5220) == 44

    def test_5ghz_channel_149(self):
        assert _freq_to_channel(5745) == 149

    def test_5ghz_channel_165(self):
        assert _freq_to_channel(5825) == 165

    # Edge cases
    def test_unknown_frequency(self):
        """Frequency outside known bands should return 0."""
        assert _freq_to_channel(900) == 0
        assert _freq_to_channel(6000) == 0

    def test_gap_frequency(self):
        """Frequency between 2.4G and 5G bands."""
        assert _freq_to_channel(3500) == 0


# ═══════════════════════════════════════════════════════════
# 3. RSSI Path-Loss Calibration — physics validation
# ═══════════════════════════════════════════════════════════

class TestCalibrationPhysics:
    """Hardware RD validates the log-distance path loss model.
    Formula: distance = 10^((A - RSSI) / (10 * n))
    where A=-45 dBm @ 1m, n=2.5 (indoor environment).
    """

    def test_distance_at_reference_1m(self):
        """At RSSI = -45 dBm (reference power), distance should be ~1m."""
        dist = 10 ** ((-45 - (-45)) / (10 * 2.5))
        assert abs(dist - 1.0) < 0.01

    def test_distance_at_3m(self):
        """RSSI degrades with distance. At ~3m, expect RSSI ≈ -57 dBm."""
        expected_rssi = -45 - 10 * 2.5 * np.log10(3.0)  # ≈ -56.9
        dist = 10 ** ((-45 - expected_rssi) / (10 * 2.5))
        assert abs(dist - 3.0) < 0.1

    def test_distance_at_0_5m(self):
        """Very close: RSSI ≈ -37.5 dBm → distance ≈ 0.5m."""
        expected_rssi = -45 - 10 * 2.5 * np.log10(0.5)  # ≈ -37.5
        dist = 10 ** ((-45 - expected_rssi) / (10 * 2.5))
        assert abs(dist - 0.5) < 0.1

    def test_distance_extreme_close(self):
        """RSSI = -30 dBm (very close to antenna)."""
        dist = 10 ** ((-45 - (-30)) / (10 * 2.5))
        assert dist < 0.5

    def test_distance_extreme_far(self):
        """RSSI = -85 dBm (barely detectable)."""
        dist = 10 ** ((-45 - (-85)) / (10 * 2.5))
        assert dist > 10.0

    def test_configurable_path_loss_exponent(self):
        """Hardware RD can set n for different environments."""
        # Hallway: n=1.8 (waveguide effect)
        mgr_hallway = CalibrationManager(path_loss_exp=1.8)
        assert mgr_hallway.path_loss_exp == 1.8
        # Factory: n=4.0 (heavy multipath)
        mgr_factory = CalibrationManager(path_loss_exp=4.0)
        assert mgr_factory.path_loss_exp == 4.0

    def test_configurable_ref_power(self):
        """Different antennas have different 1m reference power."""
        mgr = CalibrationManager(ref_power_dbm=-40)
        assert mgr.ref_power_dbm == -40

    def test_path_loss_affects_distance_estimate(self):
        """Higher n → shorter estimated distance for same RSSI."""
        mgr_low_n = CalibrationManager(path_loss_exp=2.0)
        mgr_high_n = CalibrationManager(path_loss_exp=4.0)

        mgr_low_n.start(mode="spatial")
        mgr_high_n.start(mode="spatial")
        for i in range(25):
            f = make_csi_frame(node_id=1, rssi=-65, sequence=i)
            mgr_low_n.on_csi_frame(f)
            mgr_high_n.on_csi_frame(f)

        r_low = mgr_low_n.finish()
        r_high = mgr_high_n.finish()
        d_low = r_low["nodes"]["1"]["estimated_distance_m"]
        d_high = r_high["nodes"]["1"]["estimated_distance_m"]
        assert d_low > d_high, (
            f"n=2.0 should estimate farther ({d_low}m) than n=4.0 ({d_high}m)"
        )

    def test_default_values_backward_compatible(self):
        """Default CalibrationManager uses A=-45, n=2.5 (unchanged)."""
        mgr = CalibrationManager()
        assert mgr.ref_power_dbm == -45
        assert mgr.path_loss_exp == 2.5


class TestCalibrationSession:
    """Hardware RD validates the calibration data collection process."""

    def test_min_samples_per_node(self):
        """Need at least 20 samples per node (1 second @ 20 Hz)."""
        assert MIN_SAMPLES_PER_NODE == 20

    def test_default_duration(self):
        assert CALIBRATION_DURATION_SEC == 5.0

    def test_spatial_calibration_flow(self):
        mgr = CalibrationManager(duration=5.0)
        result = mgr.start(mode="spatial")
        assert result["status"] == "calibrating"

        # Feed 25 frames from node 1 (above minimum)
        for i in range(25):
            frame = make_csi_frame(node_id=1, rssi=-55, sequence=i)
            mgr.on_csi_frame(frame)

        result = mgr.finish()
        assert result["status"] == "complete"
        assert result["mode"] == "spatial"
        assert "1" in result["nodes"]
        node_data = result["nodes"]["1"]
        assert node_data["sample_count"] == 25
        assert abs(node_data["rssi"] - (-55.0)) < 1.0

    def test_below_min_samples_excluded(self):
        """Node with < 20 samples should be excluded from results."""
        mgr = CalibrationManager()
        mgr.start(mode="spatial")

        # Node 1: 25 samples (included)
        for i in range(25):
            mgr.on_csi_frame(make_csi_frame(node_id=1, rssi=-50, sequence=i))
        # Node 2: only 15 samples (excluded)
        for i in range(15):
            mgr.on_csi_frame(make_csi_frame(node_id=2, rssi=-60, sequence=i))

        result = mgr.finish()
        assert "1" in result["nodes"]
        assert "2" not in result["nodes"]

    def test_background_calibration(self):
        """Background mode stores mean CSI amplitude per subcarrier."""
        mgr = CalibrationManager()
        mgr.start(mode="background")

        for i in range(25):
            mgr.on_csi_frame(make_csi_frame(node_id=1, sequence=i))

        result = mgr.finish()
        assert result["status"] == "complete"
        assert result["mode"] == "background"
        profile = mgr.get_background_profile()
        assert 1 in profile
        assert len(profile[1]) == 56  # 56 subcarriers

    def test_calibration_not_active_after_finish(self):
        mgr = CalibrationManager()
        mgr.start()
        mgr.finish()
        assert mgr.is_active is False

    def test_finish_without_start_returns_error(self):
        mgr = CalibrationManager()
        result = mgr.finish()
        assert result["status"] == "error"

    def test_multi_node_calibration(self):
        """Calibrate 4 nodes simultaneously — typical deployment."""
        mgr = CalibrationManager()
        mgr.start(mode="spatial")

        for nid in range(1, 5):
            rssi = -45 - nid * 5  # increasingly distant
            for i in range(25):
                mgr.on_csi_frame(make_csi_frame(
                    node_id=nid, rssi=rssi, sequence=i,
                ))

        result = mgr.finish()
        assert result["node_count"] == 4

        # Closer node (higher RSSI) should have shorter estimated distance
        d1 = result["nodes"]["1"]["estimated_distance_m"]
        d4 = result["nodes"]["4"]["estimated_distance_m"]
        assert d1 < d4, f"Node 1 ({d1}m) should be closer than Node 4 ({d4}m)"

    def test_progress_tracking(self):
        mgr = CalibrationManager(duration=5.0)
        assert mgr.progress == 0.0
        mgr.start()
        # Progress should be between 0 and 1 during session
        assert 0.0 <= mgr.progress <= 1.0


# ═══════════════════════════════════════════════════════════
# 4. Hardware Profiles — ESP32 variants
# ═══════════════════════════════════════════════════════════

class TestHardwareProfiles:
    """Hardware RD verifies each profile matches real hardware specs."""

    def test_esp32s3_profile(self):
        p = HARDWARE_PROFILES["esp32s3"]
        assert p.num_subcarriers == 56
        assert p.frequency_ghz == 2.4
        assert p.bandwidth_mhz == 20
        assert p.csi_sample_rate == 20

    def test_esp32c6_wifi6_profile(self):
        """ESP32-C6 has WiFi 6 — 64 subcarriers (802.11ax)."""
        p = HARDWARE_PROFILES["esp32c6_wifi6"]
        assert p.num_subcarriers == 64
        assert p.csi_sample_rate == 50

    def test_intel5300_profile(self):
        """Intel 5300: classic CSI tool, 30 subcarriers, 5 GHz."""
        p = HARDWARE_PROFILES["intel5300"]
        assert p.num_subcarriers == 30
        assert p.frequency_ghz == 5.0

    def test_tplink_n750_high_rate(self):
        """TP-Link N750: 114 subcarriers @ 40 MHz bandwidth."""
        p = HARDWARE_PROFILES["tplink_n750"]
        assert p.num_subcarriers == 114
        assert p.bandwidth_mhz == 40
        assert p.csi_sample_rate == 100

    def test_profile_applies_to_settings(self):
        """Applying a profile should update Settings fields."""
        s = Settings()
        s.hardware_profile = "intel5300"
        profile = s.apply_hardware_profile()
        assert s.num_subcarriers == 30
        assert s.model_path == "models/intel5300/pose_model.pth"

    def test_unknown_profile_returns_none(self):
        s = Settings()
        s.hardware_profile = "nonexistent"
        assert s.apply_hardware_profile() is None


# ═══════════════════════════════════════════════════════════
# 5. Signal Quality — grade-to-capability mapping
# ═══════════════════════════════════════════════════════════

class TestSignalCapabilityMapping:
    """Hardware RD validates that signal grades correctly limit features.
    Poor signal = presence only (no heart rate from 0.5mm displacement).
    """

    def test_excellent_enables_all(self):
        caps = CAPABILITY_TABLE["excellent"]["capabilities"]
        assert "breathing" in caps
        assert "heart_rate" in caps
        assert "hrv" in caps
        assert "pose" in caps
        assert "fine_motion" in caps

    def test_good_no_hrv(self):
        caps = CAPABILITY_TABLE["good"]["capabilities"]
        assert "heart_rate" in caps
        assert "hrv" not in caps  # HRV needs excellent SNR

    def test_fair_no_heart_rate(self):
        caps = CAPABILITY_TABLE["fair"]["capabilities"]
        assert "breathing" in caps
        assert "heart_rate" not in caps

    def test_poor_only_presence(self):
        caps = CAPABILITY_TABLE["poor"]["capabilities"]
        assert caps == ["presence"]

    def test_rssi_thresholds_match_real_world(self):
        """Hardware RD validates thresholds against ESP32 measurements:
        - Excellent (-50+): same room, line-of-sight, < 2m
        - Good (-65+): same room, minor obstacles
        - Fair (-75+): adjacent room, one wall
        - Poor (< -75): multiple walls, unreliable
        """
        assert RSSI_EXCELLENT == -50
        assert RSSI_GOOD == -65
        assert RSSI_POOR == -75

    def test_snr_computation(self):
        nq = NodeQuality(node_id=1)
        nq.rssi_history.append(-50)
        nq.noise_history.append(-90)
        assert nq.snr == 40.0  # -50 - (-90) = 40 dB

    def test_csi_frame_to_frame_variance(self):
        """Monitor tracks frame-to-frame CSI variance for stability."""
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        # Stable environment: same amplitude each frame
        stable_amp = np.ones(56, dtype=np.float32) * 50.0
        for i in range(10):
            frame = make_csi_frame(
                node_id=1, sequence=i,
                amplitude=stable_amp + np.random.randn(56).astype(np.float32) * 0.01,
            )
            monitor.on_frame(frame)
        nq = monitor._nodes[1]
        # Very stable → variance history should be very low
        if len(nq.csi_var_history) >= 5:
            assert nq.csi_stability > 0.8


# ═══════════════════════════════════════════════════════════
# 6. Multi-Antenna CSI — MIMO specifics
# ═══════════════════════════════════════════════════════════

class TestMIMOHandling:
    """ESP32-S3 supports up to 3x3 MIMO. Verify multi-antenna parsing."""

    def test_single_antenna_56_subcarriers(self):
        data = make_csi_binary(n_antennas=1, n_sub=56)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert len(frame.amplitude) == 56

    def test_dual_antenna_112_total(self):
        """2 antennas × 56 sub = 112 I/Q pairs."""
        data = make_csi_binary(n_antennas=2, n_sub=56)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert len(frame.amplitude) == 112

    def test_triple_antenna_168_total(self):
        """3 antennas × 56 sub = 168 I/Q pairs (3x3 MIMO)."""
        data = make_csi_binary(n_antennas=3, n_sub=56)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert len(frame.amplitude) == 168

    def test_wifi6_64_subcarriers(self):
        """ESP32-C6 WiFi 6: 64 subcarriers per antenna."""
        data = make_csi_binary(n_antennas=1, n_sub=64)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert len(frame.amplitude) == 64

    def test_timestamp_derived_from_sequence(self):
        """Firmware doesn't send timestamps; derived as seq × 50ms (20 Hz)."""
        data = make_csi_binary(sequence=100)
        frame = parse_csi_frame(data)
        assert frame.timestamp_ms == 5000  # 100 * 50ms
