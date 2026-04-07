"""Mixed-Chip Tour — tests for multi-chip ESP32 deployments (S3 + C3).

Covers the 2026-04-07 fixes:
  1. Flash size detection (Business: correct firmware for each chip)
  2. int8/int16 CSI parser (Seedy: different data formats from different chips)
  3. Mixed-length amplitude vectors in pipeline (Seedy: inhomogeneous arrays)
  4. Vitals buffer with mixed subcarrier counts (Historic: regression guard)
"""
import struct

import pytest
import numpy as np

from server.csi_frame import parse_csi_frame, MAGIC_HEADER, HEADER_FORMAT, HEADER_SIZE
from server.config import Settings
from server.signal_processor import SignalProcessor
from server.vital_signs import VitalSignsExtractor
# pipeline import not needed for these tests
from server.firmware_builder import CHIP_BOARDS
from tests.conftest import make_csi_frame, make_csi_binary


# ═══════════════════════════════════════════════════════════
# BUSINESS DISTRICT — core paths with mixed hardware
# ═══════════════════════════════════════════════════════════

class TestChipDetectionBusiness:
    """Flash size must match the actual hardware."""

    def test_s3_gets_8mb(self):
        assert CHIP_BOARDS["esp32s3"]["flash_size"] == "8MB"

    def test_c3_gets_4mb(self):
        assert CHIP_BOARDS["esp32c3"]["flash_size"] == "4MB"

    def test_all_chips_have_flash_size(self):
        for chip_id, info in CHIP_BOARDS.items():
            assert "flash_size" in info, f"{chip_id} missing flash_size"
            assert info["flash_size"].endswith("MB"), f"{chip_id} invalid flash_size"


class TestCSIParserFormats:
    """Parser must handle both int8 (ESP-IDF native) and int16 formats."""

    def _make_packet(self, n_sub=64, dtype=np.int8):
        """Build a raw ADR-018 packet with specified I/Q data type."""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, n_sub, 2437, 42, -50, -90, 0,
        )
        iq_count = n_sub
        iq_data = np.random.randint(-100, 100, size=iq_count * 2, dtype=dtype)
        return header + iq_data.tobytes()

    def test_int16_format(self):
        """Standard int16 I/Q pairs — the original format."""
        pkt = self._make_packet(n_sub=64, dtype=np.int16)
        frame = parse_csi_frame(pkt)
        assert frame is not None
        assert frame.num_subcarriers == 64
        assert len(frame.amplitude) == 64

    def test_int8_format(self):
        """ESP-IDF native int8 I/Q pairs — the format ESP32 actually sends."""
        pkt = self._make_packet(n_sub=64, dtype=np.int8)
        frame = parse_csi_frame(pkt)
        assert frame is not None
        assert frame.num_subcarriers == 64
        assert len(frame.amplitude) == 64

    def test_int8_values_preserved(self):
        """int8 I/Q values should produce correct amplitudes."""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, 4, 2437, 0, -40, -90, 0,
        )
        # 4 subcarriers: I=3,Q=4 → amplitude=5 for each
        iq = np.array([3, 4, 3, 4, 3, 4, 3, 4], dtype=np.int8)
        pkt = header + iq.tobytes()
        frame = parse_csi_frame(pkt)
        assert frame is not None
        np.testing.assert_allclose(frame.amplitude, [5.0, 5.0, 5.0, 5.0], atol=0.01)


# ═══════════════════════════════════════════════════════════
# SEEDY DISTRICT — edge cases with mixed-chip data
# ═══════════════════════════════════════════════════════════

class TestMixedSubcarriersFusion:
    """Pipeline must handle nodes with different subcarrier counts."""

    def test_fuse_nodes_different_lengths(self):
        """S3 (64 sub) + C3 (52 sub) should not crash."""
        s = Settings()
        s.max_nodes = 2
        s.num_subcarriers = 64
        proc = SignalProcessor(s)

        node_data = {
            1: np.random.rand(64).astype(np.float32),  # S3
            2: np.random.rand(52).astype(np.float32),  # C3 (fewer subcarriers)
        }
        result = proc.fuse_nodes(node_data, target_nodes=2)
        expected_len = 2 * 64  # target_nodes * num_subcarriers
        assert len(result) == expected_len

    def test_fuse_nodes_pads_missing(self):
        """Single node padded to target_nodes width."""
        s = Settings()
        s.max_nodes = 4
        s.num_subcarriers = 56
        proc = SignalProcessor(s)

        node_data = {1: np.ones(56, dtype=np.float32)}
        result = proc.fuse_nodes(node_data, target_nodes=4)
        assert len(result) == 4 * 56
        # First 56 should be 1.0, rest padded with 0
        assert np.all(result[:56] == 1.0)
        assert np.all(result[56:] == 0.0)

    def test_prepare_model_input_mixed_frames(self):
        """Window with frames having different node counts must not crash."""
        s = Settings()
        s.max_nodes = 2
        s.num_subcarriers = 56
        proc = SignalProcessor(s)

        # Frame 1: only node 1
        # Frame 2: node 1 + node 2
        window = [
            {1: np.random.rand(56).astype(np.float32)},
            {1: np.random.rand(56).astype(np.float32),
             2: np.random.rand(56).astype(np.float32)},
        ]
        # This used to crash with "inhomogeneous shape"
        stacked = np.array([
            proc.fuse_nodes(frame, target_nodes=2) for frame in window
        ])
        assert stacked.shape == (2, 2 * 56)


class TestVitalsBufferMixedLengths:
    """Vitals buffer must stay homogeneous even with mixed-chip input."""

    def test_push_different_lengths(self):
        """Pushing vectors of different lengths should not crash update()."""
        ext = VitalSignsExtractor(sample_rate=20.0, window_sec=5.0, num_subcarriers=64)

        # Push 100 frames alternating between 64 and 52 subcarriers
        for i in range(100):
            n_sub = 64 if i % 2 == 0 else 52
            ext.push_csi(np.random.rand(n_sub).astype(np.float32))

        # This used to crash with "inhomogeneous shape"
        result = ext.update()
        assert isinstance(result, dict)
        assert "breathing_bpm" in result

    def test_push_pads_short_vector(self):
        """Short vector should be zero-padded to num_subcarriers."""
        ext = VitalSignsExtractor(num_subcarriers=64)
        ext.push_csi(np.ones(30, dtype=np.float32))
        vec = ext.csi_buffer[0]
        assert len(vec) == 64
        assert np.all(vec[:30] == 1.0)
        assert np.all(vec[30:] == 0.0)

    def test_push_truncates_long_vector(self):
        """Long vector should be truncated to num_subcarriers."""
        ext = VitalSignsExtractor(num_subcarriers=64)
        ext.push_csi(np.ones(128, dtype=np.float32))
        vec = ext.csi_buffer[0]
        assert len(vec) == 64


# ═══════════════════════════════════════════════════════════
# HISTORIC DISTRICT — regression guards for today's fixes
# ═══════════════════════════════════════════════════════════

class TestCSIParserRegression:
    """Ensure the int16 path still works after adding int8 support."""

    def test_original_int16_binary_roundtrip(self):
        """make_csi_binary produces int16 packets that still parse correctly."""
        pkt = make_csi_binary(node_id=3, sequence=99, rssi=-45, n_sub=56)
        frame = parse_csi_frame(pkt)
        assert frame is not None
        assert frame.node_id == 3
        assert frame.sequence == 99
        assert frame.rssi == -45
        assert frame.num_subcarriers == 56

    def test_truncated_int8_packet_rejected(self):
        """Packet too short for even int8 must be rejected."""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, 64, 2437, 0, -50, -90, 0,
        )
        # Only 10 bytes of IQ data (need at least 128 for 64 sub int8)
        pkt = header + bytes(10)
        assert parse_csi_frame(pkt) is None


class TestVitalsServiceRegression:
    """VitalsService must accept num_subcarriers after constructor change."""

    def test_constructor_default(self):
        ext = VitalSignsExtractor()
        assert ext.num_subcarriers == 56

    def test_constructor_custom(self):
        ext = VitalSignsExtractor(num_subcarriers=128)
        assert ext.num_subcarriers == 128

    def test_vitals_service_passes_num_subcarriers(self):
        from server.services.event_emitter import EventEmitter
        from server.services.vitals_service import VitalsService
        svc = VitalsService(
            sample_rate=20.0,
            emitter=EventEmitter(),
            num_subcarriers=64,
        )
        assert svc.extractor.num_subcarriers == 64
