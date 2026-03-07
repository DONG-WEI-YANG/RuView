import struct
import numpy as np
from server.csi_frame import CSIFrame, parse_csi_frame, MAGIC_HEADER


def _build_fake_frame(node_id=1, seq=42, rssi=-45, num_sub=56):
    """Build a minimal ADR-018 binary frame for testing.

    Format: <IBBHIIbbH> (20 bytes) matching firmware/esp32-csi-node csi_collector.c
    """
    header = struct.pack(
        "<IBBHIIbbH",
        MAGIC_HEADER,   # magic (I)
        node_id,         # node_id (B)
        1,               # num_antennas (B)
        num_sub,         # num_subcarriers (H)
        2437,            # freq_mhz (I) -> channel 6
        seq,             # sequence (I)
        rssi,            # rssi (b, signed)
        -48,             # noise_floor (b, signed)
        0,               # reserved (H)
    )
    csi_bytes = b""
    for i in range(num_sub):
        real = int(100 * np.cos(i * 0.1))
        imag = int(100 * np.sin(i * 0.1))
        csi_bytes += struct.pack("<hh", real, imag)
    return header + csi_bytes


def test_parse_valid_frame():
    raw = _build_fake_frame(node_id=3, seq=100, rssi=-50)
    frame = parse_csi_frame(raw)
    assert frame is not None
    assert frame.node_id == 3
    assert frame.sequence == 100
    assert frame.num_subcarriers == 56
    assert frame.amplitude.shape == (56,)
    assert frame.phase.shape == (56,)
    assert np.all(frame.amplitude >= 0)


def test_parse_invalid_magic():
    raw = b"\x00\x00\x00\x00" + b"\x00" * 200
    frame = parse_csi_frame(raw)
    assert frame is None


def test_frame_complex_roundtrip():
    raw = _build_fake_frame()
    frame = parse_csi_frame(raw)
    expected_amp_0 = np.sqrt(100**2 + 0**2)  # cos(0)=1, sin(0)=0
    assert abs(frame.amplitude[0] - expected_amp_0) < 1.0
