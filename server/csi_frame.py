"""ADR-018 binary frame parser for ESP32 CSI data.

Binary Frame Format (ADR-018) — matches firmware/esp32-csi-node/main/csi_collector.c:

    Offset  Size  Field
    0       4     Magic: 0xC5110001 (LE u32)
    4       1     Node ID (u8)
    5       1     Number of antennas (u8)
    6       2     Number of subcarriers (LE u16)
    8       4     Frequency MHz (LE u32)
    12      4     Sequence number (LE u32)
    16      1     RSSI (i8)
    17      1     Noise floor (i8)
    18      2     Reserved
    20+     N*2   I/Q pairs (int16 interleaved: real, imag, real, imag, ...)

Total header: 20 bytes.  N = num_antennas * num_subcarriers.
"""
import struct
from dataclasses import dataclass
import numpy as np

MAGIC_HEADER = 0xC5110001

# ADR-018: magic(I) node_id(B) antennas(B) num_sub(H) freq_mhz(I) seq(I) rssi(b signed) noise(b signed) reserved(H)
HEADER_FORMAT = "<IBBHIIbbH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 20 bytes


@dataclass
class CSIFrame:
    node_id: int
    sequence: int
    timestamp_ms: int
    rssi: int
    noise_floor: int
    channel: int
    bandwidth: int
    num_subcarriers: int
    amplitude: np.ndarray
    phase: np.ndarray
    raw_complex: np.ndarray


def _freq_to_channel(freq_mhz: int) -> int:
    """Convert frequency in MHz to WiFi channel number."""
    if 2412 <= freq_mhz <= 2472:
        return 1 + (freq_mhz - 2412) // 5
    if freq_mhz == 2484:
        return 14
    if 5170 <= freq_mhz <= 5885:
        return (freq_mhz - 5000) // 5
    return 0


def parse_csi_frame(data: bytes) -> CSIFrame | None:
    """Parse an ADR-018 binary CSI frame. Returns None if invalid."""
    if len(data) < HEADER_SIZE:
        return None

    header = struct.unpack_from(HEADER_FORMAT, data, 0)
    magic = header[0]
    if magic != MAGIC_HEADER:
        return None

    node_id = header[1]
    num_antennas = header[2]
    num_sub = header[3]
    freq_mhz = header[4]
    sequence = header[5]
    rssi = header[6]        # signed via 'b' format
    noise_floor = header[7]  # signed via 'b' format
    # header[8] is reserved

    channel = _freq_to_channel(freq_mhz)
    bandwidth = 20  # default; ESP32 CSI is always 20MHz per channel

    # I/Q data starts at offset 20
    iq_count = num_antennas * num_sub
    csi_offset = HEADER_SIZE
    remaining = len(data) - csi_offset

    # ESP-IDF CSI uses int8 I/Q pairs (2 bytes per subcarrier).
    # Detect format by checking available data length.
    bytes_int16 = iq_count * 4  # int16 pairs: 4 bytes/subcarrier
    bytes_int8 = iq_count * 2   # int8 pairs:  2 bytes/subcarrier

    if remaining >= bytes_int16:
        csi_raw = np.frombuffer(
            data, dtype=np.int16, offset=csi_offset, count=iq_count * 2
        )
    elif remaining >= bytes_int8:
        csi_raw = np.frombuffer(
            data, dtype=np.int8, offset=csi_offset, count=iq_count * 2
        ).astype(np.int16)
    else:
        return None

    real = csi_raw[0::2].astype(np.float32)
    imag = csi_raw[1::2].astype(np.float32)

    raw_complex = real + 1j * imag
    amplitude = np.abs(raw_complex)
    phase = np.angle(raw_complex)

    # Derive a pseudo-timestamp from sequence (50ms per frame at 20Hz)
    timestamp_ms = sequence * 50

    return CSIFrame(
        node_id=node_id,
        sequence=sequence,
        timestamp_ms=timestamp_ms,
        rssi=rssi,
        noise_floor=noise_floor,
        channel=channel,
        bandwidth=bandwidth,
        num_subcarriers=num_sub,
        amplitude=amplitude,
        phase=phase,
        raw_complex=raw_complex,
    )
