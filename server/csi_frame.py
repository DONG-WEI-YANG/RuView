"""ADR-018 binary frame parser for ESP32 CSI data."""
import struct
from dataclasses import dataclass
import numpy as np

MAGIC_HEADER = 0xC5110001

HEADER_FORMAT = "<IBIIQBBBB H"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


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


def parse_csi_frame(data: bytes) -> CSIFrame | None:
    """Parse an ADR-018 binary CSI frame. Returns None if invalid."""
    if len(data) < HEADER_SIZE:
        return None

    header = struct.unpack_from(HEADER_FORMAT, data, 0)
    magic = header[0]
    if magic != MAGIC_HEADER:
        return None

    node_id = header[2]
    sequence = header[3]
    timestamp_ms = header[4]
    rssi_raw = header[5]
    noise_raw = header[6]
    channel = header[7]
    bandwidth = header[8]
    num_sub = header[9]

    rssi = rssi_raw if rssi_raw < 128 else rssi_raw - 256
    noise_floor = noise_raw if noise_raw < 128 else noise_raw - 256

    csi_offset = HEADER_SIZE
    csi_size = num_sub * 4
    if len(data) < csi_offset + csi_size:
        return None

    csi_raw = np.frombuffer(
        data, dtype=np.int16, offset=csi_offset, count=num_sub * 2
    )
    real = csi_raw[0::2].astype(np.float32)
    imag = csi_raw[1::2].astype(np.float32)

    raw_complex = real + 1j * imag
    amplitude = np.abs(raw_complex)
    phase = np.angle(raw_complex)

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
