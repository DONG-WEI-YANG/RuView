"""Shared test fixtures for Whittaker tour tests."""
import struct
import pytest
import pytest_asyncio
import numpy as np

from server.config import Settings
from server.csi_frame import CSIFrame, MAGIC_HEADER, HEADER_FORMAT
from server.services.event_emitter import EventEmitter


# ── Settings ──────────────────────────────────────────────

@pytest.fixture
def settings():
    """Default test settings (no model, no simulation)."""
    return Settings()


# ── EventEmitter ──────────────────────────────────────────

@pytest.fixture
def emitter():
    return EventEmitter()


# ── CSI Frame factories ──────────────────────────────────

def make_csi_frame(
    node_id: int = 1,
    sequence: int = 0,
    rssi: int = -50,
    noise_floor: int = -90,
    n_sub: int = 56,
    amplitude: np.ndarray | None = None,
) -> CSIFrame:
    """Build a valid CSIFrame for testing."""
    if amplitude is None:
        amplitude = np.random.rand(n_sub).astype(np.float32) * 100
    return CSIFrame(
        node_id=node_id,
        sequence=sequence,
        timestamp_ms=sequence * 50,
        rssi=rssi,
        noise_floor=noise_floor,
        channel=6,
        bandwidth=20,
        num_subcarriers=n_sub,
        amplitude=amplitude,
        phase=np.zeros(n_sub, dtype=np.float32),
        raw_complex=amplitude.astype(np.complex64),
    )


@pytest.fixture
def csi_frame():
    return make_csi_frame()


def make_csi_binary(
    node_id: int = 1,
    sequence: int = 0,
    rssi: int = -50,
    noise_floor: int = -90,
    n_sub: int = 56,
    n_antennas: int = 1,
    freq_mhz: int = 2437,
    magic: int = MAGIC_HEADER,
) -> bytes:
    """Build a raw ADR-018 binary CSI packet."""
    header = struct.pack(
        HEADER_FORMAT,
        magic, node_id, n_antennas, n_sub, freq_mhz, sequence,
        rssi, noise_floor, 0,
    )
    iq_count = n_antennas * n_sub
    iq_data = np.random.randint(-500, 500, size=iq_count * 2, dtype=np.int16)
    return header + iq_data.tobytes()


# ── FastAPI test client ───────────────────────────────────

@pytest.fixture
def app():
    """Create a FastAPI app without lifespan (no UDP receiver, no simulation)."""
    from server.api import create_app
    return create_app()


@pytest.fixture
def container(app):
    """Access the ServiceContainer wired into the app."""
    return app.state.container


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP test client."""
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
