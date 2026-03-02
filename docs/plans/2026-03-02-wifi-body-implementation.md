# WiFi Body Pose Estimation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fork wifi-densepose and build a WiFi CSI-based human pose estimation system with ESP32-S3 nodes, Python aggregator, and web dashboard for fall detection and fitness tracking.

**Architecture:** ESP32-S3 nodes capture WiFi CSI data in promiscuous mode and stream binary frames over UDP to a Python aggregator. The aggregator processes signals through bandpass filtering and multi-node fusion, feeds them into a PyTorch model that outputs 24-joint skeleton coordinates, then pushes results via WebSocket to a Three.js 3D dashboard.

**Tech Stack:** ESP-IDF 5.2 (C firmware), Python 3.10+ (PyTorch, FastAPI, SciPy), Three.js (3D viz), SQLite (local storage)

---

## Task 1: Initialize Git Repository and Clone wifi-densepose

**Files:**
- Create: `.gitignore`
- Create: `README.md`

**Step 1: Initialize git repo**

```bash
cd "D:/product/WIFI body"
git init
```

**Step 2: Clone wifi-densepose as reference**

```bash
git clone https://github.com/ruvnet/wifi-densepose.git _upstream
```

This clones into `_upstream/` so we can reference and copy files without mixing repos.

**Step 3: Create .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/

# ESP-IDF
firmware/esp32-csi-node/build/
firmware/esp32-csi-node/sdkconfig
firmware/esp32-csi-node/sdkconfig.old

# Data and Models
data/*.csv
data/*.bin
models/*.pth
models/*.pt
!models/.gitkeep

# IDE
.vscode/
.idea/

# Upstream reference
_upstream/

# OS
.DS_Store
Thumbs.db

# Environment
.env
```

**Step 4: Create README.md**

```markdown
# WiFi Body — Pose Estimation via WiFi CSI

Detect human poses, falls, and fitness activities using WiFi signals — no cameras needed.

Based on [wifi-densepose](https://github.com/ruvnet/wifi-densepose).

## Hardware
- ESP32-S3 x 4-6 (CSI sensor nodes)
- WiFi router (signal source)
- Host PC (Python aggregator)

## Quick Start
See docs/plans/ for design and implementation details.
```

**Step 5: Commit**

```bash
git add .gitignore README.md
git commit -m "chore: initialize wifi-body project"
```

---

## Task 2: Set Up Project Structure and Python Environment

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `server/__init__.py`
- Create: `server/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: directory scaffolding

**Step 1: Create directory structure**

```bash
mkdir -p server tests models data firmware dashboard docs/plans
touch server/__init__.py tests/__init__.py models/.gitkeep data/.gitkeep
```

**Step 2: Write the failing test for config**

`tests/test_config.py`:
```python
from server.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.udp_host == "0.0.0.0"
    assert settings.udp_port == 5005
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.num_subcarriers == 56
    assert settings.csi_sample_rate == 20
    assert settings.num_joints == 24


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("UDP_PORT", "6000")
    settings = Settings()
    assert settings.udp_port == 6000
```

**Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'server.config'`

**Step 4: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "wifi-body"
version = "0.1.0"
description = "WiFi CSI-based human pose estimation"
requires-python = ">=3.10"
license = {text = "MIT"}

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 5: Create requirements.txt**

```
# Core
numpy>=1.24
scipy>=1.10
torch>=2.0
torchvision>=0.15

# API
fastapi>=0.100
uvicorn[standard]>=0.22
websockets>=11.0

# Signal Processing
matplotlib>=3.7
pandas>=2.0

# Config
pydantic>=2.0
pydantic-settings>=2.0

# Database
aiosqlite>=0.19

# Dev
pytest>=7.0
pytest-asyncio>=0.21
httpx>=0.24
```

**Step 6: Implement config module**

`server/config.py`:
```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # UDP receiver
    udp_host: str = "0.0.0.0"
    udp_port: int = 5005

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CSI parameters
    num_subcarriers: int = 56
    csi_sample_rate: int = 20  # Hz per node
    max_nodes: int = 6

    # Model
    num_joints: int = 24
    model_path: str = "models/pose_model.pth"

    # Fall detection
    fall_threshold: float = 0.8
    fall_alert_cooldown: int = 30  # seconds

    class Config:
        env_prefix = ""
        env_file = ".env"
```

**Step 7: Create venv and install deps**

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
pip install -e .
```

**Step 8: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py -v
```
Expected: 2 passed

**Step 9: Commit**

```bash
git add pyproject.toml requirements.txt server/ tests/ models/.gitkeep data/.gitkeep
git commit -m "feat: add project structure, config module with tests"
```

---

## Task 3: Copy ESP32 Firmware from Upstream

**Files:**
- Copy: `_upstream/firmware/esp32-csi-node/` to `firmware/esp32-csi-node/`
- Create: `firmware/README.md`

**Step 1: Copy firmware directory**

```bash
cp -r _upstream/firmware/esp32-csi-node firmware/
```

**Step 2: Create firmware README with our build instructions**

`firmware/README.md`:
```markdown
# ESP32-S3 CSI Node Firmware

Captures WiFi CSI data and streams via UDP to the Python aggregator.

## Prerequisites
- Docker Desktop 28+ (for ESP-IDF build container)
- esptool (pip install esptool)
- CP210x USB-UART driver

## Configuration

Edit esp32-csi-node/sdkconfig.defaults:

| Parameter | Description | Default |
|-----------|-------------|---------|
| CONFIG_CSI_NODE_ID | Unique node ID (0-255) | 1 |
| CONFIG_CSI_WIFI_SSID | Your WiFi network name | -- |
| CONFIG_CSI_WIFI_PASSWORD | Your WiFi password | -- |
| CONFIG_CSI_TARGET_IP | Host PC IP address | 192.168.1.20 |
| CONFIG_CSI_TARGET_PORT | UDP port | 5005 |

## Build

    cd esp32-csi-node
    docker run --rm -v "$(pwd):/project" -w /project \
      espressif/idf:v5.2 bash -c "idf.py set-target esp32s3 && idf.py build"

## Flash

    cd esp32-csi-node/build
    python -m esptool --chip esp32s3 --port COM7 --baud 460800 \
      write_flash --flash_mode dio --flash_freq 80m --flash_size 4MB \
      0x0 bootloader/bootloader.bin \
      0x8000 partition_table/partition-table.bin \
      0x10000 esp32-csi-node.bin

Replace COM7 with your serial port.

## Firewall (Windows)

    netsh advfirewall firewall add rule name="ESP32 CSI" dir=in action=allow protocol=UDP localport=5005
```

**Step 3: Verify firmware files exist**

```bash
ls firmware/esp32-csi-node/main/
```
Expected: `CMakeLists.txt csi_collector.c csi_collector.h main.c nvs_config.c nvs_config.h stream_sender.c stream_sender.h`

**Step 4: Commit**

```bash
git add firmware/
git commit -m "feat: add ESP32-S3 CSI node firmware from wifi-densepose"
```

---

## Task 4: CSI Binary Frame Parser

**Files:**
- Create: `server/csi_frame.py`
- Create: `tests/test_csi_frame.py`

The ESP32 firmware sends CSI data in ADR-018 binary format with magic header 0xC5110001. We need a parser for this format.

**Step 1: Write the failing test**

`tests/test_csi_frame.py`:
```python
import struct
import numpy as np
from server.csi_frame import CSIFrame, parse_csi_frame, MAGIC_HEADER


def _build_fake_frame(node_id=1, seq=42, rssi=-45, num_sub=56):
    """Build a minimal ADR-018 binary frame for testing."""
    header = struct.pack(
        "<IBIIQBBBB H",
        MAGIC_HEADER,   # magic
        1,               # version
        node_id,         # node_id
        seq,             # sequence
        1000,            # timestamp_ms
        rssi & 0xFF,     # rssi (signed as unsigned byte)
        0xD0 & 0xFF,     # noise_floor -48
        6,               # channel
        20,              # bandwidth
        num_sub,         # num_subcarriers
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_csi_frame.py -v
```
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement CSI frame parser**

`server/csi_frame.py`:
```python
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
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_csi_frame.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add server/csi_frame.py tests/test_csi_frame.py
git commit -m "feat: add ADR-018 CSI binary frame parser with tests"
```

---

## Task 5: UDP CSI Receiver (Multi-Node)

**Files:**
- Create: `server/csi_receiver.py`
- Create: `tests/test_csi_receiver.py`

**Step 1: Write the failing test**

`tests/test_csi_receiver.py`:
```python
import asyncio
import struct
import numpy as np
import pytest
from server.csi_receiver import CSIReceiver
from server.csi_frame import MAGIC_HEADER
from server.config import Settings


def _build_fake_frame(node_id=1, seq=0):
    num_sub = 56
    header = struct.pack(
        "<IBIIQBBBB H",
        MAGIC_HEADER, 1, node_id, seq, 1000,
        0xD3, 0xD0, 6, 20, num_sub,
    )
    csi = b""
    for i in range(num_sub):
        csi += struct.pack("<hh", int(100 * np.cos(i)), int(100 * np.sin(i)))
    return header + csi


@pytest.mark.asyncio
async def test_receiver_processes_frames():
    settings = Settings(udp_port=15005)
    receiver = CSIReceiver(settings)
    received = []

    def on_frame(frame):
        received.append(frame)

    receiver.on_frame = on_frame
    task = asyncio.create_task(receiver.start())

    await asyncio.sleep(0.1)

    transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
        asyncio.DatagramProtocol,
        remote_addr=("127.0.0.1", 15005),
    )
    for i in range(3):
        transport.sendto(_build_fake_frame(node_id=1, seq=i))
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.2)
    receiver.stop()
    transport.close()

    assert len(received) == 3
    assert received[0].node_id == 1
    assert received[2].sequence == 2
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_csi_receiver.py -v
```
Expected: FAIL

**Step 3: Implement CSI receiver**

`server/csi_receiver.py`:
```python
"""Async UDP receiver for multi-node ESP32 CSI data."""
import asyncio
import logging
from collections import defaultdict
from typing import Callable

from server.config import Settings
from server.csi_frame import CSIFrame, parse_csi_frame

logger = logging.getLogger(__name__)


class CSIReceiver:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.on_frame: Callable[[CSIFrame], None] | None = None
        self._transport = None
        self._running = False
        self.node_stats: dict[int, int] = defaultdict(int)

    async def start(self):
        loop = asyncio.get_event_loop()
        self._running = True
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _CSIProtocol(self),
            local_addr=(self.settings.udp_host, self.settings.udp_port),
        )
        logger.info(
            "CSI receiver listening on %s:%d",
            self.settings.udp_host,
            self.settings.udp_port,
        )
        while self._running:
            await asyncio.sleep(0.1)

    def stop(self):
        self._running = False
        if self._transport:
            self._transport.close()

    def _handle_data(self, data: bytes):
        frame = parse_csi_frame(data)
        if frame is None:
            return
        self.node_stats[frame.node_id] += 1
        if self.on_frame:
            self.on_frame(frame)


class _CSIProtocol(asyncio.DatagramProtocol):
    def __init__(self, receiver: CSIReceiver):
        self.receiver = receiver

    def datagram_received(self, data: bytes, addr):
        self.receiver._handle_data(data)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_csi_receiver.py -v
```
Expected: 1 passed

**Step 5: Commit**

```bash
git add server/csi_receiver.py tests/test_csi_receiver.py
git commit -m "feat: add async UDP CSI receiver for multi-node data"
```

---

## Task 6: Signal Processing Pipeline

**Files:**
- Create: `server/signal_processor.py`
- Create: `tests/test_signal_processor.py`

**Step 1: Write the failing test**

`tests/test_signal_processor.py`:
```python
import numpy as np
import pytest
from server.signal_processor import SignalProcessor
from server.config import Settings


@pytest.fixture
def processor():
    return SignalProcessor(Settings())


def _fake_amplitudes(n_frames=100, n_sub=56):
    t = np.linspace(0, 5, n_frames)
    base = np.random.randn(n_frames, n_sub) * 0.1
    for s in range(10, 30):
        base[:, s] += np.sin(2 * np.pi * 1.0 * t) * 2.0
    return base


def test_bandpass_filter(processor):
    raw = _fake_amplitudes()
    filtered = processor.bandpass_filter(raw, low=0.5, high=3.0, fs=20)
    assert filtered.shape == raw.shape
    assert np.std(filtered[:, 15]) > 0.5


def test_normalize(processor):
    raw = np.random.randn(50, 56) * 100 + 500
    normed = processor.normalize(raw)
    assert abs(np.mean(normed)) < 0.5
    assert abs(np.std(normed) - 1.0) < 0.5


def test_fuse_nodes(processor):
    node_data = {
        1: np.random.randn(56),
        2: np.random.randn(56),
        3: np.random.randn(56),
    }
    fused = processor.fuse_nodes(node_data)
    assert fused.shape == (56 * 3,)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_signal_processor.py -v
```
Expected: FAIL

**Step 3: Implement signal processor**

`server/signal_processor.py`:
```python
"""CSI signal processing: filtering, normalization, multi-node fusion."""
import numpy as np
from scipy.signal import butter, filtfilt

from server.config import Settings


class SignalProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def bandpass_filter(
        self, data: np.ndarray, low: float, high: float, fs: float, order: int = 4
    ) -> np.ndarray:
        """Apply bandpass Butterworth filter along time axis (axis=0)."""
        nyq = fs / 2.0
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, data, axis=0).astype(np.float32)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize per subcarrier (axis=0)."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return ((data - mean) / std).astype(np.float32)

    def fuse_nodes(self, node_data: dict[int, np.ndarray]) -> np.ndarray:
        """Concatenate amplitude vectors from multiple nodes."""
        arrays = [node_data[nid] for nid in sorted(node_data.keys())]
        return np.concatenate(arrays).astype(np.float32)

    def prepare_model_input(
        self,
        window: list[dict[int, np.ndarray]],
        fs: float | None = None,
    ) -> np.ndarray:
        """Full pipeline: fuse per frame, stack, filter, normalize."""
        if fs is None:
            fs = self.settings.csi_sample_rate

        stacked = np.array([self.fuse_nodes(frame) for frame in window])
        filtered = self.bandpass_filter(stacked, low=0.1, high=8.0, fs=fs)
        return self.normalize(filtered)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_signal_processor.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add server/signal_processor.py tests/test_signal_processor.py
git commit -m "feat: add CSI signal processing (bandpass filter, normalize, node fusion)"
```

---

## Task 7: Pose Estimation Model

**Files:**
- Create: `server/pose_model.py`
- Create: `tests/test_pose_model.py`

Creates the PyTorch model that maps processed CSI data to 24 joint coordinates. Simplified version of wifi-densepose ModalityTranslationNetwork.

**Step 1: Write the failing test**

`tests/test_pose_model.py`:
```python
import torch
import pytest
from server.pose_model import WiFiPoseModel


def test_model_output_shape():
    model = WiFiPoseModel(input_dim=56 * 4, num_joints=24)
    x = torch.randn(1, 60, 56 * 4)
    joints = model(x)
    assert joints.shape == (1, 24, 3)


def test_model_deterministic():
    model = WiFiPoseModel(input_dim=56 * 3, num_joints=24)
    model.set_to_eval_mode()
    x = torch.randn(1, 60, 56 * 3)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.allclose(y1, y2)


def test_model_different_node_counts():
    for n_nodes in [2, 3, 4, 6]:
        model = WiFiPoseModel(input_dim=56 * n_nodes, num_joints=24)
        x = torch.randn(2, 40, 56 * n_nodes)
        y = model(x)
        assert y.shape == (2, 24, 3)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_pose_model.py -v
```
Expected: FAIL

**Step 3: Implement pose model**

`server/pose_model.py`:
```python
"""WiFi CSI to 24-joint pose estimation model."""
import torch
import torch.nn as nn


class WiFiPoseModel(nn.Module):
    """Maps a window of CSI features to 24 body joint coordinates.

    Architecture: 1D Conv encoder -> attention -> FC decoder -> 24 joints x 3 (xyz).
    """

    def __init__(self, input_dim: int, num_joints: int = 24, hidden: int = 256):
        super().__init__()
        self.num_joints = num_joints

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_joints * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time_steps, input_dim)
        Returns:
            joints: (batch, num_joints, 3)
        """
        x = x.transpose(1, 2)
        encoded = self.encoder(x)

        encoded_t = encoded.transpose(1, 2)
        attn_weights = self.attention(encoded_t)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (encoded_t * attn_weights).sum(dim=1)

        out = self.decoder(pooled)
        return out.view(-1, self.num_joints, 3)

    def set_to_eval_mode(self):
        """Switch model to deterministic inference mode."""
        self.eval()


def load_model(path: str, input_dim: int, device: str = "cpu") -> WiFiPoseModel:
    """Load pretrained model from checkpoint."""
    model = WiFiPoseModel(input_dim=input_dim)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.set_to_eval_mode()
    return model
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_pose_model.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add server/pose_model.py tests/test_pose_model.py
git commit -m "feat: add WiFi CSI to pose estimation PyTorch model"
```

---

## Task 8: Fall Detection Module

**Files:**
- Create: `server/fall_detector.py`
- Create: `tests/test_fall_detector.py`

**Step 1: Write the failing test**

`tests/test_fall_detector.py`:
```python
import numpy as np
import pytest
from server.fall_detector import FallDetector


def _make_joints(num_joints=24):
    joints = np.zeros((num_joints, 3))
    for i in range(num_joints):
        joints[i, 1] = 1.7 - (i / num_joints) * 1.7
    return joints


def test_no_fall_standing():
    det = FallDetector(threshold=0.8)
    standing = _make_joints()
    for _ in range(10):
        det.update(standing)
    assert det.is_fallen is False


def test_fall_detected():
    det = FallDetector(threshold=0.8)
    standing = _make_joints()
    for _ in range(5):
        det.update(standing)

    fallen = _make_joints()
    fallen[:, 1] = np.random.uniform(0.0, 0.3, 24)
    for _ in range(5):
        det.update(fallen)

    assert det.is_fallen is True


def test_cooldown():
    det = FallDetector(threshold=0.8, cooldown_sec=0.5)
    standing = _make_joints()
    fallen = _make_joints()
    fallen[:, 1] = 0.1

    for _ in range(5):
        det.update(standing)
    for _ in range(5):
        det.update(fallen)

    alerts = det.get_alerts()
    assert len(alerts) >= 1
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_fall_detector.py -v
```
Expected: FAIL

**Step 3: Implement fall detector**

`server/fall_detector.py`:
```python
"""Skeleton-based fall detection."""
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class FallAlert:
    timestamp: float
    confidence: float
    head_height: float
    velocity: float


class FallDetector:
    def __init__(self, threshold: float = 0.8, cooldown_sec: float = 30.0):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.is_fallen = False
        self._history: list[np.ndarray] = []
        self._alerts: list[FallAlert] = []
        self._last_alert_time = 0.0

    def update(self, joints: np.ndarray):
        """Update with new joint positions. joints shape: (num_joints, 3)."""
        self._history.append(joints.copy())
        if len(self._history) > 30:
            self._history.pop(0)

        if len(self._history) < 3:
            return

        confidence = self._compute_fall_confidence()
        if confidence >= self.threshold:
            self.is_fallen = True
            now = time.time()
            if now - self._last_alert_time > self.cooldown_sec:
                head_h = self._head_height(joints)
                vel = self._vertical_velocity()
                self._alerts.append(FallAlert(now, confidence, head_h, vel))
                self._last_alert_time = now
        else:
            self.is_fallen = False

    def _head_height(self, joints: np.ndarray) -> float:
        return float(joints[0, 1])

    def _vertical_velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        upper_now = np.mean(self._history[-1][:8, 1])
        upper_prev = np.mean(self._history[-2][:8, 1])
        return float(upper_prev - upper_now)

    def _compute_fall_confidence(self) -> float:
        current = self._history[-1]
        y_spread = np.std(current[:, 1])
        spread_score = max(0, 1.0 - y_spread / 0.5)

        head_h = current[0, 1]
        max_h = np.max(current[:, 1])
        height_range = max_h - np.min(current[:, 1])
        low_ratio = 1.0 - (head_h / max(height_range + 0.01, 0.5))

        velocity = self._vertical_velocity()
        vel_score = min(1.0, max(0, velocity / 0.5))

        confidence = 0.4 * low_ratio + 0.35 * spread_score + 0.25 * vel_score
        return float(np.clip(confidence, 0, 1))

    def get_alerts(self) -> list[FallAlert]:
        return list(self._alerts)

    def clear_alerts(self):
        self._alerts.clear()
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_fall_detector.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add server/fall_detector.py tests/test_fall_detector.py
git commit -m "feat: add skeleton-based fall detection with cooldown"
```

---

## Task 9: Fitness Tracker Module

**Files:**
- Create: `server/fitness_tracker.py`
- Create: `tests/test_fitness_tracker.py`

**Step 1: Write the failing test**

`tests/test_fitness_tracker.py`:
```python
import numpy as np
import pytest
from server.fitness_tracker import FitnessTracker, ActivityType


def _standing_pose():
    joints = np.zeros((24, 3))
    for i in range(24):
        joints[i, 1] = 1.7 - (i / 24) * 1.7
    return joints


def _sitting_pose():
    joints = _standing_pose()
    joints[:, 1] *= 0.6
    joints[12:, 1] *= 0.3
    return joints


def test_classify_standing():
    tracker = FitnessTracker()
    pose = _standing_pose()
    for _ in range(10):
        tracker.update(pose)
    assert tracker.current_activity == ActivityType.STANDING


def test_classify_sitting():
    tracker = FitnessTracker()
    pose = _sitting_pose()
    for _ in range(10):
        tracker.update(pose)
    assert tracker.current_activity == ActivityType.SITTING


def test_activity_duration():
    tracker = FitnessTracker()
    pose = _standing_pose()
    for _ in range(20):
        tracker.update(pose)
    stats = tracker.get_stats()
    assert ActivityType.STANDING in stats
    assert stats[ActivityType.STANDING] >= 1
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_fitness_tracker.py -v
```
Expected: FAIL

**Step 3: Implement fitness tracker**

`server/fitness_tracker.py`:
```python
"""Skeleton-based fitness and activity tracking."""
from enum import Enum
from collections import defaultdict
import numpy as np


class ActivityType(Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    EXERCISING = "exercising"


class FitnessTracker:
    def __init__(self):
        self.current_activity = ActivityType.UNKNOWN
        self._history: list[np.ndarray] = []
        self._activity_frames: dict[ActivityType, int] = defaultdict(int)
        self._rep_count = 0

    def update(self, joints: np.ndarray):
        """Update with new joint positions. joints shape: (24, 3)."""
        self._history.append(joints.copy())
        if len(self._history) > 60:
            self._history.pop(0)

        activity = self._classify_activity(joints)
        self.current_activity = activity
        self._activity_frames[activity] += 1

    def _classify_activity(self, joints: np.ndarray) -> ActivityType:
        head_y = joints[0, 1]
        hip_y = np.mean(joints[11:13, 1]) if joints.shape[0] > 12 else joints[len(joints) // 2, 1]
        feet_y = np.mean(joints[-2:, 1])

        total_height = head_y - feet_y
        torso_ratio = (head_y - hip_y) / max(total_height, 0.01)

        if len(self._history) >= 5:
            recent_x = np.array([h[0, 0] for h in self._history[-5:]])
            x_movement = np.std(recent_x)
        else:
            x_movement = 0.0

        if total_height < 0.5:
            return ActivityType.SITTING
        if torso_ratio > 0.45 and x_movement < 0.05:
            return ActivityType.STANDING
        if x_movement > 0.1:
            return ActivityType.WALKING

        return ActivityType.STANDING

    def get_stats(self) -> dict[ActivityType, int]:
        return dict(self._activity_frames)

    def get_rep_count(self) -> int:
        return self._rep_count
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_fitness_tracker.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add server/fitness_tracker.py tests/test_fitness_tracker.py
git commit -m "feat: add skeleton-based fitness tracker with activity classification"
```

---

## Task 10: FastAPI Server with WebSocket

**Files:**
- Create: `server/api.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing test**

`tests/test_api.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from server.api import create_app


@pytest.mark.asyncio
async def test_root_endpoint():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "wifi-body"


@pytest.mark.asyncio
async def test_status_endpoint():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "is_fallen" in data
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_api.py -v
```
Expected: FAIL

**Step 3: Implement FastAPI server**

`server/api.py`:
```python
"""FastAPI server with WebSocket for real-time pose streaming."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.config import Settings
from server.csi_receiver import CSIReceiver
from server.signal_processor import SignalProcessor
from server.fall_detector import FallDetector
from server.fitness_tracker import FitnessTracker

logger = logging.getLogger(__name__)

_state = {
    "settings": None,
    "receiver": None,
    "processor": None,
    "fall_detector": None,
    "fitness_tracker": None,
    "latest_joints": None,
    "connected_ws": set(),
    "node_frames": {},
}


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()
    _state["settings"] = settings
    _state["processor"] = SignalProcessor(settings)
    _state["fall_detector"] = FallDetector(threshold=settings.fall_threshold)
    _state["fitness_tracker"] = FitnessTracker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        receiver = CSIReceiver(settings)
        receiver.on_frame = _on_csi_frame
        _state["receiver"] = receiver
        task = asyncio.create_task(receiver.start())
        logger.info("WiFi Body server started")
        yield
        receiver.stop()

    app = FastAPI(title="WiFi Body", version="0.1.0", lifespan=lifespan)

    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_dir)),
            name="dashboard",
        )

    @app.get("/")
    async def root():
        return {"name": "wifi-body", "version": "0.1.0"}

    @app.get("/api/status")
    async def status():
        fd = _state["fall_detector"]
        ft = _state["fitness_tracker"]
        return {
            "nodes": {
                str(nid): {"last_seq": f.sequence, "rssi": f.rssi}
                for nid, f in _state["node_frames"].items()
            },
            "is_fallen": fd.is_fallen if fd else False,
            "current_activity": ft.current_activity.value if ft else "unknown",
            "fall_alerts": len(fd.get_alerts()) if fd else 0,
        }

    @app.get("/api/joints")
    async def get_joints():
        joints = _state["latest_joints"]
        if joints is None:
            return {"joints": None}
        return {"joints": joints.tolist()}

    @app.get("/api/alerts")
    async def get_alerts():
        fd = _state["fall_detector"]
        if fd is None:
            return {"alerts": []}
        return {
            "alerts": [
                {
                    "timestamp": a.timestamp,
                    "confidence": a.confidence,
                    "head_height": a.head_height,
                }
                for a in fd.get_alerts()
            ]
        }

    @app.websocket("/ws/pose")
    async def ws_pose(websocket: WebSocket):
        await websocket.accept()
        _state["connected_ws"].add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            _state["connected_ws"].discard(websocket)

    return app


def _on_csi_frame(frame):
    _state["node_frames"][frame.node_id] = frame


async def _broadcast_joints(joints):
    data = json.dumps({"joints": joints.tolist()})
    dead = set()
    for ws in _state["connected_ws"]:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    _state["connected_ws"] -= dead
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_api.py -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add server/api.py tests/test_api.py
git commit -m "feat: add FastAPI server with status, joints, alerts, and WebSocket"
```

---

## Task 11: Web Dashboard (3D Skeleton Viewer)

**Files:**
- Create: `dashboard/index.html`
- Create: `dashboard/skeleton3d.js`
- Create: `dashboard/styles.css`

No automated tests — visual verification via browser.

**Step 1: Create dashboard/index.html**

See design doc for full HTML content. Key structure:
- Header with status bar (connection, FPS, node count)
- Main area: Three.js canvas for 3D skeleton
- Side panel: activity card, fall alert card, node list

**Step 2: Create dashboard/skeleton3d.js**

Three.js 3D skeleton renderer with:
- 24 joint spheres connected by bone lines
- WebSocket connection for real-time updates
- Status API polling every 2 seconds
- Auto-reconnect on disconnect

**Step 3: Create dashboard/styles.css**

Dark theme UI with:
- #0f0f1a background, #00ff88 accent color
- Responsive flex layout
- Pulsing red animation for fall alerts

**Step 4: Verify by opening in browser**

```bash
python -m uvicorn server.api:create_app --factory --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/dashboard/index.html

**Step 5: Commit**

```bash
git add dashboard/
git commit -m "feat: add web dashboard with Three.js 3D skeleton viewer"
```

---

## Task 12: Integration Pipeline and Main Entry Point

**Files:**
- Create: `server/pipeline.py`
- Create: `server/__main__.py`
- Create: `tests/test_pipeline.py`

Wires everything together: CSI frames -> signal processing -> model -> fall/fitness -> WebSocket.

**Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import numpy as np
import pytest
from unittest.mock import MagicMock
from server.pipeline import PosePipeline
from server.config import Settings
from server.csi_frame import CSIFrame


def _make_frame(node_id, seq=0):
    return CSIFrame(
        node_id=node_id, sequence=seq, timestamp_ms=1000,
        rssi=-45, noise_floor=-90, channel=6, bandwidth=20,
        num_subcarriers=56,
        amplitude=np.random.randn(56).astype(np.float32),
        phase=np.random.randn(56).astype(np.float32),
        raw_complex=np.random.randn(56).astype(np.complex64),
    )


def test_pipeline_accumulates_frames():
    settings = Settings(max_nodes=3)
    pipeline = PosePipeline(settings, model=None)
    pipeline.on_csi_frame(_make_frame(1))
    pipeline.on_csi_frame(_make_frame(2))
    assert len(pipeline._current_frame_nodes) == 2


def test_pipeline_produces_joints_with_mock_model():
    settings = Settings(max_nodes=2)
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.detach.return_value.cpu.return_value.numpy.return_value = (
        np.random.randn(1, 24, 3).astype(np.float32)
    )
    pipeline = PosePipeline(settings, model=mock_model, window_size=5)

    for i in range(10):
        pipeline.on_csi_frame(_make_frame(1, seq=i))
        pipeline.on_csi_frame(_make_frame(2, seq=i))
        pipeline.flush_frame()

    assert pipeline.latest_joints is not None
    assert pipeline.latest_joints.shape == (24, 3)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_pipeline.py -v
```
Expected: FAIL

**Step 3: Implement pipeline**

`server/pipeline.py`:
```python
"""Integration pipeline: CSI -> signal processing -> model -> applications."""
import logging
from collections import deque

import numpy as np
import torch

from server.config import Settings
from server.csi_frame import CSIFrame
from server.signal_processor import SignalProcessor
from server.fall_detector import FallDetector
from server.fitness_tracker import FitnessTracker

logger = logging.getLogger(__name__)


class PosePipeline:
    def __init__(self, settings: Settings, model=None, window_size: int = 60):
        self.settings = settings
        self.model = model
        self.window_size = window_size
        self.processor = SignalProcessor(settings)
        self.fall_detector = FallDetector(
            threshold=settings.fall_threshold,
            cooldown_sec=settings.fall_alert_cooldown,
        )
        self.fitness_tracker = FitnessTracker()

        self._current_frame_nodes: dict[int, np.ndarray] = {}
        self._window: deque[dict[int, np.ndarray]] = deque(maxlen=window_size)
        self.latest_joints: np.ndarray | None = None

    def on_csi_frame(self, frame: CSIFrame):
        self._current_frame_nodes[frame.node_id] = frame.amplitude

    def flush_frame(self):
        if not self._current_frame_nodes:
            return
        self._window.append(dict(self._current_frame_nodes))
        self._current_frame_nodes = {}

        if len(self._window) >= self.window_size and self.model is not None:
            self._run_inference()

    def _run_inference(self):
        try:
            processed = self.processor.prepare_model_input(list(self._window))
            tensor = torch.from_numpy(processed).unsqueeze(0)
            with torch.no_grad():
                output = self.model(tensor)
            joints = output.detach().cpu().numpy()[0]
            self.latest_joints = joints
            self.fall_detector.update(joints)
            self.fitness_tracker.update(joints)
        except Exception as e:
            logger.error("Inference error: %s", e)

    @property
    def is_fallen(self) -> bool:
        return self.fall_detector.is_fallen

    @property
    def current_activity(self) -> str:
        return self.fitness_tracker.current_activity.value
```

**Step 4: Create main entry point**

`server/__main__.py`:
```python
"""Main entry point: python -m server"""
import logging
import uvicorn
from server.config import Settings
from server.api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def main():
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


if __name__ == "__main__":
    main()
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_pipeline.py -v
```
Expected: 2 passed

**Step 6: Commit**

```bash
git add server/pipeline.py server/__main__.py tests/test_pipeline.py
git commit -m "feat: add integration pipeline and main entry point"
```

---

## Task 13: Run All Tests and Final Verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass (20+ tests across 7 test files)

**Step 2: Verify server starts**

```bash
python -m server
```
Then test with: `curl http://localhost:8000/` and `curl http://localhost:8000/api/status`

**Step 3: Verify dashboard loads**

Open http://localhost:8000/dashboard/index.html in browser.

**Step 4: Final commit**

```bash
git add -A
git commit -m "docs: add implementation plan and finalize project structure"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Git + project init | -- |
| 2 | Python env + config | 2 |
| 3 | ESP32 firmware (copy) | -- |
| 4 | CSI binary frame parser | 3 |
| 5 | UDP CSI receiver | 1 |
| 6 | Signal processing | 3 |
| 7 | Pose estimation model | 3 |
| 8 | Fall detection | 3 |
| 9 | Fitness tracking | 3 |
| 10 | FastAPI + WebSocket | 2 |
| 11 | Web dashboard | visual |
| 12 | Integration pipeline | 2 |
| 13 | Final verification | -- |

**Total: 13 tasks, ~22 automated tests**
