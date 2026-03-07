# WiFi Body — Pose Estimation via WiFi CSI

Detect human poses, falls, vital signs, and fitness activities using WiFi Channel State Information (CSI) — no cameras needed.

ESP32-S3 nodes capture CSI from ambient WiFi signals and stream them to a Python server that reconstructs 3D body pose in real-time via a lightweight neural network.

```
ESP32 Nodes (CSI capture)  ──UDP──>  Python Server  ──WebSocket──>  Dashboard (3D viewer)
```

## Quick Start

### Option A: Docker (recommended)

```bash
docker compose up --build
# Dashboard: http://localhost:8000/dashboard/index.html
# API: http://localhost:8000/api/status
```

### Option B: Local Python

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python -m server --profile esp32s3

# Dashboard: http://localhost:8000/dashboard/index.html
```

### Option C: With hardware profile selection

```bash
# List available hardware profiles
python -m server --list-profiles

# Start with specific profile
python -m server --profile tplink_n750
```

## Hardware Setup

### Bill of Materials

| Component | Qty | Cost | Purpose |
|-----------|-----|------|---------|
| ESP32-S3-DevKitC-1 | 3-6 | ~$8 each | CSI sensor nodes |
| WiFi router | 1 | (existing) | Signal source |
| Host PC | 1 | (existing) | Python server + dashboard |
| USB cables | 3-6 | ~$2 each | Flashing + power |
| **Total** | | **~$30-60** | |

### Node Placement

Place ESP32 nodes around the room perimeter for best 3D reconstruction:

```
         N1 (1.5m)            Recommended: 4 nodes
        /    \                - Surround the subject
       /      \               - Vary heights (0.5-1.8m)
    N4 ------- N2             - Keep 2-4m from centre
  (1.0m)       (1.0m)        - Point antennas inward
       \      /
        \    /
         N3 (0.5m)
```

See the interactive placement guide in Dashboard > Hardware tab.

### Firmware

Flash ESP32 CSI firmware via the dashboard's built-in Web Flasher, or manually:

```bash
cd firmware/esp32-csi-node
# Configure WiFi in sdkconfig.defaults (see firmware/esp32-csi-node/README.md)
docker run --rm -v "$(pwd):/project" -w /project \
  espressif/idf:v5.2 bash -c "idf.py set-target esp32s3 && idf.py build"
python -m esptool --chip esp32s3 --port /dev/ttyUSB0 --baud 460800 \
  write_flash 0x0 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x10000 build/esp32-csi-node.bin
```

Full firmware docs: [`firmware/esp32-csi-node/README.md`](firmware/esp32-csi-node/README.md)

## Training

### With synthetic data (no hardware needed)

```bash
python -m server.train --synthetic --epochs 50 --batch-size 32
```

### With real data collection

```bash
# 1. Start server with ESP32 nodes connected
python -m server --profile esp32s3

# 2. Open dashboard > Hardware > Data Collection
#    Select activity, click "Start Recording"
#    Or use CLI:
python -m server.real_collector --activity walking --output data/real

# 3. Train on collected data
python -m server.train --data-dir data/real --epochs 100 --profile esp32s3
```

### With public datasets (MM-Fi, Wi-Pose)

```bash
# Download and convert MM-Fi dataset
python -m server.dataset_download --dataset mmfi --raw-dir /path/to/mmfi --output data/mmfi

# Train with matched hardware profile
python -m server.train --data-dir data/mmfi --profile tplink_n750 --epochs 100
```

## Benchmarking

```bash
python -m server.benchmark --profile esp32s3
```

Sample output (CPU, Intel i7):
```
SUMMARY
  Model params:   892,248 (3.4 MB)
  Inference:      2.5 ms (400 FPS)
  Signal proc:    0.8 ms
  End-to-end:     3.3 ms (303 FPS)
  Real-time:      YES (need <50 ms, got 3.3 ms)
```

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Dashboard (Browser)            │
                    │  3D Viewer │ Vitals │ Hardware │ Sensing    │
                    └──────────────────┬──────────────────────────┘
                                       │ WebSocket /ws/pose
                    ┌──────────────────┴──────────────────────────┐
                    │              FastAPI Server                  │
                    │                                             │
                    │  CSIReceiver ─> SignalProcessor ─> PoseModel│
                    │       │              │               │      │
                    │       ▼              ▼               ▼      │
                    │  VitalSigns    BandpassFilter    24 Joints  │
                    │  Extractor     + Normalize       (x,y,z)   │
                    │       │                             │      │
                    │       ▼                             ▼      │
                    │  Breathing/HR              FallDetector    │
                    │  HRV/Stress              FitnessTracker   │
                    └──────────────────┬──────────────────────────┘
                                       │ UDP :5005
                    ┌──────────────────┴──────────────────────────┐
                    │         ESP32-S3 Nodes (x3-6)               │
                    │  WiFi CSI callback ─> ADR-018 ─> UDP send   │
                    └─────────────────────────────────────────────┘
```

## Hardware Profiles

| Profile | Hardware | Subcarriers | Rate | Frequency | Dataset |
|---------|----------|-------------|------|-----------|---------|
| `esp32s3` | ESP32-S3 | 56 | 20 Hz | 2.4 GHz | synthetic |
| `esp32s3_mmfi` | ESP32-S3 | 56 | 20 Hz | 2.4 GHz | MM-Fi |
| `tplink_n750` | TP-Link N750 | 114 | 100 Hz | 5.0 GHz | MM-Fi |
| `intel5300` | Intel 5300 NIC | 30 | 100 Hz | 5.0 GHz | Wi-Pose |
| `esp32c6_wifi6` | ESP32-C6 | 64 | 50 Hz | 2.4 GHz | — |

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/api/status` | GET | Pipeline status, node info, vitals |
| `/api/joints` | GET | Latest 24-joint pose |
| `/api/vitals` | GET | Breathing, heart rate, HRV |
| `/api/profiles` | GET | List hardware profiles |
| `/api/alerts` | GET | Fall detection alerts |
| `/api/collect/start?activity=walking` | POST | Start data collection |
| `/api/collect/stop` | POST | Stop and save recording |
| `/api/collect/status` | GET | Collection progress |
| `/ws/pose` | WebSocket | Real-time pose + vitals stream |

## Testing

```bash
python -m pytest tests/ -v          # All 125 tests
python -m pytest tests/ -k e2e -v   # End-to-end integration only
```

## Project Structure

```
wifi-body/
  server/
    __main__.py         # Entry point: python -m server
    api.py              # FastAPI + WebSocket server
    config.py           # Settings + hardware profiles
    csi_receiver.py     # Async UDP receiver
    csi_frame.py        # ADR-018 binary frame parser
    signal_processor.py # Bandpass filter + normalisation
    pose_model.py       # 1D Conv + Attention neural network
    pipeline.py         # CSI -> model -> joints orchestrator
    train.py            # Training pipeline
    benchmark.py        # Performance benchmarking
    vital_signs.py      # Breathing/HR/HRV from CSI
    fall_detector.py    # Fall detection from joint positions
    fitness_tracker.py  # Activity classification
    data_generator.py   # Synthetic CSI data generator
    dataset.py          # PyTorch dataset + dataloaders
    dataset_download.py # MM-Fi/Wi-Pose download + convert
    real_collector.py   # Real-world CSI + camera data collection
    camera_collector.py # Camera ground truth with rtmlib
  dashboard/
    index.html          # Single-page app (7 tabs)
    styles.css          # Dark theme styles
    skeleton3d.js       # Three.js 3D viewer + WebSocket client
    smpl-mesh.js        # Body mesh renderer
    vitals-hud.js       # Vital signs HUD + CSI visualisation
  firmware/
    esp32-csi-node/     # ESP-IDF firmware for CSI capture
  models/               # Trained model weights
  data/                 # Training data (.npz)
  tests/                # 125 tests (unit + e2e integration)
```

## References

- [MM-Fi Dataset](https://mmfi.github.io/) — NeurIPS 2023, WiFi CSI + 3D pose
- [Wi-Pose](https://github.com/ysc2001/Wi-Pose) — WiFi CSI pose estimation
- [DT-Pose](https://github.com/FanJunqiao/DT-Pose) — Pretrain + fine-tune for WiFi pose
- [ESP32 CSI](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi.html#wi-fi-channel-state-information) — Espressif CSI documentation

## License

MIT
