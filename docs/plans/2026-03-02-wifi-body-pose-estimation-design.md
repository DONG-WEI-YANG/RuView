# WiFi Body Pose Estimation System Design

**Date:** 2026-03-02
**Status:** Approved
**Based on:** [wifi-densepose](https://github.com/ruvnet/wifi-densepose) (Fork approach)

## Overview

Build a WiFi-based human body pose estimation system using ESP32-S3 hardware and WiFi CSI (Channel State Information). The system detects human poses, fall events, and fitness activities without cameras — using only WiFi signal analysis.

## Goals

1. **Smart Home / Elderly Care:** Fall detection, presence monitoring, activity tracking
2. **Fitness Tracking:** Motion detection, posture analysis, exercise counting

## Architecture

```
ESP32-S3 x4-6 nodes (WiFi promiscuous mode, CSI capture)
    │
    │ UDP (ADR-018 binary frames, 56 subcarriers, ~20Hz)
    │
    ▼
Python Aggregator (Host PC)
    ├─ CSI Receiver & Parser
    ├─ Signal Processing (bandpass filter, noise removal, multi-node fusion)
    ├─ PyTorch Model (CSI → 24-joint skeleton)
    ├─ Fall Detection
    ├─ Fitness Tracking
    └─ FastAPI Server (REST + WebSocket)
    │
    ▼
Web Dashboard (Browser)
    ├─ 3D Pose Viewer (Three.js)
    ├─ Health Status
    └─ Alert History
```

## Hardware

| Component | Spec |
|-----------|------|
| CSI Nodes | ESP32-S3 x 4-6 |
| WiFi Mode | Promiscuous mode |
| CSI Channels | 56 subcarriers |
| Frame Rate | ~20 Hz per node |
| Output Format | ADR-018 binary frame |
| Transport | UDP to Host PC |
| Build System | ESP-IDF + CMake |

### Node Deployment

Nodes placed at corners/edges of room (~4m x 5m), with WiFi router as signal source in the room.

## Software Stack

| Layer | Technology |
|-------|-----------|
| ESP32 Firmware | C (ESP-IDF), from wifi-densepose |
| Aggregator | Python 3.10+, PyTorch 2.0+ |
| API Server | FastAPI + WebSocket |
| Dashboard | HTML/JS, Three.js, vanilla CSS |
| Data Storage | SQLite (local) |

## Development Strategy

- **Phase 1:** Python prototype (fast iteration, model experimentation)
- **Phase 2:** Rust port for production deployment (performance optimization)

## Project Structure

```
wifi-body/
├── firmware/              # ESP32 firmware (from wifi-densepose)
│   └── esp32-csi-node/
├── server/                # Python Aggregator
│   ├── csi_receiver.py    # UDP receive multi-node CSI data
│   ├── signal_processor.py # Signal preprocessing
│   ├── pose_model.py      # PyTorch 24-joint pose inference
│   ├── fall_detector.py   # Fall detection algorithm
│   ├── fitness_tracker.py # Fitness motion tracking
│   ├── api.py             # FastAPI WebSocket/REST endpoints
│   └── config.py          # Configuration management
├── dashboard/             # Web frontend
│   ├── index.html
│   ├── skeleton3d.js      # Three.js 3D skeleton rendering
│   ├── charts.js          # Real-time charts
│   └── styles.css
├── models/                # Pretrained model weights
├── data/                  # CSI data collection/training
├── docs/
│   └── plans/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Data Flow Pipeline

1. **CSI Capture:** ESP32-S3 nodes capture CSI data in promiscuous mode (56 subcarriers per frame)
2. **UDP Streaming:** Binary frames sent to host PC at ~20Hz per node
3. **Signal Processing:** Hardware normalization → bandpass filter → phase sanitization → multi-node fusion
4. **AI Inference:** PyTorch model maps processed CSI tensor to 24 joint coordinates (x, y, z)
5. **Application Logic:** Fall detection, fitness tracking from skeleton data
6. **Visualization:** WebSocket pushes skeleton data to browser for 3D rendering

## MVP Feature List

| Feature | Description | Priority |
|---------|-------------|----------|
| CSI Data Capture | ESP32 firmware flash, UDP streaming | P0 |
| CSI Data Receiver | Python UDP receiver, binary frame parsing | P0 |
| Signal Preprocessing | Bandpass filter, normalization, multi-node fusion | P0 |
| 24-Joint Pose Inference | PyTorch model load + real-time inference | P0 |
| 3D Skeleton Visualization | Three.js web real-time rendering | P0 |
| Fall Detection | Skeleton-based fall determination + alerts | P1 |
| Fitness Tracking | Motion counting, posture evaluation | P1 |
| Room Calibration | Per-space model fine-tuning | P2 |
| History & Reports | Activity logs, health reports | P2 |

## Performance Targets

| Metric | Target |
|--------|--------|
| Pose Update Rate | ≥10 FPS |
| Fall Detection Latency | <2 seconds |
| Supported People | 1-3 |
| Detection Range | 4m × 5m room |
| Host Minimum Spec | Python 3.10+, 8GB RAM, GPU optional |

## Key Technical Decisions

1. **Fork wifi-densepose** rather than build from scratch — leverage existing tested codebase
2. **Python first** for rapid prototyping, Rust later for production performance
3. **ESP32-S3** as CSI sensor — cheap ($9/unit), well-documented CSI API
4. **Three.js** for 3D visualization — lightweight, no extra dependencies
5. **FastAPI + WebSocket** for real-time data push to dashboard
