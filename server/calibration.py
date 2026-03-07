"""Spatial calibration for ESP32 node positions.

Calibration flow:
1. User stands at a known position (centre of room)
2. System records CSI from all nodes for ~5 seconds
3. RSSI + amplitude variance used to estimate relative node distances
4. Node positions saved for signal-processing normalisation

This replaces guesswork with measured reference data.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_DURATION_SEC = 5.0
MIN_SAMPLES_PER_NODE = 20


@dataclass
class NodeCalibration:
    """Calibration data for one ESP32 node."""
    node_id: int
    mean_rssi: float = 0.0
    mean_amplitude: float = 0.0
    std_amplitude: float = 0.0
    estimated_distance_m: float = 0.0
    sample_count: int = 0


@dataclass
class CalibrationSession:
    """Active calibration session accumulator."""
    start_time: float = 0.0
    duration: float = CALIBRATION_DURATION_SEC
    samples: dict[int, list] = field(default_factory=dict)
    rssi_samples: dict[int, list] = field(default_factory=dict)
    is_active: bool = False
    is_complete: bool = False


class CalibrationManager:
    """Manages the calibration flow.

    Usage:
        mgr = CalibrationManager()
        mgr.start()
        # ... feed CSI frames via on_csi_frame() ...
        result = mgr.finish()  # or auto-finishes after duration
    """

    def __init__(self, duration: float = CALIBRATION_DURATION_SEC):
        self.duration = duration
        self._session: CalibrationSession | None = None
        self._last_result: dict | None = None

    @property
    def is_active(self) -> bool:
        if self._session is None:
            return False
        if not self._session.is_active:
            return False
        # Auto-finish after duration
        if time.time() - self._session.start_time > self._session.duration:
            self.finish()
            return False
        return True

    @property
    def progress(self) -> float:
        if self._session is None or not self._session.is_active:
            return 0.0
        elapsed = time.time() - self._session.start_time
        return min(1.0, elapsed / self._session.duration)

    def start(self) -> dict:
        """Begin a calibration session."""
        self._session = CalibrationSession(
            start_time=time.time(),
            duration=self.duration,
            is_active=True,
        )
        logger.info("Calibration started (%.0fs duration)", self.duration)
        return {"status": "calibrating", "duration": self.duration}

    def on_csi_frame(self, frame) -> None:
        """Feed a CSI frame during calibration."""
        if self._session is None or not self._session.is_active:
            return

        # Auto-finish check
        if time.time() - self._session.start_time > self._session.duration:
            self.finish()
            return

        nid = frame.node_id
        if nid not in self._session.samples:
            self._session.samples[nid] = []
            self._session.rssi_samples[nid] = []

        if frame.amplitude is not None:
            self._session.samples[nid].append(
                np.mean(frame.amplitude).item()
            )
        self._session.rssi_samples[nid].append(frame.rssi)

    def finish(self) -> dict:
        """Complete calibration and compute node distances."""
        if self._session is None:
            return {"status": "error", "message": "No active session"}

        self._session.is_active = False
        self._session.is_complete = True

        nodes = {}
        for nid, amps in self._session.samples.items():
            rssi_list = self._session.rssi_samples.get(nid, [])
            if len(amps) < 3:
                continue

            mean_amp = float(np.mean(amps))
            std_amp = float(np.std(amps))
            mean_rssi = float(np.mean(rssi_list)) if rssi_list else -50.0

            # Estimate distance from RSSI using log-distance path loss model
            # RSSI = -10 * n * log10(d) + A
            # where n=2.5 (indoor), A=-30 (RSSI at 1m reference)
            rssi_ref = -30.0  # typical RSSI at 1 metre
            path_loss_exp = 2.5
            distance = 10 ** ((rssi_ref - mean_rssi) / (10 * path_loss_exp))
            distance = max(0.3, min(distance, 10.0))  # clamp to reasonable range

            nodes[str(nid)] = {
                "node_id": nid,
                "mean_rssi": round(mean_rssi, 1),
                "mean_amplitude": round(mean_amp, 4),
                "std_amplitude": round(std_amp, 4),
                "estimated_distance_m": round(distance, 2),
                "sample_count": len(amps),
            }

        total_samples = sum(len(a) for a in self._session.samples.values())
        result = {
            "status": "complete",
            "nodes": nodes,
            "total_samples": total_samples,
            "node_count": len(nodes),
            "duration_actual": round(
                time.time() - self._session.start_time, 1
            ),
        }
        self._last_result = result
        logger.info(
            "Calibration complete: %d nodes, %d samples",
            len(nodes), total_samples,
        )
        return result

    def get_result(self) -> dict | None:
        return self._last_result

    def get_node_positions(self) -> dict:
        """Return node position estimates (for storage)."""
        if self._last_result is None:
            return {}
        return self._last_result.get("nodes", {})

    def get_reference_csi(self) -> dict:
        """Return reference CSI stats per node (for storage)."""
        if self._last_result is None:
            return {}
        result = {}
        for nid_str, data in self._last_result.get("nodes", {}).items():
            result[nid_str] = {
                "mean_amplitude": data["mean_amplitude"],
                "std_amplitude": data["std_amplitude"],
            }
        return result

    def get_status(self) -> dict:
        """Current calibration status for API."""
        if self._session is None:
            has_cal = self._last_result is not None
            return {
                "status": "calibrated" if has_cal else "uncalibrated",
                "progress": 0,
                "last_result": self._last_result,
            }
        if self._session.is_active:
            nodes_seen = len(self._session.samples)
            total = sum(len(a) for a in self._session.samples.values())
            return {
                "status": "calibrating",
                "progress": round(self.progress, 2),
                "nodes_seen": nodes_seen,
                "samples_collected": total,
            }
        return {
            "status": "complete",
            "progress": 1.0,
            "last_result": self._last_result,
        }
