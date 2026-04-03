"""Spatial calibration for ESP32 node positions and background environment.

Calibration flow:
1. User stands at a known position (centre of room) for Spatial Calibration.
2. Empty room for Background Calibration.
3. System records CSI from all nodes for ~5 seconds.
4. Data used to estimate node distances or static background.

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
    mode: str = "spatial"  # "spatial" or "background"
    start_time: float = 0.0
    duration: float = CALIBRATION_DURATION_SEC
    samples: dict[int, list] = field(default_factory=dict) # CSI amplitudes
    rssi_samples: dict[int, list] = field(default_factory=dict)
    is_active: bool = False
    is_complete: bool = False


class CalibrationManager:
    """Manages the calibration flow.

    Usage:
        mgr = CalibrationManager()
        mgr.start(mode="background")
        # ... feed CSI frames via on_csi_frame() ...
        result = mgr.finish()
    """

    # Log-distance path loss model defaults
    DEFAULT_REF_POWER_DBM = -45   # RSSI at 1 metre (A)
    DEFAULT_PATH_LOSS_EXP = 2.5   # indoor environment (n)

    def __init__(
        self,
        duration: float = CALIBRATION_DURATION_SEC,
        ref_power_dbm: float = DEFAULT_REF_POWER_DBM,
        path_loss_exp: float = DEFAULT_PATH_LOSS_EXP,
    ):
        self.duration = duration
        self.ref_power_dbm = ref_power_dbm
        self.path_loss_exp = path_loss_exp
        self._session: CalibrationSession | None = None
        self._last_result: dict | None = None
        self._background_profile: dict[int, np.ndarray] = {} # node_id -> mean_amp

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

    def start(self, mode: str = "spatial") -> dict:
        """Begin a calibration session.
        
        Args:
            mode: "spatial" (node positions) or "background" (static environment)
        """
        self._session = CalibrationSession(
            mode=mode,
            start_time=time.time(),
            duration=self.duration,
            is_active=True,
        )
        logger.info("%s calibration started (%.0fs duration)", mode.capitalize(), self.duration)
        return {"status": "calibrating", "mode": mode, "duration": self.duration}

    def on_csi_frame(self, frame) -> None:
        """Feed a CSI frame during calibration."""
        if self._session is None or not self._session.is_active:
            return

        # Auto-finish check
        if time.time() - self._session.start_time > self._session.duration:
            self.finish()
            return

        if frame.node_id not in self._session.samples:
            self._session.samples[frame.node_id] = []
            self._session.rssi_samples[frame.node_id] = []
        
        self._session.samples[frame.node_id].append(frame.amplitude)
        self._session.rssi_samples[frame.node_id].append(frame.rssi)

    def finish(self) -> dict:
        """Process collected data and produce calibration result."""
        if self._session is None or not self._session.is_active:
            return {"status": "error", "message": "No active session"}

        self._session.is_active = False
        self._session.is_complete = True
        
        if self._session.mode == "background":
            return self._finish_background()
        else:
            return self._finish_spatial()

    def _finish_spatial(self) -> dict:
        results = {}
        total_samples = 0
        for nid, amps in self._session.samples.items():
            sample_count = len(amps)
            total_samples += sample_count
            if sample_count < MIN_SAMPLES_PER_NODE:
                continue

            amp_arr = np.array(amps)
            rssi_arr = np.array(self._session.rssi_samples[nid])

            # Simple distance estimation from RSSI (Log-distance path loss model)
            # RSSI = -10 * n * log10(d) + A
            # Assuming n=2.5 (indoor), A=-45 (1m ref)
            mean_rssi = np.mean(rssi_arr)
            dist = 10 ** ((self.ref_power_dbm - mean_rssi) / (10 * self.path_loss_exp))

            results[str(nid)] = {
                "rssi": float(mean_rssi),
                "estimated_distance_m": float(dist),
                "variance": float(np.mean(np.var(amp_arr, axis=0))),
                "sample_count": sample_count,
            }

        self._last_result = {
            "status": "complete",
            "mode": "spatial",
            "nodes": results,
            "node_count": len(results),
            "total_samples": total_samples,
        }
        logger.info("Spatial calibration complete: %d nodes", len(results))
        return self._last_result

    def _finish_background(self) -> dict:
        """Compute static background profile (mean amplitude per subcarrier)."""
        profile = {}
        for nid, amps in self._session.samples.items():
            if len(amps) < MIN_SAMPLES_PER_NODE:
                continue
            # Compute mean amplitude vector
            mean_amp = np.mean(np.array(amps), axis=0)
            profile[nid] = mean_amp
            
        self._background_profile = profile
        self._last_result = {
            "status": "complete", 
            "mode": "background", 
            "nodes_calibrated": list(profile.keys())
        }
        logger.info("Background calibration complete: %d nodes", len(profile))
        return self._last_result

    def get_result(self) -> dict | None:
        """Return the last calibration result, or None if not yet calibrated."""
        return self._last_result

    def get_background_profile(self) -> dict[int, np.ndarray]:
        return self._background_profile

    def get_node_positions(self) -> dict:
        """Return node position data from the last spatial calibration.

        Returns all nodes that had any samples, using available data even if
        below the minimum sample threshold.
        """
        if self._last_result and self._last_result.get("mode") == "spatial":
            nodes = self._last_result.get("nodes", {})
            if nodes:
                return nodes
        # Fall back: build positions from session if available
        if self._session and not self._session.is_active and self._session.mode == "spatial":
            positions = {}
            for nid, amps in self._session.samples.items():
                if not amps:
                    continue
                rssi_arr = np.array(self._session.rssi_samples.get(nid, [-50]))
                mean_rssi = float(np.mean(rssi_arr))
                dist = 10 ** ((self.ref_power_dbm - mean_rssi) / (10 * self.path_loss_exp))
                positions[str(nid)] = {
                    "rssi": mean_rssi,
                    "estimated_distance_m": float(dist),
                    "sample_count": len(amps),
                }
            return positions
        return {}

    def get_reference_csi(self) -> dict:
        # Return raw reference samples (simplified)
        return {}

    def get_status(self) -> dict:
        if self.is_active:
            return {
                "status": "calibrating",
                "progress": self.progress,
                "mode": self._session.mode
            }
        return self._last_result or {"status": "idle"}
