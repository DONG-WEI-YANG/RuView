"""Signal quality monitoring — tracks RSSI, SNR, and CSI stability per node.

Emits 'signal_quality' events so dashboard can show real-time link health
and warn when conditions degrade below usable thresholds.
"""
from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from server.csi_frame import CSIFrame
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

# ── Quality thresholds ──────────────────────────────────────
RSSI_EXCELLENT = -50
RSSI_GOOD = -65
RSSI_POOR = -75

# How it maps to detection capability
CAPABILITY_TABLE = {
    "excellent": {
        "label": "Excellent",
        "color": "#00ff00",
        "capabilities": ["breathing", "heart_rate", "hrv", "pose", "fine_motion"],
    },
    "good": {
        "label": "Good",
        "color": "#88ff00",
        "capabilities": ["breathing", "heart_rate", "pose", "gross_motion"],
    },
    "fair": {
        "label": "Fair",
        "color": "#ffaa00",
        "capabilities": ["breathing", "presence", "gross_motion"],
    },
    "poor": {
        "label": "Poor",
        "color": "#ff4444",
        "capabilities": ["presence"],
    },
}


@dataclass
class NodeQuality:
    """Rolling signal quality metrics for one ESP32 node."""
    node_id: int
    rssi_history: deque = field(default_factory=lambda: deque(maxlen=100))
    noise_history: deque = field(default_factory=lambda: deque(maxlen=100))
    csi_var_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_seen: float = 0.0
    frame_count: int = 0

    @property
    def avg_rssi(self) -> float:
        return float(np.mean(self.rssi_history)) if self.rssi_history else -100.0

    @property
    def avg_noise(self) -> float:
        return float(np.mean(self.noise_history)) if self.noise_history else -90.0

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio in dB."""
        return self.avg_rssi - self.avg_noise

    @property
    def csi_stability(self) -> float:
        """0-1 score of CSI amplitude stability (low variance = stable environment)."""
        if len(self.csi_var_history) < 5:
            return 0.0
        var = float(np.mean(self.csi_var_history))
        # Map: var < 0.01 → 1.0 (very stable), var > 0.5 → 0.0 (chaotic)
        return float(np.clip(1.0 - var * 2.0, 0.0, 1.0))

    @property
    def grade(self) -> str:
        rssi = self.avg_rssi
        if rssi >= RSSI_EXCELLENT:
            return "excellent"
        if rssi >= RSSI_GOOD:
            return "good"
        if rssi >= RSSI_POOR:
            return "fair"
        return "poor"

    def to_dict(self) -> dict:
        g = self.grade
        cap = CAPABILITY_TABLE[g]
        return {
            "node_id": self.node_id,
            "rssi": round(self.avg_rssi, 1),
            "noise_floor": round(self.avg_noise, 1),
            "snr": round(self.snr, 1),
            "csi_stability": round(self.csi_stability, 2),
            "grade": g,
            "grade_label": cap["label"],
            "grade_color": cap["color"],
            "capabilities": cap["capabilities"],
            "frames": self.frame_count,
            "last_seen": self.last_seen,
        }


class SignalQualityMonitor:
    """Monitors per-node signal quality and emits dashboard events."""

    def __init__(self, emitter: EventEmitter, emit_interval: float = 2.0):
        self._emitter = emitter
        self._nodes: dict[int, NodeQuality] = {}
        self._emit_interval = emit_interval
        self._last_emit = 0.0
        self._prev_csi: dict[int, np.ndarray] = {}

    def on_frame(self, frame: CSIFrame) -> None:
        """Feed a CSI frame to update quality metrics."""
        nid = frame.node_id
        if nid not in self._nodes:
            self._nodes[nid] = NodeQuality(node_id=nid)

        nq = self._nodes[nid]
        nq.rssi_history.append(frame.rssi)
        nq.noise_history.append(frame.noise_floor)
        nq.last_seen = time.time()
        nq.frame_count += 1

        # CSI variance (frame-to-frame change as stability measure)
        if frame.amplitude is not None:
            if nid in self._prev_csi and len(self._prev_csi[nid]) == len(frame.amplitude):
                diff = frame.amplitude - self._prev_csi[nid]
                nq.csi_var_history.append(float(np.var(diff)))
            self._prev_csi[nid] = frame.amplitude.copy()

        # Throttled emit
        now = time.time()
        if now - self._last_emit >= self._emit_interval:
            self._last_emit = now
            self._emit()

    def _emit(self) -> None:
        nodes = [nq.to_dict() for nq in self._nodes.values()]
        overall = self._overall_grade(nodes)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._emitter.emit("signal_quality", {
                    "nodes": nodes,
                    "overall_grade": overall["grade"],
                    "overall_label": overall["label"],
                    "overall_capabilities": overall["capabilities"],
                    "environment_tips": overall["tips"],
                }))
        except RuntimeError:
            pass

    def _overall_grade(self, nodes: list[dict]) -> dict:
        """Determine overall system quality from weakest link."""
        if not nodes:
            return {"grade": "poor", "label": "No nodes", "capabilities": [], "tips": ["Connect ESP32 nodes"]}

        grades = [n["grade"] for n in nodes]
        grade_order = ["poor", "fair", "good", "excellent"]
        worst = min(grades, key=lambda g: grade_order.index(g))
        cap = CAPABILITY_TABLE[worst]

        tips = []
        for n in nodes:
            if n["rssi"] < RSSI_POOR:
                tips.append(f"Node {n['node_id']}: signal too weak ({n['rssi']} dBm) — move closer or reduce obstacles")
            elif n["rssi"] < RSSI_GOOD:
                tips.append(f"Node {n['node_id']}: signal fair ({n['rssi']} dBm) — heart rate may be unreliable")
            if n["csi_stability"] < 0.3:
                tips.append(f"Node {n['node_id']}: unstable CSI — check for interference (microwave, Bluetooth)")

        if not tips:
            tips.append("Signal conditions are good for all detection modes")

        return {
            "grade": worst,
            "label": cap["label"],
            "capabilities": cap["capabilities"],
            "tips": tips,
        }

    def get_quality(self) -> dict:
        """Get current quality snapshot (for REST API)."""
        nodes = [nq.to_dict() for nq in self._nodes.values()]
        overall = self._overall_grade(nodes)
        return {"nodes": nodes, **overall}
