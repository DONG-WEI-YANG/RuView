"""CSI receive -> signal processing -> inference -> emit events.

Wraps the existing PosePipeline and adds event emission.
Also owns the simulation loop and fall detector (single source of truth).
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from server.config import Settings
from server.csi_frame import CSIFrame
from server.pipeline import PosePipeline
from server.fall_detector import FallDetector
from server.fitness_tracker import FitnessTracker
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)



# ── Scene mode presets ─────────────────────────────────────
SCENE_MODES = {
    "safety": {
        "description": "Fall detection, apnea monitoring, inactivity alerts",
        "fall_threshold": 0.6,        # more sensitive
        "fall_alert_cooldown": 15,    # faster re-alert
        "inactivity_timeout": 300,    # 5 min no-motion → alert
        "notify_on_fall": True,
        "notify_on_apnea": True,
        "track_reps": False,
    },
    "fitness": {
        "description": "Activity tracking, rep counting, posture assessment",
        "fall_threshold": 0.95,       # very strict (avoid false alarms during exercise)
        "fall_alert_cooldown": 120,   # suppress during workout
        "inactivity_timeout": 0,      # disabled
        "notify_on_fall": False,      # don't spam during burpees
        "notify_on_apnea": False,
        "track_reps": True,
    },
}


# ── Node-count → strategy mapping ──────────────────────────
# Automatically selected based on how many ESP32 nodes are detected.
STRATEGY_TABLE = {
    # (min_nodes, max_nodes): (strategy_name, description)
    (1, 2): ("basic", "Presence detection + vital signs only"),
    (3, 4): ("spatial", "Pose estimation + spatial diversity"),
    (5, 8): ("multiview", "Multi-view fusion + multi-person tracking"),
}


def _strategy_for_nodes(n: int) -> tuple[str, str]:
    for (lo, hi), (name, desc) in STRATEGY_TABLE.items():
        if lo <= n <= hi:
            return name, desc
    return "basic", "Fallback"


class PipelineService:
    def __init__(
        self,
        settings: Settings,
        emitter: EventEmitter,
        pipeline: PosePipeline | None = None,
    ):
        self.settings = settings
        self._emitter = emitter
        self.pipeline = pipeline
        self.fall_detector = FallDetector(
            threshold=settings.fall_threshold,
            cooldown_sec=settings.fall_alert_cooldown,
        )
        self.fitness_tracker = FitnessTracker()
        self.latest_joints: np.ndarray | None = None
        self.csi_frames_received: int = 0
        self._node_frames: dict[int, CSIFrame] = {}
        self._sim_task: asyncio.Task | None = None

        # ── Auto node detection ────────────────────────────
        self._detected_node_ids: set[int] = set()
        self._detection_locked: bool = False
        self._detection_start: float = 0.0
        self._strategy: str = "basic"
        self._strategy_desc: str = "Waiting for nodes..."
        self._active_node_count: int = 0
        self._DETECTION_WINDOW_SEC = 5.0  # lock after 5s of data
        self._quality_monitor = None  # set via set_quality_monitor()

        # ── Scene mode ─────────────────────────────────
        self._scene_mode: str = settings.scene_mode
        self._scene_config: dict = SCENE_MODES.get(self._scene_mode, SCENE_MODES["safety"])
        self._inactivity_start: float = 0.0
        self._apply_scene_mode()

    def _apply_scene_mode(self) -> None:
        """Apply scene-mode presets to fall detector and settings."""
        cfg = self._scene_config
        self.fall_detector.threshold = cfg["fall_threshold"]
        self.fall_detector.cooldown_sec = cfg["fall_alert_cooldown"]
        logger.info("Scene mode: %s — %s", self._scene_mode, cfg["description"])

    @property
    def scene_mode(self) -> str:
        return self._scene_mode

    @property
    def scene_config(self) -> dict:
        return self._scene_config

    def set_scene_mode(self, mode: str) -> dict:
        """Switch between safety/fitness modes. Returns new config."""
        if mode not in SCENE_MODES:
            return {"error": f"Unknown mode. Choose from: {list(SCENE_MODES.keys())}"}
        self._scene_mode = mode
        self._scene_config = SCENE_MODES[mode]
        self.settings.scene_mode = mode
        self._apply_scene_mode()
        return {"status": "switched", "scene_mode": mode, **self._scene_config}

    def get_node_weights(self) -> dict[int, float]:
        """Get signal-quality-based weights for each node (0.0-1.0).

        Used by SignalProcessor.fuse_nodes() to downweight noisy nodes.
        Requires a SignalQualityMonitor reference (set via set_quality_monitor).
        """
        if self._quality_monitor is None:
            return {}
        quality = self._quality_monitor.get_quality()
        weights = {}
        grade_weights = {"excellent": 1.0, "good": 0.85, "fair": 0.5, "poor": 0.1}
        for nq in quality.get("nodes", []):
            weights[nq["node_id"]] = grade_weights.get(nq["grade"], 0.5)
        return weights

    def set_quality_monitor(self, monitor) -> None:
        self._quality_monitor = monitor

    @property
    def detected_nodes(self) -> int:
        return self._active_node_count

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def strategy_description(self) -> str:
        return self._strategy_desc

    def _auto_detect(self, frame: CSIFrame) -> None:
        """Track unique node IDs; lock strategy after detection window."""
        if self._detection_locked:
            # Still track new nodes even after lock
            if frame.node_id not in self._detected_node_ids:
                self._detected_node_ids.add(frame.node_id)
                self._update_strategy()
            return

        now = time.time()
        if self._detection_start == 0.0:
            self._detection_start = now

        self._detected_node_ids.add(frame.node_id)
        self._update_strategy()

        if now - self._detection_start >= self._DETECTION_WINDOW_SEC:
            self._detection_locked = True
            logger.info(
                "Node auto-detection locked: %d nodes → strategy '%s' (%s)",
                self._active_node_count, self._strategy, self._strategy_desc,
            )

    def _update_strategy(self) -> None:
        n = len(self._detected_node_ids)
        self._active_node_count = n
        self._strategy, self._strategy_desc = _strategy_for_nodes(n)
        # Update settings.max_nodes so model input_dim adapts
        if n > 0:
            self.settings.max_nodes = max(self.settings.max_nodes, n)

    def on_frame(self, frame: CSIFrame, trigger_pipeline: bool = True) -> None:
        """Process an incoming CSI frame."""
        self._node_frames[frame.node_id] = frame
        self.csi_frames_received += 1
        self._auto_detect(frame)

        # Feed to pipeline
        if self.pipeline is not None:
            self.pipeline.on_csi_frame(frame)

        # Emit CSI amplitudes for waterfall
        if frame.amplitude is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self._emitter.emit("csi", {"amplitudes": frame.amplitude.tolist()})
                    )
            except RuntimeError:
                pass

        if trigger_pipeline:
            self._flush(frame)

    def _compute_joint_confidence(self) -> list[float]:
        """Estimate per-joint confidence from signal quality.

        Joints closer to nodes with better signal get higher confidence.
        This is a heuristic until the model outputs per-joint scores.

        Joint groups and their primary sensing axis:
          Head/neck (0-3): best from elevated nodes
          Arms (4-9): best from side-facing nodes
          Torso (10-11): all nodes contribute
          Legs (12-23): best from low nodes
        """
        weights = self.get_node_weights()
        if not weights:
            return [0.5] * 24  # no quality data, assume medium

        avg_w = sum(weights.values()) / len(weights) if weights else 0.5
        # Simple: overall quality applies uniformly, slight boost for torso
        conf = []
        for j in range(24):
            base = avg_w
            if 10 <= j <= 11:  # torso — all nodes see this
                base = min(1.0, avg_w * 1.1)
            elif j <= 3:  # head — needs good elevated node
                base = avg_w * 0.95
            elif 12 <= j:  # legs — harder to sense
                base = avg_w * 0.85
            conf.append(round(min(1.0, max(0.0, base)), 2))
        return conf

    def _flush(self, frame: CSIFrame) -> None:
        if self.pipeline is not None:
            self.pipeline.flush_frame()
            if self.pipeline.latest_joints is not None:
                self.latest_joints = self.pipeline.latest_joints
                self.fall_detector = self.pipeline.fall_detector
                self.fitness_tracker = self.pipeline.fitness_tracker
                joint_conf = self._compute_joint_confidence()
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self._emitter.emit("pose", {
                                "joints": self.latest_joints.tolist(),
                                "confidence": sum(joint_conf) / 24,
                                "joint_confidence": joint_conf,
                            })
                        )
                except RuntimeError:
                    pass

    def inject_joints(self, joints: np.ndarray) -> None:
        """Inject ground-truth joints (simulation mode, no model)."""
        self.latest_joints = joints
        self.fall_detector.update(joints)
        self.fitness_tracker.update(joints)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._emitter.emit("pose", {
                        "joints": joints.tolist(),
                        "confidence": 0.0,
                    })
                )
        except RuntimeError:
            pass

    @property
    def node_frames(self) -> dict[int, CSIFrame]:
        return self._node_frames

    async def start_simulation(self) -> None:
        if self._sim_task is None:
            self._sim_task = asyncio.create_task(self._simulation_loop())

    async def stop_simulation(self) -> None:
        if self._sim_task:
            self._sim_task.cancel()
            try:
                await self._sim_task
            except asyncio.CancelledError:
                pass
            self._sim_task = None

    async def _simulation_loop(self) -> None:
        from server.data_generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        activities = ["standing", "walking", "exercising", "sitting", "falling"]
        fs = self.settings.csi_sample_rate
        dt = 1.0 / fs

        while True:
            if not self.settings.simulate:
                await asyncio.sleep(1.0)
                continue

            for activity in activities:
                if not self.settings.simulate:
                    break
                logger.info("Simulating activity: %s", activity)
                try:
                    data = gen.generate_sequence(
                        activity, n_frames=100,
                        n_nodes=self.settings.max_nodes,
                        n_sub=self.settings.num_subcarriers,
                    )
                    csi_batch = data["csi"]
                    joints_batch = data["joints"]
                    n_frames, n_nodes, n_sub = csi_batch.shape

                    for t in range(n_frames):
                        loop_start = asyncio.get_event_loop().time()
                        for node_idx in range(n_nodes):
                            amp = csi_batch[t, node_idx, :]
                            frame = CSIFrame(
                                node_id=node_idx + 1, sequence=t,
                                timestamp_ms=int(t * dt * 1000),
                                rssi=-50 + int(np.random.randint(-5, 5)),
                                noise_floor=-90, channel=6, bandwidth=20,
                                num_subcarriers=n_sub,
                                amplitude=amp.astype(np.float32),
                                phase=np.zeros(n_sub, dtype=np.float32),
                                raw_complex=np.zeros(n_sub, dtype=np.complex64),
                            )
                            is_last = (node_idx == n_nodes - 1)
                            self.on_frame(frame, trigger_pipeline=is_last)

                        if self.pipeline and self.pipeline.model is None:
                            self.inject_joints(joints_batch[t].astype(np.float32))

                        elapsed = asyncio.get_event_loop().time() - loop_start
                        await asyncio.sleep(max(0, dt - elapsed))

                    await asyncio.sleep(1.0)
                except Exception as e:
                    logger.error("Simulation error: %s", e)
                    await asyncio.sleep(5.0)
