"""CSI receive -> signal processing -> inference -> emit events.

Wraps the existing PosePipeline and adds event emission.
Also owns the simulation loop, fall detector (single source of truth),
and multi-person tracking pipeline.
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
from server.vital_signs import VitalSignsExtractor, MultiPersonTracker
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

# ── Person color palette (stable assignment) ──────────────
PERSON_COLORS = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffbe0b']



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

        # ── Multi-person tracking ─────────────────────
        self._multi_person_tracker = MultiPersonTracker(
            max_persons=4, sample_rate=settings.csi_sample_rate,
        )
        self._multi_person_poses: dict[int, dict] = {}  # person_id -> person state
        self._person_color_map: dict[int, str] = {}  # stable color assignment
        self._next_color_idx: int = 0
        self._last_multi_emit: float = 0.0
        self._MULTI_EMIT_INTERVAL: float = 0.5  # emit at ~2 Hz max
        self._multi_person_task: asyncio.Task | None = None

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

        # Multi-person pipeline: activate with 3+ nodes
        if self._active_node_count >= 3:
            self._run_multi_person_pipeline()

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

    # ── Multi-person tracking ─────────────────────────────────

    def _assign_person_color(self, person_id: int) -> str:
        """Assign a stable color to a person ID. Once assigned, never changes."""
        if person_id not in self._person_color_map:
            self._person_color_map[person_id] = PERSON_COLORS[
                self._next_color_idx % len(PERSON_COLORS)
            ]
            self._next_color_idx += 1
        return self._person_color_map[person_id]

    def _estimate_room_position(self, person_idx: int, n_persons: int) -> list[float]:
        """Estimate room position for a person from CSI patterns across nodes.

        Uses centroid offset heuristic: when multiple people are detected,
        distribute them spatially based on which nodes see strongest signal
        for each person's separated CSI component.
        """
        if n_persons <= 1:
            return [0.0, 0.0]

        # Distribute persons in a circle around room center
        angle = (2.0 * np.pi * person_idx) / n_persons
        radius = 0.8  # meters from center
        x = float(radius * np.cos(angle))
        z = float(radius * np.sin(angle))

        # Refine with node signal strengths if available
        node_frames = list(self._node_frames.values())
        if len(node_frames) >= 3 and person_idx < len(node_frames):
            # Use RSSI gradient across nodes as position hint
            rssi_vals = np.array([f.rssi for f in node_frames], dtype=np.float32)
            rssi_norm = rssi_vals - rssi_vals.min()
            total = rssi_norm.sum()
            if total > 0:
                rssi_norm /= total
                # Weighted offset toward strongest-signal node
                node_positions = self.settings.node_positions
                if node_positions:
                    pos_arr = list(node_positions.values())
                    if len(pos_arr) >= len(rssi_norm):
                        wx = sum(rssi_norm[i] * pos_arr[i][0] for i in range(len(rssi_norm)))
                        wz = sum(rssi_norm[i] * pos_arr[i][2] for i in range(len(rssi_norm)))
                        # Blend heuristic position with RSSI-weighted centroid
                        x = float(x * 0.5 + (wx - self.settings.room_width / 2) * 0.5)
                        z = float(z * 0.5 + (wz - self.settings.room_depth / 2) * 0.5)

        return [round(x, 3), round(z, 3)]

    def _run_multi_person_pipeline(self) -> None:
        """Run multi-person detection and per-person pose/vitals.

        Called from on_frame when 3+ nodes are detected.
        Throttled to emit at 1-2 Hz.
        """
        now = time.time()
        if now - self._last_multi_emit < self._MULTI_EMIT_INTERVAL:
            return
        self._last_multi_emit = now

        node_frames = list(self._node_frames.values())
        n_nodes = len(node_frames)
        if n_nodes < 3:
            return

        # Build multi-antenna data from node frames
        antenna_data: dict[str, np.ndarray] = {}
        for frame in node_frames:
            key = f"TX1_RX{frame.node_id}"
            if frame.amplitude is not None:
                antenna_data[key] = frame.amplitude

        if not antenna_data:
            return

        # Feed to multi-person tracker (counts people, separates signals)
        self._multi_person_tracker.push_multi_antenna_csi(antenna_data)

        # Update vitals for all tracked persons
        person_results = self._multi_person_tracker.update_all()
        n_persons = self._multi_person_tracker.person_count

        if n_persons <= 0:
            return

        # Build per-person data with poses, vitals, and positions
        persons_data = []
        for i, pdata in enumerate(person_results):
            person_id = pdata["person_id"]
            color = self._assign_person_color(person_id)
            vitals = pdata["vitals"]

            # Generate per-person pose (offset from base pose if available)
            joints = self._generate_person_pose(i, n_persons)
            joint_conf = self._compute_joint_confidence()
            position = self._estimate_room_position(i, n_persons)

            person_state = {
                "id": person_id,
                "joints": joints,
                "confidence": sum(joint_conf) / 24 if joint_conf else 0.5,
                "joint_confidence": joint_conf,
                "vitals": vitals,
                "position": position,
                "color": color,
            }
            self._multi_person_poses[person_id] = person_state
            persons_data.append(person_state)

        # Remove stale person entries
        active_ids = {p["id"] for p in persons_data}
        stale = [pid for pid in self._multi_person_poses if pid not in active_ids]
        for pid in stale:
            del self._multi_person_poses[pid]

        # Emit persons event
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._emitter.emit("persons", {
                        "persons": persons_data,
                        "count": n_persons,
                    })
                )
        except RuntimeError:
            pass

    def _generate_person_pose(self, person_idx: int, n_persons: int) -> list[list[float]]:
        """Generate pose joints for a specific person.

        If the pipeline has a base pose (from model or simulation), offset it
        spatially for each person. Otherwise, generate a default T-pose with
        spatial offset.
        """
        # Default T-pose rest positions (24 joints)
        REST = [
            [0.0, 1.7, 0.0], [0.0, 1.55, 0.0], [0.0, 1.38, 0.0], [0.0, 1.12, 0.0],
            [-0.2, 1.4, 0.0], [-0.48, 1.4, 0.0], [-0.7, 1.4, 0.0],
            [0.2, 1.4, 0.0], [0.48, 1.4, 0.0], [0.7, 1.4, 0.0],
            [0.0, 0.95, 0.0], [0.0, 0.9, 0.0],
            [-0.1, 0.88, 0.0], [-0.1, 0.5, 0.0], [-0.1, 0.08, 0.0],
            [0.1, 0.88, 0.0], [0.1, 0.5, 0.0], [0.1, 0.08, 0.0],
            [-0.1, 0.03, 0.08], [0.1, 0.03, 0.08],
            [-0.78, 1.4, 0.0], [0.78, 1.4, 0.0],
            [-0.03, 1.72, 0.06], [0.03, 1.72, 0.06],
        ]

        # Use the pipeline base pose if available
        base_joints = None
        if self.latest_joints is not None:
            base_joints = self.latest_joints.tolist()
        elif self.pipeline and self.pipeline.latest_joints is not None:
            base_joints = self.pipeline.latest_joints.tolist()

        if base_joints is None:
            base_joints = REST

        # Spatial offset: distribute persons around room center
        position = self._estimate_room_position(person_idx, n_persons)
        offset_x = position[0]
        offset_z = position[1]

        # Add subtle variation per person (different arm/leg angles)
        t = time.time()
        phase = person_idx * 1.3  # different phase per person
        arm_swing = np.sin(t * 0.6 + phase) * 0.03
        sway = np.sin(t * 0.4 + phase) * 0.005

        joints = []
        for j, base in enumerate(base_joints):
            jx = base[0] + offset_x + sway
            jy = base[1]
            jz = base[2] + offset_z
            joints.append([round(jx, 4), round(jy, 4), round(jz, 4)])

        # Apply arm swing variation
        if len(joints) >= 10:
            joints[5][1] += arm_swing
            joints[6][1] += arm_swing * 1.4
            joints[8][1] -= arm_swing
            joints[9][1] -= arm_swing * 1.4

        return joints

    @property
    def person_count(self) -> int:
        """Number of currently tracked persons."""
        return len(self._multi_person_poses)

    def get_persons_snapshot(self) -> list[dict]:
        """Get a snapshot of all tracked persons for the REST API."""
        return list(self._multi_person_poses.values())

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
