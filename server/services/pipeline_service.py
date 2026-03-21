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

    def _flush(self, frame: CSIFrame) -> None:
        if self.pipeline is not None:
            self.pipeline.flush_frame()
            if self.pipeline.latest_joints is not None:
                self.latest_joints = self.pipeline.latest_joints
                self.fall_detector = self.pipeline.fall_detector
                self.fitness_tracker = self.pipeline.fitness_tracker
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self._emitter.emit("pose", {
                                "joints": self.latest_joints.tolist(),
                                "confidence": 0.0,
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
