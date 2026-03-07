"""Integration pipeline: CSI -> signal processing -> model -> applications.

Transparency note (addressing WiFi DensePose criticism):
- When model weights are present: runs real inference on CSI data
- When model weights are missing: logs explicit warning, pipeline receives
  CSI data but cannot produce joint estimates until weights are trained/loaded
- No "mock hardware" mode — CSI receiver is always real UDP input
"""
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
        self.csi_frames_received: int = 0  # track real CSI frames for status

        if self.model is None:
            logger.warning(
                "PosePipeline: no model weights loaded. CSI data will be "
                "received and processed but joint inference is unavailable. "
                "Train a model with `python -m server.train` or provide "
                "weights at %s",
                settings.model_path,
            )

    def on_csi_frame(self, frame: CSIFrame):
        self._current_frame_nodes[frame.node_id] = frame.amplitude
        self.csi_frames_received += 1

    def flush_frame(self):
        if not self._current_frame_nodes:
            return
        self._window.append(dict(self._current_frame_nodes))
        self._current_frame_nodes = {}

        if len(self._window) >= self.window_size and self.model is not None:
            self._run_inference()

    def _run_inference(self):
        try:
            window_list = list(self._window)
            processed = self._prepare_input(window_list)
            tensor = torch.from_numpy(processed).unsqueeze(0)
            with torch.no_grad():
                output = self.model(tensor)
            joints = output.detach().cpu().numpy()[0]
            self.latest_joints = joints
            self.fall_detector.update(joints)
            self.fitness_tracker.update(joints)
        except Exception as e:
            logger.error("Inference error: %s", e)

    def _prepare_input(self, window: list[dict[int, np.ndarray]]) -> np.ndarray:
        """Prepare model input, falling back to normalize-only for short windows."""
        try:
            return self.processor.prepare_model_input(window)
        except ValueError:
            # Window too short for bandpass filter; skip filtering
            stacked = np.array(
                [self.processor.fuse_nodes(frame) for frame in window]
            )
            return self.processor.normalize(stacked)

    @property
    def is_fallen(self) -> bool:
        return self.fall_detector.is_fallen

    @property
    def current_activity(self) -> str:
        return self.fitness_tracker.current_activity.value
