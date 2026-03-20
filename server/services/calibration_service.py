"""Calibration service — wraps CalibrationManager."""
from __future__ import annotations

from server.calibration import CalibrationManager
from server.csi_frame import CSIFrame


class CalibrationService:
    def __init__(self):
        self.manager = CalibrationManager()

    @property
    def is_active(self) -> bool:
        return self.manager.is_active

    def start(self, mode: str = "spatial") -> dict:
        return self.manager.start(mode=mode)

    def finish(self) -> dict:
        return self.manager.finish()

    def on_frame(self, frame: CSIFrame) -> None:
        if self.manager.is_active:
            self.manager.on_csi_frame(frame)

    def get_status(self) -> dict:
        return self.manager.get_status()

    def get_node_positions(self) -> dict:
        return self.manager.get_node_positions()

    def get_reference_csi(self) -> dict:
        return self.manager.get_reference_csi()

    def get_background_profile(self):
        return self.manager.get_background_profile()
