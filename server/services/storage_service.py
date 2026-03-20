"""Storage service — subscribes to events and persists data (throttled)."""
from __future__ import annotations

import logging
import time

import numpy as np

from server.storage import Storage
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

POSE_SAVE_INTERVAL = 1.0    # seconds
VITALS_SAVE_INTERVAL = 5.0  # seconds


class StorageService:
    def __init__(self, db_path: str, emitter: EventEmitter):
        self.storage = Storage(db_path)
        self._emitter = emitter
        self._last_pose_save = 0.0
        self._last_vitals_save = 0.0
        # Subscribe to events
        emitter.on("pose", self._on_pose)
        emitter.on("vitals", self._on_vitals)

    async def _on_pose(self, data: dict) -> None:
        now = time.time()
        if now - self._last_pose_save < POSE_SAVE_INTERVAL:
            return
        self._last_pose_save = now
        joints = np.array(data["joints"], dtype=np.float32)
        self.storage.save_pose(joints)

    async def _on_vitals(self, data: dict) -> None:
        now = time.time()
        if now - self._last_vitals_save < VITALS_SAVE_INTERVAL:
            return
        self._last_vitals_save = now
        self.storage.save_vitals(data)

    def close(self) -> None:
        self.storage.close()
