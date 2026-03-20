"""Vital signs extraction service — wraps VitalSignsExtractor with event emission."""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from server.vital_signs import VitalSignsExtractor, MultiPersonTracker
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class VitalsService:
    def __init__(
        self,
        sample_rate: float,
        emitter: EventEmitter,
        emit_interval_sec: float = 1.0,
    ):
        self._emitter = emitter
        self._emit_interval = emit_interval_sec
        self._last_emit = 0.0
        self.extractor = VitalSignsExtractor(sample_rate=sample_rate)
        self.multi_person = MultiPersonTracker(
            max_persons=4, sample_rate=sample_rate,
        )

    def push_csi(self, amplitudes: np.ndarray) -> None:
        """Push CSI amplitude and emit vitals at throttled rate."""
        self.extractor.push_csi(amplitudes)
        now = time.time()
        if now - self._last_emit >= self._emit_interval:
            self._last_emit = now
            vitals = self.extractor.update()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._emitter.emit("vitals", vitals))
            except RuntimeError:
                pass

    def get_vitals(self) -> dict:
        return self.extractor.update()

    def get_subcarrier_amplitudes(self) -> list[float] | None:
        return self.extractor.get_subcarrier_amplitudes()
