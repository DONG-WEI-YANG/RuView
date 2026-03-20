"""Notification service — subscribes to alert events."""
from __future__ import annotations

import logging

from server.notifier import Notifier, FallNotification
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self, notifier: Notifier, emitter: EventEmitter):
        self.notifier = notifier
        self._emitter = emitter
        emitter.on("fall_alert", self._on_alert)

    async def _on_alert(self, data: dict) -> None:
        if not self.notifier.enabled:
            return
        notif = FallNotification(
            timestamp=data.get("timestamp", 0),
            confidence=data.get("confidence", 0),
            head_height=data.get("head_height", 0),
            velocity=data.get("velocity", 0),
            alert_id=data.get("alert_id", 0),
        )
        self.notifier.send_fall_alert(notif)

    def close(self) -> None:
        self.notifier.close()
