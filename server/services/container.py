"""Dependency injection container — manages service lifecycles.

FastAPI lifespan creates this, stores in app.state.container.
Routes access services via Depends(get_container).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.websocket_service import WebSocketService

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    settings: Settings
    emitter: EventEmitter = field(default_factory=EventEmitter)
    websocket: WebSocketService = field(init=False)

    def __post_init__(self):
        self.websocket = WebSocketService(
            emitter=self.emitter,
            server_version="0.2.0",
        )

    async def startup(self) -> None:
        logger.info("ServiceContainer starting up")
        await self.websocket.start_heartbeat()

    async def shutdown(self) -> None:
        logger.info("ServiceContainer shutting down")
        await self.websocket.stop_heartbeat()
