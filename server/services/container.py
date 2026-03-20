"""Dependency injection container — manages service lifecycles."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from server.config import Settings
from server.notifier import Notifier
from server.services.event_emitter import EventEmitter
from server.services.websocket_service import WebSocketService
from server.services.pipeline_service import PipelineService
from server.services.vitals_service import VitalsService
from server.services.calibration_service import CalibrationService
from server.services.storage_service import StorageService
from server.services.notification_service import NotificationService

logger = logging.getLogger(__name__)


def _load_pipeline(settings):
    """Create PosePipeline, loading model weights if available."""
    from server.pipeline import PosePipeline
    model = None
    model_path = Path(settings.model_path)
    if model_path.exists():
        try:
            from server.pose_model import load_model
            import torch
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
            first_key = [k for k in ckpt.keys() if "encoder.0.weight" in k]
            input_dim = ckpt[first_key[0]].shape[1] if first_key else settings.num_subcarriers * settings.max_nodes
            model = load_model(str(model_path), input_dim=input_dim)
            logger.info("Pose model loaded from %s (input_dim=%d)", model_path, input_dim)
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", model_path, e)
    else:
        logger.info("No model weights at %s — pipeline will run without inference", model_path)
    return PosePipeline(settings, model=model)


@dataclass
class ServiceContainer:
    settings: Settings
    emitter: EventEmitter = field(default_factory=EventEmitter)
    websocket: WebSocketService = field(init=False)
    pipeline_svc: PipelineService = field(init=False)
    vitals: VitalsService = field(init=False)
    calibration: CalibrationService = field(init=False)
    storage: StorageService = field(init=False)
    notification: NotificationService = field(init=False)

    def __post_init__(self):
        s = self.settings
        self.websocket = WebSocketService(emitter=self.emitter)
        pipeline = _load_pipeline(s)
        # Wire calibration + vitals into pipeline settings
        self.calibration = CalibrationService()
        self.vitals = VitalsService(sample_rate=s.csi_sample_rate, emitter=self.emitter)
        s.calibration_manager = self.calibration.manager
        s.vitals_extractor = self.vitals.extractor

        self.pipeline_svc = PipelineService(
            settings=s, emitter=self.emitter, pipeline=pipeline,
        )
        self.storage = StorageService(db_path=s.db_path, emitter=self.emitter)
        notifier = Notifier(
            webhook_url=s.notify_webhook_url,
            line_token=s.notify_line_token,
            telegram_bot_token=s.notify_telegram_bot_token,
            telegram_chat_id=s.notify_telegram_chat_id,
        )
        self.notification = NotificationService(notifier=notifier, emitter=self.emitter)

        # Subscribe WebSocket to all streams
        self.emitter.on("pose", self._on_pose_for_ws)
        self.emitter.on("vitals", self._on_vitals_for_ws)
        self.emitter.on("csi", self._on_csi_for_ws)

    async def _on_pose_for_ws(self, data):
        from server.protocol.envelope import make_envelope, PoseData
        env = make_envelope("pose", PoseData(**data))
        await self.websocket.broadcast_envelope(env)

    async def _on_vitals_for_ws(self, data):
        from server.protocol.envelope import make_envelope, VitalsData
        env = make_envelope("vitals", VitalsData(**data))
        await self.websocket.broadcast_envelope(env)

    async def _on_csi_for_ws(self, data):
        from server.protocol.envelope import make_envelope, CsiData
        env = make_envelope("csi", CsiData(**data))
        await self.websocket.broadcast_envelope(env)

    async def startup(self) -> None:
        logger.info("ServiceContainer starting up")
        await self.websocket.start_heartbeat()
        if self.settings.simulate:
            await self.pipeline_svc.start_simulation()

    async def shutdown(self) -> None:
        logger.info("ServiceContainer shutting down")
        await self.pipeline_svc.stop_simulation()
        await self.websocket.stop_heartbeat()
        self.storage.close()
        self.notification.close()
