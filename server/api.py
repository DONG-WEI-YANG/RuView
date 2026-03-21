"""FastAPI server -- slim app shell with router mounts and service container."""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import Settings
from server.csi_receiver import CSIReceiver
from server.services.container import ServiceContainer, _load_pipeline  # noqa: F401 — re-export for backward compat
from server.routes import system, calibration, data, ws

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    profile = settings.apply_hardware_profile()
    if profile:
        logger.info(
            "Hardware profile: %s (%s, %d sub, %d Hz, %.1f GHz)",
            profile.id, profile.name, profile.num_subcarriers,
            profile.csi_sample_rate, profile.frequency_ghz,
        )

    container = ServiceContainer(settings=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start CSI receiver
        receiver = CSIReceiver(settings)

        def on_csi(frame, trigger_pipeline=True):
            if container.calibration.is_active:
                container.calibration.on_frame(frame)
            if frame.amplitude is not None:
                import numpy as np
                container.vitals.push_csi(frame.amplitude.astype(np.float32))
            container.pipeline_svc.on_frame(frame, trigger_pipeline=trigger_pipeline)

        receiver.on_frame = on_csi
        recv_task = asyncio.create_task(receiver.start())

        await container.startup()

        logger.info(
            "WiFi Body server started (model_loaded=%s, simulate=%s)",
            container.pipeline_svc.pipeline is not None
            and container.pipeline_svc.pipeline.model is not None,
            settings.simulate,
        )
        yield

        receiver.stop()
        await container.shutdown()

    app = FastAPI(title="WiFi Body", version="0.2.0", lifespan=lifespan)

    # Make container available immediately (tests may skip lifespan)
    app.state.container = container

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    # Mount routes
    app.include_router(system.router)
    app.include_router(calibration.router)
    app.include_router(data.router)
    app.include_router(ws.router)

    # Static files — prefer Vite build output, fall back to source
    dist_dir = Path(__file__).parent.parent / "dist" / "dashboard"
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    static_dir = dist_dir if dist_dir.exists() else dashboard_dir
    if static_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(static_dir)), name="dashboard")

    return app
