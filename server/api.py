"""FastAPI server with WebSocket for real-time pose streaming."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.config import Settings
from server.csi_receiver import CSIReceiver
from server.signal_processor import SignalProcessor
from server.fall_detector import FallDetector
from server.fitness_tracker import FitnessTracker
from server.vital_signs import VitalSignsExtractor, MultiPersonTracker

logger = logging.getLogger(__name__)

_state = {
    "settings": None,
    "receiver": None,
    "processor": None,
    "fall_detector": None,
    "fitness_tracker": None,
    "latest_joints": None,
    "connected_ws": set(),
    "node_frames": {},
    "vitals": None,
    "multi_person": None,
}


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()
    _state["settings"] = settings
    _state["processor"] = SignalProcessor(settings)
    _state["fall_detector"] = FallDetector(threshold=settings.fall_threshold)
    _state["fitness_tracker"] = FitnessTracker()
    _state["vitals"] = VitalSignsExtractor(sample_rate=settings.csi_sample_rate)
    _state["multi_person"] = MultiPersonTracker(
        max_persons=4, sample_rate=settings.csi_sample_rate
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        receiver = CSIReceiver(settings)
        receiver.on_frame = _on_csi_frame
        _state["receiver"] = receiver
        task = asyncio.create_task(receiver.start())
        logger.info("WiFi Body server started")
        yield
        receiver.stop()

    app = FastAPI(title="WiFi Body", version="0.1.0", lifespan=lifespan)

    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_dir)),
            name="dashboard",
        )

    @app.get("/")
    async def root():
        return {"name": "wifi-body", "version": "0.1.0"}

    @app.get("/api/status")
    async def status():
        fd = _state["fall_detector"]
        ft = _state["fitness_tracker"]
        return {
            "nodes": {
                str(nid): {"last_seq": f.sequence, "rssi": f.rssi}
                for nid, f in _state["node_frames"].items()
            },
            "is_fallen": fd.is_fallen if fd else False,
            "current_activity": ft.current_activity.value if ft else "unknown",
            "fall_alerts": len(fd.get_alerts()) if fd else 0,
            "vitals": _state["vitals"].update() if _state["vitals"] else None,
            "person_count": _state["multi_person"].person_count if _state["multi_person"] else 0,
        }

    @app.get("/api/joints")
    async def get_joints():
        joints = _state["latest_joints"]
        if joints is None:
            return {"joints": None}
        return {"joints": joints.tolist()}

    @app.get("/api/vitals")
    async def get_vitals():
        vs = _state["vitals"]
        mp = _state["multi_person"]
        return {
            "primary": vs.update() if vs else None,
            "csi_amplitudes": vs.get_subcarrier_amplitudes() if vs else None,
            "persons": mp.update_all() if mp else [],
        }

    @app.get("/api/alerts")
    async def get_alerts():
        fd = _state["fall_detector"]
        if fd is None:
            return {"alerts": []}
        return {
            "alerts": [
                {
                    "timestamp": a.timestamp,
                    "confidence": a.confidence,
                    "head_height": a.head_height,
                }
                for a in fd.get_alerts()
            ]
        }

    @app.websocket("/ws/pose")
    async def ws_pose(websocket: WebSocket):
        await websocket.accept()
        _state["connected_ws"].add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            _state["connected_ws"].discard(websocket)

    return app


def _on_csi_frame(frame):
    _state["node_frames"][frame.node_id] = frame
    # Feed CSI amplitude to vital signs extractor
    if _state["vitals"] and hasattr(frame, "amplitudes") and frame.amplitudes is not None:
        import numpy as np
        _state["vitals"].push_csi(np.array(frame.amplitudes, dtype=np.float32))


async def _broadcast_joints(joints):
    payload = {"joints": joints.tolist()}
    vs = _state["vitals"]
    if vs:
        vitals = vs.update()
        payload["vitals"] = vitals
        payload["csi_amplitudes"] = vs.get_subcarrier_amplitudes()
    data = json.dumps(payload)
    dead = set()
    for ws in _state["connected_ws"]:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    _state["connected_ws"] -= dead
