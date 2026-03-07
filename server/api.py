"""FastAPI server with WebSocket for real-time pose streaming."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import Settings, HARDWARE_PROFILES, HardwareProfile
from server.csi_receiver import CSIReceiver
from server.signal_processor import SignalProcessor
from server.fall_detector import FallDetector
from server.fitness_tracker import FitnessTracker
from server.pipeline import PosePipeline
from server.vital_signs import VitalSignsExtractor, MultiPersonTracker

logger = logging.getLogger(__name__)

_state = {
    "settings": None,
    "receiver": None,
    "processor": None,
    "pipeline": None,
    "fall_detector": None,
    "fitness_tracker": None,
    "latest_joints": None,
    "connected_ws": set(),
    "node_frames": {},
    "vitals": None,
    "multi_person": None,
    "collector": None,
    "collecting": False,
    "collect_activity": "",
    "collect_frames": 0,
    "collect_saved": [],
}


def _load_pipeline(settings: Settings) -> PosePipeline:
    """Create PosePipeline, loading model weights if available."""
    model = None
    model_path = Path(settings.model_path)
    if model_path.exists():
        try:
            from server.pose_model import load_model
            # Detect input_dim from checkpoint
            import torch
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
            # Infer input_dim from first conv layer weight shape
            first_key = [k for k in ckpt.keys() if "encoder.0.weight" in k]
            if first_key:
                input_dim = ckpt[first_key[0]].shape[1]
            else:
                input_dim = settings.num_subcarriers * settings.max_nodes
            model = load_model(str(model_path), input_dim=input_dim)
            logger.info("Pose model loaded from %s (input_dim=%d)", model_path, input_dim)
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", model_path, e)
    else:
        logger.info("No model weights at %s — pipeline will run without inference", model_path)

    return PosePipeline(settings, model=model)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    # Apply hardware profile (sets num_subcarriers, sample_rate, model_path)
    profile = settings.apply_hardware_profile()
    if profile:
        logger.info(
            "Hardware profile: %s (%s, %d sub, %d Hz, %.1f GHz)",
            profile.id, profile.name, profile.num_subcarriers,
            profile.csi_sample_rate, profile.frequency_ghz,
        )
    _state["settings"] = settings

    _state["processor"] = SignalProcessor(settings)
    _state["fall_detector"] = FallDetector(threshold=settings.fall_threshold)
    _state["fitness_tracker"] = FitnessTracker()
    _state["vitals"] = VitalSignsExtractor(sample_rate=settings.csi_sample_rate)
    _state["multi_person"] = MultiPersonTracker(
        max_persons=4, sample_rate=settings.csi_sample_rate
    )

    # Load pose pipeline (with or without model weights)
    pipeline = _load_pipeline(settings)
    _state["pipeline"] = pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        receiver = CSIReceiver(settings)
        receiver.on_frame = _on_csi_frame
        _state["receiver"] = receiver
        task = asyncio.create_task(receiver.start())
        logger.info(
            "WiFi Body server started (model_loaded=%s)",
            pipeline.model is not None,
        )
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
        # Pipeline transparency: report what's real vs unavailable
        pipeline = _state.get("pipeline")
        has_model = pipeline is not None and pipeline.model is not None if pipeline else False
        csi_frames = pipeline.csi_frames_received if pipeline else 0
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
            # Transparency fields — what's real and what's not
            "pipeline_status": {
                "csi_receiver": "listening",
                "model_loaded": has_model,
                "csi_frames_received": csi_frames,
                "inference_active": has_model and csi_frames > 0,
                "hardware_profile": _state["settings"].hardware_profile,
            },
        }

    @app.get("/api/profiles")
    async def get_profiles():
        """List available hardware profiles and which is active."""
        settings = _state["settings"]
        active_id = settings.hardware_profile if settings else ""
        profiles = []
        for pid, p in HARDWARE_PROFILES.items():
            model_exists = Path(p.model_path).exists()
            profiles.append({
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "active": pid == active_id,
                "num_subcarriers": p.num_subcarriers,
                "max_nodes": p.max_nodes,
                "csi_sample_rate": p.csi_sample_rate,
                "frequency_ghz": p.frequency_ghz,
                "bandwidth_mhz": p.bandwidth_mhz,
                "model_path": p.model_path,
                "model_ready": model_exists,
                "dataset": p.dataset,
            })
        return {"profiles": profiles, "active": active_id}

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

    # ── Data Collection API ─────────────────────────────────
    @app.post("/api/collect/start")
    async def collect_start(activity: str = "standing"):
        """Start collecting CSI + pose data for training."""
        from server.real_collector import RealDataCollector, ACTIVITIES
        if activity not in ACTIVITIES:
            return JSONResponse(
                {"error": f"Unknown activity. Choose from: {ACTIVITIES}"},
                status_code=400,
            )
        if _state["collecting"]:
            return JSONResponse({"error": "Already collecting"}, status_code=409)

        collector = _state.get("collector")
        if collector is None:
            collector = RealDataCollector(
                settings=settings,
                n_nodes=settings.max_nodes,
            )
            # Wire CSI frames to collector
            _state["collector"] = collector
        collector.start_recording(activity)
        _state["collecting"] = True
        _state["collect_activity"] = activity
        _state["collect_frames"] = 0
        return {"status": "recording", "activity": activity}

    @app.post("/api/collect/stop")
    async def collect_stop():
        """Stop collecting and save the .npz file."""
        collector = _state.get("collector")
        if not _state["collecting"] or collector is None:
            return JSONResponse({"error": "Not collecting"}, status_code=409)

        data = collector.stop_recording()
        _state["collecting"] = False
        if data is None:
            return {"status": "stopped", "frames": 0, "file": None}

        path = collector.save_sequence(data, "data/real")
        _state["collect_saved"].append(path)
        return {
            "status": "saved",
            "frames": len(data["joints"]),
            "file": path,
            "activity": _state["collect_activity"],
        }

    @app.get("/api/collect/status")
    async def collect_status():
        """Get current collection status."""
        from server.real_collector import ACTIVITIES
        collector = _state.get("collector")
        csi_count = collector._csi_frame_count if collector else 0
        nodes_seen = len(collector._csi_buffer) if collector else 0
        return {
            "collecting": _state["collecting"],
            "activity": _state["collect_activity"],
            "frames": _state["collect_frames"],
            "csi_frames": csi_count,
            "nodes_seen": nodes_seen,
            "saved_files": _state["collect_saved"],
            "activities": ACTIVITIES,
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

    # Feed CSI to data collector
    collector = _state.get("collector")
    if collector is not None:
        collector.on_csi_frame(frame)

    # Feed CSI amplitude to vital signs extractor
    if _state["vitals"] and hasattr(frame, "amplitudes") and frame.amplitudes is not None:
        import numpy as np
        _state["vitals"].push_csi(np.array(frame.amplitudes, dtype=np.float32))

    # Feed frame to pose pipeline for inference
    pipeline = _state.get("pipeline")
    if pipeline is not None:
        pipeline.on_csi_frame(frame)
        pipeline.flush_frame()
        if pipeline.latest_joints is not None:
            _state["latest_joints"] = pipeline.latest_joints
            # Update fall/fitness from pipeline's own detectors
            _state["fall_detector"] = pipeline.fall_detector
            _state["fitness_tracker"] = pipeline.fitness_tracker
            # Feed joints to data collector if recording
            if _state["collecting"] and _state.get("collector"):
                _state["collector"].add_frame(pipeline.latest_joints)
                _state["collect_frames"] += 1
            # Broadcast to connected WebSocket clients
            asyncio.ensure_future(_broadcast_joints(pipeline.latest_joints))


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
