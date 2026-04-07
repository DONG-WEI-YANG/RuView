"""Data routes: history, collection, OTA, notifications, training, models."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, FileResponse

from server.services.container import ServiceContainer
from server.notifier import FallNotification

router = APIRouter()


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


# -- History ---------------------------------------------------------
@router.get("/api/history/poses")
async def history_poses(limit: int = 100, container: ServiceContainer = Depends(get_container)):
    return {"poses": container.storage.storage.get_recent_poses(limit)}


@router.get("/api/history/vitals")
async def history_vitals(limit: int = 100, container: ServiceContainer = Depends(get_container)):
    return {"vitals": container.storage.storage.get_recent_vitals(limit)}


@router.get("/api/history/alerts")
async def history_alerts(limit: int = 50, container: ServiceContainer = Depends(get_container)):
    return {"alerts": container.storage.storage.get_fall_alerts(limit)}


# -- Notifications ---------------------------------------------------
@router.get("/api/notifications/status")
async def notification_status(container: ServiceContainer = Depends(get_container)):
    n = container.notification.notifier
    return {"enabled": n.enabled, "channels": n._channels}


@router.post("/api/notifications/test")
async def notification_test(container: ServiceContainer = Depends(get_container)):
    import time as _time
    n = container.notification.notifier
    if not n.enabled:
        return JSONResponse({"error": "No notification channels configured"}, status_code=400)
    results = n.send_fall_alert(FallNotification(
        timestamp=_time.time(), confidence=0.95,
        head_height=0.3, velocity=1.5, alert_id=0,
    ))
    return {"results": results}


# -- Camera preview + Data Collection -----------------------------------
import threading
import io
import time as _time
import cv2
import numpy as np


def _ensure_camera(container):
    """Lazily initialise shared camera + pose detector on container."""
    if getattr(container, '_cam', None) is not None:
        return
    container._cam = cv2.VideoCapture(0)
    container._cam_lock = threading.Lock()
    container._cam_frame = None       # latest raw frame (BGR)
    container._cam_joints24 = None    # latest 24-joint pose
    container._cam_keypoints = None   # latest COCO-17 pixel keypoints
    container._cam_running = False
    container._cam_thread = None

    # Collector for CSI sync
    from server.real_collector import RealDataCollector
    s = container.settings
    collector = getattr(container, '_collector', None)
    if collector is None:
        collector = RealDataCollector(settings=s, n_nodes=s.max_nodes)
        container._collector = collector

    # Wire CSI callback so collector gets real frames
    orig_on_frame = container.pipeline_svc.on_frame.__func__ if hasattr(container.pipeline_svc.on_frame, '__func__') else None

    def _csi_tap(frame, **kw):
        collector.on_csi_frame(frame)
    container._csi_tap = _csi_tap


def _camera_loop(container):
    """Background thread: capture frames + run pose detection."""
    try:
        from rtmlib import Body
        from server.camera_collector import coco17_to_24joint, MIN_CONFIDENCE, SKELETON_CONNECTIONS
        body = Body(mode='lightweight')
    except ImportError:
        return

    cam = container._cam
    collector = container._collector

    while container._cam_running:
        ret, frame = cam.read()
        if not ret:
            _time.sleep(0.05)
            continue

        keypoints, scores = body(frame)
        joints24 = None
        kp_px = None

        if keypoints is not None and len(keypoints) > 0:
            kp = keypoints[0]
            sc = scores[0]
            if np.sum(sc >= MIN_CONFIDENCE) >= 6:
                for i in range(17):
                    if sc[i] < MIN_CONFIDENCE:
                        nbrs = [j for j in range(17) if j != i and sc[j] >= MIN_CONFIDENCE]
                        if nbrs:
                            kp[i] = np.mean(kp[nbrs], axis=0)
                joints24 = coco17_to_24joint(kp[:, :2], standing_height=1.70)
                kp_px = kp[:, :2].astype(int)

                # If recording, add synchronised frame
                if getattr(container, '_collecting', False):
                    collector.add_frame(joints24)
                    container._collect_frames = len(collector._frames_joints)

        with container._cam_lock:
            container._cam_frame = frame
            container._cam_joints24 = joints24
            container._cam_keypoints = kp_px

        _time.sleep(0.03)  # ~30 FPS cap


@router.post("/api/camera/start")
async def camera_start(container: ServiceContainer = Depends(get_container)):
    _ensure_camera(container)
    if container._cam_running:
        return {"status": "already_running"}
    # Hook CSI tap
    container._emitter_unsub = container.emitter.on("csi", lambda d: None)  # keep emitter active
    container._cam_running = True
    container._cam_thread = threading.Thread(target=_camera_loop, args=(container,), daemon=True)
    container._cam_thread.start()
    return {"status": "started"}


@router.post("/api/camera/stop")
async def camera_stop(container: ServiceContainer = Depends(get_container)):
    container._cam_running = False
    if getattr(container, '_cam', None):
        container._cam.release()
        container._cam = None
    return {"status": "stopped"}


@router.get("/api/camera/frame")
async def camera_frame(container: ServiceContainer = Depends(get_container)):
    """Return JPEG snapshot with skeleton overlay."""
    from fastapi.responses import Response
    from server.camera_collector import SKELETON_CONNECTIONS

    if not getattr(container, '_cam_running', False):
        return JSONResponse({"error": "Camera not started. POST /api/camera/start first."}, status_code=400)

    with container._cam_lock:
        frame = container._cam_frame
        kp = container._cam_keypoints
        collecting = getattr(container, '_collecting', False)
        activity = getattr(container, '_collect_activity', '')
        n_frames = getattr(container, '_collect_frames', 0)

    if frame is None:
        return JSONResponse({"error": "No frame yet"}, status_code=503)

    vis = frame.copy()

    # Draw skeleton
    if kp is not None:
        for i, j in SKELETON_CONNECTIONS:
            cv2.line(vis, tuple(kp[i]), tuple(kp[j]), (0, 255, 0), 2)
        for pt in kp:
            cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)
        cv2.putText(vis, "POSE OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "NO POSE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Recording indicator
    if collecting:
        cv2.circle(vis, (vis.shape[1] - 25, 25), 10, (0, 0, 255), -1)
        cv2.putText(vis, f"REC {activity} #{n_frames}", (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    _, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.post("/api/collect/start")
async def collect_start(activity: str = "standing",
                        container: ServiceContainer = Depends(get_container)):
    from server.real_collector import ACTIVITIES
    if activity not in ACTIVITIES:
        return JSONResponse({"error": f"Unknown activity. Choose from: {ACTIVITIES}"}, status_code=400)
    if getattr(container, '_collecting', False):
        return JSONResponse({"error": "Already collecting"}, status_code=409)

    # Auto-start camera if not running
    if not getattr(container, '_cam_running', False):
        _ensure_camera(container)
        container._cam_running = True
        container._cam_thread = threading.Thread(target=_camera_loop, args=(container,), daemon=True)
        container._cam_thread.start()
        _time.sleep(1)  # let camera warm up

    collector = container._collector
    collector.start_recording(activity)
    container._collecting = True
    container._collect_activity = activity
    container._collect_frames = 0
    return {"status": "recording", "activity": activity}


@router.post("/api/collect/stop")
async def collect_stop(container: ServiceContainer = Depends(get_container)):
    collector = getattr(container, '_collector', None)
    if not getattr(container, '_collecting', False) or collector is None:
        return JSONResponse({"error": "Not collecting"}, status_code=409)
    data = collector.stop_recording()
    container._collecting = False
    if data is None:
        return {"status": "stopped", "frames": 0, "file": None}
    path = collector.save_sequence(data, "data/real")
    return {"status": "saved", "frames": len(data["joints"]), "file": path,
            "activity": getattr(container, '_collect_activity', '')}


@router.get("/api/collect/status")
async def collect_status(container: ServiceContainer = Depends(get_container)):
    from server.real_collector import ACTIVITIES
    collector = getattr(container, '_collector', None)
    return {
        "collecting": getattr(container, '_collecting', False),
        "activity": getattr(container, '_collect_activity', ''),
        "frames": getattr(container, '_collect_frames', 0),
        "csi_frames": collector._csi_frame_count if collector else 0,
        "nodes_seen": len(collector._csi_buffer) if collector else 0,
        "camera_running": getattr(container, '_cam_running', False),
        "pose_detected": getattr(container, '_cam_joints24', None) is not None,
        "activities": ACTIVITIES,
    }


# -- OTA -------------------------------------------------------------
def _firmware_dirs() -> list[Path]:
    """All directories that may contain firmware binaries."""
    base = Path(__file__).parent.parent.parent
    return [
        base / "dashboard" / "public" / "firmware",  # Vite build output
        base / "dist" / "dashboard" / "firmware",     # Production build
        base / "firmware" / "esp32-csi-node" / "build",  # Legacy path
    ]


@router.get("/api/ota/firmware")
async def ota_firmware_list():
    bins = []
    seen = set()
    for fw_dir in _firmware_dirs():
        if fw_dir.exists():
            for f in sorted(fw_dir.glob("*.bin")):
                if f.name not in seen:
                    seen.add(f.name)
                    bins.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "path": f"/api/ota/download/{f.name}",
                        "modified": f.stat().st_mtime,
                    })
    bins.sort(key=lambda x: x.get("modified", 0), reverse=True)
    return {"firmware": bins, "ota_endpoint": "/api/ota/download/<filename>"}


@router.get("/api/ota/download/{filename}")
async def ota_download(filename: str):
    if not re.match(r'^[\w\-\.]+$', filename):
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    for fw_dir in _firmware_dirs():
        fw_path = fw_dir / filename
        if fw_path.exists():
            return FileResponse(str(fw_path), media_type="application/octet-stream")
    return JSONResponse({"error": "Firmware not found"}, status_code=404)


@router.post("/api/ota/push")
async def ota_push(node_id: int | None = None):
    """Trigger OTA update for online nodes.

    ESP32 OTA works by the device pulling firmware from a URL.
    This endpoint returns the OTA download URL that nodes should fetch.
    In a full implementation, the server would notify nodes via WebSocket
    to trigger the pull. For now, return the URL for manual configuration.
    """
    bins = []
    for fw_dir in _firmware_dirs():
        if fw_dir.exists():
            bins.extend(fw_dir.glob("firmware-*.bin"))
    if not bins:
        return JSONResponse({"error": "No firmware binaries available. Build firmware first."}, status_code=404)

    # Find the newest firmware binary
    newest = max(bins, key=lambda f: f.stat().st_mtime)
    ota_url = f"/api/ota/download/{newest.name}"

    if node_id:
        return {
            "message": f"OTA ready for Node {node_id}",
            "firmware": newest.name,
            "ota_url": ota_url,
            "node_id": node_id,
            "instructions": f"Configure ESP32 Node {node_id} to fetch OTA from this server's {ota_url}",
        }
    return {
        "message": "OTA ready for all nodes",
        "firmware": newest.name,
        "ota_url": ota_url,
        "instructions": "Configure ESP32 nodes to fetch OTA from this server",
    }


# -- Training ------------------------------------------------------------
@router.post("/api/train/start")
async def train_start(epochs: int = 50, container: ServiceContainer = Depends(get_container)):
    """Start model training in background."""
    import subprocess, sys, threading
    if getattr(container, '_train_process', None) and container._train_process.poll() is None:
        return JSONResponse({"error": "Training already in progress"}, status_code=409)

    data_dir = "data/real" if Path("data/real").exists() else "data/synthetic"
    cmd = [sys.executable, "-m", "server.train",
           "--data-dir", data_dir, "--epochs", str(epochs),
           "--profile", container.settings.hardware_profile]
    container._train_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    container._train_log = []

    def reader():
        for line in container._train_process.stdout:
            container._train_log.append(line.strip())
            if len(container._train_log) > 200:
                container._train_log.pop(0)
    threading.Thread(target=reader, daemon=True).start()

    return {"status": "started", "data_dir": data_dir, "epochs": epochs}


@router.get("/api/train/status")
async def train_status(container: ServiceContainer = Depends(get_container)):
    """Return current training status and recent log lines."""
    proc = getattr(container, '_train_process', None)
    log = getattr(container, '_train_log', [])
    if proc is None:
        return {"status": "idle", "log": []}
    if proc.poll() is None:
        return {"status": "running", "log": log[-20:]}
    return {"status": "complete" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode, "log": log[-20:]}


# -- Model versioning ----------------------------------------------------
@router.get("/api/models")
async def list_models(container: ServiceContainer = Depends(get_container)):
    """List available model weight files."""
    models_dir = Path("models")
    result = []
    if models_dir.exists():
        for p in sorted(models_dir.rglob("*.pth")):
            active = str(p) == container.settings.model_path
            result.append({
                "path": str(p), "name": p.stem, "profile": p.parent.name,
                "size_mb": round(p.stat().st_size / 1048576, 1),
                "active": active,
            })
    return {"models": result, "active": container.settings.model_path}
