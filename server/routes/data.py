"""Data routes: history, collection, OTA, notifications."""
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


# -- Data Collection -------------------------------------------------
@router.post("/api/collect/start")
async def collect_start(activity: str = "standing",
                        container: ServiceContainer = Depends(get_container)):
    from server.real_collector import RealDataCollector, ACTIVITIES
    if activity not in ACTIVITIES:
        return JSONResponse({"error": f"Unknown activity. Choose from: {ACTIVITIES}"}, status_code=400)
    # Use a simple flag on container for collection state
    if getattr(container, '_collecting', False):
        return JSONResponse({"error": "Already collecting"}, status_code=409)
    collector = getattr(container, '_collector', None)
    if collector is None:
        s = container.settings
        collector = RealDataCollector(settings=s, n_nodes=s.max_nodes)
        container._collector = collector
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
        "activities": ACTIVITIES,
    }


# -- OTA -------------------------------------------------------------
@router.get("/api/ota/firmware")
async def ota_firmware_list():
    fw_dir = Path(__file__).parent.parent.parent / "firmware" / "esp32-csi-node" / "build"
    bins = []
    if fw_dir.exists():
        for f in sorted(fw_dir.glob("*.bin")):
            bins.append({"name": f.name, "size": f.stat().st_size, "path": f"/api/ota/download/{f.name}"})
    return {"firmware": bins, "ota_endpoint": "/api/ota/download/<filename>"}


@router.get("/api/ota/download/{filename}")
async def ota_download(filename: str):
    if not re.match(r'^[\w\-\.]+$', filename):
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    fw_path = Path(__file__).parent.parent.parent / "firmware" / "esp32-csi-node" / "build" / filename
    if not fw_path.exists():
        return JSONResponse({"error": "Firmware not found"}, status_code=404)
    return FileResponse(str(fw_path), media_type="application/octet-stream")
