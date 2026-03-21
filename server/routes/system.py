"""System routes: /, /api/status, /api/profiles, /api/joints, /api/vitals, /api/alerts, /api/system/mode."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from server.config import HARDWARE_PROFILES
from server.services.container import ServiceContainer

router = APIRouter()


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


@router.get("/")
async def root():
    return {"name": "wifi-body", "version": "0.2.0"}


@router.get("/api/network")
async def network_info(request: Request):
    """Return server's LAN IPs so tablets/phones can discover the dashboard URL."""
    import socket
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    ips = list(dict.fromkeys(ips))
    port = request.url.port or 8000
    urls = [f"http://{ip}:{port}/dashboard/" for ip in ips]
    mdns_url = f"http://wifi-body.local:{port}/dashboard/"
    return {"ips": ips, "port": port, "urls": urls, "mdns": mdns_url, "hostname": socket.gethostname()}


@router.get("/api/status")
async def status(container: ServiceContainer = Depends(get_container)):
    s = container.settings
    ps = container.pipeline_svc
    fd = ps.fall_detector
    ft = ps.fitness_tracker
    cal = container.calibration
    has_model = ps.pipeline is not None and ps.pipeline.model is not None
    return {
        "nodes": {
            str(nid): {"last_seq": f.sequence, "rssi": f.rssi}
            for nid, f in ps.node_frames.items()
        },
        "is_fallen": fd.is_fallen if fd else False,
        "current_activity": ft.current_activity.value if ft else "unknown",
        "fall_alerts": len(fd.get_alerts()) if fd else 0,
        "vitals": container.vitals.get_vitals(),
        "person_count": container.vitals.multi_person.person_count,
        "pipeline_status": {
            "csi_receiver": "listening",
            "model_loaded": has_model,
            "csi_frames_received": ps.csi_frames_received,
            "inference_active": has_model and ps.csi_frames_received > 0,
            "hardware_profile": s.hardware_profile,
            "is_simulating": s.simulate,
            "detected_nodes": ps.detected_nodes,
            "strategy": ps.strategy,
            "strategy_description": ps.strategy_description,
        },
        "node_positions": s.node_positions,
        "room_dimensions": {
            "width": s.room_width,
            "depth": s.room_depth,
            "height": s.room_height,
        },
        "storage": container.storage.storage.get_stats(),
        "calibration": cal.get_status(),
        "notifications_enabled": container.notification.notifier.enabled,
        "scene_mode": ps.scene_mode,
    }


@router.get("/api/profiles")
async def get_profiles(container: ServiceContainer = Depends(get_container)):
    s = container.settings
    active_id = s.hardware_profile
    profiles = []
    for pid, p in HARDWARE_PROFILES.items():
        model_exists = Path(p.model_path).exists()
        profiles.append({
            "id": p.id, "name": p.name, "description": p.description,
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


@router.get("/api/joints")
async def get_joints(container: ServiceContainer = Depends(get_container)):
    joints = container.pipeline_svc.latest_joints
    if joints is None:
        return {"joints": None}
    return {"joints": joints.tolist()}


@router.get("/api/vitals")
async def get_vitals(container: ServiceContainer = Depends(get_container)):
    vs = container.vitals
    return {
        "primary": vs.get_vitals(),
        "csi_amplitudes": vs.get_subcarrier_amplitudes(),
        "persons": vs.multi_person.update_all(),
    }


@router.get("/api/alerts")
async def get_alerts(container: ServiceContainer = Depends(get_container)):
    fd = container.pipeline_svc.fall_detector
    if fd is None:
        return {"alerts": []}
    return {
        "alerts": [
            {"timestamp": a.timestamp, "confidence": a.confidence, "head_height": a.head_height}
            for a in fd.get_alerts()
        ]
    }


@router.post("/api/system/mode")
async def set_mode(mode: str, container: ServiceContainer = Depends(get_container)):
    if mode not in ["simulation", "real"]:
        return JSONResponse({"error": "Invalid mode"}, status_code=400)
    is_sim = (mode == "simulation")
    if is_sim == container.settings.simulate:
        return {"status": "unchanged", "mode": mode}
    container.settings.simulate = is_sim
    if is_sim:
        await container.pipeline_svc.start_simulation()
    else:
        await container.pipeline_svc.stop_simulation()
    return {"status": "switched", "mode": mode}


@router.get("/api/system/scene")
async def get_scene(container: ServiceContainer = Depends(get_container)):
    ps = container.pipeline_svc
    return {
        "scene_mode": ps.scene_mode,
        **ps.scene_config,
    }


@router.post("/api/system/scene")
async def set_scene(scene: str, container: ServiceContainer = Depends(get_container)):
    from server.services.pipeline_service import SCENE_MODES
    if scene not in SCENE_MODES:
        return JSONResponse(
            {"error": f"Unknown scene. Choose from: {list(SCENE_MODES.keys())}"},
            status_code=400,
        )
    result = container.pipeline_svc.set_scene_mode(scene)
    return result
