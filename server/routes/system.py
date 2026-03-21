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
            "real_nodes": ps.real_node_count,
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


@router.get("/api/persons")
async def get_persons(container: ServiceContainer = Depends(get_container)):
    """Get all currently tracked persons with their poses, vitals, and positions."""
    ps = container.pipeline_svc
    return {
        "count": ps.person_count,
        "persons": ps.get_persons_snapshot(),
    }


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


@router.get("/api/network/wifi")
async def wifi_config():
    """Auto-detect current WiFi SSID, password, and server IP from host PC."""
    from server.wifi_detect import detect_wifi
    result = detect_wifi()

    # Check if firmware was flashed with different WiFi config
    from server.firmware_builder import BUILD_DIR
    defaults_path = BUILD_DIR / "sdkconfig.defaults"
    result["firmware_match"] = True
    result["firmware_ssid"] = ""
    result["firmware_ip"] = ""
    if defaults_path.exists():
        import re
        text = defaults_path.read_text(encoding="utf-8", errors="replace")
        m_ssid = re.search(r'CONFIG_CSI_WIFI_SSID="(.+?)"', text)
        m_ip = re.search(r'CONFIG_CSI_TARGET_IP="(.+?)"', text)
        if m_ssid:
            result["firmware_ssid"] = m_ssid.group(1)
        if m_ip:
            result["firmware_ip"] = m_ip.group(1)
        if m_ssid and m_ssid.group(1) != result["ssid"]:
            result["firmware_match"] = False
        if m_ip and m_ip.group(1) != result["server_ip"]:
            result["firmware_match"] = False

    return result


@router.get("/api/firmware/detect")
async def firmware_detect_devices():
    """Auto-detect all connected ESP32 boards — chip type, port, board name."""
    from server.firmware_builder import detect_all_ports
    devices = detect_all_ports()
    return {"devices": devices, "count": len(devices)}


@router.post("/api/firmware/auto")
async def firmware_auto_build_flash(
    port: str,
    node_id: int = 1,
    container: ServiceContainer = Depends(get_container),
):
    """One-click: detect chip → auto WiFi → build → flash. Runs in background."""
    import threading
    from server.firmware_builder import auto_build_and_flash

    if getattr(container, '_fw_building', False):
        return JSONResponse({"error": "Build already in progress"}, status_code=409)

    container._fw_building = True
    container._fw_build_result = None

    def do_all():
        try:
            result = auto_build_and_flash(port=port, node_id=node_id)
            container._fw_build_result = {"status": "complete" if result["success"] else "failed", **result}
        except Exception as e:
            container._fw_build_result = {"status": "failed", "error": str(e)}
        finally:
            container._fw_building = False

    threading.Thread(target=do_all, daemon=True).start()
    return {"status": "started", "port": port, "node_id": node_id}


@router.post("/api/firmware/build")
async def firmware_build(
    node_ids: str = "1,2",
    container: ServiceContainer = Depends(get_container),
):
    """Build firmware with auto-detected WiFi (without flashing)."""
    import threading
    from server.wifi_detect import detect_wifi
    from server.firmware_builder import build_all_nodes, detect_all_ports, update_platformio_ini, clean_build

    wifi = detect_wifi()
    if not wifi["detected"]:
        return JSONResponse({"error": "WiFi not detected on this PC"}, status_code=400)

    ids = [int(x.strip()) for x in node_ids.split(",") if x.strip().isdigit()]
    if not ids:
        return JSONResponse({"error": "No valid node IDs"}, status_code=400)

    # Auto-detect chip from first connected device
    devices = detect_all_ports()
    chip = "esp32c3"  # fallback
    port = "COM10"
    if devices:
        dev = devices[0]
        chip = dev.get("chip", chip)
        port = dev.get("port", port)

    if getattr(container, '_fw_building', False):
        return JSONResponse({"error": "Build already in progress"}, status_code=409)

    container._fw_building = True
    container._fw_build_result = None

    def do_build():
        try:
            for nid in ids:
                update_platformio_ini(chip, port, nid)
                clean_build(nid)
            result = build_all_nodes(
                ssid=wifi["ssid"], password=wifi["password"],
                server_ip=wifi["server_ip"], node_ids=ids,
            )
            container._fw_build_result = {
                "status": "complete", "wifi": wifi, "chip": chip, "nodes": result,
            }
        except Exception as e:
            container._fw_build_result = {"status": "failed", "error": str(e)}
        finally:
            container._fw_building = False

    threading.Thread(target=do_build, daemon=True).start()
    return {"status": "building", "wifi_ssid": wifi["ssid"], "chip": chip, "node_ids": ids}


@router.get("/api/firmware/status")
async def firmware_build_status(container: ServiceContainer = Depends(get_container)):
    """Check firmware build/flash progress."""
    if getattr(container, '_fw_building', False):
        return {"status": "building"}
    result = getattr(container, '_fw_build_result', None)
    if result:
        return result
    return {"status": "idle"}


@router.get("/api/signal-quality")
async def signal_quality(container: ServiceContainer = Depends(get_container)):
    """Per-node signal quality, overall grade, and environment tips."""
    return container.signal_quality.get_quality()


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
