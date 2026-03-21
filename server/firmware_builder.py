"""Build ESP32 CSI firmware with auto-detected WiFi credentials.

Writes SSID, password, server IP, and node ID into sdkconfig,
then triggers PlatformIO build. Output binaries go to
dashboard/public/firmware/ for web flashing.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths (avoid spaces — ESP-IDF can't handle them)
BUILD_DIR = Path("C:/temp/csi-node")
PIO_EXE = "C:/temp/pio-env/Scripts/pio.exe"
OUTPUT_DIR = Path(__file__).parent.parent / "dashboard" / "public" / "firmware"


def update_sdkconfig(
    ssid: str,
    password: str,
    server_ip: str,
    udp_port: int = 5005,
    node_id: int = 1,
) -> Path:
    """Write WiFi + server config into sdkconfig.defaults and return build dir."""
    if not BUILD_DIR.exists():
        raise FileNotFoundError(
            f"Build directory {BUILD_DIR} not found. "
            f"Copy firmware/esp32-csi-node/ to C:\\temp\\csi-node\\ first."
        )

    defaults = BUILD_DIR / "sdkconfig.defaults"
    lines = []
    if defaults.exists():
        lines = defaults.read_text().splitlines()

    # Remove old WiFi/server lines
    lines = [
        l for l in lines
        if not any(k in l for k in [
            "CONFIG_WIFI_SSID", "CONFIG_WIFI_PASSWORD",
            "CONFIG_CSI_TARGET_IP", "CONFIG_CSI_TARGET_PORT",
            "CONFIG_CSI_NODE_ID",
        ])
    ]

    # Append new config
    lines.extend([
        f'CONFIG_WIFI_SSID="{ssid}"',
        f'CONFIG_WIFI_PASSWORD="{password}"',
        f'CONFIG_CSI_TARGET_IP="{server_ip}"',
        f'CONFIG_CSI_TARGET_PORT={udp_port}',
        f'CONFIG_CSI_NODE_ID={node_id}',
    ])

    defaults.write_text("\n".join(lines) + "\n")
    logger.info("sdkconfig.defaults updated: SSID=%s, IP=%s, node=%d", ssid, server_ip, node_id)
    return BUILD_DIR


def build_firmware(node_id: int = 1) -> dict:
    """Run PlatformIO build for the given node. Returns build result."""
    env_name = f"node{node_id}" if node_id <= 2 else "node1"

    if not Path(PIO_EXE).exists():
        return {"success": False, "error": f"PlatformIO not found at {PIO_EXE}"}

    logger.info("Building firmware env=%s ...", env_name)
    try:
        result = subprocess.run(
            [PIO_EXE, "run", "-e", env_name],
            cwd=str(BUILD_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Build timed out (5 min)"}

    if result.returncode != 0:
        return {
            "success": False,
            "error": "Build failed",
            "log": result.stdout[-2000:] + "\n" + result.stderr[-500:],
        }

    # Copy binaries to dashboard/public/firmware/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_out = BUILD_DIR / ".pio" / "build" / env_name

    copied = {}
    for name, src in [
        ("bootloader.bin", build_out / "bootloader.bin"),
        ("partitions.bin", build_out / "partitions.bin"),
        (f"firmware-node{node_id}.bin", build_out / "firmware.bin"),
    ]:
        if src.exists():
            dst = OUTPUT_DIR / name
            shutil.copy2(src, dst)
            copied[name] = dst.stat().st_size
            logger.info("Copied %s (%d bytes)", name, copied[name])

    return {"success": True, "files": copied, "env": env_name}


def build_all_nodes(
    ssid: str,
    password: str,
    server_ip: str,
    udp_port: int = 5005,
    node_ids: list[int] | None = None,
) -> list[dict]:
    """Build firmware for multiple nodes with the same WiFi config."""
    if node_ids is None:
        node_ids = [1, 2]

    results = []
    for nid in node_ids:
        update_sdkconfig(ssid, password, server_ip, udp_port, nid)
        result = build_firmware(nid)
        result["node_id"] = nid
        results.append(result)

    return results
