"""Build & flash ESP32 CSI firmware with auto-detected chip type and WiFi credentials.

Flow:
1. Detect chip type via esptool (ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C6)
2. Auto-select matching PlatformIO board + sdkconfig
3. Bake in WiFi SSID/password/server IP from PC
4. Build via PlatformIO
5. Flash to device

Supports multiple ESP32 variants without manual configuration.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths (avoid spaces — ESP-IDF can't handle them)
BUILD_DIR = Path("C:/temp/csi-node")
PIO_EXE = "C:/temp/pio-env/Scripts/pio.exe"
OUTPUT_DIR = Path(__file__).parent.parent / "dashboard" / "public" / "firmware"

# ── Chip → PlatformIO board mapping ─────────────────────────
CHIP_BOARDS = {
    "esp32":    {"board": "esp32dev",            "flash_size": "4MB",  "name": "ESP32 (Xtensa)"},
    "esp32s2":  {"board": "esp32-s2-saola-1",    "flash_size": "4MB",  "name": "ESP32-S2"},
    "esp32s3":  {"board": "esp32-s3-devkitc-1",  "flash_size": "8MB",  "name": "ESP32-S3 (Xtensa)"},
    "esp32c3":  {"board": "esp32-c3-devkitm-1",  "flash_size": "4MB",  "name": "ESP32-C3 (RISC-V)"},
    "esp32c6":  {"board": "esp32-c6-devkitc-1",  "flash_size": "4MB",  "name": "ESP32-C6 (RISC-V, WiFi 6)"},
    "esp32h2":  {"board": "esp32-h2-devkitm-1",  "flash_size": "4MB",  "name": "ESP32-H2 (Thread/Zigbee)"},
}


def detect_chip(port: str) -> dict:
    """Detect ESP32 chip type on a serial port via esptool.

    Returns {"chip": "esp32c3", "port": "COM10", "board": "esp32-c3-devkitm-1", ...}
    """
    esptool_cmd = _find_esptool()
    if not esptool_cmd:
        return {"error": "esptool not found", "chip": None, "port": port}

    logger.info("Detecting chip on %s ...", port)
    try:
        result = subprocess.run(
            [*esptool_cmd, "--port", port, "chip_id"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"error": "Detection timed out", "chip": None, "port": port}
    except Exception as e:
        return {"error": str(e), "chip": None, "port": port}

    # Parse chip type from esptool output
    # Examples: "Chip is ESP32-C3 (QFN32)" or "Detecting chip type... ESP32-S3"
    chip_id = None
    for pattern in [
        r"Chip is (ESP32[^\s(]*)",
        r"Detecting chip type[.\s]*(ESP32[^\s]*)",
    ]:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            raw = m.group(1).upper().replace("-", "").replace("ESP32", "esp32")
            # Normalize: ESP32C3 → esp32c3, ESP32-S3 → esp32s3
            chip_id = raw.lower().replace("esp32", "esp32", 1)
            break

    if not chip_id:
        # Fallback: look for "esp32" anywhere
        m = re.search(r"(esp32\w*)", output, re.IGNORECASE)
        if m:
            chip_id = m.group(1).lower().replace("-", "")

    if not chip_id or chip_id not in CHIP_BOARDS:
        return {
            "error": f"Unknown chip: {chip_id or 'not detected'}",
            "chip": chip_id,
            "port": port,
            "raw_output": output[-500:],
        }

    info = CHIP_BOARDS[chip_id]
    return {
        "chip": chip_id,
        "port": port,
        "board": info["board"],
        "flash_size": info["flash_size"],
        "name": info["name"],
        "detected": True,
    }


def detect_all_ports() -> list[dict]:
    """Detect chip type on all connected serial ports."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return []

    ports = list(serial.tools.list_ports.comports())
    results = []
    for p in ports:
        # Only probe Espressif / common USB-UART chips
        if p.vid in (0x303A, 0x10C4, 0x1A86, 0x0403, None):
            info = detect_chip(p.device)
            info["description"] = p.description
            info["vid"] = p.vid
            info["pid"] = p.pid
            results.append(info)

    return results


def update_platformio_ini(chip: str, port: str, node_id: int = 1) -> None:
    """Rewrite platformio.ini for the detected chip."""
    if chip not in CHIP_BOARDS:
        raise ValueError(f"Unsupported chip: {chip}")

    board = CHIP_BOARDS[chip]["board"]
    env_name = f"node{node_id}"

    ini_path = BUILD_DIR / "platformio.ini"
    ini_path.write_text(
        f"; Auto-generated for {CHIP_BOARDS[chip]['name']} on {port}\n"
        f"[env:{env_name}]\n"
        f"platform = espressif32\n"
        f"framework = espidf\n"
        f"board = {board}\n"
        f"monitor_speed = 115200\n"
        f"upload_port = {port}\n",
        encoding="utf-8",
    )
    logger.info("platformio.ini updated: chip=%s, board=%s, port=%s", chip, board, port)


def update_sdkconfig(
    ssid: str,
    password: str,
    server_ip: str,
    udp_port: int = 5005,
    node_id: int = 1,
) -> Path:
    """Write WiFi + server config into sdkconfig.defaults."""
    if not BUILD_DIR.exists():
        raise FileNotFoundError(
            f"Build directory {BUILD_DIR} not found. "
            f"Copy firmware/esp32-csi-node/ to C:\\temp\\csi-node\\ first."
        )

    defaults = BUILD_DIR / "sdkconfig.defaults"
    lines = []
    if defaults.exists():
        lines = defaults.read_text(encoding="utf-8", errors="replace").splitlines()

    # Remove old WiFi/server/target lines (both CSI_ and non-CSI_ prefixed)
    remove_keys = [
        "CONFIG_WIFI_SSID", "CONFIG_WIFI_PASSWORD",
        "CONFIG_CSI_WIFI_SSID", "CONFIG_CSI_WIFI_PASSWORD",
        "CONFIG_CSI_TARGET_IP", "CONFIG_CSI_TARGET_PORT",
        "CONFIG_CSI_NODE_ID", "CONFIG_IDF_TARGET",
    ]
    lines = [l for l in lines if not any(k in l for k in remove_keys)]

    lines.extend([
        f'CONFIG_CSI_WIFI_SSID="{ssid}"',
        f'CONFIG_CSI_WIFI_PASSWORD="{password}"',
        f'CONFIG_CSI_TARGET_IP="{server_ip}"',
        f'CONFIG_CSI_TARGET_PORT={udp_port}',
        f'CONFIG_CSI_NODE_ID={node_id}',
    ])

    defaults.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("sdkconfig.defaults updated: SSID=%s, IP=%s, node=%d", ssid, server_ip, node_id)
    return BUILD_DIR


def clean_build(node_id: int = 1) -> None:
    """Remove build artifacts and sdkconfig for a clean rebuild (needed when chip changes)."""
    env_name = f"node{node_id}"
    build_out = BUILD_DIR / ".pio" / "build" / env_name
    sdkconfig = BUILD_DIR / f"sdkconfig.{env_name}"

    if build_out.exists():
        shutil.rmtree(build_out, ignore_errors=True)
    if sdkconfig.exists():
        sdkconfig.unlink()

    # Also remove the main sdkconfig (forces regeneration from defaults)
    main_sdkconfig = BUILD_DIR / "sdkconfig"
    if main_sdkconfig.exists():
        main_sdkconfig.unlink()

    logger.info("Cleaned build for %s", env_name)


def build_firmware(node_id: int = 1) -> dict:
    """Run PlatformIO build. Returns build result."""
    env_name = f"node{node_id}"

    if not Path(PIO_EXE).exists():
        return {"success": False, "error": f"PlatformIO not found at {PIO_EXE}"}

    logger.info("Building firmware env=%s ...", env_name)
    try:
        result = subprocess.run(
            [PIO_EXE, "run", "-e", env_name],
            cwd=str(BUILD_DIR),
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Build timed out (5 min)"}

    if result.returncode != 0:
        return {
            "success": False, "error": "Build failed",
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

    return {"success": True, "files": copied, "env": env_name}


def flash_firmware(port: str, node_id: int = 1) -> dict:
    """Flash pre-built firmware to a device via esptool."""
    esptool_cmd = _find_esptool()
    if not esptool_cmd:
        return {"success": False, "error": "esptool not found"}

    env_name = f"node{node_id}"
    build_out = BUILD_DIR / ".pio" / "build" / env_name

    bootloader = build_out / "bootloader.bin"
    partitions = build_out / "partitions.bin"
    firmware = build_out / "firmware.bin"

    if not firmware.exists():
        return {"success": False, "error": f"Firmware not built yet for {env_name}"}

    args = [
        *esptool_cmd, "--port", port, "--baud", "460800",
        "write_flash",
        "0x0", str(bootloader),
        "0x8000", str(partitions),
        "0x10000", str(firmware),
    ]

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Flash timed out"}

    if result.returncode != 0:
        return {"success": False, "error": "Flash failed", "log": result.stdout[-1000:]}

    return {"success": True, "port": port, "node_id": node_id}


def auto_build_and_flash(
    port: str,
    node_id: int = 1,
    ssid: str = "",
    password: str = "",
    server_ip: str = "",
    udp_port: int = 5005,
) -> dict:
    """Full pipeline: detect chip → configure → clean → build → flash.

    This is the one-click function that handles everything.
    """
    # 1. Detect chip
    chip_info = detect_chip(port)
    if not chip_info.get("detected"):
        return {"success": False, "step": "detect", **chip_info}

    chip = chip_info["chip"]
    logger.info("Auto-build: %s on %s (node %d)", chip_info["name"], port, node_id)

    # 2. Auto-detect WiFi if not provided
    if not ssid:
        from server.wifi_detect import detect_wifi
        wifi = detect_wifi()
        ssid = wifi["ssid"]
        password = wifi["password"]
        server_ip = wifi["server_ip"]
        if not ssid:
            return {"success": False, "step": "wifi", "error": "WiFi not detected"}

    # 3. Configure
    update_platformio_ini(chip, port, node_id)
    update_sdkconfig(ssid, password, server_ip, udp_port, node_id)
    clean_build(node_id)

    # 4. Build
    build_result = build_firmware(node_id)
    if not build_result["success"]:
        return {"success": False, "step": "build", **build_result}

    # 5. Flash
    flash_result = flash_firmware(port, node_id)
    if not flash_result["success"]:
        return {"success": False, "step": "flash", **flash_result}

    return {
        "success": True,
        "chip": chip_info,
        "wifi_ssid": ssid,
        "server_ip": server_ip,
        "node_id": node_id,
        "files": build_result.get("files", {}),
    }


def _find_esptool() -> list[str] | None:
    """Find esptool.py and return a command list to invoke it.

    Returns e.g. [sys.executable, "path/to/esptool.py"] so that
    subprocess.run() works on Windows (where .py files are not directly executable).
    """
    import sys

    # PlatformIO's bundled esptool
    pio_esptool = Path(PIO_EXE).parent.parent / "packages" / "tool-esptoolpy" / "esptool.py"
    if pio_esptool.exists():
        return [sys.executable, str(pio_esptool)]

    # Check platformio packages dir
    import glob
    for p in glob.glob(str(Path.home() / ".platformio/packages/tool-esptoolpy/esptool.py")):
        return [sys.executable, p]

    # System esptool (might be a real .exe)
    esptool_path = shutil.which("esptool.py") or shutil.which("esptool")
    if esptool_path:
        if esptool_path.endswith(".py"):
            return [sys.executable, esptool_path]
        return [esptool_path]
    return None
