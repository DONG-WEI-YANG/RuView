"""Auto-detect current WiFi SSID, password, and server IP from the host PC.

Works on Windows (netsh), macOS (networksetup/security), and Linux (nmcli).
Falls back gracefully if detection fails.
"""
from __future__ import annotations

import logging
import platform
import re
import socket
import subprocess

logger = logging.getLogger(__name__)


def detect_wifi() -> dict:
    """Return {"ssid": str, "password": str, "server_ip": str, "detected": bool}."""
    result = {"ssid": "", "password": "", "server_ip": "", "detected": False}

    # Server IP
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                result["server_ip"] = ip
                break
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Windows":
            _detect_windows(result)
        elif system == "Darwin":
            _detect_macos(result)
        else:
            _detect_linux(result)
    except Exception as e:
        logger.warning("WiFi detection failed: %s", e)

    result["detected"] = bool(result["ssid"])
    return result


def _detect_windows(result: dict) -> None:
    # Get SSID
    out = subprocess.check_output(
        ["netsh", "wlan", "show", "interfaces"], text=True, timeout=5
    )
    for line in out.splitlines():
        if "SSID" in line and "BSSID" not in line:
            match = re.search(r"SSID\s*[:：]\s*(.+)", line)
            if match:
                result["ssid"] = match.group(1).strip()
                break

    if not result["ssid"]:
        return

    # Get password
    out = subprocess.check_output(
        ["netsh", "wlan", "show", "profile", f'name={result["ssid"]}', "key=clear"],
        text=True, timeout=5,
    )
    for line in out.splitlines():
        # Match English "Key Content" or Chinese "金鑰內容"
        if re.search(r"Key Content|金鑰內容|密碼|密钥内容", line):
            match = re.search(r"[:：]\s*(.+)", line)
            if match:
                result["password"] = match.group(1).strip()
                break


def _detect_macos(result: dict) -> None:
    out = subprocess.check_output(
        ["/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport", "-I"],
        text=True, timeout=5,
    )
    for line in out.splitlines():
        if " SSID:" in line:
            result["ssid"] = line.split("SSID:")[-1].strip()
            break
    if result["ssid"]:
        try:
            out = subprocess.check_output(
                ["security", "find-generic-password", "-wa", result["ssid"]],
                text=True, timeout=5,
            )
            result["password"] = out.strip()
        except subprocess.CalledProcessError:
            pass


def _detect_linux(result: dict) -> None:
    out = subprocess.check_output(
        ["nmcli", "-t", "-f", "ACTIVE,SSID,DEVICE", "connection", "show", "--active"],
        text=True, timeout=5,
    )
    for line in out.splitlines():
        parts = line.split(":")
        if parts[0] == "yes" and len(parts) >= 2:
            result["ssid"] = parts[1]
            break
    if result["ssid"]:
        try:
            out = subprocess.check_output(
                ["nmcli", "-s", "-g", "802-11-wireless-security.psk",
                 "connection", "show", result["ssid"]],
                text=True, timeout=5,
            )
            result["password"] = out.strip()
        except subprocess.CalledProcessError:
            pass
