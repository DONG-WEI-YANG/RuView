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

    # Server IP — prefer the WiFi adapter's IP over virtual adapters (Hyper-V, WSL, etc.)
    result["server_ip"] = _detect_wifi_ip()

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


def _detect_wifi_ip() -> str:
    """Get the IP address of the WiFi adapter, avoiding virtual adapters."""
    system = platform.system()

    if system == "Windows":
        # Use netsh to find the WiFi adapter's IP specifically
        try:
            out = subprocess.check_output(
                ["netsh", "interface", "ipv4", "show", "addresses", "Wi-Fi"],
                timeout=5, encoding="utf-8", errors="replace",
            )
            for line in out.splitlines():
                m = re.search(r"IP Address[:：]\s*([\d.]+)", line, re.IGNORECASE)
                if not m:
                    m = re.search(r"IP 位址[:：]\s*([\d.]+)", line)  # Chinese Windows
                if m:
                    return m.group(1)
        except Exception:
            pass

    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["ipconfig", "getifaddr", "en0"], text=True, timeout=5,
            )
            ip = out.strip()
            if ip:
                return ip
        except Exception:
            pass

    else:  # Linux
        try:
            out = subprocess.check_output(
                ["hostname", "-I"], text=True, timeout=5,
            )
            for ip in out.strip().split():
                if not ip.startswith("127."):
                    return ip
        except Exception:
            pass

    # Fallback: first non-loopback, non-virtual IP
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127.") or ip.startswith("172.16.") or ip.startswith("172.17."):
                continue  # skip loopback and common virtual adapter ranges
            return ip
    except Exception:
        pass
    return ""


def _detect_windows(result: dict) -> None:
    # Get SSID
    out = subprocess.check_output(
        ["netsh", "wlan", "show", "interfaces"],
        timeout=5, encoding="utf-8", errors="replace",
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
        timeout=5, encoding="utf-8", errors="replace",
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
