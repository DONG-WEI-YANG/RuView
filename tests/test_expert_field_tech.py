"""Field Technician Tour — on-site installation, flashing, calibration, networking.

A field installation technician reviews: Can I flash firmware on a fresh laptop?
What happens when WiFi detection fails? Does calibration work with slow-booting
nodes? Are COM port and path assumptions valid on all machines?

Expert focus: deployment robustness, error messages, hardcoded assumptions.
"""
import struct
import re
from pathlib import Path

import pytest
import numpy as np

from server.config import Settings, HARDWARE_PROFILES, DEFAULT_PROFILE
from server.csi_frame import parse_csi_frame, MAGIC_HEADER, HEADER_FORMAT
from server.calibration import (
    CalibrationManager, CALIBRATION_DURATION_SEC, MIN_SAMPLES_PER_NODE,
)
from server.services.signal_quality import (
    SignalQualityMonitor, RSSI_EXCELLENT, RSSI_GOOD, RSSI_POOR,
)
from server.services.event_emitter import EventEmitter
from server.wifi_detect import detect_wifi
from tests.conftest import make_csi_frame, make_csi_binary


# ═══════════════════════════════════════════════════════════
# 1. Firmware Builder — path assumptions
# ═══════════════════════════════════════════════════════════

class TestFirmwareBuilderPaths:
    """Technician says: 'I plugged in a new laptop and nothing works.'
    Verify that hardcoded paths are documented and detectable.
    """

    def test_build_dir_constant_exists(self):
        """BUILD_DIR must be importable — it's referenced by routes."""
        from server.firmware_builder import BUILD_DIR
        assert isinstance(BUILD_DIR, Path)

    def test_pio_exe_constant_exists(self):
        from server.firmware_builder import PIO_EXE
        assert isinstance(PIO_EXE, str)
        assert PIO_EXE.endswith("pio.exe") or PIO_EXE.endswith("pio")

    def test_chip_boards_covers_all_common_chips(self):
        """Technician may encounter any ESP32 variant."""
        from server.firmware_builder import CHIP_BOARDS
        required = {"esp32", "esp32s2", "esp32s3", "esp32c3", "esp32c6"}
        assert required.issubset(set(CHIP_BOARDS.keys()))

    def test_output_dir_is_under_dashboard(self):
        """Firmware binaries go to dashboard/public/firmware for OTA."""
        from server.firmware_builder import OUTPUT_DIR
        assert "firmware" in str(OUTPUT_DIR)

    def test_detect_chip_handles_missing_esptool(self):
        """If esptool is not installed, detect_chip should return error dict."""
        from server.firmware_builder import detect_chip
        # This will fail on machines without esptool — should return error, not crash
        result = detect_chip("COM99")  # non-existent port
        assert isinstance(result, dict)
        assert "port" in result

    def test_paths_configurable_via_env(self):
        """BUILD_DIR and PIO_EXE should be overridable via environment variables."""
        source = Path("server/firmware_builder.py").read_text(encoding="utf-8")
        assert "CSI_BUILD_DIR" in source, "BUILD_DIR should read CSI_BUILD_DIR env var"
        assert "CSI_PIO_EXE" in source, "PIO_EXE should read CSI_PIO_EXE env var"


# ═══════════════════════════════════════════════════════════
# 2. WiFi Detection — platform quirks
# ═══════════════════════════════════════════════════════════

class TestWiFiDetection:
    """Technician's WiFi might be WPA3, hidden SSID, or Chinese Windows."""

    def test_detect_wifi_returns_dict(self):
        """detect_wifi() must always return a dict, even on failure."""
        result = detect_wifi()
        assert isinstance(result, dict)
        assert "detected" in result
        assert "ssid" in result
        assert "password" in result
        assert "server_ip" in result

    def test_detect_wifi_graceful_on_no_wifi(self):
        """If WiFi is off, should return detected=False, not crash."""
        result = detect_wifi()
        assert isinstance(result["detected"], bool)

    def test_chinese_windows_password_parsing(self):
        """wifi_detect.py handles Chinese Windows ('金鑰內容')."""
        source = Path("server/wifi_detect.py").read_text(encoding="utf-8")
        assert "金鑰內容" in source or "密碼" in source, \
            "WiFi detect must handle Chinese Windows key labels"

    def test_server_ip_filters_virtual_adapters(self):
        """Server IP should exclude Hyper-V/WSL virtual adapter IPs."""
        source = Path("server/wifi_detect.py").read_text(encoding="utf-8")
        assert "172.16" in source or "172.17" in source, \
            "Should filter out virtual adapter IPs (172.16.x.x, 172.17.x.x)"


# ═══════════════════════════════════════════════════════════
# 3. Calibration — field deployment scenarios
# ═══════════════════════════════════════════════════════════

class TestCalibrationFieldScenarios:
    """Technician runs calibration on-site. Nodes may boot slowly,
    environment may have interference, some nodes may be unreachable.
    """

    def test_calibration_with_exactly_min_samples(self):
        """Edge case: exactly 20 samples — should be included."""
        mgr = CalibrationManager()
        mgr.start(mode="spatial")
        for i in range(MIN_SAMPLES_PER_NODE):
            mgr.on_csi_frame(make_csi_frame(node_id=1, rssi=-55, sequence=i))
        result = mgr.finish()
        assert "1" in result["nodes"], "Exactly 20 samples should be accepted"

    def test_calibration_with_19_samples_excluded(self):
        """19 samples (one short) — node must be excluded."""
        mgr = CalibrationManager()
        mgr.start(mode="spatial")
        for i in range(MIN_SAMPLES_PER_NODE - 1):
            mgr.on_csi_frame(make_csi_frame(node_id=1, rssi=-55, sequence=i))
        result = mgr.finish()
        assert "1" not in result["nodes"]

    def test_calibration_partial_nodes(self):
        """Some nodes respond, some don't — result should include only responsive ones."""
        mgr = CalibrationManager()
        mgr.start(mode="spatial")
        # Node 1: responsive (25 frames)
        for i in range(25):
            mgr.on_csi_frame(make_csi_frame(node_id=1, rssi=-50, sequence=i))
        # Node 2: slow boot (only 5 frames)
        for i in range(5):
            mgr.on_csi_frame(make_csi_frame(node_id=2, rssi=-60, sequence=i))
        # Node 3: completely silent (0 frames)
        result = mgr.finish()
        assert result["node_count"] == 1  # only node 1 qualifies
        assert "1" in result["nodes"]
        assert "2" not in result["nodes"]
        assert "3" not in result["nodes"]

    def test_calibration_distance_reasonable_range(self):
        """Typical room: nodes 1-5 meters from center. Distance should be in range."""
        mgr = CalibrationManager()
        mgr.start(mode="spatial")
        # RSSI -55 dBm ≈ 2-3m in typical indoor
        for i in range(25):
            mgr.on_csi_frame(make_csi_frame(node_id=1, rssi=-55, sequence=i))
        result = mgr.finish()
        dist = result["nodes"]["1"]["estimated_distance_m"]
        assert 1.0 < dist < 10.0, f"Distance {dist}m is unreasonable for indoor"

    def test_background_calibration_stores_profile(self):
        """Background calibration: technician stands in empty room."""
        mgr = CalibrationManager()
        mgr.start(mode="background")
        for i in range(25):
            mgr.on_csi_frame(make_csi_frame(node_id=1, sequence=i))
        result = mgr.finish()
        assert result["mode"] == "background"
        profile = mgr.get_background_profile()
        assert 1 in profile
        assert len(profile[1]) == 56

    def test_double_finish_returns_error(self):
        """Finishing an already-finished calibration should return error."""
        mgr = CalibrationManager()
        mgr.start()
        mgr.finish()
        result = mgr.finish()
        assert result["status"] == "error"


# ═══════════════════════════════════════════════════════════
# 4. Node Positioning & Room Setup
# ═══════════════════════════════════════════════════════════

class TestNodePositioning:
    """Technician places 4 nodes around a room. Default positions must make sense."""

    def test_default_4_nodes(self):
        s = Settings()
        assert len(s.node_positions) == 4

    def test_default_positions_are_symmetric(self):
        """Nodes should be placed symmetrically around room center (0,0,0)."""
        s = Settings()
        xs = [pos[0] for pos in s.node_positions.values()]
        zs = [pos[2] for pos in s.node_positions.values()]
        # X values should be balanced: some positive, some negative
        assert any(x > 0 for x in xs) and any(x < 0 for x in xs)
        assert any(z > 0 for z in zs) and any(z < 0 for z in zs)

    def test_default_room_dimensions(self):
        s = Settings()
        assert s.room_width == 4.0   # 4m wide
        assert s.room_depth == 4.0   # 4m deep
        assert s.room_height == 2.8  # 2.8m ceiling

    def test_nodes_within_room_bounds(self):
        """Default node positions should be within the default room."""
        s = Settings()
        hw = s.room_width / 2
        hd = s.room_depth / 2
        for nid, pos in s.node_positions.items():
            assert -hw <= pos[0] <= hw, f"Node {nid} X={pos[0]} outside room"
            assert 0 <= pos[1] <= s.room_height, f"Node {nid} Y={pos[1]} outside room"
            assert -hd <= pos[2] <= hd, f"Node {nid} Z={pos[2]} outside room"

    def test_node_heights_vary(self):
        """Nodes should be at different heights for spatial diversity."""
        s = Settings()
        heights = [pos[1] for pos in s.node_positions.values()]
        assert len(set(heights)) >= 2, "Nodes should be at varied heights"


# ═══════════════════════════════════════════════════════════
# 5. Network & Port Configuration
# ═══════════════════════════════════════════════════════════

class TestNetworkConfig:
    """Technician needs to know: what ports are used, can they be changed?"""

    def test_default_udp_port(self):
        s = Settings()
        assert s.udp_port == 5005

    def test_default_api_port(self):
        s = Settings()
        assert s.api_port == 8000

    def test_udp_host_binds_all_interfaces(self):
        s = Settings()
        assert s.udp_host == "0.0.0.0"

    def test_default_hardware_profile(self):
        s = Settings()
        assert s.hardware_profile == DEFAULT_PROFILE
        assert DEFAULT_PROFILE in HARDWARE_PROFILES

    def test_hardware_profile_apply_updates_settings(self):
        s = Settings()
        s.hardware_profile = "esp32s3"
        profile = s.apply_hardware_profile()
        assert profile is not None
        assert s.num_subcarriers == 56
        assert s.csi_sample_rate == 20


# ═══════════════════════════════════════════════════════════
# 6. HTTP Endpoints — technician's installation flow
# ═══════════════════════════════════════════════════════════

class TestInstallationFlowEndpoints:
    """Technician uses dashboard to detect, flash, calibrate, and verify."""

    @pytest.mark.asyncio
    async def test_firmware_detect_endpoint(self, client):
        r = await client.get("/api/firmware/detect")
        assert r.status_code == 200
        d = r.json()
        assert "devices" in d
        assert "count" in d

    @pytest.mark.asyncio
    async def test_firmware_status_idle(self, client):
        r = await client.get("/api/firmware/status")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    @pytest.mark.asyncio
    async def test_wifi_config_endpoint(self, client):
        """Technician checks WiFi config before flashing."""
        r = await client.get("/api/network/wifi")
        assert r.status_code == 200
        d = r.json()
        assert "detected" in d

    @pytest.mark.asyncio
    async def test_calibration_full_cycle(self, client):
        """Start → check status → finish."""
        r = await client.post("/api/calibration/start", params={"mode": "spatial"})
        assert r.status_code == 200

        r = await client.get("/api/calibration/status")
        assert r.status_code == 200

        r = await client.post("/api/calibration/finish")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_ota_firmware_list(self, client):
        """Technician needs to see available firmware for OTA."""
        r = await client.get("/api/ota/firmware")
        assert r.status_code == 200
        assert "firmware" in r.json()

    @pytest.mark.asyncio
    async def test_status_shows_real_nodes(self, client):
        """After installation, status should reflect detected nodes."""
        r = await client.get("/api/status")
        d = r.json()
        assert "nodes" in d
        assert "pipeline_status" in d
        ps = d["pipeline_status"]
        assert "detected_nodes" in ps
        assert "real_nodes" in ps


# ═══════════════════════════════════════════════════════════
# 7. Signal Quality — on-site troubleshooting
# ═══════════════════════════════════════════════════════════

class TestSignalQualityFieldUse:
    """Technician moves nodes around and checks signal quality feedback."""

    def test_excellent_signal_near_node(self):
        """RSSI -40 dBm = node very close, clear line of sight."""
        nq = SignalQualityMonitor(emitter=EventEmitter())
        for i in range(10):
            nq.on_frame(make_csi_frame(node_id=1, rssi=-40, sequence=i))
        q = nq.get_quality()
        assert q["nodes"][0]["grade"] == "excellent"

    def test_poor_signal_through_walls(self):
        """RSSI -80 dBm = multiple walls, signal barely usable."""
        nq = SignalQualityMonitor(emitter=EventEmitter())
        for i in range(10):
            nq.on_frame(make_csi_frame(node_id=1, rssi=-80, sequence=i))
        q = nq.get_quality()
        assert q["nodes"][0]["grade"] == "poor"

    def test_tips_guide_technician(self):
        """Poor signal should produce actionable tips for the technician."""
        nq = SignalQualityMonitor(emitter=EventEmitter())
        for i in range(10):
            nq.on_frame(make_csi_frame(node_id=1, rssi=-80, sequence=i))
        q = nq.get_quality()
        tips = q["tips"]
        assert any("move closer" in tip or "obstacles" in tip for tip in tips)

    def test_good_tips_when_all_nodes_strong(self):
        """All signals excellent + stable CSI → positive feedback."""
        nq = SignalQualityMonitor(emitter=EventEmitter())
        stable_amp = np.ones(56, dtype=np.float32) * 50.0
        for nid in range(1, 3):
            for i in range(10):
                # Use stable amplitude to avoid "unstable CSI" tip
                nq.on_frame(make_csi_frame(
                    node_id=nid, rssi=-40, sequence=i,
                    amplitude=stable_amp + np.random.randn(56).astype(np.float32) * 0.001,
                ))
        q = nq.get_quality()
        assert any("good" in tip.lower() for tip in q["tips"])


# ═══════════════════════════════════════════════════════════
# 8. ADR-018 Frame — firmware protocol on the wire
# ═══════════════════════════════════════════════════════════

class TestFirmwareProtocolOnWire:
    """Technician's firmware sends UDP frames. Verify server can parse them."""

    def test_esp32s3_typical_frame(self):
        """ESP32-S3: 1 antenna, 56 subcarriers, 2.4GHz ch6."""
        data = make_csi_binary(
            node_id=1, n_antennas=1, n_sub=56,
            freq_mhz=2437, rssi=-55, noise_floor=-90,
        )
        frame = parse_csi_frame(data)
        assert frame is not None
        assert frame.node_id == 1
        assert frame.num_subcarriers == 56
        assert frame.channel == 6

    def test_esp32c6_wifi6_frame(self):
        """ESP32-C6 WiFi 6: 64 subcarriers."""
        data = make_csi_binary(n_antennas=1, n_sub=64, freq_mhz=2437)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert len(frame.amplitude) == 64

    def test_two_nodes_different_ids(self):
        """Two ESP32s on the same network, different node IDs."""
        f1 = parse_csi_frame(make_csi_binary(node_id=1, sequence=0))
        f2 = parse_csi_frame(make_csi_binary(node_id=2, sequence=0))
        assert f1.node_id != f2.node_id

    def test_sequence_number_increments(self):
        """Firmware sends incrementing sequence numbers."""
        frames = []
        for seq in range(5):
            f = parse_csi_frame(make_csi_binary(sequence=seq))
            frames.append(f)
        seqs = [f.sequence for f in frames]
        assert seqs == [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════
# 9. Quick Setup — technician's first-day installation flow
# ═══════════════════════════════════════════════════════════

class TestQuickSetupInstallationFlow:
    """Technician arrives on-site, opens the Hardware tab, and runs Quick Setup
    before doing spatial calibration.  These tests walk the exact same path.
    """

    @pytest.mark.asyncio
    async def test_defaults_match_typical_room(self, client):
        """Default 4×4×2.8 m is a typical bedroom — no guesswork for tech."""
        r = await client.get("/api/settings/quick")
        d = r.json()
        assert 2.0 <= d["room_width"] <= 8.0
        assert 2.0 <= d["room_depth"] <= 8.0
        assert 2.0 <= d["room_height"] <= 4.0

    @pytest.mark.asyncio
    async def test_profiles_listed_for_hardware_selection(self, client):
        """Technician picks the right ESP32 variant from the list.
        /api/settings/quick returns profiles as a list of string IDs.
        """
        r = await client.get("/api/settings/quick")
        profiles = r.json()["profiles"]  # list of str IDs, e.g. ["esp32s3", ...]
        assert "esp32s3" in profiles  # standard board in BOM

    @pytest.mark.asyncio
    async def test_scene_modes_listed_for_customer_use_case(self, client):
        """Tech sets scene based on customer: elderly home vs rehab clinic."""
        r = await client.get("/api/settings/quick")
        modes = r.json()["scene_modes"]
        assert "safety" in modes
        assert "fitness" in modes

    @pytest.mark.asyncio
    async def test_large_room_accepted(self, client, tmp_path, monkeypatch):
        """Hospital ward: 10×8×3.5 m — must be accepted, not rejected."""
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={
            "room_width": 10.0, "room_depth": 8.0, "room_height": 3.5,
        })
        d = r.json()
        assert d["applied"]["room_width"] == 10.0
        assert d["applied"]["room_depth"] == 8.0

    @pytest.mark.asyncio
    async def test_line_notify_token_saved(self, client, tmp_path, monkeypatch):
        """Tech enters LINE token for the nursing station group."""
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={
            "notify_line_token": "LINE_TOKEN_HOSPITAL_ICU",
        })
        assert r.json()["applied"]["notify_line_token"] == "LINE_TOKEN_HOSPITAL_ICU"

    @pytest.mark.asyncio
    async def test_telegram_bot_and_chat_id_saved(self, client, tmp_path, monkeypatch):
        """Tech enters Telegram credentials for on-call alert channel."""
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={
            "notify_telegram_bot_token": "7654321:AAB_TOKEN",
            "notify_telegram_chat_id": "-100999888",
        })
        applied = r.json()["applied"]
        assert "notify_telegram_bot_token" in applied
        assert "notify_telegram_chat_id" in applied

    @pytest.mark.asyncio
    async def test_fall_threshold_lower_for_elderly(self, client, tmp_path, monkeypatch):
        """Elderly ward: threshold 0.5 (more sensitive) vs default 0.6."""
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"fall_threshold": 0.5})
        assert r.json()["applied"]["fall_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_full_day1_installation_sequence(self, client, tmp_path, monkeypatch):
        """Complete day-1 checklist: configure → verify → calibrate → done."""
        monkeypatch.chdir(tmp_path)

        # Step 1 — Configure room for a 3×4 ward
        r = await client.post("/api/settings/quick", json={
            "room_width": 3.0, "room_depth": 4.0,
            "room_height": 2.8, "scene_mode": "safety",
        })
        assert r.json()["status"] == "saved"

        # Step 2 — Re-open Quick Setup, verify settings round-tripped
        g = await client.get("/api/settings/quick")
        d = g.json()
        assert d["room_width"] == 3.0
        assert d["scene_mode"] == "safety"

        # Step 3 — Spatial calibration (nodes already mounted)
        r = await client.post("/api/calibration/start", params={"mode": "spatial"})
        assert r.status_code == 200

        # Step 4 — Finish calibration
        r = await client.post("/api/calibration/finish")
        assert r.status_code == 200

        # Step 5 — Confirm calibration completed
        r = await client.get("/api/status")
        assert r.json()["calibration"]["status"] == "complete"
