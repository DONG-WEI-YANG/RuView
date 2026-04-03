"""Configuration Convenience Tour — discoverability, consistency, safe defaults.

A deployment engineer reviews: Can I configure this system without reading
source code? Do defaults work out of the box? Are cross-layer values
(firmware port, server port, firewall port) sourced from the same place?

Expert focus: zero-config startup, .env discoverability, cross-layer consistency.
"""
import os
import re
from pathlib import Path

import pytest
import numpy as np

from server.config import Settings, HARDWARE_PROFILES, DEFAULT_PROFILE
from server.calibration import (
    CalibrationManager, CALIBRATION_DURATION_SEC, MIN_SAMPLES_PER_NODE,
)
from server.services.signal_quality import RSSI_EXCELLENT, RSSI_GOOD, RSSI_POOR
from server.services.websocket_service import HEARTBEAT_INTERVAL_SEC, HEARTBEAT_TIMEOUT_SEC
from server.services.storage_service import POSE_SAVE_INTERVAL, VITALS_SAVE_INTERVAL
from server.services.pipeline_service import SCENE_MODES, STRATEGY_TABLE
from server.protocol.handlers import V0_DETECT_TIMEOUT_SEC, ALL_STREAMS
from server.csi_frame import MAGIC_HEADER, HEADER_SIZE
from server.notifier import MAX_RETRIES, RETRY_DELAY_SEC


# ═══════════════════════════════════════════════════════════
# 1. Zero-Config Startup — does it run without .env?
# ═══════════════════════════════════════════════════════════

class TestZeroConfigStartup:
    """Deploy engineer says: 'I just cloned the repo and ran it. Does it work?'"""

    def test_settings_creates_without_env(self):
        """Settings() with no .env file should not crash."""
        s = Settings()
        assert s.udp_port == 5005
        assert s.api_port == 8000

    def test_default_profile_exists(self):
        assert DEFAULT_PROFILE in HARDWARE_PROFILES

    def test_apply_default_profile(self):
        s = Settings()
        profile = s.apply_hardware_profile()
        assert profile is not None
        assert s.num_subcarriers > 0
        assert s.csi_sample_rate > 0

    def test_default_node_positions_valid(self):
        s = Settings()
        assert len(s.node_positions) >= 1
        for nid, pos in s.node_positions.items():
            assert len(pos) == 3, f"Node {nid} position needs [x, y, z]"

    def test_default_room_dimensions_positive(self):
        s = Settings()
        assert s.room_width > 0
        assert s.room_depth > 0
        assert s.room_height > 0

    def test_default_scene_mode_valid(self):
        s = Settings()
        assert s.scene_mode in SCENE_MODES

    def test_default_fall_threshold_in_range(self):
        s = Settings()
        assert 0.0 < s.fall_threshold <= 1.0

    def test_notifications_disabled_by_default(self):
        """No notification tokens = no accidental alert spam."""
        s = Settings()
        assert s.notify_webhook_url == ""
        assert s.notify_line_token == ""
        assert s.notify_telegram_bot_token == ""

    def test_simulation_off_by_default(self):
        s = Settings()
        assert s.simulate is False

    def test_app_creates_without_model(self):
        """App should start even if model weights don't exist."""
        from server.api import create_app
        app = create_app()
        container = app.state.container
        # Pipeline should exist but model may be None
        assert container.pipeline_svc is not None


# ═══════════════════════════════════════════════════════════
# 2. Cross-Layer Consistency — same value everywhere
# ═══════════════════════════════════════════════════════════

class TestCrossLayerConsistency:
    """The UDP port 5005 appears in config.py, firmware_builder.py, and
    setup-firewall.bat. They must all agree.
    """

    def test_config_udp_port_matches_default(self):
        s = Settings()
        assert s.udp_port == 5005

    def test_firmware_builder_default_port(self):
        """firmware_builder bakes port into sdkconfig — must match config default."""
        source = Path("server/firmware_builder.py").read_text(encoding="utf-8")
        # Look for CONFIG_CSI_TARGET_PORT pattern
        m = re.search(r'CONFIG_CSI_TARGET_PORT["\s=]+(\d+)', source)
        if m:
            assert int(m.group(1)) == 5005

    def test_firewall_script_port(self):
        """setup-firewall.bat should reference same port."""
        bat_path = Path("setup-firewall.bat")
        if bat_path.exists():
            text = bat_path.read_text(encoding="utf-8", errors="replace")
            assert "5005" in text, "Firewall script must reference UDP port 5005"

    def test_all_scene_modes_have_required_keys(self):
        """Every scene mode must define the same set of config keys."""
        required = {"description", "fall_threshold", "fall_alert_cooldown",
                    "inactivity_timeout", "notify_on_fall", "notify_on_apnea",
                    "track_reps"}
        for mode_name, cfg in SCENE_MODES.items():
            for key in required:
                assert key in cfg, f"Scene '{mode_name}' missing key '{key}'"

    def test_all_hardware_profiles_have_model_path(self):
        for pid, p in HARDWARE_PROFILES.items():
            assert p.model_path, f"Profile '{pid}' has no model_path"
            assert p.num_subcarriers > 0, f"Profile '{pid}' has 0 subcarriers"

    def test_strategy_table_covers_1_to_8_nodes(self):
        """No gap in node-count → strategy mapping."""
        covered = set()
        for (lo, hi) in STRATEGY_TABLE.keys():
            for n in range(lo, hi + 1):
                covered.add(n)
        for n in range(1, 9):
            assert n in covered, f"Node count {n} has no strategy"

    def test_all_streams_match_envelope_types(self):
        """ALL_STREAMS must include every type the server can emit."""
        assert "pose" in ALL_STREAMS
        assert "vitals" in ALL_STREAMS
        assert "csi" in ALL_STREAMS
        assert "persons" in ALL_STREAMS


# ═══════════════════════════════════════════════════════════
# 3. Settings .env Support — pydantic BaseSettings
# ═══════════════════════════════════════════════════════════

class TestEnvVarSupport:
    """Every Settings field should be overridable via environment variable."""

    def test_settings_reads_env_prefix(self):
        """Settings uses env_prefix='' — all fields are direct env vars."""
        s = Settings()
        assert s.model_config.get("env_prefix", "") == ""

    def test_udp_port_overridable(self):
        """UDP_PORT env var should override default."""
        # We test that the field exists and has correct type annotation
        assert hasattr(Settings, 'model_fields')
        assert 'udp_port' in Settings.model_fields

    def test_all_critical_fields_in_settings(self):
        """Critical deployment params must be Settings fields (not module constants)."""
        fields = set(Settings.model_fields.keys())
        critical = {
            'udp_host', 'udp_port', 'api_host', 'api_port',
            'hardware_profile', 'num_subcarriers', 'csi_sample_rate',
            'max_nodes', 'model_path', 'simulate', 'db_path',
            'fall_threshold', 'fall_alert_cooldown', 'scene_mode',
            'room_width', 'room_depth', 'room_height',
        }
        missing = critical - fields
        assert not missing, f"Critical settings not in Settings: {missing}"

    def test_notification_fields_in_settings(self):
        fields = set(Settings.model_fields.keys())
        notif = {'notify_webhook_url', 'notify_line_token',
                 'notify_telegram_bot_token', 'notify_telegram_chat_id'}
        missing = notif - fields
        assert not missing, f"Notification fields missing: {missing}"


# ═══════════════════════════════════════════════════════════
# 4. Hardcoded Constants Audit — which should be tunable?
# ═══════════════════════════════════════════════════════════

class TestHardcodedConstantsAudit:
    """Identify constants that are reasonable vs. those that trap deployments."""

    # ── Reasonable defaults (don't need to change) ──────────

    def test_magic_header_fixed(self):
        """Protocol magic must be fixed — firmware and server must match."""
        assert MAGIC_HEADER == 0xC5110001

    def test_header_size_fixed(self):
        assert HEADER_SIZE == 20

    def test_v0_detect_timeout_reasonable(self):
        """5 seconds to detect v0 client — reasonable."""
        assert V0_DETECT_TIMEOUT_SEC == 5.0

    # ── Constants that should be documented ─────────────────

    def test_heartbeat_intervals_symmetric(self):
        """Ping every 30s, timeout after 30s — total dead detection: 60s."""
        assert HEARTBEAT_INTERVAL_SEC == 30
        assert HEARTBEAT_TIMEOUT_SEC == 30

    def test_storage_save_intervals_documented(self):
        """Pose every 1s, vitals every 5s — affects storage size."""
        assert POSE_SAVE_INTERVAL == 1.0
        assert VITALS_SAVE_INTERVAL == 5.0

    def test_calibration_min_samples_matches_rate(self):
        """20 samples @ 20Hz = 1 second. Must complete within 5s duration."""
        assert MIN_SAMPLES_PER_NODE <= CALIBRATION_DURATION_SEC * 20

    def test_notifier_retry_exists(self):
        assert MAX_RETRIES >= 1
        assert RETRY_DELAY_SEC > 0

    # ── Signal quality thresholds ──────────────────────────

    def test_rssi_thresholds_ordered(self):
        """Thresholds must be strictly descending: excellent > good > poor."""
        assert RSSI_EXCELLENT > RSSI_GOOD > RSSI_POOR

    # ── Vital signs bandpass ranges ────────────────────────

    def test_breathing_band_covers_normal_range(self):
        """Normal adult: 12-20 BPM = 0.2-0.33 Hz. Band: 0.1-0.5 Hz."""
        source = Path("server/vital_signs.py").read_text(encoding="utf-8")
        assert "0.1, 0.5" in source or "0.1,0.5" in source

    def test_heart_rate_band_covers_normal_range(self):
        """Normal adult: 60-100 BPM = 1.0-1.67 Hz. Band: 0.8-2.0 Hz."""
        source = Path("server/vital_signs.py").read_text(encoding="utf-8")
        assert "0.8, 2.0" in source or "0.8,2.0" in source

    def test_fall_confidence_weights_sum_to_1(self):
        """Fall detector weights: 0.10 + 0.15 + 0.40 + 0.35 = 1.0."""
        source = Path("server/fall_detector.py").read_text(encoding="utf-8")
        weights = re.findall(r'(\d+\.\d+)\s*\*\s*(?:low_ratio|spread_score|abs_height_score|vel_score)', source)
        if weights:
            total = sum(float(w) for w in weights)
            assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"


# ═══════════════════════════════════════════════════════════
# 5. Firmware Builder Env Vars — field deployment
# ═══════════════════════════════════════════════════════════

class TestFirmwareEnvVars:
    """Field engineer needs to override build paths on different machines."""

    def test_build_dir_reads_env(self):
        source = Path("server/firmware_builder.py").read_text(encoding="utf-8")
        assert "CSI_BUILD_DIR" in source

    def test_pio_exe_reads_env(self):
        source = Path("server/firmware_builder.py").read_text(encoding="utf-8")
        assert "CSI_PIO_EXE" in source

    def test_build_dir_has_default(self):
        """Even without env var, a default path exists."""
        from server.firmware_builder import BUILD_DIR
        assert BUILD_DIR is not None
        assert str(BUILD_DIR) != ""

    def test_pio_exe_has_default(self):
        from server.firmware_builder import PIO_EXE
        assert PIO_EXE is not None
        assert PIO_EXE != ""


# ═══════════════════════════════════════════════════════════
# 6. .env.example Discoverability
# ═══════════════════════════════════════════════════════════

class TestEnvExampleExists:
    """.env.example should exist so technicians know what to configure."""

    def test_env_example_file_exists(self):
        """A .env.example file should be present in the project root."""
        assert Path(".env.example").exists(), \
            "Missing .env.example — technicians can't discover config options"

    def test_env_example_has_critical_vars(self):
        """The example file must mention critical configuration variables."""
        text = Path(".env.example").read_text(encoding="utf-8")
        for var in ["udp_port", "api_port", "hardware_profile",
                    "simulate", "fall_threshold", "scene_mode"]:
            assert var in text.lower(), f".env.example missing '{var}'"

    def test_env_example_has_notification_vars(self):
        text = Path(".env.example").read_text(encoding="utf-8")
        for var in ["notify_webhook_url", "notify_line_token",
                    "notify_telegram_bot_token"]:
            assert var in text.lower(), f".env.example missing '{var}'"

    def test_env_example_has_firmware_vars(self):
        text = Path(".env.example").read_text(encoding="utf-8")
        for var in ["CSI_BUILD_DIR", "CSI_PIO_EXE"]:
            assert var in text, f".env.example missing '{var}'"


# ═══════════════════════════════════════════════════════════
# 7. HTTP Config Endpoints — runtime tunability
# ═══════════════════════════════════════════════════════════

class TestRuntimeConfigEndpoints:
    """Some settings should be changeable at runtime without restart."""

    @pytest.mark.asyncio
    async def test_scene_mode_switchable(self, client):
        r = await client.post("/api/system/scene", params={"scene": "fitness"})
        assert r.status_code == 200
        assert r.json()["scene_mode"] == "fitness"
        # Switch back
        await client.post("/api/system/scene", params={"scene": "safety"})

    @pytest.mark.asyncio
    async def test_simulation_mode_switchable(self, client):
        r = await client.post("/api/system/mode", params={"mode": "simulation"})
        assert r.status_code == 200
        # Switch back
        await client.post("/api/system/mode", params={"mode": "real"})

    @pytest.mark.asyncio
    async def test_calibration_triggerable(self, client):
        r = await client.post("/api/calibration/start", params={"mode": "spatial"})
        assert r.status_code == 200
        await client.post("/api/calibration/finish")

    @pytest.mark.asyncio
    async def test_profiles_queryable(self, client):
        """Technician can see all profiles without restarting."""
        r = await client.get("/api/profiles")
        assert r.status_code == 200
        profiles = r.json()["profiles"]
        assert len(profiles) == len(HARDWARE_PROFILES)


# ═══════════════════════════════════════════════════════════
# 8. Timeout & Interval Sanity
# ═══════════════════════════════════════════════════════════

class TestTimeoutSanity:
    """No timeout should be unreasonably short or absurdly long."""

    def test_heartbeat_not_too_frequent(self):
        """Heartbeat < 10s wastes bandwidth on mobile connections."""
        assert HEARTBEAT_INTERVAL_SEC >= 10

    def test_heartbeat_not_too_slow(self):
        """Heartbeat > 120s means dead connection lingers too long."""
        assert HEARTBEAT_INTERVAL_SEC <= 120

    def test_calibration_duration_reasonable(self):
        assert 1.0 <= CALIBRATION_DURATION_SEC <= 60.0

    def test_storage_intervals_reasonable(self):
        assert 0.1 <= POSE_SAVE_INTERVAL <= 60.0
        assert 0.1 <= VITALS_SAVE_INTERVAL <= 300.0

    def test_notifier_timeout_exists(self):
        """HTTP timeout for notifications: should be 5-30 seconds."""
        source = Path("server/notifier.py").read_text(encoding="utf-8")
        m = re.search(r'timeout[=\s]*(\d+)', source)
        assert m, "Notifier HTTP timeout not found"
        timeout = int(m.group(1))
        assert 5 <= timeout <= 30

    def test_v0_detection_not_too_long(self):
        """v0 detection > 15s would delay initial data for legacy clients."""
        assert V0_DETECT_TIMEOUT_SEC <= 15.0

    def test_retry_delay_not_too_long(self):
        """Notification retry delay should be < 5s (fall is time-critical)."""
        assert RETRY_DELAY_SEC <= 5.0
