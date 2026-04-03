"""Business District — core revenue paths that must never break.

Tests the main HTTP routes, SignalQualityMonitor grading, PipelineService
scene modes, and strategy auto-detection.
"""
import asyncio
import time

import pytest
import numpy as np
from unittest.mock import patch

from server.config import Settings, HARDWARE_PROFILES
from server.services.event_emitter import EventEmitter
from server.services.pipeline_service import (
    PipelineService, SCENE_MODES, _strategy_for_nodes,
)
from server.services.signal_quality import (
    SignalQualityMonitor, NodeQuality,
    RSSI_EXCELLENT, RSSI_GOOD, RSSI_POOR, CAPABILITY_TABLE,
)
from tests.conftest import make_csi_frame


# ═══════════════════════════════════════════════════════════
# 1. HTTP Route Integration Tests
# ═══════════════════════════════════════════════════════════

class TestRouteSystem:
    """System routes: /, /api/status, /api/profiles, /api/joints, /api/vitals, /api/alerts."""

    @pytest.mark.asyncio
    async def test_root(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "wifi-body"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_status_has_all_fields(self, client):
        r = await client.get("/api/status")
        assert r.status_code == 200
        d = r.json()
        for key in [
            "nodes", "is_fallen", "current_activity", "fall_alerts",
            "vitals", "person_count", "pipeline_status", "node_positions",
            "room_dimensions", "storage", "calibration",
            "notifications_enabled", "scene_mode",
        ]:
            assert key in d, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_status_pipeline_fields(self, client):
        r = await client.get("/api/status")
        ps = r.json()["pipeline_status"]
        for key in ["csi_receiver", "model_loaded", "csi_frames_received",
                     "hardware_profile", "is_simulating", "detected_nodes",
                     "real_nodes", "strategy", "strategy_description"]:
            assert key in ps, f"Missing pipeline_status.{key}"

    @pytest.mark.asyncio
    async def test_profiles_lists_all(self, client):
        r = await client.get("/api/profiles")
        assert r.status_code == 200
        data = r.json()
        assert "profiles" in data
        assert "active" in data
        profile_ids = {p["id"] for p in data["profiles"]}
        for pid in HARDWARE_PROFILES:
            assert pid in profile_ids

    @pytest.mark.asyncio
    async def test_profiles_has_model_ready(self, client):
        r = await client.get("/api/profiles")
        for p in r.json()["profiles"]:
            assert "model_ready" in p
            assert isinstance(p["model_ready"], bool)

    @pytest.mark.asyncio
    async def test_joints_empty_initially(self, client):
        r = await client.get("/api/joints")
        assert r.status_code == 200
        assert r.json()["joints"] is None

    @pytest.mark.asyncio
    async def test_vitals_structure(self, client):
        r = await client.get("/api/vitals")
        assert r.status_code == 200
        d = r.json()
        assert "primary" in d
        assert "persons" in d

    @pytest.mark.asyncio
    async def test_alerts_empty_initially(self, client):
        r = await client.get("/api/alerts")
        assert r.status_code == 200
        assert r.json()["alerts"] == []

    @pytest.mark.asyncio
    async def test_persons_empty_initially(self, client):
        r = await client.get("/api/persons")
        assert r.status_code == 200
        d = r.json()
        assert d["count"] == 0
        assert d["persons"] == []

    @pytest.mark.asyncio
    async def test_signal_quality_endpoint(self, client):
        r = await client.get("/api/signal-quality")
        assert r.status_code == 200
        d = r.json()
        assert "nodes" in d
        assert "grade" in d


class TestRouteSceneMode:
    """Scene mode switch: POST /api/system/scene."""

    @pytest.mark.asyncio
    async def test_get_scene(self, client):
        r = await client.get("/api/system/scene")
        assert r.status_code == 200
        d = r.json()
        assert d["scene_mode"] in SCENE_MODES

    @pytest.mark.asyncio
    async def test_set_scene_safety(self, client):
        r = await client.post("/api/system/scene", params={"scene": "safety"})
        assert r.status_code == 200
        assert r.json()["scene_mode"] == "safety"

    @pytest.mark.asyncio
    async def test_set_scene_fitness(self, client):
        r = await client.post("/api/system/scene", params={"scene": "fitness"})
        assert r.status_code == 200
        assert r.json()["scene_mode"] == "fitness"

    @pytest.mark.asyncio
    async def test_set_scene_invalid(self, client):
        r = await client.post("/api/system/scene", params={"scene": "party"})
        assert r.status_code == 400


class TestRouteSystemMode:
    """POST /api/system/mode — simulation/real toggle."""

    @pytest.mark.asyncio
    async def test_set_mode_simulation(self, client):
        r = await client.post("/api/system/mode", params={"mode": "simulation"})
        assert r.status_code == 200
        assert r.json()["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_set_mode_real(self, client):
        r = await client.post("/api/system/mode", params={"mode": "real"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_set_mode_invalid(self, client):
        r = await client.post("/api/system/mode", params={"mode": "magic"})
        assert r.status_code == 400
        assert "error" in r.json()

    @pytest.mark.asyncio
    async def test_set_mode_unchanged(self, client, container):
        """Setting same mode returns 'unchanged'."""
        container.settings.simulate = False
        r = await client.post("/api/system/mode", params={"mode": "real"})
        assert r.json()["status"] == "unchanged"


class TestRouteCalibration:
    """Calibration routes: start, status, finish."""

    @pytest.mark.asyncio
    async def test_calibration_status(self, client):
        r = await client.get("/api/calibration/status")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_calibration_start_and_finish(self, client):
        r = await client.post("/api/calibration/start", params={"mode": "spatial"})
        assert r.status_code == 200
        r2 = await client.post("/api/calibration/finish")
        assert r2.status_code == 200

    @pytest.mark.asyncio
    async def test_calibration_double_start(self, client):
        """Starting calibration twice should fail."""
        await client.post("/api/calibration/start")
        r = await client.post("/api/calibration/start")
        assert r.status_code == 409


class TestRouteData:
    """Data routes: history, OTA, notifications."""

    @pytest.mark.asyncio
    async def test_history_poses(self, client):
        r = await client.get("/api/history/poses")
        assert r.status_code == 200
        assert "poses" in r.json()

    @pytest.mark.asyncio
    async def test_history_vitals(self, client):
        r = await client.get("/api/history/vitals")
        assert r.status_code == 200
        assert "vitals" in r.json()

    @pytest.mark.asyncio
    async def test_history_alerts(self, client):
        r = await client.get("/api/history/alerts")
        assert r.status_code == 200
        assert "alerts" in r.json()

    @pytest.mark.asyncio
    async def test_notification_status(self, client):
        r = await client.get("/api/notifications/status")
        assert r.status_code == 200
        d = r.json()
        assert "enabled" in d

    @pytest.mark.asyncio
    async def test_ota_firmware_list(self, client):
        r = await client.get("/api/ota/firmware")
        assert r.status_code == 200
        assert "firmware" in r.json()

    @pytest.mark.asyncio
    async def test_ota_download_not_found(self, client):
        r = await client.get("/api/ota/download/nonexistent.bin")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_models_list(self, client):
        r = await client.get("/api/models")
        assert r.status_code == 200
        d = r.json()
        assert "models" in d
        assert "active" in d


# ═══════════════════════════════════════════════════════════
# 2. SignalQualityMonitor — the system's health dashboard
# ═══════════════════════════════════════════════════════════

class TestSignalQualityMonitor:
    """Signal grading is the dashboard's primary health indicator."""

    def test_grade_excellent(self):
        nq = NodeQuality(node_id=1)
        for _ in range(10):
            nq.rssi_history.append(-40)
            nq.noise_history.append(-90)
        assert nq.grade == "excellent"
        assert nq.snr == 50.0

    def test_grade_good(self):
        nq = NodeQuality(node_id=1)
        for _ in range(10):
            nq.rssi_history.append(-60)
        assert nq.grade == "good"

    def test_grade_fair(self):
        nq = NodeQuality(node_id=1)
        for _ in range(10):
            nq.rssi_history.append(-70)
        assert nq.grade == "fair"

    def test_grade_poor(self):
        nq = NodeQuality(node_id=1)
        for _ in range(10):
            nq.rssi_history.append(-80)
        assert nq.grade == "poor"

    def test_capabilities_degrade_with_signal(self):
        """Excellent sees all; poor sees only presence."""
        excellent = CAPABILITY_TABLE["excellent"]["capabilities"]
        poor = CAPABILITY_TABLE["poor"]["capabilities"]
        assert "heart_rate" in excellent
        assert "heart_rate" not in poor
        assert "presence" in poor

    def test_to_dict_structure(self):
        nq = NodeQuality(node_id=2)
        for _ in range(10):
            nq.rssi_history.append(-55)
            nq.noise_history.append(-90)
        nq.frame_count = 42
        d = nq.to_dict()
        assert d["node_id"] == 2
        assert d["frames"] == 42
        assert "grade" in d
        assert "capabilities" in d

    def test_csi_stability_insufficient_data(self):
        """Stability is 0.0 when fewer than 5 samples."""
        nq = NodeQuality(node_id=1)
        nq.csi_var_history.append(0.01)
        assert nq.csi_stability == 0.0

    def test_csi_stability_stable_environment(self):
        """Very low variance → stability near 1.0."""
        nq = NodeQuality(node_id=1)
        for _ in range(10):
            nq.csi_var_history.append(0.005)
        assert nq.csi_stability >= 0.95

    def test_csi_stability_chaotic_environment(self):
        """High variance → stability near 0.0."""
        nq = NodeQuality(node_id=1)
        for _ in range(10):
            nq.csi_var_history.append(1.0)
        assert nq.csi_stability == 0.0

    def test_on_frame_creates_node(self):
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        frame = make_csi_frame(node_id=3, rssi=-55)
        monitor.on_frame(frame)
        q = monitor.get_quality()
        assert len(q["nodes"]) == 1
        assert q["nodes"][0]["node_id"] == 3

    def test_on_frame_updates_frame_count(self):
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        for i in range(5):
            monitor.on_frame(make_csi_frame(sequence=i))
        assert monitor._nodes[1].frame_count == 5

    def test_overall_grade_weakest_link(self):
        """Overall grade is the worst node's grade."""
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        # Node 1: excellent
        for i in range(10):
            monitor.on_frame(make_csi_frame(node_id=1, rssi=-40, sequence=i))
        # Node 2: poor
        for i in range(10):
            monitor.on_frame(make_csi_frame(node_id=2, rssi=-80, sequence=i))
        q = monitor.get_quality()
        assert q["grade"] == "poor"

    def test_overall_grade_no_nodes(self):
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        q = monitor.get_quality()
        assert q["grade"] == "poor"
        assert "Connect ESP32" in q["tips"][0]

    def test_tips_generated_for_weak_signal(self):
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        for i in range(10):
            monitor.on_frame(make_csi_frame(node_id=1, rssi=-80, sequence=i))
        q = monitor.get_quality()
        assert any("too weak" in tip for tip in q["tips"])

    def test_throttled_emit(self):
        """Events are throttled by emit_interval."""
        emitter = EventEmitter()
        emitted = []
        async def on_sq(data): emitted.append(data)
        emitter.on("signal_quality", on_sq)
        monitor = SignalQualityMonitor(emitter=emitter, emit_interval=0.0)
        monitor.on_frame(make_csi_frame())
        # With interval=0 it should emit immediately
        assert monitor._last_emit > 0


# ═══════════════════════════════════════════════════════════
# 3. PipelineService — scene modes & strategy auto-detection
# ═══════════════════════════════════════════════════════════

class TestPipelineSceneMode:
    """Scene mode directly controls fall detection sensitivity — business-critical."""

    def test_default_scene_is_safety(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        assert svc.scene_mode == "safety"

    def test_switch_to_fitness(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        result = svc.set_scene_mode("fitness")
        assert result["scene_mode"] == "fitness"
        assert svc.fall_detector.threshold == 0.95  # very strict during exercise

    def test_switch_to_safety(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        svc.set_scene_mode("fitness")
        result = svc.set_scene_mode("safety")
        assert svc.fall_detector.threshold == 0.6  # more sensitive for elderly

    def test_invalid_scene(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        result = svc.set_scene_mode("disco")
        assert "error" in result


class TestPipelineStrategyAutoDetect:
    """Node count → strategy mapping ensures correct capability level."""

    def test_strategy_1_node(self):
        name, _ = _strategy_for_nodes(1)
        assert name == "basic"

    def test_strategy_2_nodes(self):
        name, _ = _strategy_for_nodes(2)
        assert name == "basic"

    def test_strategy_3_nodes(self):
        name, _ = _strategy_for_nodes(3)
        assert name == "spatial"

    def test_strategy_5_nodes(self):
        name, _ = _strategy_for_nodes(5)
        assert name == "multiview"

    def test_strategy_fallback(self):
        name, _ = _strategy_for_nodes(99)
        assert name == "basic"

    def test_auto_detect_on_frames(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        for nid in range(1, 4):
            svc.on_frame(make_csi_frame(node_id=nid))
        assert svc.detected_nodes == 3
        assert svc.strategy == "spatial"

    def test_auto_detect_locks_after_window(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        svc._DETECTION_WINDOW_SEC = 0.0  # immediate lock
        svc.on_frame(make_csi_frame(node_id=1))
        assert svc._detection_locked is True

    def test_real_node_tracking(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        svc.on_frame(make_csi_frame(node_id=1), simulated=False)
        svc.on_frame(make_csi_frame(node_id=2), simulated=True)
        assert svc.real_node_count == 1


class TestPipelineJointConfidence:
    """Joint confidence varies by body region and signal quality."""

    def test_default_confidence_medium(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        conf = svc._compute_joint_confidence()
        assert len(conf) == 24
        assert all(c == 0.5 for c in conf)  # no quality data

    def test_confidence_with_quality_monitor(self):
        emitter = EventEmitter()
        svc = PipelineService(settings=Settings(), emitter=emitter)
        monitor = SignalQualityMonitor(emitter=emitter)
        svc.set_quality_monitor(monitor)
        # Feed some excellent signal
        for i in range(10):
            monitor.on_frame(make_csi_frame(rssi=-40, sequence=i))
        conf = svc._compute_joint_confidence()
        # Torso joints (10-11) should have highest confidence
        assert conf[10] >= conf[14]  # torso >= legs
