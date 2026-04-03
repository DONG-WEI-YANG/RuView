"""Historic District — regression tests for refactored architecture.

After the 2026-03-21 refactoring (monolith → Service Layer + DI + EventBus),
these tests verify that the wiring still works and nothing silently broke.
"""
import asyncio
import json
import time

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.container import ServiceContainer
from server.services.websocket_service import WebSocketService
from server.services.pipeline_service import PipelineService
from server.services.vitals_service import VitalsService
from server.services.storage_service import StorageService
from server.services.signal_quality import SignalQualityMonitor
from server.protocol.envelope import (
    make_envelope, parse_client_message, Envelope,
    PoseData, VitalsData, CsiData, PersonsData, PersonData,
    HelloMessage, PongMessage,
)
from server.protocol.v0_adapter import v1_to_v0, v0_to_v1_parts
from server.protocol.handlers import (
    ConnectionState, handle_client_message, ALL_STREAMS, V0_DETECT_TIMEOUT_SEC,
)
from tests.conftest import make_csi_frame


# ═══════════════════════════════════════════════════════════
# 1. ServiceContainer DI Wiring — the refactoring's core risk
# ═══════════════════════════════════════════════════════════

class TestServiceContainerWiring:
    """Verify all services are constructed and event subscriptions are active."""

    def test_container_creates_all_services(self):
        settings = Settings()
        container = ServiceContainer(settings=settings)
        assert container.websocket is not None
        assert container.pipeline_svc is not None
        assert container.vitals is not None
        assert container.calibration is not None
        assert container.storage is not None
        assert container.notification is not None
        assert container.signal_quality is not None

    def test_container_emitter_shared(self):
        """All services share the same EventEmitter instance."""
        container = ServiceContainer(settings=Settings())
        assert container.websocket._emitter is container.emitter
        assert container.pipeline_svc._emitter is container.emitter
        assert container.vitals._emitter is container.emitter

    def test_quality_monitor_wired_to_pipeline(self):
        """PipelineService should have the SignalQualityMonitor set."""
        container = ServiceContainer(settings=Settings())
        assert container.pipeline_svc._quality_monitor is container.signal_quality

    def test_calibration_manager_injected_into_settings(self):
        settings = Settings()
        container = ServiceContainer(settings=settings)
        assert settings.calibration_manager is container.calibration.manager

    def test_vitals_extractor_injected_into_settings(self):
        settings = Settings()
        container = ServiceContainer(settings=settings)
        assert settings.vitals_extractor is container.vitals.extractor

    @pytest.mark.asyncio
    async def test_pose_event_reaches_websocket(self):
        """emitter.emit('pose') → container handler → ws.broadcast_envelope()."""
        container = ServiceContainer(settings=Settings())
        mock_ws = AsyncMock()
        conn = container.websocket.register(mock_ws)
        # Force v0 so it receives everything without hello
        conn.connected_at = 0

        await container.emitter.emit("pose", {
            "joints": [[0.0, 1.5, 0.0]] * 24,
            "confidence": 0.8,
        })
        await asyncio.sleep(0.1)

        # v0 client should have received a broadcast
        assert mock_ws.send_text.called
        payload = json.loads(mock_ws.send_text.call_args[0][0])
        assert "joints" in payload

    @pytest.mark.asyncio
    async def test_vitals_event_reaches_websocket(self):
        container = ServiceContainer(settings=Settings())
        mock_ws = AsyncMock()
        conn = container.websocket.register(mock_ws)
        conn.connected_at = 0

        await container.emitter.emit("vitals", {
            "heart_bpm": 72.0,
            "breathing_bpm": 16.0,
        })
        await asyncio.sleep(0.1)

        assert mock_ws.send_text.called
        payload = json.loads(mock_ws.send_text.call_args[0][0])
        assert "vitals" in payload

    @pytest.mark.asyncio
    async def test_csi_event_reaches_websocket(self):
        container = ServiceContainer(settings=Settings())
        mock_ws = AsyncMock()
        conn = container.websocket.register(mock_ws)
        conn.connected_at = 0

        await container.emitter.emit("csi", {
            "amplitudes": [1.0, 2.0, 3.0],
        })
        await asyncio.sleep(0.1)

        assert mock_ws.send_text.called

    @pytest.mark.asyncio
    async def test_persons_event_reaches_websocket(self):
        container = ServiceContainer(settings=Settings())
        mock_ws = AsyncMock()
        conn = container.websocket.register(mock_ws)
        conn.connected_at = 0

        await container.emitter.emit("persons", {
            "persons": [{"id": 1, "joints": [], "vitals": {}}],
            "count": 1,
        })
        await asyncio.sleep(0.1)

        assert mock_ws.send_text.called

    @pytest.mark.asyncio
    async def test_storage_subscribes_to_pose(self):
        """StorageService auto-subscribes to 'pose' events in constructor."""
        emitter = EventEmitter()
        storage_svc = StorageService(db_path=":memory:", emitter=emitter)
        # Check that the subscription exists
        assert "pose" in emitter._handlers
        assert len(emitter._handlers["pose"]) >= 1

    @pytest.mark.asyncio
    async def test_storage_subscribes_to_vitals(self):
        emitter = EventEmitter()
        storage_svc = StorageService(db_path=":memory:", emitter=emitter)
        assert "vitals" in emitter._handlers
        assert len(emitter._handlers["vitals"]) >= 1


# ═══════════════════════════════════════════════════════════
# 2. v0/v1 Protocol — backward compatibility regression
# ═══════════════════════════════════════════════════════════

class TestV0AdapterRegression:
    """v0 dashboard must keep working after v1 refactoring."""

    def test_pose_v1_to_v0(self):
        env = make_envelope("pose", PoseData(
            joints=[[1.0, 2.0, 3.0]] * 24, confidence=0.9,
        ))
        v0 = v1_to_v0(env)
        assert "joints" in v0
        assert len(v0["joints"]) == 24
        assert v0["joints"][0] == [1.0, 2.0, 3.0]

    def test_vitals_v1_to_v0(self):
        env = make_envelope("vitals", VitalsData(
            heart_bpm=72.0, breathing_bpm=16.0,
        ))
        v0 = v1_to_v0(env)
        assert "vitals" in v0
        assert v0["vitals"]["heart_bpm"] == 72.0

    def test_csi_v1_to_v0(self):
        env = make_envelope("csi", CsiData(amplitudes=[1.0, 2.0]))
        v0 = v1_to_v0(env)
        assert "csi_amplitudes" in v0
        assert v0["csi_amplitudes"] == [1.0, 2.0]

    def test_persons_v1_to_v0(self):
        env = make_envelope("persons", PersonsData(
            persons=[PersonData(id=1, joints=[[0.0]*3]*24)],
            count=1,
        ))
        v0 = v1_to_v0(env)
        assert "persons" in v0
        assert "person_count" in v0
        assert v0["person_count"] == 1

    def test_unknown_type_v1_to_v0(self):
        """Unknown envelope types produce empty v0 payload."""
        from server.protocol.envelope import Envelope, StatusData
        env = make_envelope("status", StatusData(model_loaded=True))
        v0 = v1_to_v0(env)
        assert v0 == {}

    def test_v0_to_v1_pose(self):
        parts = v0_to_v1_parts({"joints": [[0.0]*3]*24})
        assert "pose" in parts
        assert isinstance(parts["pose"], PoseData)

    def test_v0_to_v1_vitals(self):
        parts = v0_to_v1_parts({"vitals": {"heart_bpm": 60}})
        assert "vitals" in parts
        assert parts["vitals"].heart_bpm == 60

    def test_v0_to_v1_csi(self):
        parts = v0_to_v1_parts({"csi_amplitudes": [1.0, 2.0]})
        assert "csi" in parts

    def test_v0_to_v1_empty(self):
        parts = v0_to_v1_parts({})
        assert parts == {}

    def test_v0_to_v1_combined(self):
        """v0 payload with all fields should split into separate parts."""
        parts = v0_to_v1_parts({
            "joints": [[0.0]*3]*24,
            "vitals": {"heart_bpm": 72},
            "csi_amplitudes": [1.0],
        })
        assert len(parts) == 3

    def test_roundtrip_pose(self):
        """v1 → v0 → v1 should preserve data."""
        original = PoseData(joints=[[1.0, 2.0, 3.0]] * 24, confidence=0.9)
        env = make_envelope("pose", original)
        v0 = v1_to_v0(env)
        parts = v0_to_v1_parts(v0)
        assert parts["pose"].joints == original.joints


# ═══════════════════════════════════════════════════════════
# 3. Connection State — v0 detection after timeout
# ═══════════════════════════════════════════════════════════

class TestConnectionStateRegression:
    """Protocol detection logic was refactored — verify it still works."""

    def test_new_connection_is_unknown(self):
        conn = ConnectionState()
        assert conn.protocol_version is None
        assert conn.is_v1 is False

    def test_v0_detected_after_timeout(self):
        conn = ConnectionState()
        conn.connected_at = time.time() - V0_DETECT_TIMEOUT_SEC - 1
        assert conn.is_v0 is True

    def test_v0_not_detected_before_timeout(self):
        conn = ConnectionState()
        conn.connected_at = time.time()
        assert conn.is_v0 is False

    def test_hello_upgrades_to_v1(self):
        conn = ConnectionState()
        result = handle_client_message(conn, {
            "v": 1, "type": "hello", "capabilities": ["pose", "vitals"],
        })
        assert conn.is_v1 is True
        assert conn.subscriptions == {"pose", "vitals"}
        assert result["type"] == "welcome"
        assert set(result["streams"]) == {"pose", "vitals"}

    def test_hello_without_capabilities_subscribes_all(self):
        conn = ConnectionState()
        handle_client_message(conn, {"v": 1, "type": "hello"})
        assert conn.subscriptions == ALL_STREAMS

    def test_hello_with_empty_capabilities_subscribes_all(self):
        conn = ConnectionState()
        handle_client_message(conn, {"v": 1, "type": "hello", "capabilities": []})
        assert conn.subscriptions == ALL_STREAMS

    def test_pong_updates_timestamp(self):
        conn = ConnectionState()
        conn.protocol_version = 1
        before = conn.last_pong_ts
        handle_client_message(conn, {"v": 1, "type": "pong", "ts": 12345})
        assert conn.last_pong_ts > before

    def test_v0_message_returns_none(self):
        """Non-v1 messages (e.g. v=0 or missing v) return None."""
        conn = ConnectionState()
        result = handle_client_message(conn, {"type": "hello"})
        assert result is None
        assert conn.protocol_version is None

    def test_unknown_v1_type_returns_none(self):
        conn = ConnectionState()
        result = handle_client_message(conn, {"v": 1, "type": "subscribe"})
        assert result is None


# ═══════════════════════════════════════════════════════════
# 4. Envelope — sequence counter and serialization
# ═══════════════════════════════════════════════════════════

class TestEnvelopeRegression:
    """Envelope creation was extracted into its own module — verify behavior."""

    def test_envelope_sequence_increments(self):
        e1 = make_envelope("pose", PoseData(joints=[[0.0]*3]*24))
        e2 = make_envelope("pose", PoseData(joints=[[0.0]*3]*24))
        assert e2.seq > e1.seq

    def test_envelope_has_timestamp(self):
        e = make_envelope("pose", PoseData(joints=[[0.0]*3]*24))
        assert e.ts > 0
        assert e.v == 1

    def test_envelope_json_serialization(self):
        e = make_envelope("csi", CsiData(amplitudes=[1.0, 2.0, 3.0]))
        j = json.loads(e.model_dump_json())
        assert j["v"] == 1
        assert j["type"] == "csi"
        assert j["data"]["amplitudes"] == [1.0, 2.0, 3.0]

    def test_parse_hello_message(self):
        msg = parse_client_message({"v": 1, "type": "hello", "capabilities": ["pose"]})
        assert isinstance(msg, HelloMessage)
        assert msg.capabilities == ["pose"]

    def test_parse_pong_message(self):
        msg = parse_client_message({"v": 1, "type": "pong", "ts": 999})
        assert isinstance(msg, PongMessage)
        assert msg.ts == 999

    def test_parse_invalid_version(self):
        assert parse_client_message({"v": 2, "type": "hello"}) is None

    def test_parse_unknown_type(self):
        assert parse_client_message({"v": 1, "type": "unknown"}) is None


# ═══════════════════════════════════════════════════════════
# 5. VitalsService — event emission throttle regression
# ═══════════════════════════════════════════════════════════

class TestVitalsServiceRegression:
    """VitalsService throttles event emission — verify after refactoring."""

    def test_get_vitals_returns_dict(self):
        svc = VitalsService(sample_rate=20, emitter=EventEmitter())
        v = svc.get_vitals()
        assert isinstance(v, dict)

    def test_push_csi_does_not_crash(self):
        svc = VitalsService(sample_rate=20, emitter=EventEmitter())
        amp = np.random.rand(56).astype(np.float32)
        svc.push_csi(amp)

    @pytest.mark.asyncio
    async def test_throttle_prevents_rapid_emit(self):
        emitter = EventEmitter()
        results = []
        async def on_vitals(data): results.append(data)
        emitter.on("vitals", on_vitals)

        svc = VitalsService(sample_rate=20, emitter=emitter, emit_interval_sec=10.0)
        amp = np.random.rand(56).astype(np.float32)
        # Push twice rapidly — only first should emit
        svc.push_csi(amp)
        svc.push_csi(amp)
        await asyncio.sleep(0.1)
        assert len(results) <= 1

    def test_subcarrier_amplitudes_initially_none(self):
        svc = VitalsService(sample_rate=20, emitter=EventEmitter())
        result = svc.get_subcarrier_amplitudes()
        # Before any CSI is pushed, may return None or empty
        assert result is None or isinstance(result, list)


# ═══════════════════════════════════════════════════════════
# 6. Full App Wiring — integration regression
# ═══════════════════════════════════════════════════════════

class TestAppWiringRegression:
    """The FastAPI app with all routers mounted — smoke test the full stack."""

    @pytest.mark.asyncio
    async def test_all_system_routes_reachable(self, client):
        """Every system route should return 200 (not 404 or 500)."""
        routes = [
            "/", "/api/status", "/api/profiles", "/api/joints",
            "/api/vitals", "/api/alerts", "/api/persons",
            "/api/signal-quality", "/api/system/scene",
        ]
        for route in routes:
            r = await client.get(route)
            assert r.status_code == 200, f"Route {route} returned {r.status_code}"

    @pytest.mark.asyncio
    async def test_calibration_routes_reachable(self, client):
        r = await client.get("/api/calibration/status")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_data_routes_reachable(self, client):
        routes = [
            "/api/history/poses", "/api/history/vitals",
            "/api/history/alerts", "/api/notifications/status",
            "/api/ota/firmware", "/api/models",
            "/api/train/status", "/api/firmware/status",
        ]
        for route in routes:
            r = await client.get(route)
            assert r.status_code == 200, f"Route {route} returned {r.status_code}"

    @pytest.mark.asyncio
    async def test_container_accessible_from_app(self, app):
        """ServiceContainer is set on app.state before lifespan."""
        assert hasattr(app.state, "container")
        assert isinstance(app.state.container, ServiceContainer)
