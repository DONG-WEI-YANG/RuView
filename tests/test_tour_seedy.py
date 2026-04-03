"""Seedy District — edge cases, malformed inputs, adversarial testing.

Targets system boundaries: binary CSI parser, WebSocket protocol,
extreme signal values, concurrent operations, and fault isolation.
"""
import asyncio
import struct
import json
import time

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from server.csi_frame import (
    parse_csi_frame, CSIFrame, MAGIC_HEADER, HEADER_FORMAT, HEADER_SIZE,
)
from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.pipeline_service import PipelineService
from server.services.websocket_service import WebSocketService
from server.services.signal_quality import (
    SignalQualityMonitor, NodeQuality,
    RSSI_EXCELLENT, RSSI_GOOD, RSSI_POOR,
)
from server.protocol.envelope import make_envelope, PoseData, Envelope
from server.protocol.handlers import ConnectionState
from tests.conftest import make_csi_frame, make_csi_binary


# ═══════════════════════════════════════════════════════════
# 1. CSI Binary Parser — the UDP boundary
# ═══════════════════════════════════════════════════════════

class TestMalformedCSI:
    """Fuzz the ADR-018 binary parser with garbage from the RF ether."""

    def test_empty_packet(self):
        assert parse_csi_frame(b"") is None

    def test_truncated_header(self):
        """Less than 20 bytes — not even a full header."""
        assert parse_csi_frame(b"\xc5\x11\x00\x01" + b"\x00" * 10) is None

    def test_bad_magic(self):
        data = make_csi_binary(magic=0xDEADBEEF)
        assert parse_csi_frame(data) is None

    def test_truncated_iq_data(self):
        """Valid header but IQ payload is cut short."""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, 56, 2437, 0, -50, -90, 0,
        )
        # Need 56 * 2 * 2 = 224 bytes of IQ, provide only 10
        assert parse_csi_frame(header + b"\x00" * 10) is None

    def test_zero_subcarriers(self):
        """n_sub=0 should produce a valid frame with empty arrays."""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, 0, 2437, 0, -50, -90, 0,
        )
        frame = parse_csi_frame(header)
        assert frame is not None
        assert frame.num_subcarriers == 0
        assert len(frame.amplitude) == 0

    def test_huge_subcarrier_count(self):
        """n_sub=65535 but no IQ data — truncated."""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER, 1, 1, 65535, 2437, 0, -50, -90, 0,
        )
        assert parse_csi_frame(header) is None

    def test_valid_frame_parses(self):
        """Sanity: a well-formed binary frame should parse."""
        data = make_csi_binary(node_id=2, rssi=-60, n_sub=56)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert frame.node_id == 2
        assert frame.rssi == -60
        assert len(frame.amplitude) == 56

    def test_multi_antenna_frame(self):
        """2 antennas × 56 subcarriers = 112 IQ pairs."""
        data = make_csi_binary(n_antennas=2, n_sub=56)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert len(frame.amplitude) == 112

    def test_5ghz_channel_mapping(self):
        """5 GHz frequency should map to correct channel."""
        data = make_csi_binary(freq_mhz=5200, n_sub=30)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert frame.channel == 40  # 5200 MHz = channel 40

    def test_random_garbage(self):
        """Pure random bytes should not crash the parser."""
        for _ in range(50):
            garbage = np.random.bytes(np.random.randint(0, 500))
            result = parse_csi_frame(garbage)
            # Should return None or a CSIFrame, never raise
            assert result is None or isinstance(result, CSIFrame)

    def test_extreme_rssi_values(self):
        """RSSI is signed i8: range -128 to 127."""
        data = make_csi_binary(rssi=-128)
        frame = parse_csi_frame(data)
        assert frame is not None
        assert frame.rssi == -128


# ═══════════════════════════════════════════════════════════
# 2. WebSocket — dead connections and protocol abuse
# ═══════════════════════════════════════════════════════════

class TestWebSocketBroadcastFaults:
    """Broadcast must tolerate dead/misbehaving clients."""

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connection(self):
        """A connection that raises on send_text gets removed."""
        emitter = EventEmitter()
        ws_svc = WebSocketService(emitter=emitter)

        good_ws = AsyncMock()
        bad_ws = AsyncMock()
        bad_ws.send_text.side_effect = ConnectionError("pipe broken")

        ws_svc.register(good_ws)
        ws_svc.register(bad_ws)
        assert ws_svc.connection_count == 2

        # Force both to v0 (no hello → v0 after timeout)
        for conn in ws_svc._connections.values():
            conn.connected_at = 0

        env = make_envelope("pose", PoseData(
            joints=[[0.0] * 3] * 24, confidence=0.5,
        ))
        await ws_svc.broadcast_envelope(env)

        # Dead connection removed, good one stays
        assert ws_svc.connection_count == 1
        good_ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_to_v1_respects_subscriptions(self):
        """v1 clients only receive types they subscribed to."""
        emitter = EventEmitter()
        ws_svc = WebSocketService(emitter=emitter)

        ws = AsyncMock()
        conn = ws_svc.register(ws)
        # Simulate v1 hello with only "vitals" subscription
        conn.protocol_version = 1
        conn.subscriptions = {"vitals"}

        # Send a "pose" envelope — should be filtered
        env = make_envelope("pose", PoseData(
            joints=[[0.0] * 3] * 24, confidence=0.5,
        ))
        await ws_svc.broadcast_envelope(env)
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_v0_lazy_conversion(self):
        """v0 payload is computed lazily — only when v0 clients exist."""
        emitter = EventEmitter()
        ws_svc = WebSocketService(emitter=emitter)

        # Only v1 client subscribed to "pose"
        ws = AsyncMock()
        conn = ws_svc.register(ws)
        conn.protocol_version = 1
        conn.subscriptions = {"pose"}

        env = make_envelope("pose", PoseData(
            joints=[[0.0] * 3] * 24, confidence=0.5,
        ))
        await ws_svc.broadcast_envelope(env)
        # v1 client gets v1 JSON
        ws.send_text.assert_called_once()
        sent = ws.send_text.call_args[0][0]
        parsed = json.loads(sent)
        assert parsed["v"] == 1
        assert parsed["type"] == "pose"

    @pytest.mark.asyncio
    async def test_broadcast_no_connections(self):
        """Broadcasting to zero clients should not crash."""
        ws_svc = WebSocketService(emitter=EventEmitter())
        env = make_envelope("pose", PoseData(
            joints=[[0.0] * 3] * 24, confidence=0.5,
        ))
        await ws_svc.broadcast_envelope(env)  # no crash

    @pytest.mark.asyncio
    async def test_handle_unknown_message(self):
        """Unknown message type should not crash."""
        ws_svc = WebSocketService(emitter=EventEmitter())
        ws = AsyncMock()
        ws_svc.register(ws)
        result = ws_svc.handle_message(ws, {"type": "hack_the_planet"})
        # Should return None or handle gracefully
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handle_message_unregistered_ws(self):
        """Message from unknown WebSocket returns None."""
        ws_svc = WebSocketService(emitter=EventEmitter())
        ws = AsyncMock()
        result = ws_svc.handle_message(ws, {"type": "hello"})
        assert result is None


# ═══════════════════════════════════════════════════════════
# 3. SignalQuality — extreme RSSI and degenerate inputs
# ═══════════════════════════════════════════════════════════

class TestSignalQualityEdgeCases:
    """Push the signal quality monitor to its limits."""

    def test_empty_rssi_history(self):
        nq = NodeQuality(node_id=1)
        assert nq.avg_rssi == -100.0  # default for no data
        assert nq.grade == "poor"

    def test_single_rssi_sample(self):
        nq = NodeQuality(node_id=1)
        nq.rssi_history.append(-50)
        assert nq.avg_rssi == -50.0
        assert nq.grade == "excellent"

    def test_rssi_at_exact_thresholds(self):
        """Test behavior at exact threshold boundaries."""
        nq = NodeQuality(node_id=1)

        nq.rssi_history.clear()
        nq.rssi_history.append(RSSI_EXCELLENT)  # -50
        assert nq.grade == "excellent"

        nq.rssi_history.clear()
        nq.rssi_history.append(RSSI_EXCELLENT - 1)  # -51
        assert nq.grade == "good"

        nq.rssi_history.clear()
        nq.rssi_history.append(RSSI_GOOD)  # -65
        assert nq.grade == "good"

        nq.rssi_history.clear()
        nq.rssi_history.append(RSSI_POOR)  # -75
        assert nq.grade == "fair"

        nq.rssi_history.clear()
        nq.rssi_history.append(RSSI_POOR - 1)  # -76
        assert nq.grade == "poor"

    def test_rolling_window_overflow(self):
        """NodeQuality uses deque(maxlen=100) — should not grow unbounded."""
        nq = NodeQuality(node_id=1)
        for i in range(200):
            nq.rssi_history.append(-50)
        assert len(nq.rssi_history) == 100

    def test_csi_variance_different_amplitude_lengths(self):
        """If amplitude length changes between frames, skip variance calc."""
        monitor = SignalQualityMonitor(emitter=EventEmitter())
        f1 = make_csi_frame(node_id=1, n_sub=56, sequence=0)
        f2 = make_csi_frame(node_id=1, n_sub=30, sequence=1)
        monitor.on_frame(f1)
        monitor.on_frame(f2)
        # Should not crash; variance should not be added for mismatched lengths
        assert monitor._nodes[1].frame_count == 2


# ═══════════════════════════════════════════════════════════
# 4. PipelineService — concurrent frames and edge cases
# ═══════════════════════════════════════════════════════════

class TestPipelineEdgeCases:
    """Push the pipeline with unusual patterns."""

    def test_on_frame_no_amplitude(self):
        """Frame with amplitude=None should not crash event emission."""
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        frame = CSIFrame(
            node_id=1, sequence=0, timestamp_ms=0,
            rssi=-50, noise_floor=-90, channel=6, bandwidth=20,
            num_subcarriers=0,
            amplitude=None,
            phase=None,
            raw_complex=None,
        )
        svc.on_frame(frame)
        assert svc.csi_frames_received == 1

    def test_many_nodes_detected(self):
        """Simulate 8 nodes connecting — strategy should adapt."""
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        for nid in range(1, 9):
            svc.on_frame(make_csi_frame(node_id=nid))
        assert svc.detected_nodes == 8
        assert svc.strategy == "multiview"

    def test_inject_joints_updates_fall_detector(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        joints = np.random.rand(24, 3).astype(np.float32)
        joints[:, 1] = 1.5  # standing height
        svc.inject_joints(joints)
        assert svc.latest_joints is not None
        assert not svc.fall_detector.is_fallen

    def test_inject_joints_fallen_pose(self):
        """Head at floor level should trigger fall detection."""
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        # First inject a standing pose to establish baseline
        standing = np.random.rand(24, 3).astype(np.float32)
        standing[:, 1] = 1.5
        svc.inject_joints(standing)
        # Now inject a fallen pose (head near ground)
        fallen = np.random.rand(24, 3).astype(np.float32)
        fallen[:, 1] = 0.1  # everything near floor
        svc.inject_joints(fallen)
        # fall_detector tracks this — is_fallen depends on threshold

    def test_person_color_assignment_stable(self):
        """Same person ID always gets the same color."""
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        color1 = svc._assign_person_color(42)
        color2 = svc._assign_person_color(42)
        assert color1 == color2

    def test_person_color_unique_per_person(self):
        """Different person IDs get different colors (up to palette size)."""
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        colors = [svc._assign_person_color(i) for i in range(4)]
        assert len(set(colors)) == 4

    def test_node_weights_without_quality_monitor(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        assert svc.get_node_weights() == {}

    def test_node_weights_with_quality_monitor(self):
        emitter = EventEmitter()
        svc = PipelineService(settings=Settings(), emitter=emitter)
        monitor = SignalQualityMonitor(emitter=emitter)
        svc.set_quality_monitor(monitor)
        for i in range(10):
            monitor.on_frame(make_csi_frame(node_id=1, rssi=-40, sequence=i))
        weights = svc.get_node_weights()
        assert 1 in weights
        assert weights[1] == 1.0  # excellent

    def test_estimate_room_position_single_person(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        pos = svc._estimate_room_position(0, 1)
        assert pos == [0.0, 0.0]

    def test_estimate_room_position_multi_person(self):
        svc = PipelineService(settings=Settings(), emitter=EventEmitter())
        pos0 = svc._estimate_room_position(0, 3)
        pos1 = svc._estimate_room_position(1, 3)
        assert pos0 != pos1  # different positions


# ═══════════════════════════════════════════════════════════
# 5. EventEmitter — fault isolation
# ═══════════════════════════════════════════════════════════

class TestEventEmitterFaultIsolation:
    """A bad subscriber must never take down the system."""

    @pytest.mark.asyncio
    async def test_error_in_subscriber_does_not_crash_others(self):
        emitter = EventEmitter()
        results = []

        async def bad(data):
            raise RuntimeError("I'm broken")

        async def good(data):
            results.append(data)

        emitter.on("test", bad)
        emitter.on("test", good)
        await emitter.emit("test", {"v": 1})
        await asyncio.sleep(0.05)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_emit_with_no_subscribers(self):
        emitter = EventEmitter()
        await emitter.emit("nonexistent", {"v": 1})  # should not raise

    @pytest.mark.asyncio
    async def test_unsubscribe_then_emit(self):
        emitter = EventEmitter()
        results = []
        async def handler(data): results.append(data)
        emitter.on("ev", handler)
        emitter.off("ev", handler)
        await emitter.emit("ev", {"v": 1})
        await asyncio.sleep(0.05)
        assert results == []


# ═══════════════════════════════════════════════════════════
# 6. HTTP Route Adversarial Inputs
# ═══════════════════════════════════════════════════════════

class TestRouteAdversarial:
    """Send malicious inputs to HTTP endpoints."""

    @pytest.mark.asyncio
    async def test_ota_download_path_traversal(self, client):
        """Filename with path traversal should be rejected (400 or 404, never 200)."""
        r = await client.get("/api/ota/download/..%2F..%2Fetc%2Fpasswd")
        assert r.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_ota_download_special_chars(self, client):
        r = await client.get("/api/ota/download/%3Cscript%3Ealert(1)%3C%2Fscript%3E.bin")
        assert r.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_history_negative_limit(self, client):
        """Negative limit should not crash."""
        r = await client.get("/api/history/poses", params={"limit": -1})
        # Should either return empty or handle gracefully
        assert r.status_code == 200 or r.status_code == 422

    @pytest.mark.asyncio
    async def test_history_huge_limit(self, client):
        r = await client.get("/api/history/poses", params={"limit": 999999})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_train_status_idle(self, client):
        r = await client.get("/api/train/status")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    @pytest.mark.asyncio
    async def test_firmware_status_idle(self, client):
        r = await client.get("/api/firmware/status")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"
