# Architecture & UI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the WiFi Body server and dashboard from monolithic files into a modular, service-layered architecture with a versioned WebSocket protocol.

**Architecture:** Protocol-First strategy — define the v1 WebSocket contract first, then refactor the server around services and DI, then modularize the dashboard with Vite + ES6 modules. Each phase leaves the system fully functional.

**Tech Stack:** Python 3.11+ / FastAPI / Pydantic v2 / asyncio, JavaScript ES6 modules / Vite / Three.js

**Spec:** `docs/superpowers/specs/2026-03-20-architecture-ui-optimization-design.md`

---

## Phase 1: WebSocket Protocol v1

### Task 1: Protocol Envelope — Pydantic Models

**Files:**
- Create: `server/protocol/__init__.py`
- Create: `server/protocol/envelope.py`
- Test: `tests/test_protocol_envelope.py`

- [ ] **Step 1: Write failing tests for envelope models**

```python
# tests/test_protocol_envelope.py
import time
import pytest
from server.protocol.envelope import (
    Envelope, PoseData, VitalsData, CsiData, StatusData, ErrorData,
    HelloMessage, WelcomeMessage, PingMessage, PongMessage,
    make_envelope, parse_client_message,
)


def test_make_pose_envelope():
    joints = [[0.1, 0.2, 0.3]] * 24
    env = make_envelope("pose", PoseData(joints=joints, confidence=0.92))
    assert env.v == 1
    assert env.type == "pose"
    assert env.seq > 0
    assert env.ts > 0
    assert env.data.joints == joints
    assert env.data.confidence == 0.92


def test_make_vitals_envelope():
    data = VitalsData(
        heart_bpm=72.0, heart_confidence=0.85,
        breathing_bpm=16.0, breathing_confidence=0.90,
        hrv_rmssd=45.0, hrv_sdnn=55.0,
        stress_index=30.0, motion_intensity=5.0,
        body_movement="still", breath_regularity=0.85,
        sleep_stage="awake", respiratory_distress=False,
        apnea_events=0,
    )
    env = make_envelope("vitals", data)
    assert env.type == "vitals"
    assert env.data.heart_bpm == 72.0


def test_make_csi_envelope():
    amps = [0.5] * 56
    env = make_envelope("csi", CsiData(amplitudes=amps))
    assert env.type == "csi"
    assert len(env.data.amplitudes) == 56


def test_make_status_envelope():
    data = StatusData(
        model_loaded=True, csi_frames_received=100,
        inference_active=True, is_simulating=False,
        connected_clients=2, hardware_profile="esp32s3",
    )
    env = make_envelope("status", data)
    assert env.data.model_loaded is True


def test_make_error_envelope():
    env = make_envelope("error", ErrorData(code="INFERENCE_FAIL", message="Model error"))
    assert env.data.code == "INFERENCE_FAIL"


def test_envelope_to_json():
    env = make_envelope("pose", PoseData(joints=[[0, 0, 0]] * 24, confidence=0.5))
    j = env.model_dump_json()
    assert '"v": 1' in j or '"v":1' in j


def test_seq_increments():
    e1 = make_envelope("csi", CsiData(amplitudes=[1.0]))
    e2 = make_envelope("csi", CsiData(amplitudes=[2.0]))
    assert e2.seq > e1.seq


def test_parse_hello():
    msg = parse_client_message({"v": 1, "type": "hello", "capabilities": ["pose", "vitals"]})
    assert isinstance(msg, HelloMessage)
    assert "pose" in msg.capabilities


def test_parse_pong():
    msg = parse_client_message({"v": 1, "type": "pong", "ts": 12345})
    assert isinstance(msg, PongMessage)


def test_parse_unknown_returns_none():
    msg = parse_client_message({"v": 1, "type": "unknown_type"})
    assert msg is None


def test_parse_v0_returns_none():
    msg = parse_client_message({"joints": [[0, 0, 0]]})
    assert msg is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_protocol_envelope.py -v`
Expected: ModuleNotFoundError for `server.protocol.envelope`

- [ ] **Step 3: Implement envelope models**

```python
# server/protocol/__init__.py
"""WebSocket protocol v1 — envelope, models, and message handling."""

# server/protocol/envelope.py
"""v1 protocol envelope and message models.

All WebSocket messages use a common envelope:
  {"v": 1, "ts": <ms>, "seq": <int>, "type": "<type>", "data": {...}}
"""
from __future__ import annotations

import time
import threading
from typing import Union, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Sequence counter (thread-safe) ────────────────────────
_seq_lock = threading.Lock()
_seq_counter = 0


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


# ── Data payloads ─────────────────────────────────────────

class PoseData(BaseModel):
    joints: list[list[float]]      # 24 × [x, y, z]
    confidence: float = 0.0

class VitalsData(BaseModel):
    model_config = ConfigDict(extra="ignore")  # ignore buffer_fullness etc.
    heart_bpm: float = 0.0
    heart_confidence: float = 0.0
    breathing_bpm: float = 0.0
    breathing_confidence: float = 0.0
    hrv_rmssd: float = 0.0
    hrv_sdnn: float = 0.0
    stress_index: float = 0.0
    motion_intensity: float = 0.0
    body_movement: str = "still"
    breath_regularity: float = 0.0
    sleep_stage: str = "awake"
    respiratory_distress: bool = False
    apnea_events: int = 0

class CsiData(BaseModel):
    amplitudes: list[float]

class StatusData(BaseModel):
    model_loaded: bool = False
    csi_frames_received: int = 0
    inference_active: bool = False
    is_simulating: bool = False
    connected_clients: int = 0
    hardware_profile: str = ""

class ErrorData(BaseModel):
    code: str
    message: str


# ── Envelope ──────────────────────────────────────────────

DataType = Union[PoseData, VitalsData, CsiData, StatusData, ErrorData]

class Envelope(BaseModel):
    v: Literal[1] = 1
    ts: int                         # server-side ms timestamp
    seq: int                        # monotonic sequence number
    type: str
    data: DataType


def make_envelope(msg_type: str, data: DataType) -> Envelope:
    """Create a new v1 envelope with auto-assigned ts and seq."""
    return Envelope(
        ts=int(time.time() * 1000),
        seq=_next_seq(),
        type=msg_type,
        data=data,
    )


# ── Client messages ───────────────────────────────────────

class HelloMessage(BaseModel):
    v: Literal[1] = 1
    type: Literal["hello"] = "hello"
    capabilities: list[str] = Field(default_factory=list)

class PongMessage(BaseModel):
    v: Literal[1] = 1
    type: Literal["pong"] = "pong"
    ts: int = 0

class PingMessage(BaseModel):
    v: Literal[1] = 1
    type: Literal["ping"] = "ping"
    ts: int = 0

class WelcomeMessage(BaseModel):
    v: Literal[1] = 1
    type: Literal["welcome"] = "welcome"
    server_version: str = "0.2.0"
    streams: list[str] = Field(default_factory=list)


ClientMessage = Union[HelloMessage, PongMessage]


def parse_client_message(raw: dict) -> ClientMessage | None:
    """Parse a client-sent JSON dict into a typed message, or None."""
    if raw.get("v") != 1:
        return None
    msg_type = raw.get("type")
    if msg_type == "hello":
        return HelloMessage(**raw)
    if msg_type == "pong":
        return PongMessage(**raw)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_protocol_envelope.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/protocol/__init__.py server/protocol/envelope.py tests/test_protocol_envelope.py
git commit -m "feat(protocol): add v1 envelope Pydantic models and message parsing"
```

---

### Task 2: v0 Adapter — Bidirectional Conversion

**Files:**
- Create: `server/protocol/v0_adapter.py`
- Test: `tests/test_protocol_v0_adapter.py`

- [ ] **Step 1: Write failing tests for v0 adapter**

```python
# tests/test_protocol_v0_adapter.py
import pytest
from server.protocol.v0_adapter import v1_to_v0, v0_to_v1_parts
from server.protocol.envelope import (
    make_envelope, PoseData, VitalsData, CsiData,
)


def test_v1_pose_to_v0():
    """v1 pose envelope → v0 single-payload dict."""
    joints = [[i * 0.1, i * 0.2, i * 0.3] for i in range(24)]
    env = make_envelope("pose", PoseData(joints=joints, confidence=0.9))
    v0 = v1_to_v0(env)
    assert "joints" in v0
    assert len(v0["joints"]) == 24
    assert v0["joints"][0] == joints[0]


def test_v1_vitals_merged_into_v0():
    """v0 format has vitals nested inside the same payload."""
    vitals_data = VitalsData(heart_bpm=72.0, breathing_bpm=16.0)
    env = make_envelope("vitals", vitals_data)
    v0 = v1_to_v0(env)
    assert v0["vitals"]["heart_bpm"] == 72.0


def test_v1_csi_merged_into_v0():
    env = make_envelope("csi", CsiData(amplitudes=[0.5] * 30))
    v0 = v1_to_v0(env)
    assert len(v0["csi_amplitudes"]) == 30


def test_v0_payload_to_v1_parts():
    """Legacy v0 payload → separated v1 data objects."""
    v0_payload = {
        "joints": [[0, 0, 0]] * 24,
        "vitals": {"heart_bpm": 72, "breathing_bpm": 16},
        "csi_amplitudes": [0.5] * 30,
    }
    parts = v0_to_v1_parts(v0_payload)
    assert "pose" in parts
    assert "vitals" in parts
    assert "csi" in parts
    assert parts["pose"].joints == [[0, 0, 0]] * 24


def test_v0_payload_missing_vitals():
    v0_payload = {"joints": [[0, 0, 0]] * 24}
    parts = v0_to_v1_parts(v0_payload)
    assert "pose" in parts
    assert "vitals" not in parts
    assert "csi" not in parts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_protocol_v0_adapter.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement v0 adapter**

```python
# server/protocol/v0_adapter.py
"""Bidirectional conversion between v0 (legacy) and v1 protocol formats.

v0 format (single payload):
  {"joints": [...], "vitals": {...}, "csi_amplitudes": [...]}

v1 format (separate envelopes per stream type).
"""
from __future__ import annotations

from server.protocol.envelope import (
    Envelope, PoseData, VitalsData, CsiData,
)


def v1_to_v0(envelope: Envelope) -> dict:
    """Convert a v1 envelope to v0 legacy payload dict.

    v0 clients expect all data in one dict. We merge each stream
    type into the appropriate v0 key.
    """
    result = {}
    if envelope.type == "pose":
        result["joints"] = envelope.data.joints
    elif envelope.type == "vitals":
        result["vitals"] = envelope.data.model_dump()
    elif envelope.type == "csi":
        result["csi_amplitudes"] = envelope.data.amplitudes
    # status and error types have no v0 equivalent — skip
    return result


def v0_to_v1_parts(payload: dict) -> dict[str, PoseData | VitalsData | CsiData]:
    """Split a v0 payload into separate v1 data objects.

    Returns a dict mapping stream type name → data model.
    Only includes keys that are present in the payload.
    """
    parts = {}
    if "joints" in payload:
        parts["pose"] = PoseData(joints=payload["joints"])
    if "vitals" in payload and payload["vitals"]:
        parts["vitals"] = VitalsData(**payload["vitals"])
    if "csi_amplitudes" in payload and payload["csi_amplitudes"]:
        parts["csi"] = CsiData(amplitudes=payload["csi_amplitudes"])
    return parts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_protocol_v0_adapter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/protocol/v0_adapter.py tests/test_protocol_v0_adapter.py
git commit -m "feat(protocol): add v0 adapter for bidirectional format conversion"
```

---

### Task 3: Client Message Handlers

**Files:**
- Create: `server/protocol/handlers.py`
- Test: `tests/test_protocol_handlers.py`

- [ ] **Step 1: Write failing tests for handlers**

```python
# tests/test_protocol_handlers.py
import pytest
import asyncio
from server.protocol.handlers import ConnectionState, handle_client_message


def test_connection_starts_as_unknown():
    conn = ConnectionState()
    assert conn.protocol_version is None
    assert conn.subscriptions == set()


def test_hello_sets_v1_and_subscriptions():
    conn = ConnectionState()
    response = handle_client_message(conn, {"v": 1, "type": "hello", "capabilities": ["pose", "vitals"]})
    assert conn.protocol_version == 1
    assert conn.subscriptions == {"pose", "vitals"}
    assert response is not None
    assert response["type"] == "welcome"
    assert "pose" in response["streams"]


def test_hello_empty_capabilities_subscribes_all():
    conn = ConnectionState()
    response = handle_client_message(conn, {"v": 1, "type": "hello", "capabilities": []})
    assert conn.subscriptions == {"pose", "vitals", "csi", "status"}


def test_pong_updates_last_pong():
    conn = ConnectionState()
    conn.protocol_version = 1
    response = handle_client_message(conn, {"v": 1, "type": "pong", "ts": 12345})
    assert conn.last_pong_ts > 0
    assert response is None  # pong has no response


def test_v0_message_detected():
    conn = ConnectionState()
    response = handle_client_message(conn, {"some_key": "some_value"})
    assert conn.protocol_version is None  # stays unknown
    assert response is None


def test_is_v0_after_timeout():
    conn = ConnectionState()
    assert conn.is_v0 is False  # not yet decided
    conn.connected_at = 0  # pretend connected long ago
    assert conn.is_v0 is True  # 5-second timeout passed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_protocol_handlers.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement handlers**

```python
# server/protocol/handlers.py
"""Handle incoming client WebSocket messages (hello, pong).

Each WebSocket connection has a ConnectionState that tracks:
- Protocol version (None = unknown, 1 = v1)
- Subscribed streams
- Heartbeat timing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from server.protocol.envelope import (
    parse_client_message, HelloMessage, PongMessage,
    WelcomeMessage,
)

ALL_STREAMS = {"pose", "vitals", "csi", "status"}
V0_DETECT_TIMEOUT_SEC = 5.0


@dataclass
class ConnectionState:
    """Per-connection protocol state."""
    protocol_version: int | None = None    # None=unknown, 1=v1
    subscriptions: set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_pong_ts: float = 0.0

    @property
    def is_v1(self) -> bool:
        return self.protocol_version == 1

    @property
    def is_v0(self) -> bool:
        """True if enough time has passed without a hello → assume v0."""
        if self.protocol_version == 1:
            return False
        if self.protocol_version is None:
            return (time.time() - self.connected_at) > V0_DETECT_TIMEOUT_SEC
        return True


def handle_client_message(conn: ConnectionState, raw: dict) -> dict | None:
    """Process a client message, update connection state, return response or None."""
    msg = parse_client_message(raw)
    if msg is None:
        return None

    if isinstance(msg, HelloMessage):
        conn.protocol_version = 1
        caps = set(msg.capabilities) if msg.capabilities else ALL_STREAMS
        conn.subscriptions = caps if caps else ALL_STREAMS
        welcome = WelcomeMessage(streams=sorted(conn.subscriptions))
        return welcome.model_dump()

    if isinstance(msg, PongMessage):
        conn.last_pong_ts = time.time()
        return None

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_protocol_handlers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/protocol/handlers.py tests/test_protocol_handlers.py
git commit -m "feat(protocol): add connection state and client message handlers"
```

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -x -q`
Expected: All 125 existing tests + new protocol tests pass

- [ ] **Step 7: Commit phase 1 complete marker**

```bash
git commit --allow-empty -m "milestone: Phase 1 (Protocol Layer) complete"
```

---

## Phase 2: Server Service Layer

### Task 4: Event Emitter

**Files:**
- Create: `server/services/__init__.py`
- Create: `server/services/event_emitter.py`
- Test: `tests/test_event_emitter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_emitter.py
import asyncio
import pytest
from server.services.event_emitter import EventEmitter


@pytest.mark.asyncio
async def test_emit_calls_subscriber():
    emitter = EventEmitter()
    results = []
    async def handler(data):
        results.append(data)
    emitter.on("test", handler)
    await emitter.emit("test", {"value": 42})
    # Give tasks time to run
    await asyncio.sleep(0.05)
    assert results == [{"value": 42}]


@pytest.mark.asyncio
async def test_multiple_subscribers():
    emitter = EventEmitter()
    a, b = [], []
    async def ha(d): a.append(d)
    async def hb(d): b.append(d)
    emitter.on("ev", ha)
    emitter.on("ev", hb)
    await emitter.emit("ev", "hello")
    await asyncio.sleep(0.05)
    assert a == ["hello"]
    assert b == ["hello"]


@pytest.mark.asyncio
async def test_off_removes_subscriber():
    emitter = EventEmitter()
    results = []
    async def handler(data): results.append(data)
    emitter.on("ev", handler)
    emitter.off("ev", handler)
    await emitter.emit("ev", "ignored")
    await asyncio.sleep(0.05)
    assert results == []


@pytest.mark.asyncio
async def test_subscriber_error_does_not_crash():
    emitter = EventEmitter()
    results = []
    async def bad_handler(data): raise ValueError("boom")
    async def good_handler(data): results.append(data)
    emitter.on("ev", bad_handler)
    emitter.on("ev", good_handler)
    await emitter.emit("ev", "data")
    await asyncio.sleep(0.05)
    # good_handler should still have run
    assert results == ["data"]


@pytest.mark.asyncio
async def test_emit_no_subscribers():
    emitter = EventEmitter()
    # Should not raise
    await emitter.emit("nobody_listening", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_event_emitter.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement event emitter**

```python
# server/services/__init__.py
"""Service layer for WiFi Body server."""

# server/services/event_emitter.py
"""Lightweight async event emitter for inter-service communication.

Subscribers are async callbacks invoked via asyncio.create_task.
If a subscriber raises, the error is logged and other subscribers
continue unaffected.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

AsyncHandler = Callable[[Any], Coroutine[Any, Any, None]]


class EventEmitter:
    def __init__(self):
        self._handlers: dict[str, list[AsyncHandler]] = defaultdict(list)

    def on(self, event: str, handler: AsyncHandler) -> None:
        self._handlers[event].append(handler)

    def off(self, event: str, handler: AsyncHandler) -> None:
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: str, data: Any = None) -> None:
        for handler in self._handlers.get(event, []):
            asyncio.create_task(self._safe_call(handler, data, event))

    async def _safe_call(self, handler: AsyncHandler, data: Any, event: str) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Event handler error on '%s'", event)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_event_emitter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/services/__init__.py server/services/event_emitter.py tests/test_event_emitter.py
git commit -m "feat(services): add async EventEmitter for inter-service communication"
```

---

### Task 5: Service Container + WebSocket Service

**Files:**
- Create: `server/services/container.py`
- Create: `server/services/websocket_service.py`
- Test: `tests/test_websocket_service.py`

- [ ] **Step 1: Write failing tests for WebSocket service**

```python
# tests/test_websocket_service.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from server.services.event_emitter import EventEmitter
from server.services.websocket_service import WebSocketService
from server.protocol.handlers import ConnectionState


@pytest.fixture
def emitter():
    return EventEmitter()


@pytest.fixture
def ws_service(emitter):
    return WebSocketService(emitter=emitter, server_version="0.2.0")


@pytest.mark.asyncio
async def test_register_connection(ws_service):
    ws = AsyncMock()
    conn = ws_service.register(ws)
    assert isinstance(conn, ConnectionState)
    assert ws_service.connection_count == 1


@pytest.mark.asyncio
async def test_unregister_connection(ws_service):
    ws = AsyncMock()
    ws_service.register(ws)
    ws_service.unregister(ws)
    assert ws_service.connection_count == 0


@pytest.mark.asyncio
async def test_broadcast_v0_when_no_hello(ws_service):
    """Connections without hello get v0 format after timeout."""
    ws = AsyncMock()
    conn = ws_service.register(ws)
    # Force v0 detection
    conn.connected_at = 0
    assert conn.is_v0 is True


@pytest.mark.asyncio
async def test_handle_hello(ws_service):
    ws = AsyncMock()
    conn = ws_service.register(ws)
    response = ws_service.handle_message(ws, {"v": 1, "type": "hello", "capabilities": ["pose"]})
    assert conn.is_v1 is True
    assert response is not None
    assert response["type"] == "welcome"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_websocket_service.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement WebSocket service and container**

```python
# server/services/websocket_service.py
"""WebSocket connection management, v0/v1 dispatch, and heartbeat."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket

from server.protocol.envelope import Envelope, make_envelope, PingMessage
from server.protocol.v0_adapter import v1_to_v0
from server.protocol.handlers import ConnectionState, handle_client_message
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SEC = 30
HEARTBEAT_TIMEOUT_SEC = 30


class WebSocketService:
    def __init__(self, emitter: EventEmitter, server_version: str = "0.2.0"):
        self._emitter = emitter
        self._server_version = server_version
        self._connections: dict[WebSocket, ConnectionState] = {}
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def register(self, ws: WebSocket) -> ConnectionState:
        conn = ConnectionState()
        self._connections[ws] = conn
        return conn

    def unregister(self, ws: WebSocket) -> None:
        self._connections.pop(ws, None)

    def handle_message(self, ws: WebSocket, raw: dict) -> dict | None:
        conn = self._connections.get(ws)
        if conn is None:
            return None
        return handle_client_message(conn, raw)

    async def broadcast_envelope(self, envelope: Envelope) -> None:
        """Send an envelope to all subscribed connections."""
        v1_json = envelope.model_dump_json()
        v0_payload = None  # lazy-compute only if needed
        dead = set()

        for ws, conn in self._connections.items():
            try:
                if conn.is_v1 and envelope.type in conn.subscriptions:
                    await ws.send_text(v1_json)
                elif conn.is_v0 or conn.protocol_version is None:
                    if v0_payload is None:
                        v0_payload = json.dumps(v1_to_v0(envelope))
                    await ws.send_text(v0_payload)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self.unregister(ws)

    async def start_heartbeat(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            now = time.time()
            ping = PingMessage(ts=int(now * 1000))
            ping_json = json.dumps(ping.model_dump())
            dead = set()

            for ws, conn in self._connections.items():
                if not conn.is_v1:
                    continue
                # Check if previous ping timed out
                if conn.last_pong_ts > 0 and (now - conn.last_pong_ts) > HEARTBEAT_TIMEOUT_SEC:
                    dead.add(ws)
                    continue
                try:
                    await ws.send_text(ping_json)
                except Exception:
                    dead.add(ws)

            for ws in dead:
                logger.info("Removing dead WebSocket connection (heartbeat timeout)")
                self.unregister(ws)
```

```python
# server/services/container.py
"""Dependency injection container — manages service lifecycles.

FastAPI lifespan creates this, stores in app.state.container.
Routes access services via Depends(get_container).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.websocket_service import WebSocketService

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    settings: Settings
    emitter: EventEmitter = field(default_factory=EventEmitter)
    websocket: WebSocketService = field(init=False)

    def __post_init__(self):
        self.websocket = WebSocketService(
            emitter=self.emitter,
            server_version="0.2.0",
        )

    async def startup(self) -> None:
        logger.info("ServiceContainer starting up")
        await self.websocket.start_heartbeat()

    async def shutdown(self) -> None:
        logger.info("ServiceContainer shutting down")
        await self.websocket.stop_heartbeat()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_websocket_service.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/services/container.py server/services/websocket_service.py tests/test_websocket_service.py
git commit -m "feat(services): add WebSocketService with v0/v1 dispatch and heartbeat"
```

---

### Task 6: Pipeline Service + Vitals Service

**Files:**
- Create: `server/services/pipeline_service.py`
- Create: `server/services/vitals_service.py`
- Test: `tests/test_pipeline_service.py`
- Test: `tests/test_vitals_service.py`

- [ ] **Step 1: Write failing tests for pipeline service**

```python
# tests/test_pipeline_service.py
import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock
from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.pipeline_service import PipelineService
from server.csi_frame import CSIFrame


def _make_frame(node_id=1, n_sub=56):
    return CSIFrame(
        node_id=node_id, sequence=0, timestamp_ms=0,
        rssi=-50, noise_floor=-90, channel=6, bandwidth=20,
        num_subcarriers=n_sub,
        amplitude=np.random.rand(n_sub).astype(np.float32),
        phase=np.zeros(n_sub, dtype=np.float32),
        raw_complex=np.zeros(n_sub, dtype=np.complex64),
    )


@pytest.fixture
def emitter():
    return EventEmitter()


@pytest.fixture
def service(emitter):
    settings = Settings()
    return PipelineService(settings=settings, emitter=emitter)


def test_on_frame_increments_count(service):
    frame = _make_frame()
    service.on_frame(frame)
    assert service.csi_frames_received == 1


@pytest.mark.asyncio
async def test_on_frame_emits_csi(emitter, service):
    results = []
    async def on_csi(data): results.append(data)
    emitter.on("csi", on_csi)
    frame = _make_frame()
    service.on_frame(frame, trigger_pipeline=True)
    await asyncio.sleep(0.1)
    assert len(results) > 0
    assert "amplitudes" in results[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_service.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement pipeline service**

```python
# server/services/pipeline_service.py
"""CSI receive → signal processing → inference → emit events.

Wraps the existing PosePipeline and adds event emission.
Also owns the simulation loop and fall detector (single source of truth).
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from server.config import Settings
from server.csi_frame import CSIFrame
from server.pipeline import PosePipeline
from server.fall_detector import FallDetector
from server.fitness_tracker import FitnessTracker
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(
        self,
        settings: Settings,
        emitter: EventEmitter,
        pipeline: PosePipeline | None = None,
    ):
        self.settings = settings
        self._emitter = emitter
        self.pipeline = pipeline
        self.fall_detector = FallDetector(
            threshold=settings.fall_threshold,
            cooldown_sec=settings.fall_alert_cooldown,
        )
        self.fitness_tracker = FitnessTracker()
        self.latest_joints: np.ndarray | None = None
        self.csi_frames_received: int = 0
        self._node_frames: dict[int, CSIFrame] = {}
        self._sim_task: asyncio.Task | None = None

    def on_frame(self, frame: CSIFrame, trigger_pipeline: bool = True) -> None:
        """Process an incoming CSI frame."""
        self._node_frames[frame.node_id] = frame
        self.csi_frames_received += 1

        # Feed to pipeline
        if self.pipeline is not None:
            self.pipeline.on_csi_frame(frame)

        # Emit CSI amplitudes for waterfall
        if frame.amplitude is not None:
            asyncio.ensure_future(
                self._emitter.emit("csi", {"amplitudes": frame.amplitude.tolist()})
            )

        if trigger_pipeline:
            self._flush(frame)

    def _flush(self, frame: CSIFrame) -> None:
        if self.pipeline is not None:
            self.pipeline.flush_frame()
            if self.pipeline.latest_joints is not None:
                self.latest_joints = self.pipeline.latest_joints
                self.fall_detector = self.pipeline.fall_detector
                self.fitness_tracker = self.pipeline.fitness_tracker
                asyncio.ensure_future(
                    self._emitter.emit("pose", {
                        "joints": self.latest_joints.tolist(),
                        "confidence": 0.0,
                    })
                )

    def inject_joints(self, joints: np.ndarray) -> None:
        """Inject ground-truth joints (simulation mode, no model)."""
        self.latest_joints = joints
        self.fall_detector.update(joints)
        self.fitness_tracker.update(joints)
        asyncio.ensure_future(
            self._emitter.emit("pose", {
                "joints": joints.tolist(),
                "confidence": 0.0,
            })
        )

    @property
    def node_frames(self) -> dict[int, CSIFrame]:
        return self._node_frames

    async def start_simulation(self) -> None:
        if self._sim_task is None:
            self._sim_task = asyncio.create_task(self._simulation_loop())

    async def stop_simulation(self) -> None:
        if self._sim_task:
            self._sim_task.cancel()
            try:
                await self._sim_task
            except asyncio.CancelledError:
                pass
            self._sim_task = None

    async def _simulation_loop(self) -> None:
        from server.data_generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        activities = ["standing", "walking", "exercising", "sitting", "falling"]
        fs = self.settings.csi_sample_rate
        dt = 1.0 / fs

        while True:
            if not self.settings.simulate:
                await asyncio.sleep(1.0)
                continue

            for activity in activities:
                if not self.settings.simulate:
                    break
                logger.info("Simulating activity: %s", activity)
                try:
                    data = gen.generate_sequence(
                        activity, n_frames=100,
                        n_nodes=self.settings.max_nodes,
                        n_sub=self.settings.num_subcarriers,
                    )
                    csi_batch = data["csi"]
                    joints_batch = data["joints"]
                    n_frames, n_nodes, n_sub = csi_batch.shape

                    for t in range(n_frames):
                        loop_start = asyncio.get_event_loop().time()
                        for node_idx in range(n_nodes):
                            amp = csi_batch[t, node_idx, :]
                            frame = CSIFrame(
                                node_id=node_idx + 1, sequence=t,
                                timestamp_ms=int(t * dt * 1000),
                                rssi=-50 + int(np.random.randint(-5, 5)),
                                noise_floor=-90, channel=6, bandwidth=20,
                                num_subcarriers=n_sub,
                                amplitude=amp.astype(np.float32),
                                phase=np.zeros(n_sub, dtype=np.float32),
                                raw_complex=np.zeros(n_sub, dtype=np.complex64),
                            )
                            is_last = (node_idx == n_nodes - 1)
                            self.on_frame(frame, trigger_pipeline=is_last)

                        if self.pipeline and self.pipeline.model is None:
                            self.inject_joints(joints_batch[t].astype(np.float32))

                        elapsed = asyncio.get_event_loop().time() - loop_start
                        await asyncio.sleep(max(0, dt - elapsed))

                    await asyncio.sleep(1.0)
                except Exception as e:
                    logger.error("Simulation error: %s", e)
                    await asyncio.sleep(5.0)
```

- [ ] **Step 4: Implement vitals service**

```python
# server/services/vitals_service.py
"""Vital signs extraction service — wraps VitalSignsExtractor with event emission."""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from server.vital_signs import VitalSignsExtractor, MultiPersonTracker
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class VitalsService:
    def __init__(
        self,
        sample_rate: float,
        emitter: EventEmitter,
        emit_interval_sec: float = 1.0,
    ):
        self._emitter = emitter
        self._emit_interval = emit_interval_sec
        self._last_emit = 0.0
        self.extractor = VitalSignsExtractor(sample_rate=sample_rate)
        self.multi_person = MultiPersonTracker(
            max_persons=4, sample_rate=sample_rate,
        )

    def push_csi(self, amplitudes: np.ndarray) -> None:
        """Push CSI amplitude and emit vitals at throttled rate."""
        self.extractor.push_csi(amplitudes)
        now = time.time()
        if now - self._last_emit >= self._emit_interval:
            self._last_emit = now
            vitals = self.extractor.update()
            asyncio.ensure_future(self._emitter.emit("vitals", vitals))

    def get_vitals(self) -> dict:
        return self.extractor.update()

    def get_subcarrier_amplitudes(self) -> list[float] | None:
        return self.extractor.get_subcarrier_amplitudes()
```

```python
# tests/test_vitals_service.py
import asyncio
import pytest
import numpy as np
from server.services.event_emitter import EventEmitter
from server.services.vitals_service import VitalsService


@pytest.fixture
def emitter():
    return EventEmitter()


@pytest.fixture
def service(emitter):
    return VitalsService(sample_rate=100.0, emitter=emitter)


def test_push_csi(service):
    amp = np.random.rand(56).astype(np.float32)
    service.push_csi(amp)
    assert len(service.extractor.csi_buffer) == 1


@pytest.mark.asyncio
async def test_emits_vitals_on_interval(emitter):
    results = []
    async def on_vitals(data): results.append(data)
    emitter.on("vitals", on_vitals)
    svc = VitalsService(sample_rate=100.0, emitter=emitter, emit_interval_sec=0.0)
    svc.push_csi(np.random.rand(56).astype(np.float32))
    await asyncio.sleep(0.1)
    assert len(results) >= 1
    assert "breathing_bpm" in results[0]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_pipeline_service.py tests/test_vitals_service.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add server/services/pipeline_service.py server/services/vitals_service.py tests/test_pipeline_service.py tests/test_vitals_service.py
git commit -m "feat(services): add PipelineService and VitalsService with event emission"
```

---

### Task 7: Remaining Services (Calibration, Storage, Notification)

**Files:**
- Create: `server/services/calibration_service.py`
- Create: `server/services/storage_service.py`
- Create: `server/services/notification_service.py`
- Test: `tests/test_storage_service.py`

- [ ] **Step 1: Write failing test for storage service**

```python
# tests/test_storage_service.py
import asyncio
import pytest
import numpy as np
from server.services.event_emitter import EventEmitter
from server.services.storage_service import StorageService


@pytest.fixture
def emitter():
    return EventEmitter()


@pytest.fixture
def service(emitter, tmp_path):
    db_path = str(tmp_path / "test.db")
    return StorageService(db_path=db_path, emitter=emitter)


@pytest.mark.asyncio
async def test_subscribes_to_pose_events(emitter, service):
    """StorageService should save poses from events (throttled)."""
    service._last_pose_save = 0  # force save
    await emitter.emit("pose", {"joints": [[0, 0, 0]] * 24, "confidence": 0.5})
    await asyncio.sleep(0.1)
    stats = service.storage.get_stats()
    assert stats["poses"] >= 1


@pytest.mark.asyncio
async def test_subscribes_to_vitals_events(emitter, service):
    service._last_vitals_save = 0
    await emitter.emit("vitals", {"breathing_bpm": 16.0, "heart_bpm": 72.0})
    await asyncio.sleep(0.1)
    stats = service.storage.get_stats()
    assert stats["vitals"] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_storage_service.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement remaining services**

```python
# server/services/calibration_service.py
"""Calibration service — wraps CalibrationManager."""
from __future__ import annotations

from server.calibration import CalibrationManager
from server.csi_frame import CSIFrame


class CalibrationService:
    def __init__(self):
        self.manager = CalibrationManager()

    @property
    def is_active(self) -> bool:
        return self.manager.is_active

    def start(self, mode: str = "spatial") -> dict:
        return self.manager.start(mode=mode)

    def finish(self) -> dict:
        return self.manager.finish()

    def on_frame(self, frame: CSIFrame) -> None:
        if self.manager.is_active:
            self.manager.on_csi_frame(frame)

    def get_status(self) -> dict:
        return self.manager.get_status()

    def get_node_positions(self) -> dict:
        return self.manager.get_node_positions()

    def get_reference_csi(self) -> dict:
        return self.manager.get_reference_csi()

    def get_background_profile(self):
        return self.manager.get_background_profile()
```

```python
# server/services/storage_service.py
"""Storage service — subscribes to events and persists data (throttled)."""
from __future__ import annotations

import logging
import time

import numpy as np

from server.storage import Storage
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

POSE_SAVE_INTERVAL = 1.0    # seconds
VITALS_SAVE_INTERVAL = 5.0  # seconds


class StorageService:
    def __init__(self, db_path: str, emitter: EventEmitter):
        self.storage = Storage(db_path)
        self._emitter = emitter
        self._last_pose_save = 0.0
        self._last_vitals_save = 0.0
        # Subscribe to events
        emitter.on("pose", self._on_pose)
        emitter.on("vitals", self._on_vitals)

    async def _on_pose(self, data: dict) -> None:
        now = time.time()
        if now - self._last_pose_save < POSE_SAVE_INTERVAL:
            return
        self._last_pose_save = now
        joints = np.array(data["joints"], dtype=np.float32)
        self.storage.save_pose(joints)

    async def _on_vitals(self, data: dict) -> None:
        now = time.time()
        if now - self._last_vitals_save < VITALS_SAVE_INTERVAL:
            return
        self._last_vitals_save = now
        self.storage.save_vitals(data)

    def close(self) -> None:
        self.storage.close()
```

```python
# server/services/notification_service.py
"""Notification service — subscribes to alert events."""
from __future__ import annotations

import logging

from server.notifier import Notifier, FallNotification
from server.services.event_emitter import EventEmitter

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self, notifier: Notifier, emitter: EventEmitter):
        self.notifier = notifier
        self._emitter = emitter
        emitter.on("fall_alert", self._on_alert)

    async def _on_alert(self, data: dict) -> None:
        if not self.notifier.enabled:
            return
        notif = FallNotification(
            timestamp=data.get("timestamp", 0),
            confidence=data.get("confidence", 0),
            head_height=data.get("head_height", 0),
            velocity=data.get("velocity", 0),
            alert_id=data.get("alert_id", 0),
        )
        self.notifier.send_fall_alert(notif)

    def close(self) -> None:
        self.notifier.close()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_storage_service.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/services/calibration_service.py server/services/storage_service.py server/services/notification_service.py tests/test_storage_service.py
git commit -m "feat(services): add CalibrationService, StorageService, NotificationService"
```

---

### Task 8: Update Container with All Services

**Files:**
- Modify: `server/services/container.py`

- [ ] **Step 1: Update container to wire all services**

```python
# server/services/container.py — REPLACE full file
"""Dependency injection container — manages service lifecycles."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from server.config import Settings
from server.notifier import Notifier
from server.services.event_emitter import EventEmitter
from server.services.websocket_service import WebSocketService
from server.services.pipeline_service import PipelineService
from server.services.vitals_service import VitalsService
from server.services.calibration_service import CalibrationService
from server.services.storage_service import StorageService
from server.services.notification_service import NotificationService

logger = logging.getLogger(__name__)


def _load_pipeline(settings):
    """Create PosePipeline, loading model weights if available."""
    from server.pipeline import PosePipeline
    model = None
    model_path = Path(settings.model_path)
    if model_path.exists():
        try:
            from server.pose_model import load_model
            import torch
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
            first_key = [k for k in ckpt.keys() if "encoder.0.weight" in k]
            input_dim = ckpt[first_key[0]].shape[1] if first_key else settings.num_subcarriers * settings.max_nodes
            model = load_model(str(model_path), input_dim=input_dim)
            logger.info("Pose model loaded from %s (input_dim=%d)", model_path, input_dim)
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", model_path, e)
    else:
        logger.info("No model weights at %s — pipeline will run without inference", model_path)
    return PosePipeline(settings, model=model)


@dataclass
class ServiceContainer:
    settings: Settings
    emitter: EventEmitter = field(default_factory=EventEmitter)
    websocket: WebSocketService = field(init=False)
    pipeline_svc: PipelineService = field(init=False)
    vitals: VitalsService = field(init=False)
    calibration: CalibrationService = field(init=False)
    storage: StorageService = field(init=False)
    notification: NotificationService = field(init=False)

    def __post_init__(self):
        s = self.settings
        self.websocket = WebSocketService(emitter=self.emitter)
        pipeline = _load_pipeline(s)
        # Wire calibration + vitals into pipeline settings
        self.calibration = CalibrationService()
        self.vitals = VitalsService(sample_rate=s.csi_sample_rate, emitter=self.emitter)
        s.calibration_manager = self.calibration.manager
        s.vitals_extractor = self.vitals.extractor

        self.pipeline_svc = PipelineService(
            settings=s, emitter=self.emitter, pipeline=pipeline,
        )
        self.storage = StorageService(db_path=s.db_path, emitter=self.emitter)
        notifier = Notifier(
            webhook_url=s.notify_webhook_url,
            line_token=s.notify_line_token,
            telegram_bot_token=s.notify_telegram_bot_token,
            telegram_chat_id=s.notify_telegram_chat_id,
        )
        self.notification = NotificationService(notifier=notifier, emitter=self.emitter)

        # Subscribe WebSocket to all streams
        self.emitter.on("pose", self._on_pose_for_ws)
        self.emitter.on("vitals", self._on_vitals_for_ws)
        self.emitter.on("csi", self._on_csi_for_ws)

    async def _on_pose_for_ws(self, data):
        from server.protocol.envelope import make_envelope, PoseData
        env = make_envelope("pose", PoseData(**data))
        await self.websocket.broadcast_envelope(env)

    async def _on_vitals_for_ws(self, data):
        from server.protocol.envelope import make_envelope, VitalsData
        env = make_envelope("vitals", VitalsData(**data))
        await self.websocket.broadcast_envelope(env)

    async def _on_csi_for_ws(self, data):
        from server.protocol.envelope import make_envelope, CsiData
        env = make_envelope("csi", CsiData(**data))
        await self.websocket.broadcast_envelope(env)

    async def startup(self) -> None:
        logger.info("ServiceContainer starting up")
        await self.websocket.start_heartbeat()
        if self.settings.simulate:
            await self.pipeline_svc.start_simulation()

    async def shutdown(self) -> None:
        logger.info("ServiceContainer shutting down")
        await self.pipeline_svc.stop_simulation()
        await self.websocket.stop_heartbeat()
        self.storage.close()
        self.notification.close()
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All existing + new tests pass

- [ ] **Step 3: Commit**

```bash
git add server/services/container.py
git commit -m "feat(services): wire all services into ServiceContainer with event subscriptions"
```

---

### Task 9: Routes — Extract from api.py

**Files:**
- Create: `server/routes/__init__.py`
- Create: `server/routes/system.py`
- Create: `server/routes/calibration.py`
- Create: `server/routes/data.py`
- Create: `server/routes/ws.py`
- Modify: `server/api.py`

- [ ] **Step 1: Create route files extracting endpoints from api.py**

```python
# server/routes/__init__.py
"""API route modules."""

# server/routes/system.py
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
```

```python
# server/routes/calibration.py
"""Calibration routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from server.services.container import ServiceContainer

router = APIRouter(prefix="/api/calibration")


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


@router.post("/start")
async def calibration_start(mode: str = "spatial", duration: float = 5.0,
                            container: ServiceContainer = Depends(get_container)):
    cal = container.calibration
    if cal.is_active:
        return JSONResponse({"error": "Calibration already in progress"}, status_code=409)
    return cal.start(mode=mode)


@router.post("/finish")
async def calibration_finish(container: ServiceContainer = Depends(get_container)):
    cal = container.calibration
    result = cal.finish()
    if result.get("status") == "complete":
        container.storage.storage.save_calibration(
            container.settings.hardware_profile,
            cal.get_node_positions(),
            cal.get_reference_csi(),
        )
    return result


@router.get("/status")
async def calibration_status(container: ServiceContainer = Depends(get_container)):
    return container.calibration.get_status()
```

```python
# server/routes/data.py
"""Data routes: history, collection, OTA, notifications."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, FileResponse

from server.services.container import ServiceContainer
from server.notifier import FallNotification

router = APIRouter()


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


# ── History ─────────────────────────────────────────
@router.get("/api/history/poses")
async def history_poses(limit: int = 100, container: ServiceContainer = Depends(get_container)):
    return {"poses": container.storage.storage.get_recent_poses(limit)}

@router.get("/api/history/vitals")
async def history_vitals(limit: int = 100, container: ServiceContainer = Depends(get_container)):
    return {"vitals": container.storage.storage.get_recent_vitals(limit)}

@router.get("/api/history/alerts")
async def history_alerts(limit: int = 50, container: ServiceContainer = Depends(get_container)):
    return {"alerts": container.storage.storage.get_fall_alerts(limit)}


# ── Notifications ───────────────────────────────────
@router.get("/api/notifications/status")
async def notification_status(container: ServiceContainer = Depends(get_container)):
    n = container.notification.notifier
    return {"enabled": n.enabled, "channels": n._channels}

@router.post("/api/notifications/test")
async def notification_test(container: ServiceContainer = Depends(get_container)):
    import time as _time
    n = container.notification.notifier
    if not n.enabled:
        return JSONResponse({"error": "No notification channels configured"}, status_code=400)
    results = n.send_fall_alert(FallNotification(
        timestamp=_time.time(), confidence=0.95,
        head_height=0.3, velocity=1.5, alert_id=0,
    ))
    return {"results": results}


# ── Data Collection ─────────────────────────────────
@router.post("/api/collect/start")
async def collect_start(activity: str = "standing",
                        container: ServiceContainer = Depends(get_container)):
    from server.real_collector import RealDataCollector, ACTIVITIES
    if activity not in ACTIVITIES:
        return JSONResponse({"error": f"Unknown activity. Choose from: {ACTIVITIES}"}, status_code=400)
    # Use a simple flag on container for collection state
    if getattr(container, '_collecting', False):
        return JSONResponse({"error": "Already collecting"}, status_code=409)
    collector = getattr(container, '_collector', None)
    if collector is None:
        s = container.settings
        collector = RealDataCollector(settings=s, n_nodes=s.max_nodes)
        container._collector = collector
    collector.start_recording(activity)
    container._collecting = True
    container._collect_activity = activity
    container._collect_frames = 0
    return {"status": "recording", "activity": activity}

@router.post("/api/collect/stop")
async def collect_stop(container: ServiceContainer = Depends(get_container)):
    collector = getattr(container, '_collector', None)
    if not getattr(container, '_collecting', False) or collector is None:
        return JSONResponse({"error": "Not collecting"}, status_code=409)
    data = collector.stop_recording()
    container._collecting = False
    if data is None:
        return {"status": "stopped", "frames": 0, "file": None}
    path = collector.save_sequence(data, "data/real")
    return {"status": "saved", "frames": len(data["joints"]), "file": path,
            "activity": getattr(container, '_collect_activity', '')}

@router.get("/api/collect/status")
async def collect_status(container: ServiceContainer = Depends(get_container)):
    from server.real_collector import ACTIVITIES
    collector = getattr(container, '_collector', None)
    return {
        "collecting": getattr(container, '_collecting', False),
        "activity": getattr(container, '_collect_activity', ''),
        "frames": getattr(container, '_collect_frames', 0),
        "csi_frames": collector._csi_frame_count if collector else 0,
        "nodes_seen": len(collector._csi_buffer) if collector else 0,
        "activities": ACTIVITIES,
    }


# ── OTA ─────────────────────────────────────────────
@router.get("/api/ota/firmware")
async def ota_firmware_list():
    fw_dir = Path(__file__).parent.parent.parent / "firmware" / "esp32-csi-node" / "build"
    bins = []
    if fw_dir.exists():
        for f in sorted(fw_dir.glob("*.bin")):
            bins.append({"name": f.name, "size": f.stat().st_size, "path": f"/api/ota/download/{f.name}"})
    return {"firmware": bins, "ota_endpoint": "/api/ota/download/<filename>"}

@router.get("/api/ota/download/{filename}")
async def ota_download(filename: str):
    if not re.match(r'^[\w\-\.]+$', filename):
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    fw_path = Path(__file__).parent.parent.parent / "firmware" / "esp32-csi-node" / "build" / filename
    if not fw_path.exists():
        return JSONResponse({"error": "Firmware not found"}, status_code=404)
    return FileResponse(str(fw_path), media_type="application/octet-stream")
```

```python
# server/routes/ws.py
"""WebSocket endpoint."""
from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.services.container import ServiceContainer

router = APIRouter()


@router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    container: ServiceContainer = websocket.app.state.container
    await websocket.accept()
    conn = container.websocket.register(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            try:
                raw = json.loads(text)
                response = container.websocket.handle_message(websocket, raw)
                if response:
                    await websocket.send_text(json.dumps(response))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        container.websocket.unregister(websocket)
```

- [ ] **Step 2: Rewrite api.py to use routes + container**

```python
# server/api.py — REPLACE full file (slim version ~80 lines)
"""FastAPI server — slim app shell with router mounts and service container."""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import Settings
from server.csi_receiver import CSIReceiver
from server.services.container import ServiceContainer
from server.routes import system, calibration, data, ws

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    profile = settings.apply_hardware_profile()
    if profile:
        logger.info(
            "Hardware profile: %s (%s, %d sub, %d Hz, %.1f GHz)",
            profile.id, profile.name, profile.num_subcarriers,
            profile.csi_sample_rate, profile.frequency_ghz,
        )

    container = ServiceContainer(settings=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container

        # Start CSI receiver
        receiver = CSIReceiver(settings)
        def on_csi(frame, trigger_pipeline=True):
            if container.calibration.is_active:
                container.calibration.on_frame(frame)
            if frame.amplitude is not None:
                import numpy as np
                container.vitals.push_csi(frame.amplitude.astype(np.float32))
            container.pipeline_svc.on_frame(frame, trigger_pipeline=trigger_pipeline)
        receiver.on_frame = on_csi
        recv_task = asyncio.create_task(receiver.start())

        await container.startup()

        logger.info(
            "WiFi Body server started (model_loaded=%s, simulate=%s)",
            container.pipeline_svc.pipeline is not None
            and container.pipeline_svc.pipeline.model is not None,
            settings.simulate,
        )
        yield

        receiver.stop()
        await container.shutdown()

    app = FastAPI(title="WiFi Body", version="0.2.0", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    # Mount routes
    app.include_router(system.router)
    app.include_router(calibration.router)
    app.include_router(data.router)
    app.include_router(ws.router)

    # Static files
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir)), name="dashboard")

    return app
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All 125+ tests pass (existing test_api.py tests unchanged — they call `create_app()` and test HTTP endpoints, which still work)

- [ ] **Step 4: Commit**

```bash
git add server/routes/ server/api.py
git commit -m "refactor: extract routes from api.py into router modules, wire ServiceContainer"
```

- [ ] **Step 5: Phase 2 complete marker**

```bash
git commit --allow-empty -m "milestone: Phase 2 (Server Service Layer) complete"
```

---

## Phase 3: Dashboard Vite + ES6 Modules

### Task 10: Vite Scaffold + EventBus + Utils

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/vite.config.js`
- Create: `dashboard/src/main.js`
- Create: `dashboard/src/events.js`
- Create: `dashboard/src/utils/resize.js`
- Create: `dashboard/src/utils/format.js`

- [ ] **Step 1: Initialize npm project and install Vite**

```bash
cd "D:/product/WIFI body/dashboard"
npm init -y
npm install --save-dev vite
npm install three
```

- [ ] **Step 2: Create vite.config.js**

```javascript
// dashboard/vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: '../dist/dashboard',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
```

- [ ] **Step 3: Create EventBus**

```javascript
// dashboard/src/events.js
/**
 * Lightweight EventEmitter — app-wide communication backbone.
 * Replaces global variable coupling between modules.
 */
export class EventBus {
  constructor() {
    this._handlers = {};
  }

  on(event, handler) {
    if (!this._handlers[event]) this._handlers[event] = [];
    this._handlers[event].push(handler);
  }

  off(event, handler) {
    const list = this._handlers[event];
    if (!list) return;
    const idx = list.indexOf(handler);
    if (idx >= 0) list.splice(idx, 1);
  }

  emit(event, data) {
    const list = this._handlers[event];
    if (!list) return;
    for (const handler of list) {
      try {
        handler(data);
      } catch (e) {
        console.error(`EventBus error on '${event}':`, e);
      }
    }
  }
}

export const bus = new EventBus();
```

- [ ] **Step 4: Create utils**

```javascript
// dashboard/src/utils/resize.js
/**
 * ResizeObserver wrapper — replaces per-frame resize hack.
 * Only fires callback when dimensions are positive (tab visible).
 */
export function observeResize(element, callback) {
  const ro = new ResizeObserver((entries) => {
    const { width, height } = entries[0].contentRect;
    if (width > 0 && height > 0) callback(width, height);
  });
  ro.observe(element);
  return () => ro.disconnect();
}

// dashboard/src/utils/format.js
/**
 * Number formatting utilities.
 */
export function formatBpm(value) {
  return value > 0 ? value.toFixed(1) : '--';
}

export function formatPercent(value) {
  return value > 0 ? (value * 100).toFixed(0) + '%' : '--';
}

export function formatConfidence(value) {
  if (value >= 0.8) return 'High';
  if (value >= 0.5) return 'Medium';
  return 'Low';
}
```

- [ ] **Step 5: Create main.js entry point (skeleton)**

```javascript
// dashboard/src/main.js
/**
 * WiFi Body Dashboard — entry point.
 * Initializes EventBus, WebSocket client, and tab router.
 */
import { bus } from './events.js';

// Will be populated in subsequent tasks
console.log('WiFi Body Dashboard initialized');
console.log('EventBus ready:', bus);
```

- [ ] **Step 6: Verify Vite serves the app**

```bash
cd "D:/product/WIFI body/dashboard" && npx vite --host 0.0.0.0 &
# Open http://localhost:5173 — should show blank page with console log
```

- [ ] **Step 7: Commit**

```bash
git add dashboard/package.json dashboard/vite.config.js dashboard/src/
git commit -m "feat(dashboard): scaffold Vite project with EventBus and utils"
```

---

### Task 11: WebSocket Client (v1 Protocol)

**Files:**
- Create: `dashboard/src/connection/ws-client.js`
- Create: `dashboard/src/connection/protocol.js`

- [ ] **Step 1: Create protocol parser**

```javascript
// dashboard/src/connection/protocol.js
/**
 * v1 envelope parsing and v0 fallback detection.
 */

export function isV1Envelope(data) {
  return data && data.v === 1 && typeof data.type === 'string';
}

export function parseEnvelope(raw) {
  try {
    const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (isV1Envelope(data)) {
      return { version: 1, type: data.type, data: data.data, seq: data.seq, ts: data.ts };
    }
    // v0 fallback: legacy single-payload format
    return { version: 0, type: 'legacy', data: data };
  } catch (e) {
    return null;
  }
}
```

- [ ] **Step 2: Create WebSocket client**

```javascript
// dashboard/src/connection/ws-client.js
/**
 * WebSocket client with v1 protocol, auto-reconnect, and heartbeat.
 */
import { bus } from '../events.js';
import { parseEnvelope } from './protocol.js';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;
const HEARTBEAT_INTERVAL_MS = 30000;

export class WsClient {
  constructor(url) {
    this._url = url;
    this._ws = null;
    this._reconnectAttempt = 0;
    this._heartbeatTimer = null;
    this._capabilities = ['pose', 'vitals', 'csi', 'status'];
  }

  connect() {
    try {
      this._ws = new WebSocket(this._url);
      this._ws.onopen = () => this._onOpen();
      this._ws.onmessage = (e) => this._onMessage(e.data);
      this._ws.onclose = () => this._onClose();
      this._ws.onerror = () => {}; // onclose will fire
    } catch (e) {
      this._scheduleReconnect();
    }
  }

  send(obj) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(obj));
    }
  }

  _onOpen() {
    this._reconnectAttempt = 0;
    bus.emit('ws:connected');
    // Send v1 hello
    this.send({ v: 1, type: 'hello', capabilities: this._capabilities });
  }

  _onMessage(raw) {
    const parsed = parseEnvelope(raw);
    if (!parsed) return;

    if (parsed.version === 1) {
      if (parsed.type === 'ping') {
        this.send({ v: 1, type: 'pong', ts: Date.now() });
        return;
      }
      if (parsed.type === 'welcome') {
        bus.emit('ws:welcome', parsed.data);
        return;
      }
      // Emit by type: pose, vitals, csi, status, error
      bus.emit(parsed.type, parsed.data);
    } else {
      // v0 legacy: emit as individual streams
      const d = parsed.data;
      if (d.joints) bus.emit('pose', { joints: d.joints, confidence: 0 });
      if (d.vitals) bus.emit('vitals', d.vitals);
      if (d.csi_amplitudes) bus.emit('csi', { amplitudes: d.csi_amplitudes });
    }
  }

  _onClose() {
    bus.emit('ws:disconnected');
    this._scheduleReconnect();
  }

  _scheduleReconnect() {
    const delay = Math.min(
      RECONNECT_BASE_MS * Math.pow(2, this._reconnectAttempt),
      RECONNECT_MAX_MS,
    );
    this._reconnectAttempt++;
    setTimeout(() => this.connect(), delay);
  }

  // Heartbeat: server sends ping, client responds with pong in _onMessage
}
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/connection/
git commit -m "feat(dashboard): add WsClient with v1 protocol and auto-reconnect"
```

---

### Task 12: Tab Manager + Skeleton Tab Controllers

**Files:**
- Create: `dashboard/src/tabs/tab-manager.js`
- Create: `dashboard/src/tabs/viewer.js` (stub)
- Create: `dashboard/src/tabs/dashboard.js` (stub)
- Create: `dashboard/src/tabs/hardware.js` (stub)
- Create: `dashboard/src/tabs/demo.js` (stub)
- Create: `dashboard/src/tabs/sensing.js` (stub)
- Create: `dashboard/src/tabs/architecture.js` (stub)
- Create: `dashboard/src/tabs/performance.js` (stub)

- [ ] **Step 1: Create tab manager**

```javascript
// dashboard/src/tabs/tab-manager.js
/**
 * Tab switching with lazy initialization.
 * Each tab controller must export: { id, label, init(), activate(), deactivate() }
 */
import { bus } from '../events.js';

const tabs = {};
const initialized = new Set();
let activeTabId = null;

export function registerTab(controller) {
  tabs[controller.id] = controller;
}

export function switchTab(tabId) {
  if (!tabs[tabId]) return;
  if (activeTabId && tabs[activeTabId]) {
    tabs[activeTabId].deactivate();
  }
  if (!initialized.has(tabId)) {
    tabs[tabId].init();
    initialized.add(tabId);
  }
  tabs[tabId].activate();
  activeTabId = tabId;
  bus.emit('tab:changed', tabId);
}

export function getRegisteredTabs() {
  return Object.values(tabs);
}

export function getActiveTabId() {
  return activeTabId;
}
```

- [ ] **Step 2: Create stub tab controllers** (one example shown; repeat pattern for all 7)

```javascript
// dashboard/src/tabs/viewer.js
/**
 * 3D Viewer tab — Three.js scene with skeleton + body mesh.
 * Full implementation will be migrated from skeleton3d.js in a later step.
 */
export default {
  id: 'viewer',
  label: '3D Viewer',

  init() {
    // Will initialize Three.js scene
    console.log('Viewer tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-viewer');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-viewer');
    if (el) el.style.display = 'none';
  },
};
```

Create similar stubs for: `dashboard.js`, `hardware.js`, `demo.js`, `sensing.js`, `architecture.js`, `performance.js` — each with matching id/label and empty init().

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/tabs/
git commit -m "feat(dashboard): add tab manager with lazy init and stub controllers"
```

---

### Task 13: Migrate Three.js Scene Modules

**Files:**
- Create: `dashboard/src/scene/three-setup.js`
- Create: `dashboard/src/scene/skeleton.js`
- Create: `dashboard/src/scene/body-mesh.js`
- Create: `dashboard/src/scene/room.js`

This is the largest migration task. Extract from `skeleton3d.js`:
- Three.js init, camera, lighting, controls → `three-setup.js`
- Skeleton rendering (24 joints) → `skeleton.js`
- SMPL body mesh → `body-mesh.js`
- Room geometry + node markers → `room.js`

- [ ] **Step 1: Read current skeleton3d.js carefully**

Read: `dashboard/skeleton3d.js` in full. Map each function/block to its target module.

- [ ] **Step 2: Extract three-setup.js**

Extract: Scene creation, camera, renderer, OrbitControls, lighting, animation loop.
Key: Use `observeResize()` from utils instead of per-frame resize.

- [ ] **Step 3: Extract skeleton.js, body-mesh.js, room.js**

Each module exports functions that operate on the shared scene. They subscribe to `bus.on('pose', ...)` to update.

- [ ] **Step 4: Update viewer.js tab controller to wire scene modules**

The viewer tab's `init()` calls `createScene()`, `createSkeleton()`, `createRoom()`, and subscribes to events.

- [ ] **Step 5: Verify 3D viewer works**

Run Vite dev server, open viewer tab, confirm skeleton renders with simulated data.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/scene/
git commit -m "feat(dashboard): extract Three.js scene into ES6 modules"
```

---

### Task 14: Migrate Vitals Rendering Modules

**Files:**
- Create: `dashboard/src/vitals/hud.js`
- Create: `dashboard/src/vitals/waveform.js`
- Create: `dashboard/src/vitals/waterfall.js`
- Create: `dashboard/src/vitals/heatmap.js`
- Create: `dashboard/src/simulation/demo-data.js`

- [ ] **Step 1: Read current vitals-hud.js carefully**

Read: `dashboard/vitals-hud.js` in full. Map simulation logic vs rendering logic.

- [ ] **Step 2: Extract demo-data.js (simulation)**

Move client-side vital signs simulation (lines ~14-112 of vitals-hud.js) into `simulation/demo-data.js`. This runs when no WebSocket data is available.

- [ ] **Step 3: Extract hud.js, waveform.js, waterfall.js, heatmap.js**

Each is a pure rendering module that subscribes to `bus.on('vitals', ...)` or `bus.on('csi', ...)`.

- [ ] **Step 4: Update tab controllers to wire vitals modules**

Dashboard tab and Sensing tab wire up the appropriate vitals renderers.

- [ ] **Step 5: Verify vitals HUD displays**

Run Vite dev server, confirm HUD overlay, waveforms, waterfall render correctly.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/vitals/ dashboard/src/simulation/
git commit -m "feat(dashboard): extract vitals rendering into ES6 modules"
```

---

### Task 15: Split CSS + Update index.html

**Files:**
- Create: `dashboard/styles/main.css`
- Create: `dashboard/styles/tabs.css`
- Create: `dashboard/styles/hud.css`
- Create: `dashboard/styles/cards.css`
- Create: `dashboard/styles/effects.css`
- Modify: `dashboard/index.html`

- [ ] **Step 1: Split styles.css into component files**

Read `dashboard/styles.css`. Split by concern:
- `main.css`: custom properties, reset, body, typography
- `tabs.css`: tab-bar, tab-content, navigation
- `hud.css`: HUD overlay, vital sign displays
- `cards.css`: health-card, status-card, profile-card
- `effects.css`: CRT scanline, glow, animations

- [ ] **Step 2: Update index.html**

Slim down to ~50 lines. Import `src/main.js` as module. CSS imports via Vite.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WiFi Body Sensing Dashboard</title>
  <link rel="stylesheet" href="/styles/main.css">
  <link rel="stylesheet" href="/styles/tabs.css">
  <link rel="stylesheet" href="/styles/hud.css">
  <link rel="stylesheet" href="/styles/cards.css">
  <link rel="stylesheet" href="/styles/effects.css">
</head>
<body>
  <div id="app">
    <header id="app-header">
      <h1>WiFi Body</h1>
      <div id="connection-status"></div>
      <div id="mode-badge"></div>
    </header>
    <nav id="tab-bar"></nav>
    <main id="tab-content">
      <section id="tab-viewer" class="tab-panel"></section>
      <section id="tab-dashboard" class="tab-panel"></section>
      <section id="tab-hardware" class="tab-panel"></section>
      <section id="tab-demo" class="tab-panel"></section>
      <section id="tab-sensing" class="tab-panel"></section>
      <section id="tab-architecture" class="tab-panel"></section>
      <section id="tab-performance" class="tab-panel"></section>
    </main>
  </div>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

- [ ] **Step 3: Update main.js to register tabs and connect WS**

```javascript
// dashboard/src/main.js
import { bus } from './events.js';
import { WsClient } from './connection/ws-client.js';
import { registerTab, switchTab, getRegisteredTabs } from './tabs/tab-manager.js';
import viewer from './tabs/viewer.js';
import dashboard from './tabs/dashboard.js';
import hardware from './tabs/hardware.js';
import demo from './tabs/demo.js';
import sensing from './tabs/sensing.js';
import architecture from './tabs/architecture.js';
import performance from './tabs/performance.js';

// Register all tabs
[viewer, dashboard, hardware, demo, sensing, architecture, performance]
  .forEach(registerTab);

// Build tab bar
const tabBar = document.getElementById('tab-bar');
getRegisteredTabs().forEach((tab) => {
  const btn = document.createElement('button');
  btn.textContent = tab.label;
  btn.dataset.tab = tab.id;
  btn.addEventListener('click', () => switchTab(tab.id));
  tabBar.appendChild(btn);
});

// Highlight active tab
bus.on('tab:changed', (tabId) => {
  tabBar.querySelectorAll('button').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
});

// Connect WebSocket
const wsUrl = `ws://${location.hostname}:${location.port || 8000}/ws/pose`;
const client = new WsClient(wsUrl);
client.connect();

// Connection status indicator
const statusEl = document.getElementById('connection-status');
bus.on('ws:connected', () => { statusEl.textContent = 'Connected'; statusEl.className = 'connected'; });
bus.on('ws:disconnected', () => { statusEl.textContent = 'Reconnecting...'; statusEl.className = 'disconnected'; });

// Default to viewer tab
switchTab('viewer');
```

- [ ] **Step 4: Verify full dashboard works via Vite**

```bash
cd "D:/product/WIFI body/dashboard" && npx vite
```

Open http://localhost:5173 — verify all 7 tabs render, WebSocket connects to backend.

- [ ] **Step 5: Commit**

```bash
git add dashboard/styles/ dashboard/index.html dashboard/src/main.js
git commit -m "feat(dashboard): split CSS, slim index.html, wire tab manager and WsClient"
```

- [ ] **Step 6: Phase 3 complete marker**

```bash
git commit --allow-empty -m "milestone: Phase 3 (Dashboard Vite + ES6 Modules) complete"
```

---

## Phase 4: Cleanup & Polish

### Task 16: Remove Old Files + v0 Adapter

**Files:**
- Delete: `dashboard/skeleton3d.js`
- Delete: `dashboard/vitals-hud.js`
- Delete: `dashboard/styles.css`

- [ ] **Step 1: Verify new dashboard fully replaces old files**

Run full manual smoke test: all 7 tabs, WebSocket connection, 3D viewer, vitals HUD, waterfall.

- [ ] **Step 2: Delete old files**

```bash
git rm dashboard/skeleton3d.js dashboard/vitals-hud.js dashboard/styles.css
```

- [ ] **Step 3: Update FastAPI static mount for production**

In `server/api.py`, check for `dist/dashboard/` first (Vite build output), fallback to `dashboard/`:

```python
dist_dir = Path(__file__).parent.parent / "dist" / "dashboard"
dashboard_dir = Path(__file__).parent.parent / "dashboard"
static_dir = dist_dir if dist_dir.exists() else dashboard_dir
if static_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(static_dir)), name="dashboard")
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "cleanup: remove old monolithic dashboard files, add production static mount"
```

---

### Task 17: Final Regression + Documentation

- [ ] **Step 1: Run full server test suite**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 2: Build dashboard for production**

```bash
cd "D:/product/WIFI body/dashboard" && npx vite build
```

- [ ] **Step 3: Start server and verify production build**

```bash
python -m server --simulate
```

Open http://localhost:8000/dashboard/ — verify all tabs work with bundled assets.

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "milestone: Phase 4 (Cleanup) complete — architecture optimization done"
```
