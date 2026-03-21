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


class PersonData(BaseModel):
    id: int
    joints: list[list[float]] = []
    confidence: float = 0.0
    joint_confidence: list[float] = []
    vitals: dict = {}
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    color: str = "#00ff88"


class PersonsData(BaseModel):
    persons: list[PersonData] = []
    count: int = 0


# ── Envelope ──────────────────────────────────────────────

DataType = Union[PoseData, VitalsData, CsiData, StatusData, ErrorData, PersonsData]

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
