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
    protocol_version: int | None = None
    subscriptions: set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_pong_ts: float = 0.0

    @property
    def is_v1(self) -> bool:
        return self.protocol_version == 1

    @property
    def is_v0(self) -> bool:
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
