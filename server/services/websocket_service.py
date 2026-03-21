"""WebSocket connection management, v0/v1 dispatch, and heartbeat.

All outbound server → client broadcasts MUST go through
``WebSocketService.broadcast_envelope()``.  Direct ``send_text`` calls
outside of this class are not permitted for broadcast traffic; they are only
used here internally and in ws.py for point-to-point client message responses.
"""
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
