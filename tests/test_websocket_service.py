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
