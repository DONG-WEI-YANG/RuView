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
    assert conn.subscriptions == {"pose", "vitals", "csi", "status", "persons"}


def test_pong_updates_last_pong():
    conn = ConnectionState()
    conn.protocol_version = 1
    response = handle_client_message(conn, {"v": 1, "type": "pong", "ts": 12345})
    assert conn.last_pong_ts > 0
    assert response is None


def test_v0_message_detected():
    conn = ConnectionState()
    response = handle_client_message(conn, {"some_key": "some_value"})
    assert conn.protocol_version is None
    assert response is None


def test_is_v0_after_timeout():
    conn = ConnectionState()
    assert conn.is_v0 is False
    conn.connected_at = 0  # pretend connected long ago
    assert conn.is_v0 is True
