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
