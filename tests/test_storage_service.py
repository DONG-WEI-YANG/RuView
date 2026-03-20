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
