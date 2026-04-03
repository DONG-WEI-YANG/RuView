"""Quick Setup API tests — GET/POST /api/settings/quick, .env persistence."""
import pytest
from pathlib import Path


class TestQuickSetupGet:
    """GET /api/settings/quick returns all tuneable fields."""

    @pytest.mark.asyncio
    async def test_returns_room_dimensions(self, client):
        r = await client.get("/api/settings/quick")
        assert r.status_code == 200
        d = r.json()
        assert "room_width" in d
        assert "room_depth" in d
        assert "room_height" in d
        assert d["room_width"] > 0

    @pytest.mark.asyncio
    async def test_returns_scene_mode(self, client):
        d = (await client.get("/api/settings/quick")).json()
        assert d["scene_mode"] in ("safety", "fitness")

    @pytest.mark.asyncio
    async def test_returns_available_options(self, client):
        d = (await client.get("/api/settings/quick")).json()
        assert "profiles" in d
        assert "scene_modes" in d
        assert len(d["profiles"]) >= 3

    @pytest.mark.asyncio
    async def test_notification_fields_present(self, client):
        d = (await client.get("/api/settings/quick")).json()
        assert "notify_webhook_url" in d
        assert "notify_line_token" in d
        assert "notify_telegram_bot_token" in d
        assert "notify_telegram_chat_id" in d


class TestQuickSetupPost:
    """POST /api/settings/quick applies changes live.
    All POST tests use tmp_path to avoid polluting the project .env.
    """

    @pytest.mark.asyncio
    async def test_change_room_dimensions(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={
            "room_width": 5.0, "room_depth": 6.0, "room_height": 3.0,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "saved"
        assert d["applied"]["room_width"] == 5.0
        # Verify via GET (live settings were updated)
        g = (await client.get("/api/settings/quick")).json()
        assert g["room_width"] == 5.0

    @pytest.mark.asyncio
    async def test_change_scene_mode(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"scene_mode": "fitness"})
        d = r.json()
        assert d["applied"]["scene_mode"] == "fitness"

    @pytest.mark.asyncio
    async def test_invalid_scene_ignored(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"scene_mode": "party"})
        d = r.json()
        assert "scene_mode" not in d["applied"]

    @pytest.mark.asyncio
    async def test_invalid_room_ignored(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"room_width": -1})
        d = r.json()
        assert "room_width" not in d["applied"]

    @pytest.mark.asyncio
    async def test_partial_update(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"room_height": 3.5})
        d = r.json()
        assert len(d["applied"]) == 1
        assert d["applied"]["room_height"] == 3.5

    @pytest.mark.asyncio
    async def test_empty_payload_no_changes(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={})
        d = r.json()
        assert d["applied"] == {}

    @pytest.mark.asyncio
    async def test_notification_url_saved(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={
            "notify_webhook_url": "http://hook.example.com/alert",
        })
        d = r.json()
        assert d["applied"]["notify_webhook_url"] == "http://hook.example.com/alert"

    @pytest.mark.asyncio
    async def test_fall_threshold_in_range(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"fall_threshold": 0.7})
        d = r.json()
        assert d["applied"]["fall_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_fall_threshold_out_of_range(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"fall_threshold": 1.5})
        d = r.json()
        assert "fall_threshold" not in d["applied"]


class TestEnvPersistence:
    """.env file should be written after POST."""

    @pytest.mark.asyncio
    async def test_env_file_created(self, client, tmp_path, monkeypatch):
        """Save creates .env if it doesn't exist."""
        monkeypatch.chdir(tmp_path)
        r = await client.post("/api/settings/quick", json={"room_width": 7.0})
        assert r.json()["status"] == "saved"
        env = tmp_path / ".env"
        assert env.exists()
        text = env.read_text()
        assert "room_width=7.0" in text

    @pytest.mark.asyncio
    async def test_env_file_merges(self, client, tmp_path, monkeypatch):
        """Existing .env lines are preserved; only changed keys are updated."""
        monkeypatch.chdir(tmp_path)
        env = tmp_path / ".env"
        env.write_text("api_port=9000\nroom_width=4.0\n")
        r = await client.post("/api/settings/quick", json={"room_width": 8.0})
        assert r.json()["status"] == "saved"
        text = env.read_text()
        assert "api_port=9000" in text   # preserved
        assert "room_width=8.0" in text  # updated
        assert text.count("room_width") == 1  # not duplicated
