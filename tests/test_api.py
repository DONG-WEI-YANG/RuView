import pytest
from httpx import AsyncClient, ASGITransport
from server.api import create_app


@pytest.mark.asyncio
async def test_root_endpoint():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "wifi-body"


@pytest.mark.asyncio
async def test_status_endpoint():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "is_fallen" in data
