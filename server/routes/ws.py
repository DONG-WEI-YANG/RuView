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
