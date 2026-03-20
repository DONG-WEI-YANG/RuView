"""Calibration routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from server.services.container import ServiceContainer

router = APIRouter(prefix="/api/calibration")


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


@router.post("/start")
async def calibration_start(mode: str = "spatial", duration: float = 5.0,
                            container: ServiceContainer = Depends(get_container)):
    cal = container.calibration
    if cal.is_active:
        return JSONResponse({"error": "Calibration already in progress"}, status_code=409)
    return cal.start(mode=mode)


@router.post("/finish")
async def calibration_finish(container: ServiceContainer = Depends(get_container)):
    cal = container.calibration
    result = cal.finish()
    if result.get("status") == "complete":
        container.storage.storage.save_calibration(
            container.settings.hardware_profile,
            cal.get_node_positions(),
            cal.get_reference_csi(),
        )
    return result


@router.get("/status")
async def calibration_status(container: ServiceContainer = Depends(get_container)):
    return container.calibration.get_status()
