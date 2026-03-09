@echo off
title WiFi Body - Server + Dashboard
cd /d "%~dp0"

:: Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] No .venv found, using system Python
)

:: Open dashboard in default browser after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

:: Start backend (also serves dashboard static files)
echo.
echo  ==========================================
echo   WiFi Body Pose Estimation Server
echo   Dashboard: http://localhost:8000
echo   Press Ctrl+C to stop
echo  ==========================================
echo.
python -m server %*

pause
