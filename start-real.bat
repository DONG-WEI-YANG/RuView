@echo off
title WiFi Body - REAL HARDWARE
cd /d "%~dp0"

:: Activate virtual environment
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

:: Build dashboard if dist doesn't exist
if not exist "dist\dashboard\index.html" (
    echo Building dashboard...
    cd dashboard
    call npx vite build
    cd ..
)

:: Open dashboard in browser after server starts
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8000/dashboard/"

:: Start server in real hardware mode
python -m server --port 8000
pause
