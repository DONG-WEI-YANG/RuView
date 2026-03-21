@echo off
title WiFi Body - REAL HARDWARE MODE
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

:: Kill any existing server on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo Stopping old server (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

:: Wait for port release
timeout /t 1 /nobreak >nul

:: Open dashboard in browser
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000/dashboard/"

:: Start server
python -m server --port 8000
pause
