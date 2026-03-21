@echo off
title WiFi Body - REAL HARDWARE
cd /d "%~dp0"

:: Ensure firewall rules (elevate once if needed, non-blocking)
netsh advfirewall firewall show rule name="WiFi Body CSI UDP" >nul 2>&1
if %errorlevel% neq 0 (
    echo Opening firewall for UDP 5005...
    powershell -Command "Start-Process '%~dp0setup-firewall.bat' -Verb RunAs" 2>nul
    timeout /t 3 /nobreak >nul
)

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
