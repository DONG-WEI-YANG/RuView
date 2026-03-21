@echo off
title WiFi Body - REAL HARDWARE
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8000/dashboard/"
python -m server --port 8000
pause
