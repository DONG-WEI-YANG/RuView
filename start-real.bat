@echo off
title WiFi Body - REAL HARDWARE MODE
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000/dashboard/"
python -m server --profile esp32s3 --port 8000
pause
