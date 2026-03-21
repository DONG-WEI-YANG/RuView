@echo off
:: Setup firewall rules for WiFi Body (requires admin)
:: Called automatically by start-*.bat via UAC elevation

netsh advfirewall firewall show rule name="WiFi Body CSI UDP" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="WiFi Body CSI UDP" dir=in action=allow protocol=UDP localport=5005
    echo [OK] Firewall rule added: UDP 5005
) else (
    echo [OK] Firewall rule exists: UDP 5005
)

netsh advfirewall firewall show rule name="WiFi Body API TCP" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="WiFi Body API TCP" dir=in action=allow protocol=TCP localport=8000
    echo [OK] Firewall rule added: TCP 8000
) else (
    echo [OK] Firewall rule exists: TCP 8000
)
