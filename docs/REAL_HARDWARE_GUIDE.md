# Real Hardware Guide

Your WiFi Body system is ready for real hardware deployment. The current simulation mode is just a layer on top of the real processing pipeline.

## Architecture

The system is designed to run in hybrid mode:
1. **Real CSI Receiver**: Always listening on UDP port 5005.
2. **Simulation Loop**: Optionally injects synthetic data if `--simulate` is used.

This means you can switch to real hardware seamlessly.

## Step-by-Step Deployment

### 1. Hardware Preparation
You need:
- 3-6 ESP32-S3 development boards.
- A WiFi router (2.4GHz) to serve as the signal source.

### 2. Firmware Flashing
The firmware source code is located in `firmware/esp32-csi-node`.

1. **Configure WiFi**:
   Edit `firmware/esp32-csi-node/sdkconfig.defaults`:
   ```properties
   CONFIG_CSI_WIFI_SSID="Your_WiFi_Name"
   CONFIG_CSI_WIFI_PASSWORD="Your_WiFi_Password"
   CONFIG_CSI_TARGET_IP="192.168.1.100"  <-- Your PC's IP address
   CONFIG_CSI_TARGET_PORT=5005
   ```

2. **Flash Nodes**:
   You can use the built-in `esptool` or PlatformIO.
   ```bash
   cd firmware/esp32-csi-node
   idf.py build
   idf.py -p COM3 flash monitor
   ```

### 3. Start Server in Real Mode
Stop the simulation server and run:

```bash
python -m server --profile esp32s3
```

(Do NOT use `--simulate`)

### 4. Verify Connection
Watch the server logs. You should see:
```
[INFO] server.csi_receiver: CSI receiver listening on 0.0.0.0:5005
```
When ESP32 nodes are powered on, you will see packet stats in the Dashboard > **Hardware** tab.

## Troubleshooting

Run the readiness check tool:
```bash
python tools/check_hardware_ready.py
```
