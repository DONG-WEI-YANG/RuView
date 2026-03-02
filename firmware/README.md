# ESP32-S3 CSI Node Firmware

Captures WiFi CSI data and streams via UDP to the Python aggregator.

## Prerequisites
- Docker Desktop 28+ (for ESP-IDF build container)
- esptool (pip install esptool)
- CP210x USB-UART driver

## Configuration

Edit esp32-csi-node/sdkconfig.defaults:

| Parameter | Description | Default |
|-----------|-------------|---------|
| CONFIG_CSI_NODE_ID | Unique node ID (0-255) | 1 |
| CONFIG_CSI_WIFI_SSID | Your WiFi network name | -- |
| CONFIG_CSI_WIFI_PASSWORD | Your WiFi password | -- |
| CONFIG_CSI_TARGET_IP | Host PC IP address | 192.168.1.20 |
| CONFIG_CSI_TARGET_PORT | UDP port | 5005 |

## Build

    cd esp32-csi-node
    docker run --rm -v "$(pwd):/project" -w /project \
      espressif/idf:v5.2 bash -c "idf.py set-target esp32s3 && idf.py build"

## Flash

    cd esp32-csi-node/build
    python -m esptool --chip esp32s3 --port COM7 --baud 460800 \
      write_flash --flash_mode dio --flash_freq 80m --flash_size 4MB \
      0x0 bootloader/bootloader.bin \
      0x8000 partition_table/partition-table.bin \
      0x10000 esp32-csi-node.bin

Replace COM7 with your serial port.

## Firewall (Windows)

    netsh advfirewall firewall add rule name="ESP32 CSI" dir=in action=allow protocol=UDP localport=5005
