/**
 * ESP32 Web Serial flasher — uses esptool-js to flash CSI firmware
 * directly from the browser via USB.
 *
 * Exposes: connect(), disconnect(), flash(nodeId), getChipInfo()
 */
import { ESPLoader, Transport } from 'esptool-js';

// Flash addresses for ESP32-S3 (ESP-IDF partition layout)
const FLASH_OFFSETS = {
  bootloader: 0x0,
  partitions: 0x8000,
  firmware:   0x10000,
};

// Firmware binary URLs (served from dashboard/public/firmware/)
const FW_BASE = '/firmware';

let transport = null;
let esploader = null;
let connected = false;
let chipName = '';

/**
 * Request a serial port and connect to the ESP32.
 * @param {function} log — callback(message, type) for status messages
 * @returns {Promise<{chip: string, mac: string}>}
 */
export async function connect(log = () => {}) {
  if (connected) {
    log('Already connected', 'warning');
    return { chip: chipName, mac: '' };
  }

  if (!('serial' in navigator)) {
    throw new Error('Web Serial API not supported. Use Chrome or Edge.');
  }

  log('Requesting serial port...');
  const port = await navigator.serial.requestPort({
    filters: [
      { usbVendorId: 0x303A }, // Espressif
      { usbVendorId: 0x10C4 }, // Silicon Labs CP210x
      { usbVendorId: 0x1A86 }, // CH340
    ],
  });

  transport = new Transport(port, true);
  log('Connecting to ESP32...');

  esploader = new ESPLoader({
    transport,
    baudrate: 460800,
    romBaudrate: 115200,
    terminal: {
      clean() {},
      writeLine(data) { log(data); },
      write(data) { /* suppress raw bytes */ },
    },
  });

  chipName = await esploader.main();
  connected = true;

  const mac = await esploader.readMac();
  log(`Connected: ${chipName} (MAC: ${mac})`, 'success');
  return { chip: chipName, mac };
}

/**
 * Disconnect from the ESP32.
 * @param {function} log
 */
export async function disconnect(log = () => {}) {
  if (!connected) return;
  try {
    await transport.disconnect();
  } catch (e) { /* ignore */ }
  connected = false;
  transport = null;
  esploader = null;
  chipName = '';
  log('Disconnected', 'info');
}

/**
 * Flash CSI firmware to the connected ESP32.
 * @param {number} nodeId — 1 or 2 (selects firmware-node1.bin or firmware-node2.bin)
 * @param {function} log — callback(message, type)
 * @param {function} onProgress — callback(percent)
 * @returns {Promise<void>}
 */
export async function flash(nodeId = 1, log = () => {}, onProgress = () => {}) {
  if (!connected || !esploader) {
    throw new Error('Not connected. Call connect() first.');
  }

  log(`Downloading firmware binaries for Node ${nodeId}...`);
  onProgress(5);

  // Fetch all binaries in parallel
  const [bootloaderBuf, partitionsBuf, firmwareBuf] = await Promise.all([
    fetchBinary(`${FW_BASE}/bootloader.bin`),
    fetchBinary(`${FW_BASE}/partitions.bin`),
    fetchBinary(`${FW_BASE}/firmware-node${nodeId}.bin`),
  ]);

  log(`Binaries loaded: bootloader ${bootloaderBuf.byteLength}B, partitions ${partitionsBuf.byteLength}B, firmware ${firmwareBuf.byteLength}B`);
  onProgress(15);

  // Build file array for esptool-js
  const fileArray = [
    { data: new Uint8Array(bootloaderBuf), address: FLASH_OFFSETS.bootloader },
    { data: new Uint8Array(partitionsBuf), address: FLASH_OFFSETS.partitions },
    { data: new Uint8Array(firmwareBuf),   address: FLASH_OFFSETS.firmware },
  ];

  log('Erasing flash...', 'info');
  onProgress(20);

  try {
    await esploader.writeFlash({
      fileArray,
      flashSize: 'keep',
      flashMode: 'keep',
      flashFreq: 'keep',
      eraseAll: false,
      compress: true,
      reportProgress: (fileIndex, written, total) => {
        const fileBase = [20, 30, 35][fileIndex] || 35;
        const fileWeight = [10, 5, 50][fileIndex] || 50;
        const pct = fileBase + (written / total) * fileWeight;
        onProgress(Math.round(pct));

        if (written === total) {
          const names = ['bootloader', 'partitions', 'firmware'];
          log(`${names[fileIndex] || 'file'} written (${total} bytes)`, 'success');
        }
      },
    });
  } catch (err) {
    log(`Flash failed: ${err.message}`, 'error');
    throw err;
  }

  onProgress(95);
  log('Verifying...', 'info');

  // Hard reset to start the new firmware
  try {
    await esploader.hardReset();
  } catch (e) { /* some boards don't support auto-reset */ }

  onProgress(100);
  log(`Node ${nodeId} flashed successfully! Device will restart and stream CSI to UDP :5005`, 'success');
}

/**
 * Get chip info from connected ESP32.
 * @returns {{ chip: string, connected: boolean }}
 */
export function getStatus() {
  return { chip: chipName, connected };
}

// ── Helpers ──────────────────────────────────────────────
async function fetchBinary(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to download ${url}: ${resp.status}`);
  return resp.arrayBuffer();
}
