// dashboard/src/tabs/hardware.js
/**
 * Hardware tab — ESP32 device management, AP placement guide, flash tool,
 * calibration, notifications, OTA.
 *
 * All DOM is built via safe DOM API calls (no innerHTML).
 */

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

// ── Built-in hardware profiles (fallback when server is unreachable) ──
const BUILTIN_PROFILES = [
  { id: 'esp32s3', name: 'ESP32-S3 (Default)', description: 'ESP32-S3 with LWIP CSI, 56 subcarriers, 2.4 GHz, 3x3 MIMO', num_subcarriers: 56, max_nodes: 6, csi_sample_rate: 20, frequency_ghz: 2.4, bandwidth_mhz: 20, model_ready: false, dataset: 'synthetic', active: true },
  { id: 'esp32s3_mmfi', name: 'ESP32-S3 + MM-Fi', description: 'ESP32-S3 trained on MM-Fi dataset (NeurIPS 2023)', num_subcarriers: 56, max_nodes: 4, csi_sample_rate: 20, frequency_ghz: 2.4, bandwidth_mhz: 20, model_ready: false, dataset: 'mmfi', active: false },
  { id: 'tplink_n750', name: 'TP-Link N750 (MM-Fi)', description: 'TP-Link N750 AP with Atheros CSI Tool', num_subcarriers: 114, max_nodes: 2, csi_sample_rate: 100, frequency_ghz: 5.0, bandwidth_mhz: 40, model_ready: false, dataset: 'mmfi', active: false },
  { id: 'intel5300', name: 'Intel 5300 NIC', description: 'Intel 5300 with Linux CSI Tool \u2014 classic WiFi sensing', num_subcarriers: 30, max_nodes: 3, csi_sample_rate: 100, frequency_ghz: 5.0, bandwidth_mhz: 20, model_ready: false, dataset: 'wipose', active: false },
  { id: 'esp32c6_wifi6', name: 'ESP32-C6 (WiFi 6)', description: 'ESP32-C6 RISC-V with WiFi 6 (802.11ax) CSI', num_subcarriers: 64, max_nodes: 4, csi_sample_rate: 50, frequency_ghz: 2.4, bandwidth_mhz: 20, model_ready: false, dataset: '', active: false },
];

// ── AP Placement presets ──
const PLACEMENT_PRESETS = {
  3: [
    { angle: 0, h: 1.5, label: 'N1' },
    { angle: 120, h: 1.0, label: 'N2' },
    { angle: 240, h: 0.6, label: 'N3' },
  ],
  4: [
    { angle: 0, h: 1.5, label: 'N1' },
    { angle: 90, h: 1.0, label: 'N2' },
    { angle: 180, h: 0.5, label: 'N3' },
    { angle: 270, h: 1.0, label: 'N4' },
  ],
  6: [
    { angle: 0, h: 1.8, label: 'N1' },
    { angle: 60, h: 1.0, label: 'N2' },
    { angle: 120, h: 0.5, label: 'N3' },
    { angle: 180, h: 1.8, label: 'N4' },
    { angle: 240, h: 1.0, label: 'N5' },
    { angle: 300, h: 0.5, label: 'N6' },
  ],
};

const NODE_COLORS = ['#00c8ff', '#3ddc84', '#ff6b6b', '#ffb400', '#c084fc', '#5bc0eb'];

/**
 * Build the full hardware tab DOM.
 */
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);

  const scroll = makeEl('div', { className: 'tab-scroll' });

  // ── Hardware Profile Selector ──────────────────────────────
  const profilePanel = makeEl('div', { className: 'panel profile-panel' });
  profilePanel.appendChild(makeEl('h2', { textContent: 'Hardware Profile' }));
  profilePanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Select a CSI hardware configuration. Each profile has matched model weights.' }));
  const profileCards = makeEl('div', { className: 'profile-cards', id: 'profile-cards' });
  profilePanel.appendChild(profileCards);
  scroll.appendChild(profilePanel);

  // ── AP Placement Guide ─────────────────────────────────────
  const placementPanel = makeEl('div', { className: 'panel placement-panel' });
  placementPanel.appendChild(makeEl('h2', { textContent: 'AP Placement Guide' }));
  placementPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Optimal ESP32 node positions for 3D pose reconstruction. Spatial diversity is key.' }));

  const controls = makeEl('div', { className: 'placement-controls' });
  controls.appendChild(makeEl('label', { textContent: 'Node count:' }));
  [3, 4, 6].forEach((n, i) => {
    const btn = makeEl('button', { className: 'placement-btn' + (i === 0 ? ' active' : ''), textContent: n + ' Nodes' });
    btn.dataset.nodes = String(n);
    controls.appendChild(btn);
  });
  placementPanel.appendChild(controls);

  const layout = makeEl('div', { className: 'placement-layout' });
  const roomDiv = makeEl('div', { className: 'placement-room' });
  const topCanvas = document.createElement('canvas');
  topCanvas.id = 'placement-canvas';
  topCanvas.width = 320;
  topCanvas.height = 320;
  roomDiv.appendChild(topCanvas);
  roomDiv.appendChild(makeEl('div', { className: 'placement-room-label', textContent: 'Top-down view (4m x 4m room)' }));
  layout.appendChild(roomDiv);

  const sideDiv = makeEl('div', { className: 'placement-side' });
  const sideCanvas = document.createElement('canvas');
  sideCanvas.id = 'placement-side-canvas';
  sideCanvas.width = 320;
  sideCanvas.height = 200;
  sideDiv.appendChild(sideCanvas);
  sideDiv.appendChild(makeEl('div', { className: 'placement-room-label', textContent: 'Side view (height stagger)' }));
  layout.appendChild(sideDiv);
  placementPanel.appendChild(layout);

  // Placement rules
  const rulesDiv = makeEl('div', { className: 'placement-rules' });
  const rules = [
    ['1', 'Surround the subject', ' \u2014 distribute nodes around the perimeter, not clustered on one side'],
    ['2', 'Vary heights (0.5m~2.0m)', ' \u2014 same-height nodes can\'t distinguish standing from sitting'],
    ['3', 'Keep 2~4m from centre', ' \u2014 within the Fresnel zone for maximum body influence on CSI'],
    ['4', 'Point antennas inward', ' \u2014 ESP32 PCB antenna is directional; aim the main lobe at the activity area'],
    ['5', 'Avoid metal/mirrors', ' \u2014 large reflective surfaces create multipath interference that degrades CSI quality'],
  ];
  rules.forEach(([num, bold, rest]) => {
    const rule = makeEl('div', { className: 'placement-rule' });
    rule.appendChild(makeEl('span', { className: 'rule-num', textContent: num }));
    const desc = document.createElement('div');
    const b = document.createElement('b');
    b.textContent = bold;
    desc.appendChild(b);
    desc.appendChild(document.createTextNode(rest));
    rule.appendChild(desc);
    rulesDiv.appendChild(rule);
  });
  placementPanel.appendChild(rulesDiv);
  scroll.appendChild(placementPanel);

  // ── Data Collection Panel ──────────────────────────────────
  const collectPanel = makeEl('div', { className: 'panel collect-panel' });
  collectPanel.appendChild(makeEl('h2', { textContent: 'Data Collection' }));
  collectPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Record real CSI + pose data for model training. Requires backend server running.' }));

  const collectGrid = makeEl('div', { className: 'collect-grid' });

  // Controls side
  const collectControls = makeEl('div', { className: 'collect-controls' });
  const collectRow = makeEl('div', { className: 'collect-row' });
  collectRow.appendChild(makeEl('label', { textContent: 'Activity:' }));
  const actSelect = document.createElement('select');
  actSelect.id = 'collect-activity';
  ['standing', 'walking', 'sitting', 'falling', 'exercising', 'waving', 'stretching', 'turning'].forEach(a => {
    const opt = document.createElement('option');
    opt.value = a;
    opt.textContent = a.charAt(0).toUpperCase() + a.slice(1);
    actSelect.appendChild(opt);
  });
  collectRow.appendChild(actSelect);
  collectControls.appendChild(collectRow);

  const collectActions = makeEl('div', { className: 'collect-actions' });
  const startRecBtn = makeEl('button', { className: 'collect-btn start', id: 'collect-start-btn', textContent: 'Start Recording' });
  collectActions.appendChild(startRecBtn);
  const stopRecBtn = makeEl('button', { className: 'collect-btn stop', id: 'collect-stop-btn', textContent: 'Stop & Save' });
  stopRecBtn.disabled = true;
  collectActions.appendChild(stopRecBtn);
  collectControls.appendChild(collectActions);
  collectGrid.appendChild(collectControls);

  // Status side
  const collectStatus = makeEl('div', { className: 'collect-status' });
  [['Status', 'collect-state', 'Idle'], ['Frames', 'collect-frame-count', '0'], ['CSI Received', 'collect-csi-count', '0'], ['Nodes Seen', 'collect-nodes-seen', '0']].forEach(([label, id, val]) => {
    const stat = makeEl('div', { className: 'collect-stat' });
    stat.appendChild(makeEl('span', { className: 'collect-stat-label', textContent: label }));
    stat.appendChild(makeEl('span', { className: 'collect-stat-value', id: id, textContent: val }));
    collectStatus.appendChild(stat);
  });
  collectGrid.appendChild(collectStatus);
  collectPanel.appendChild(collectGrid);

  const collectFiles = makeEl('div', { className: 'collect-files' });
  collectFiles.appendChild(makeEl('h3', { className: 'sub-heading', textContent: 'Saved Sequences' }));
  const fileList = makeEl('div', { className: 'collect-file-list', id: 'collect-file-list' });
  fileList.appendChild(makeEl('span', { className: 'help-text', textContent: 'No recordings yet' }));
  collectFiles.appendChild(fileList);
  collectPanel.appendChild(collectFiles);
  scroll.appendChild(collectPanel);

  // ── Antenna Array + WiFi Config grid ───────────────────────
  const hwGrid = makeEl('div', { className: 'hw-grid' });

  // Antenna Array
  const antennaPanel = makeEl('div', { className: 'panel' });
  antennaPanel.appendChild(makeEl('h2', { textContent: '3x3 Antenna Array' }));
  antennaPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Click antennas to toggle active/inactive' }));
  const antennaArray = makeEl('div', { className: 'antenna-array' });
  const antennaGrid = makeEl('div', { className: 'antenna-grid' });
  ['TX1', 'TX2', 'TX3', 'RX1', 'RX2', 'RX3', 'RX4', 'RX5', 'RX6'].forEach(name => {
    const type = name.startsWith('TX') ? 'tx' : 'rx';
    const ant = makeEl('div', { className: 'antenna ' + type + ' active', textContent: name });
    ant.dataset.type = name;
    antennaGrid.appendChild(ant);
  });
  antennaArray.appendChild(antennaGrid);
  const legend = makeEl('div', { className: 'antenna-legend' });
  legend.appendChild(makeEl('span', { className: 'legend-tx', textContent: 'TX (Transmitters)' }));
  legend.appendChild(makeEl('span', { className: 'legend-rx', textContent: 'RX (Receivers)' }));
  antennaArray.appendChild(legend);
  const antennaStatus = makeEl('div', { className: 'antenna-status' });
  const txSpan = document.createElement('span');
  txSpan.textContent = 'Active TX: ';
  txSpan.appendChild(makeEl('b', { id: 'active-tx', textContent: '3' }));
  txSpan.appendChild(document.createTextNode('/3'));
  antennaStatus.appendChild(txSpan);
  const rxSpan = document.createElement('span');
  rxSpan.textContent = 'Active RX: ';
  rxSpan.appendChild(makeEl('b', { id: 'active-rx', textContent: '6' }));
  rxSpan.appendChild(document.createTextNode('/6'));
  antennaStatus.appendChild(rxSpan);
  const sigSpan = document.createElement('span');
  sigSpan.textContent = 'Signal Quality: ';
  sigSpan.appendChild(makeEl('b', { id: 'signal-quality', textContent: '100%' }));
  antennaStatus.appendChild(sigSpan);
  antennaArray.appendChild(antennaStatus);
  antennaPanel.appendChild(antennaArray);
  hwGrid.appendChild(antennaPanel);

  // WiFi Configuration — auto-detected from PC
  const wifiPanel = makeEl('div', { className: 'panel' });
  wifiPanel.appendChild(makeEl('h2', { textContent: 'WiFi Configuration' }));
  wifiPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Auto-detected from this PC. Used when building firmware for ESP32.' }));

  const wifiGrid = makeEl('div', { className: 'config-grid' });
  const wifiFields = [
    ['SSID', 'wifi-ssid', 'Detecting...'],
    ['Password', 'wifi-pass', '...'],
    ['Server IP', 'wifi-server-ip', '...'],
    ['UDP Port', 'wifi-udp-port', '5005'],
  ];
  wifiFields.forEach(([label, id, val]) => {
    const item = makeEl('div', { className: 'config-item' });
    item.appendChild(makeEl('label', { textContent: label }));
    item.appendChild(makeEl('div', { className: 'config-value', id: id, textContent: val }));
    wifiGrid.appendChild(item);
  });
  wifiPanel.appendChild(wifiGrid);

  // ── One-Click Auto Flash Panel ────────────────────────────
  wifiPanel.appendChild(makeEl('h3', { className: 'sub-heading', textContent: 'One-Click Flash', style: 'margin-top:16px' }));
  wifiPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Detect chip → auto WiFi → build → flash. Just plug in USB and click.' }));

  // Detected devices list
  const devicesDiv = makeEl('div', { id: 'auto-devices', style: 'margin:8px 0' });
  devicesDiv.appendChild(makeEl('span', { textContent: 'Scanning USB ports...', style: 'color:#888;font-size:12px' }));
  wifiPanel.appendChild(devicesDiv);

  // Refresh + auto flash buttons
  const autoRow = makeEl('div', { style: 'display:flex;gap:8px;align-items:center;margin-top:8px' });
  autoRow.appendChild(makeEl('button', { className: 'btn', id: 'auto-detect-btn', textContent: 'Scan Devices' }));
  autoRow.appendChild(makeEl('button', { className: 'btn btn-primary', id: 'auto-flash-btn', textContent: 'Auto Flash All', disabled: true }));
  wifiPanel.appendChild(autoRow);

  // Progress steps
  const stepsDiv = makeEl('div', { id: 'auto-flash-steps', style: 'margin-top:12px;display:none' });
  const stepNames = ['Detect Chip', 'Read WiFi', 'Configure', 'Build', 'Flash', 'Done'];
  stepNames.forEach((name, i) => {
    const step = makeEl('div', { style: 'display:flex;align-items:center;gap:8px;padding:4px 0' });
    const dot = makeEl('span', { id: 'auto-step-' + i, style: 'display:inline-block;width:10px;height:10px;border-radius:50%;background:#333;flex-shrink:0' });
    step.appendChild(dot);
    step.appendChild(makeEl('span', { textContent: name, style: 'font-size:12px;color:#888' }));
    stepsDiv.appendChild(step);
  });
  wifiPanel.appendChild(stepsDiv);

  // Progress bar
  const progWrap = makeEl('div', { id: 'auto-progress-wrap', style: 'margin-top:8px;display:none;background:#222;border-radius:2px;height:6px;overflow:hidden' });
  const progFill = makeEl('div', { id: 'auto-progress-fill', style: 'height:100%;background:var(--accent-green,#0f0);width:0%;transition:width 0.3s' });
  progWrap.appendChild(progFill);
  wifiPanel.appendChild(progWrap);

  // Status text
  wifiPanel.appendChild(makeEl('div', { id: 'auto-flash-status', style: 'font-size:11px;color:#888;margin-top:4px' }));

  hwGrid.appendChild(wifiPanel);
  scroll.appendChild(hwGrid);

  // ── Calibration Panel ──────────────────────────────────────
  const calPanel = makeEl('div', { className: 'panel calibration-panel' });
  calPanel.appendChild(makeEl('h2', { textContent: 'Spatial Calibration' }));
  calPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Stand at the centre of the room and calibrate node distances. Required for optimal pose accuracy.' }));
  const calStatusRow = makeEl('div', { className: 'cal-status-row' });
  const calIndicator = makeEl('div', { className: 'cal-indicator', id: 'cal-indicator' });
  calIndicator.appendChild(makeEl('span', { className: 'cal-dot uncalibrated' }));
  calIndicator.appendChild(makeEl('span', { id: 'cal-status-text', textContent: 'Uncalibrated' }));
  calStatusRow.appendChild(calIndicator);
  calStatusRow.appendChild(makeEl('button', { className: 'btn btn-primary', id: 'cal-start-btn', textContent: 'Start Calibration' }));
  calPanel.appendChild(calStatusRow);

  const calProgressSection = makeEl('div', { id: 'cal-progress-section' });
  calProgressSection.style.display = 'none';
  const calProgressBar = makeEl('div', { className: 'cal-progress-bar' });
  const calProgressFill = makeEl('div', { className: 'cal-progress-fill', id: 'cal-progress-fill' });
  calProgressFill.style.width = '0%';
  calProgressBar.appendChild(calProgressFill);
  calProgressSection.appendChild(calProgressBar);
  calProgressSection.appendChild(makeEl('p', { className: 'help-text', id: 'cal-progress-text', textContent: 'Stand still at the centre of the room...' }));
  calPanel.appendChild(calProgressSection);

  const calResultSection = makeEl('div', { id: 'cal-result-section' });
  calResultSection.style.display = 'none';
  calResultSection.appendChild(makeEl('h3', { className: 'sub-heading', textContent: 'Calibration Results' }));
  calResultSection.appendChild(makeEl('div', { className: 'cal-nodes-grid', id: 'cal-nodes-grid' }));
  calPanel.appendChild(calResultSection);
  scroll.appendChild(calPanel);

  // ── Notification Settings ──────────────────────────────────
  const notifyPanel = makeEl('div', { className: 'panel notify-panel' });
  notifyPanel.appendChild(makeEl('h2', { textContent: 'Fall Alert Notifications' }));
  notifyPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Configure push notifications for fall detection alerts. Set tokens via environment variables.' }));
  const notifyChannels = makeEl('div', { className: 'notify-channels', id: 'notify-channels' });
  [['Webhook', 'notify-webhook-status', 'NOTIFY_WEBHOOK_URL'], ['LINE', 'notify-line-status', 'NOTIFY_LINE_TOKEN'], ['Telegram', 'notify-telegram-status', 'NOTIFY_TELEGRAM_BOT_TOKEN']].forEach(([name, id, code]) => {
    const ch = makeEl('div', { className: 'notify-channel' });
    ch.appendChild(makeEl('span', { className: 'notify-icon', textContent: name }));
    ch.appendChild(makeEl('span', { className: 'notify-status', id: id, textContent: 'Not configured' }));
    const codeEl = document.createElement('code');
    codeEl.textContent = code;
    ch.appendChild(codeEl);
    notifyChannels.appendChild(ch);
  });
  notifyPanel.appendChild(notifyChannels);
  const testNotifyBtn = makeEl('button', { className: 'btn', id: 'notify-test-btn', textContent: 'Send Test Notification' });
  testNotifyBtn.disabled = true;
  notifyPanel.appendChild(testNotifyBtn);
  notifyPanel.appendChild(makeEl('span', { className: 'help-text', id: 'notify-test-result' }));
  scroll.appendChild(notifyPanel);

  // ── ESP32 Web Flasher ──────────────────────────────────────
  const flashPanel = makeEl('div', { className: 'panel flash-panel' });
  flashPanel.appendChild(makeEl('h2', { textContent: 'ESP32 Firmware Flasher' }));
  flashPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Flash WiFi-DensePose CSI firmware directly from your browser via USB' }));
  const flashWarn = makeEl('div', { className: 'flash-warn', id: 'flash-browser-warn' });
  flashWarn.textContent = 'Web Serial API not supported. Please use Chrome or Edge.';
  flashWarn.style.display = 'none';
  flashPanel.appendChild(flashWarn);

  const flashSteps = makeEl('div', { className: 'flash-steps' });

  // Step 1
  const step1 = makeEl('div', { className: 'flash-step' });
  step1.appendChild(makeEl('div', { className: 'flash-step-num', textContent: '1' }));
  const step1Body = makeEl('div', { className: 'flash-step-body' });
  step1Body.appendChild(makeEl('h3', { textContent: 'Select ESP32 Variant' }));
  const flashTarget = document.createElement('select');
  flashTarget.id = 'flash-target';
  [
    ['esp32', 'ESP32 (Xtensa LX6, 520KB SRAM)'],
    ['esp32s2', 'ESP32-S2 (Xtensa LX7, USB OTG)'],
    ['esp32s3', 'ESP32-S3 (Recommended \u2014 SIMD, CSI support)'],
    ['esp32c3', 'ESP32-C3 (RISC-V, low power)'],
    ['esp32c6', 'ESP32-C6 (RISC-V, WiFi 6)'],
  ].forEach(([val, text]) => {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = text;
    if (val === 'esp32s3') opt.selected = true;
    flashTarget.appendChild(opt);
  });
  step1Body.appendChild(flashTarget);
  const flashFeats = makeEl('div', { className: 'flash-features', id: 'flash-features' });
  ['CSI Capture', 'UDP Stream', 'SIMD Accel', '3x3 MIMO'].forEach(f => {
    flashFeats.appendChild(makeEl('span', { className: 'flash-feat', textContent: f }));
  });
  step1Body.appendChild(flashFeats);
  step1.appendChild(step1Body);
  flashSteps.appendChild(step1);

  // Step 2
  const step2 = makeEl('div', { className: 'flash-step' });
  step2.appendChild(makeEl('div', { className: 'flash-step-num', textContent: '2' }));
  const step2Body = makeEl('div', { className: 'flash-step-body' });
  step2Body.appendChild(makeEl('h3', { textContent: 'Connect Device' }));
  const connRow = makeEl('div', { className: 'flash-conn-row' });
  connRow.appendChild(makeEl('div', { className: 'flash-conn-status disconnected', id: 'flash-conn-status', textContent: 'Not connected' }));
  connRow.appendChild(makeEl('button', { className: 'btn btn-primary', id: 'flash-connect-btn', textContent: 'Connect' }));
  step2Body.appendChild(connRow);
  const holdNote = makeEl('p', { className: 'help-text' });
  holdNote.style.marginTop = '8px';
  holdNote.style.marginBottom = '0';
  holdNote.textContent = 'Hold BOOT button while clicking Connect if device doesn\'t appear';
  step2Body.appendChild(holdNote);
  step2.appendChild(step2Body);
  flashSteps.appendChild(step2);

  // Step 3
  const step3 = makeEl('div', { className: 'flash-step' });
  step3.appendChild(makeEl('div', { className: 'flash-step-num', textContent: '3' }));
  const step3Body = makeEl('div', { className: 'flash-step-body' });
  step3Body.appendChild(makeEl('h3', { textContent: 'Flash Firmware' }));
  const flashStartBtn = makeEl('button', { className: 'btn btn-primary', id: 'flash-start-btn', textContent: 'Flash CSI Firmware' });
  flashStartBtn.disabled = true;
  step3Body.appendChild(flashStartBtn);
  const flashProgress = makeEl('div', { className: 'flash-progress', id: 'flash-progress' });
  flashProgress.style.display = 'none';
  flashProgress.appendChild(makeEl('div', { className: 'flash-progress-bar', id: 'flash-progress-bar' }));
  step3Body.appendChild(flashProgress);
  step3Body.appendChild(makeEl('p', { className: 'flash-progress-text', id: 'flash-progress-text' }));
  step3.appendChild(step3Body);
  flashSteps.appendChild(step3);
  flashPanel.appendChild(flashSteps);

  const flashLog = makeEl('div', { className: 'flash-log', id: 'flash-log' });
  flashLog.appendChild(makeEl('div', { className: 'flash-log-entry info', textContent: 'Ready. Select target and connect device.' }));
  flashPanel.appendChild(flashLog);
  scroll.appendChild(flashPanel);

  // ── OTA Firmware Update ────────────────────────────────────
  const otaPanel = makeEl('div', { className: 'panel ota-panel' });
  otaPanel.appendChild(makeEl('h2', { textContent: 'OTA Firmware Update' }));
  otaPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Update ESP32 firmware wirelessly via HTTP OTA. Nodes must be connected to the same network.' }));
  const otaList = makeEl('div', { className: 'ota-firmware-list', id: 'ota-firmware-list' });
  otaList.appendChild(makeEl('span', { className: 'help-text', textContent: 'Checking for firmware binaries...' }));
  otaPanel.appendChild(otaList);
  const otaInstr = makeEl('div', { className: 'ota-instructions' });
  otaInstr.appendChild(makeEl('h3', { className: 'sub-heading', textContent: 'How OTA Works' }));
  const otaSteps = document.createElement('ol');
  otaSteps.className = 'ota-steps';
  const otaStepTexts = [
    'Build firmware with OTA partition: idf.py menuconfig \u2192 enable OTA',
    'Place built .bin in firmware/esp32-csi-node/build/',
    'ESP32 nodes fetch updates from http://<server-ip>:8000/api/ota/download/<file>',
    'Configure OTA URL in firmware sdkconfig or via serial command',
  ];
  otaStepTexts.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    otaSteps.appendChild(li);
  });
  otaInstr.appendChild(otaSteps);
  otaPanel.appendChild(otaInstr);
  scroll.appendChild(otaPanel);

  container.appendChild(scroll);
}

// ── Interactive behaviour initialization ─────────────────────
function initAntennaToggle() {
  document.querySelectorAll('.antenna').forEach(el => {
    el.addEventListener('click', () => {
      el.classList.toggle('active');
      updateAntennaStatus();
    });
  });
}

function updateAntennaStatus() {
  const txActive = document.querySelectorAll('.antenna.tx.active').length;
  const rxActive = document.querySelectorAll('.antenna.rx.active').length;
  const txEl = document.getElementById('active-tx');
  const rxEl = document.getElementById('active-rx');
  const sigEl = document.getElementById('signal-quality');
  if (txEl) txEl.textContent = txActive;
  if (rxEl) rxEl.textContent = rxActive;
  if (sigEl) sigEl.textContent = Math.round((txActive * rxActive) / (3 * 6) * 100) + '%';
}

function initPlacement() {
  const topCanvas = document.getElementById('placement-canvas');
  const sideCanvas = document.getElementById('placement-side-canvas');
  if (!topCanvas || !sideCanvas) return;
  const topCtx = topCanvas.getContext('2d');
  const sideCtx = sideCanvas.getContext('2d');
  let currentNodes = 3;

  function drawTop(nodes) {
    const w = topCanvas.width, h = topCanvas.height;
    const cx = w / 2, cy = h / 2;
    const roomR = 130;
    topCtx.clearRect(0, 0, w, h);

    topCtx.strokeStyle = '#2a2a4a';
    topCtx.lineWidth = 2;
    topCtx.strokeRect(cx - roomR, cy - roomR, roomR * 2, roomR * 2);

    topCtx.strokeStyle = '#1a1a30';
    topCtx.lineWidth = 1;
    for (let g = -1; g <= 1; g++) {
      topCtx.beginPath(); topCtx.moveTo(cx + g * (roomR / 2), cy - roomR); topCtx.lineTo(cx + g * (roomR / 2), cy + roomR); topCtx.stroke();
      topCtx.beginPath(); topCtx.moveTo(cx - roomR, cy + g * (roomR / 2)); topCtx.lineTo(cx + roomR, cy + g * (roomR / 2)); topCtx.stroke();
    }

    topCtx.fillStyle = '#555'; topCtx.font = '10px sans-serif'; topCtx.fillText('1m', cx + roomR / 2 - 6, cy + roomR + 14);
    topCtx.fillStyle = 'rgba(255,255,255,0.08)'; topCtx.beginPath(); topCtx.arc(cx, cy, 16, 0, Math.PI * 2); topCtx.fill();
    topCtx.fillStyle = '#555'; topCtx.font = '11px sans-serif'; topCtx.textAlign = 'center'; topCtx.fillText('Person', cx, cy + 28);

    // Fresnel zones
    topCtx.globalAlpha = 0.06;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a1 = nodes[i].angle * Math.PI / 180, a2 = nodes[j].angle * Math.PI / 180;
        const x1 = cx + Math.sin(a1) * roomR * 0.85, y1 = cy - Math.cos(a1) * roomR * 0.85;
        const x2 = cx + Math.sin(a2) * roomR * 0.85, y2 = cy - Math.cos(a2) * roomR * 0.85;
        topCtx.strokeStyle = '#00c8ff'; topCtx.lineWidth = 18;
        topCtx.beginPath(); topCtx.moveTo(x1, y1); topCtx.lineTo(x2, y2); topCtx.stroke();
      }
    }
    topCtx.globalAlpha = 1.0; topCtx.lineWidth = 1;

    // Dashed links
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a1 = nodes[i].angle * Math.PI / 180, a2 = nodes[j].angle * Math.PI / 180;
        const x1 = cx + Math.sin(a1) * roomR * 0.85, y1 = cy - Math.cos(a1) * roomR * 0.85;
        const x2 = cx + Math.sin(a2) * roomR * 0.85, y2 = cy - Math.cos(a2) * roomR * 0.85;
        topCtx.strokeStyle = 'rgba(0,200,255,0.2)'; topCtx.lineWidth = 1; topCtx.setLineDash([4, 4]);
        topCtx.beginPath(); topCtx.moveTo(x1, y1); topCtx.lineTo(x2, y2); topCtx.stroke();
      }
    }
    topCtx.setLineDash([]);

    // Nodes
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i].angle * Math.PI / 180;
      const nx = cx + Math.sin(a) * roomR * 0.85, ny = cy - Math.cos(a) * roomR * 0.85;
      const col = NODE_COLORS[i % NODE_COLORS.length];

      topCtx.fillStyle = col; topCtx.globalAlpha = 0.15; topCtx.beginPath(); topCtx.arc(nx, ny, 18, 0, Math.PI * 2); topCtx.fill(); topCtx.globalAlpha = 1.0;

      // Antenna direction
      const dirAngle = Math.atan2(cy - ny, cx - nx);
      topCtx.fillStyle = col; topCtx.globalAlpha = 0.4;
      topCtx.beginPath();
      topCtx.moveTo(nx + Math.cos(dirAngle) * 20, ny + Math.sin(dirAngle) * 20);
      topCtx.lineTo(nx + Math.cos(dirAngle + 0.4) * 12, ny + Math.sin(dirAngle + 0.4) * 12);
      topCtx.lineTo(nx + Math.cos(dirAngle - 0.4) * 12, ny + Math.sin(dirAngle - 0.4) * 12);
      topCtx.closePath(); topCtx.fill(); topCtx.globalAlpha = 1.0;

      topCtx.fillStyle = col; topCtx.beginPath(); topCtx.arc(nx, ny, 8, 0, Math.PI * 2); topCtx.fill();
      topCtx.fillStyle = '#fff'; topCtx.font = 'bold 9px sans-serif'; topCtx.textAlign = 'center'; topCtx.fillText(nodes[i].label, nx, ny + 3);
      topCtx.fillStyle = '#888'; topCtx.font = '9px sans-serif'; topCtx.fillText(nodes[i].h + 'm', nx, ny - 14);
    }
    topCtx.textAlign = 'start';
  }

  function drawSide(nodes) {
    const w = sideCanvas.width, h = sideCanvas.height;
    sideCtx.clearRect(0, 0, w, h);
    const floorY = h - 30, ceilY = 20, maxH = 2.5;
    const leftX = 40, rightX = w - 20;
    const personX = (leftX + rightX) / 2;

    sideCtx.strokeStyle = '#2a2a4a'; sideCtx.lineWidth = 1;
    sideCtx.beginPath(); sideCtx.moveTo(leftX, floorY); sideCtx.lineTo(rightX, floorY); sideCtx.stroke();

    sideCtx.fillStyle = '#444'; sideCtx.font = '10px sans-serif';
    for (let hm = 0; hm <= 2.0; hm += 0.5) {
      const py = floorY - (hm / maxH) * (floorY - ceilY);
      sideCtx.fillText(hm + 'm', 4, py + 4);
      sideCtx.strokeStyle = '#1a1a30'; sideCtx.beginPath(); sideCtx.moveTo(leftX, py); sideCtx.lineTo(rightX, py); sideCtx.stroke();
    }

    // Person stick figure
    const personTop = floorY - (1.7 / maxH) * (floorY - ceilY);
    const personMid = floorY - (0.9 / maxH) * (floorY - ceilY);
    sideCtx.strokeStyle = 'rgba(255,255,255,0.15)'; sideCtx.lineWidth = 2;
    sideCtx.beginPath(); sideCtx.moveTo(personX, personTop + 10); sideCtx.lineTo(personX, personMid + 20); sideCtx.stroke();
    sideCtx.beginPath(); sideCtx.arc(personX, personTop + 5, 6, 0, Math.PI * 2); sideCtx.stroke();
    sideCtx.beginPath(); sideCtx.moveTo(personX - 15, personMid - 10); sideCtx.lineTo(personX, personTop + 20); sideCtx.lineTo(personX + 15, personMid - 10); sideCtx.stroke();
    sideCtx.beginPath(); sideCtx.moveTo(personX - 10, floorY); sideCtx.lineTo(personX, personMid + 20); sideCtx.lineTo(personX + 10, floorY); sideCtx.stroke();

    const spacing = (rightX - leftX - 40) / Math.max(nodes.length - 1, 1);
    for (let i = 0; i < nodes.length; i++) {
      const nx = leftX + 20 + i * spacing;
      const ny = floorY - (nodes[i].h / maxH) * (floorY - ceilY);
      const col = NODE_COLORS[i % NODE_COLORS.length];

      sideCtx.strokeStyle = col; sideCtx.globalAlpha = 0.3; sideCtx.lineWidth = 1; sideCtx.setLineDash([3, 3]);
      sideCtx.beginPath(); sideCtx.moveTo(nx, ny); sideCtx.lineTo(nx, floorY); sideCtx.stroke();
      sideCtx.setLineDash([]); sideCtx.globalAlpha = 1.0;

      sideCtx.fillStyle = col; sideCtx.beginPath(); sideCtx.arc(nx, ny, 7, 0, Math.PI * 2); sideCtx.fill();
      sideCtx.fillStyle = '#fff'; sideCtx.font = 'bold 8px sans-serif'; sideCtx.textAlign = 'center'; sideCtx.fillText(nodes[i].label, nx, ny + 3);
      sideCtx.fillStyle = col; sideCtx.font = '10px sans-serif'; sideCtx.fillText(nodes[i].h + 'm', nx, ny - 12);
    }
    sideCtx.textAlign = 'start';
  }

  function render(count) {
    const nodes = PLACEMENT_PRESETS[count] || PLACEMENT_PRESETS[3];
    drawTop(nodes);
    drawSide(nodes);
  }

  document.querySelectorAll('.placement-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.placement-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentNodes = parseInt(btn.dataset.nodes, 10);
      render(currentNodes);
    });
  });

  render(currentNodes);
}

function initProfiles() {
  const container = document.getElementById('profile-cards');
  if (!container) return;

  function renderProfiles(profiles) {
    while (container.firstChild) container.removeChild(container.firstChild);
    profiles.forEach(p => {
      const card = makeEl('div', { className: 'profile-card' + (p.active ? ' active' : '') });
      card.dataset.profile = p.id;
      card.appendChild(makeEl('div', { className: 'profile-name', textContent: p.name }));
      card.appendChild(makeEl('div', { className: 'profile-desc', textContent: p.description }));
      const specs = makeEl('div', { className: 'profile-specs' });
      [['Sub', p.num_subcarriers], ['Nodes', p.max_nodes], ['Rate', p.csi_sample_rate + ' Hz'], ['Freq', p.frequency_ghz + ' GHz'], ['BW', p.bandwidth_mhz + ' MHz']].forEach(([label, val]) => {
        const sp = makeEl('span', { className: 'profile-spec' });
        sp.appendChild(makeEl('b', { textContent: String(val) }));
        sp.appendChild(document.createTextNode(' ' + label));
        specs.appendChild(sp);
      });
      card.appendChild(specs);
      const badges = makeEl('div', { className: 'profile-badges' });
      if (p.active) badges.appendChild(makeEl('span', { className: 'profile-badge active-badge', textContent: 'Active' }));
      badges.appendChild(makeEl('span', { className: 'profile-badge ' + (p.model_ready ? 'ready' : 'no-weights'), textContent: p.model_ready ? 'Model Ready' : 'No Weights' }));
      if (p.dataset) badges.appendChild(makeEl('span', { className: 'profile-badge dataset', textContent: p.dataset }));
      card.appendChild(badges);
      card.addEventListener('click', () => { selectProfile(p.id, profiles); });
      container.appendChild(card);
    });
    const active = profiles.find(p => p.active);
    if (active) updateHwConfig(active);
  }

  function selectProfile(id, profiles) {
    profiles.forEach(p => { p.active = (p.id === id); });
    renderProfiles(profiles);
  }

  function updateHwConfig(p) {
    const items = document.querySelectorAll('#tab-hardware .config-value');
    if (items.length >= 3) {
      items[0].textContent = p.frequency_ghz + 'GHz +/- ' + p.bandwidth_mhz + 'MHz';
      items[1].textContent = p.num_subcarriers;
      items[2].textContent = p.csi_sample_rate + ' Hz';
    }
  }

  fetch('/api/profiles').then(r => {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  }).then(data => {
    renderProfiles(data.profiles);
  }).catch(() => {
    renderProfiles(BUILTIN_PROFILES);
  });
}

function initWifiConfig() {
  const statusEl = document.getElementById('auto-flash-status');
  const stepsDiv = document.getElementById('auto-flash-steps');
  const progWrap = document.getElementById('auto-progress-wrap');
  const progFill = document.getElementById('auto-progress-fill');
  const devicesDiv = document.getElementById('auto-devices');
  const detectBtn = document.getElementById('auto-detect-btn');
  const flashAllBtn = document.getElementById('auto-flash-btn');

  let detectedDevices = [];

  function setStatus(msg, color) {
    if (statusEl) { statusEl.textContent = msg; statusEl.style.color = color || '#888'; }
  }

  function setStep(idx, state) {
    // state: 'pending' | 'active' | 'done' | 'error'
    const dot = document.getElementById('auto-step-' + idx);
    if (!dot) return;
    const colors = { pending: '#333', active: '#ffaa00', done: '#00ff00', error: '#ff4444' };
    dot.style.background = colors[state] || '#333';
    if (state === 'active') dot.style.boxShadow = '0 0 6px ' + colors.active;
    else dot.style.boxShadow = 'none';
  }

  function setProgress(pct) {
    if (progFill) progFill.style.width = pct + '%';
  }

  // Auto-fetch WiFi config
  fetch('/api/network/wifi').then(r => r.json()).then(data => {
    const ssidEl = document.getElementById('wifi-ssid');
    const passEl = document.getElementById('wifi-pass');
    const ipEl = document.getElementById('wifi-server-ip');
    if (ssidEl) { ssidEl.textContent = data.ssid || 'Not detected'; if (data.detected) ssidEl.style.color = 'var(--accent-green,#0f0)'; }
    if (passEl) passEl.textContent = data.password ? '\u2022\u2022\u2022\u2022\u2022\u2022' : 'Not found';
    if (ipEl) ipEl.textContent = data.server_ip || 'Unknown';
  }).catch(() => {});

  // Scan devices
  function scanDevices() {
    if (devicesDiv) devicesDiv.textContent = '';
    setStatus('Scanning USB ports...');
    fetch('/api/firmware/detect').then(r => r.json()).then(data => {
      detectedDevices = data.devices || [];
      if (devicesDiv) devicesDiv.textContent = '';
      if (detectedDevices.length === 0) {
        devicesDiv.appendChild(makeEl('div', { textContent: 'No ESP32 devices found. Plug in USB cable.', style: 'color:#ff8800;font-size:12px' }));
        if (flashAllBtn) flashAllBtn.disabled = true;
        setStatus('No devices detected');
        return;
      }
      detectedDevices.forEach((dev, i) => {
        const row = makeEl('div', { style: 'display:flex;align-items:center;gap:8px;padding:4px 0' });
        const dot = makeEl('span', { style: 'width:8px;height:8px;border-radius:50%;background:' + (dev.detected ? '#00ff00' : '#ff4444') });
        row.appendChild(dot);
        const label = dev.detected
          ? dev.name + ' on ' + dev.port
          : (dev.error || 'Unknown') + ' on ' + dev.port;
        row.appendChild(makeEl('span', { textContent: label, style: 'font-size:12px;color:#ccc' }));
        // Node ID selector
        const nsel = document.createElement('select');
        nsel.style.cssText = 'padding:2px 4px;background:#111;color:#ccc;border:1px solid #333;font-size:11px;margin-left:auto';
        for (let n = 1; n <= 8; n++) {
          const opt = document.createElement('option');
          opt.value = n; opt.textContent = 'Node ' + n;
          if (n === i + 1) opt.selected = true;
          nsel.appendChild(opt);
        }
        nsel.id = 'auto-node-sel-' + i;
        row.appendChild(nsel);
        devicesDiv.appendChild(row);
      });
      if (flashAllBtn) flashAllBtn.disabled = false;
      setStatus(detectedDevices.length + ' device(s) found', '#00ff00');
    }).catch(err => {
      setStatus('Scan failed: ' + err.message, '#ff4444');
    });
  }

  if (detectBtn) detectBtn.addEventListener('click', scanDevices);
  scanDevices(); // auto-scan on init

  // Auto Flash All
  if (flashAllBtn) flashAllBtn.addEventListener('click', async () => {
    const devices = detectedDevices.filter(d => d.detected);
    if (devices.length === 0) { setStatus('No flashable devices', '#ff4444'); return; }

    flashAllBtn.disabled = true;
    if (detectBtn) detectBtn.disabled = true;
    if (stepsDiv) stepsDiv.style.display = 'block';
    if (progWrap) progWrap.style.display = 'block';

    for (let di = 0; di < devices.length; di++) {
      const dev = devices[di];
      const nsel = document.getElementById('auto-node-sel-' + detectedDevices.indexOf(dev));
      const nodeId = nsel ? parseInt(nsel.value) : di + 1;

      setStatus('Flashing ' + dev.name + ' on ' + dev.port + ' as Node ' + nodeId + ' (' + (di + 1) + '/' + devices.length + ')');

      // Reset steps
      for (let s = 0; s < 6; s++) setStep(s, 'pending');

      // Step 0: Detect
      setStep(0, 'active'); setProgress(5);
      await sleep(300);
      setStep(0, 'done');

      // Step 1: WiFi
      setStep(1, 'active'); setProgress(10);
      await sleep(300);
      setStep(1, 'done');

      // Step 2: Configure
      setStep(2, 'active'); setProgress(15);

      // Step 3: Build + Flash (trigger server)
      setStep(3, 'active'); setProgress(20);
      try {
        const resp = await fetch('/api/firmware/auto?port=' + encodeURIComponent(dev.port) + '&node_id=' + nodeId, { method: 'POST' });
        const startData = await resp.json();
        if (startData.error) {
          setStep(3, 'error');
          setStatus('Error: ' + startData.error, '#ff4444');
          continue;
        }

        setStep(2, 'done');

        // Poll for completion
        let done = false;
        while (!done) {
          await sleep(3000);
          const statusResp = await fetch('/api/firmware/status');
          const s = await statusResp.json();
          if (s.status === 'building') {
            setProgress(20 + (di / devices.length) * 60);
          } else {
            done = true;
            if (s.status === 'complete' && s.success) {
              setStep(3, 'done');
              setStep(4, 'done');
              setStep(5, 'done');
              setProgress(100);
              setStatus(dev.name + ' Node ' + nodeId + ' flashed!', '#00ff00');
            } else {
              setStep(3, 'error');
              setStatus('Failed: ' + (s.error || s.step || 'unknown'), '#ff4444');
            }
          }
        }
      } catch (err) {
        setStep(3, 'error');
        setStatus('Error: ' + err.message, '#ff4444');
      }

      // Small delay between devices
      if (di < devices.length - 1) await sleep(1000);
    }

    flashAllBtn.disabled = false;
    if (detectBtn) detectBtn.disabled = false;
    setStatus('All devices processed!', '#00ff00');
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function initFlasher() {
  const connectBtn = document.getElementById('flash-connect-btn');
  const flashBtn = document.getElementById('flash-start-btn');
  const connStatus = document.getElementById('flash-conn-status');
  const progContainer = document.getElementById('flash-progress');
  const progBar = document.getElementById('flash-progress-bar');
  const progText = document.getElementById('flash-progress-text');
  const logDiv = document.getElementById('flash-log');
  const targetSel = document.getElementById('flash-target');
  const featuresDiv = document.getElementById('flash-features');

  if (!connectBtn) return;

  // Node ID selector (which firmware binary to flash)
  let selectedNodeId = 1;
  const nodeSelect = document.createElement('select');
  nodeSelect.id = 'flash-node-id';
  nodeSelect.style.cssText = 'margin-left:8px;padding:4px;background:#111;color:#ccc;border:1px solid #333';
  [1, 2, 3, 4, 5, 6].forEach(n => {
    const opt = document.createElement('option');
    opt.value = n; opt.textContent = 'Node ' + n;
    nodeSelect.appendChild(opt);
  });
  nodeSelect.addEventListener('change', () => { selectedNodeId = parseInt(nodeSelect.value); });
  // Insert after target selector
  if (targetSel && targetSel.parentNode) {
    const label = makeEl('span', { textContent: '  Node ID: ', style: 'color:#888;font-size:12px;margin-left:12px' });
    targetSel.parentNode.appendChild(label);
    targetSel.parentNode.appendChild(nodeSelect);
  }

  if (!('serial' in navigator)) {
    const warn = document.getElementById('flash-browser-warn');
    if (warn) warn.style.display = 'block';
    connectBtn.disabled = true;
    logMsg('Web Serial API not supported. Use Chrome or Edge.', 'error');
  }

  function logMsg(msg, type) {
    const entry = makeEl('div', { className: 'flash-log-entry ' + (type || 'info') });
    entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
    logDiv.appendChild(entry);
    logDiv.scrollTop = logDiv.scrollHeight;
  }

  function setFeatures(feats) {
    while (featuresDiv.firstChild) featuresDiv.removeChild(featuresDiv.firstChild);
    feats.forEach(f => {
      featuresDiv.appendChild(makeEl('span', { className: 'flash-feat', textContent: f }));
    });
  }

  // ── Real esptool-js flashing ──────────────────────────────
  let flasher = null; // lazy-loaded module

  connectBtn.addEventListener('click', async () => {
    // Lazy-load esptool module
    if (!flasher) {
      try {
        flasher = await import('../hardware/esp-flasher.js');
      } catch (e) {
        logMsg('Failed to load flasher module: ' + e.message, 'error');
        return;
      }
    }

    const status = flasher.getStatus();
    if (status.connected) {
      await flasher.disconnect(logMsg);
      connStatus.className = 'flash-conn-status disconnected';
      connStatus.textContent = 'Not connected';
      connectBtn.textContent = 'Connect';
      flashBtn.disabled = true;
      return;
    }

    try {
      const info = await flasher.connect(logMsg);
      connStatus.className = 'flash-conn-status connected';
      connStatus.textContent = info.chip || 'Connected';
      connectBtn.textContent = 'Disconnect';
      flashBtn.disabled = false;
    } catch (err) {
      logMsg('Connection failed: ' + err.message, 'error');
    }
  });

  flashBtn.addEventListener('click', async () => {
    if (!flasher || !flasher.getStatus().connected) {
      logMsg('Please connect device first', 'warning');
      return;
    }

    // Only node 1 and 2 have pre-built binaries; others use node 1 as base
    const nodeId = selectedNodeId <= 2 ? selectedNodeId : 1;
    if (selectedNodeId > 2) {
      logMsg('Using Node 1 firmware as base (Node ' + selectedNodeId + ' config will be set via serial after flash)', 'info');
    }

    progContainer.style.display = 'block';
    flashBtn.disabled = true;
    connectBtn.disabled = true;

    try {
      await flasher.flash(
        nodeId,
        logMsg,
        (pct) => {
          progBar.style.width = pct + '%';
          progText.textContent = pct < 100 ? 'Flashing... ' + pct + '%' : 'Complete!';
        },
      );
    } catch (err) {
      logMsg('Flash failed: ' + err.message, 'error');
    } finally {
      flashBtn.disabled = false;
      connectBtn.disabled = false;
    }
  });

  targetSel.addEventListener('change', () => {
    const t = targetSel.value;
    const feats = ['CSI Capture', 'UDP Stream'];
    if (t.indexOf('s3') !== -1) feats.push('SIMD Accel', '3x3 MIMO');
    else if (t.indexOf('c6') !== -1) feats.push('WiFi 6', 'Low Power');
    else if (t.indexOf('c3') !== -1) feats.push('RISC-V', 'Low Power');
    else if (t.indexOf('s2') !== -1) feats.push('USB OTG');
    else feats.push('Classic BT');
    setFeatures(feats);
  });

  logMsg('Web flasher ready (esptool-js)');
}

function initDataCollector() {
  const startBtn = document.getElementById('collect-start-btn');
  const stopBtn = document.getElementById('collect-stop-btn');
  const actSelect = document.getElementById('collect-activity');
  const stateEl = document.getElementById('collect-state');
  const frameEl = document.getElementById('collect-frame-count');
  const csiEl = document.getElementById('collect-csi-count');
  const nodesEl = document.getElementById('collect-nodes-seen');
  const fileList = document.getElementById('collect-file-list');
  if (!startBtn) return;
  let polling = null;

  function updateStatus() {
    fetch('/api/collect/status').then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(d => {
      stateEl.textContent = d.collecting ? 'Recording' : 'Idle';
      stateEl.className = 'collect-stat-value' + (d.collecting ? ' recording' : '');
      frameEl.textContent = d.frames;
      csiEl.textContent = d.csi_frames;
      nodesEl.textContent = d.nodes_seen;
      if (d.saved_files && d.saved_files.length > 0) {
        while (fileList.firstChild) fileList.removeChild(fileList.firstChild);
        d.saved_files.forEach(f => {
          const entry = makeEl('div', { className: 'collect-file-entry' });
          entry.appendChild(makeEl('span', { textContent: f.split('/').pop().split('\\').pop() }));
          fileList.appendChild(entry);
        });
      }
    }).catch(() => { stateEl.textContent = 'Offline'; });
  }

  startBtn.addEventListener('click', () => {
    const activity = actSelect.value;
    fetch('/api/collect/start?activity=' + encodeURIComponent(activity), { method: 'POST' })
      .then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.error); }); return r.json(); })
      .then(() => {
        startBtn.disabled = true; stopBtn.disabled = false; actSelect.disabled = true;
        stateEl.textContent = 'Recording'; stateEl.className = 'collect-stat-value recording';
        if (polling) clearInterval(polling);
        polling = setInterval(updateStatus, 500);
      })
      .catch(err => { stateEl.textContent = 'Error: ' + err.message; });
  });

  stopBtn.addEventListener('click', () => {
    fetch('/api/collect/stop', { method: 'POST' }).then(r => r.json()).then(d => {
      startBtn.disabled = false; stopBtn.disabled = true; actSelect.disabled = false;
      stateEl.textContent = 'Idle'; stateEl.className = 'collect-stat-value';
      if (polling) { clearInterval(polling); polling = null; }
      if (d.file) {
        const noRec = fileList.querySelector('.help-text');
        if (noRec) fileList.removeChild(noRec);
        const entry = makeEl('div', { className: 'collect-file-entry' });
        entry.appendChild(makeEl('span', { textContent: d.file.split('/').pop().split('\\').pop() }));
        const meta = document.createElement('span');
        const act = makeEl('span', { className: 'collect-file-activity', textContent: d.activity || '' });
        meta.appendChild(act);
        meta.appendChild(makeEl('span', { className: 'collect-file-frames', textContent: ' ' + d.frames + ' frames' }));
        entry.appendChild(meta);
        fileList.appendChild(entry);
        frameEl.textContent = d.frames;
      }
    }).catch(() => { stateEl.textContent = 'Error'; });
  });

  updateStatus();
}

function initCalibration() {
  const startBtn = document.getElementById('cal-start-btn');
  const progressSection = document.getElementById('cal-progress-section');
  const progressFill = document.getElementById('cal-progress-fill');
  const progressText = document.getElementById('cal-progress-text');
  const resultSection = document.getElementById('cal-result-section');
  const nodesGrid = document.getElementById('cal-nodes-grid');
  const statusText = document.getElementById('cal-status-text');
  const calDot = document.querySelector('.cal-dot');
  if (!startBtn) return;
  let pollTimer = null;

  function showResult(result) {
    if (!result || !result.nodes) return;
    resultSection.style.display = '';
    statusText.textContent = 'Calibrated (' + (result.node_count || 0) + ' nodes)';
    calDot.className = 'cal-dot calibrated';
    while (nodesGrid.firstChild) nodesGrid.removeChild(nodesGrid.firstChild);
    Object.keys(result.nodes).forEach(nid => {
      const n = result.nodes[nid];
      const card = makeEl('div', { className: 'cal-node-card' });
      card.appendChild(makeEl('div', { className: 'cal-node-title', textContent: 'Node ' + nid }));
      [['Distance', n.estimated_distance_m + ' m'], ['RSSI', n.mean_rssi + ' dBm'], ['Amplitude', n.mean_amplitude.toFixed(2)], ['Samples', n.sample_count]].forEach(([label, val]) => {
        const row = makeEl('div', { className: 'cal-node-stat' });
        row.appendChild(makeEl('span', { textContent: label }));
        row.appendChild(makeEl('b', { textContent: String(val) }));
        card.appendChild(row);
      });
      nodesGrid.appendChild(card);
    });
  }

  function pollCalibration() {
    pollTimer = setInterval(() => {
      fetch('/api/calibration/status').then(r => r.json()).then(data => {
        if (data.status === 'calibrating') {
          const pct = Math.round((data.progress || 0) * 100);
          progressFill.style.width = pct + '%';
          progressText.textContent = 'Collecting... ' + (data.samples_collected || 0) + ' samples from ' + (data.nodes_seen || 0) + ' nodes';
        } else if (data.status === 'complete' || data.last_result) {
          clearInterval(pollTimer);
          progressSection.style.display = 'none';
          startBtn.disabled = false; startBtn.textContent = 'Re-calibrate';
          if (data.status === 'calibrating') {
            fetch('/api/calibration/finish', { method: 'POST' }).then(r => r.json()).then(showResult);
          } else {
            showResult(data.last_result || data);
          }
        }
      });
    }, 500);

    setTimeout(() => {
      clearInterval(pollTimer);
      fetch('/api/calibration/finish', { method: 'POST' }).then(r => r.json()).then(data => {
        progressSection.style.display = 'none';
        startBtn.disabled = false; startBtn.textContent = 'Re-calibrate';
        showResult(data);
      }).catch(() => { startBtn.disabled = false; startBtn.textContent = 'Start Calibration'; });
    }, 6000);
  }

  startBtn.addEventListener('click', () => {
    fetch('/api/calibration/start', { method: 'POST' }).then(r => r.json()).then(data => {
      if (data.status === 'calibrating') {
        progressSection.style.display = '';
        resultSection.style.display = 'none';
        startBtn.disabled = true; startBtn.textContent = 'Calibrating...';
        pollCalibration();
      }
    }).catch(() => { statusText.textContent = 'Server unavailable'; });
  });

  fetch('/api/calibration/status').then(r => r.json()).then(data => {
    if (data.last_result) { showResult(data.last_result); startBtn.textContent = 'Re-calibrate'; }
  }).catch(() => {});
}

function initNotifications() {
  const testBtn = document.getElementById('notify-test-btn');
  const testResult = document.getElementById('notify-test-result');
  if (!testBtn) return;

  fetch('/api/notifications/status').then(r => r.json()).then(data => {
    const channels = data.channels || [];
    if (channels.indexOf('webhook') >= 0) { const el = document.getElementById('notify-webhook-status'); if (el) { el.textContent = 'Active'; el.style.color = '#4ade80'; } }
    if (channels.indexOf('line') >= 0) { const el = document.getElementById('notify-line-status'); if (el) { el.textContent = 'Active'; el.style.color = '#4ade80'; } }
    if (channels.indexOf('telegram') >= 0) { const el = document.getElementById('notify-telegram-status'); if (el) { el.textContent = 'Active'; el.style.color = '#4ade80'; } }
    if (data.enabled) testBtn.disabled = false;
  }).catch(() => {});

  testBtn.addEventListener('click', () => {
    testBtn.disabled = true; testResult.textContent = 'Sending...';
    fetch('/api/notifications/test', { method: 'POST' }).then(r => r.json()).then(data => {
      testResult.textContent = (data.results || []).join(', ');
      testBtn.disabled = false;
    }).catch(e => { testResult.textContent = 'Error: ' + e; testBtn.disabled = false; });
  });
}

function initOTA() {
  const listEl = document.getElementById('ota-firmware-list');
  if (!listEl) return;

  fetch('/api/ota/firmware').then(r => r.json()).then(data => {
    while (listEl.firstChild) listEl.removeChild(listEl.firstChild);
    const bins = data.firmware || [];
    if (bins.length === 0) {
      listEl.appendChild(makeEl('span', { className: 'help-text', textContent: 'No firmware binaries found. Build firmware first.' }));
      return;
    }
    bins.forEach(fw => {
      const row = makeEl('div', { className: 'ota-firmware-row' });
      row.appendChild(makeEl('span', { className: 'ota-fw-name', textContent: fw.name }));
      row.appendChild(makeEl('span', { className: 'ota-fw-size', textContent: (fw.size / 1024).toFixed(0) + ' KB' }));
      const link = document.createElement('a');
      link.className = 'btn btn-sm';
      link.href = fw.path;
      link.textContent = 'Download';
      link.setAttribute('download', '');
      row.appendChild(link);
      listEl.appendChild(row);
    });
  }).catch(() => { listEl.textContent = 'Server unavailable'; });
}

export default {
  id: 'hardware',
  label: 'Hardware',

  init() {
    const el = document.getElementById('tab-hardware');
    if (!el) return;

    // Build DOM
    buildDOM(el);

    // Initialize interactive behaviours
    initAntennaToggle();
    initPlacement();
    initProfiles();
    initWifiConfig();
    initFlasher();
    initDataCollector();
    initCalibration();
    initNotifications();
    initOTA();

    console.log('Hardware tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-hardware');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-hardware');
    if (el) el.style.display = 'none';
  },
};
