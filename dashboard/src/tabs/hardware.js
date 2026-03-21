// dashboard/src/tabs/hardware.js
/**
 * Hardware tab — Device-Centric layout with inline wizard.
 * Three sections: DEVICES, NODE PLACEMENT + SETUP, FIRMWARE & OTA.
 */

function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

// ── State ────────────────────────────────────────────────
let detectedDevices = [];
let wifiConfig = { ssid: '', password: '', server_ip: '', detected: false };
let wizardOpen = false;
let wizardStep = 0;
let flashingDevice = null;

const NODE_COLORS = ['#00c8ff', '#3ddc84', '#ff6b6b', '#ffb400', '#c084fc', '#5bc0eb'];

// ── Helpers ──────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function statusDot(online) {
  const dot = makeEl('span');
  dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${online ? '#0f0' : '#f44'};margin-right:4px`;
  return dot;
}

function badge(text, color, bg) {
  const b = makeEl('span', { textContent: text });
  b.style.cssText = `background:${bg || '#222'};color:${color || '#888'};padding:2px 8px;border-radius:3px;font-size:10px`;
  return b;
}

function actionBtn(text, color, onclick) {
  const btn = makeEl('button', { textContent: text });
  btn.style.cssText = `background:#111;border:1px solid #333;color:${color || '#888'};padding:4px 10px;border-radius:3px;font-size:11px;cursor:pointer`;
  if (onclick) btn.addEventListener('click', onclick);
  return btn;
}

// ══════════════════════════════════════════════════════════
// BUILD DOM
// ══════════════════════════════════════════════════════════
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);
  const scroll = makeEl('div', { className: 'tab-scroll' });

  // ── SECTION 1: DEVICES ──────────────────────────────────
  const devSection = makeEl('div', { className: 'panel', id: 'hw-devices-panel' });
  const devHeader = makeEl('div', { className: 'hw-section-header' });
  devHeader.appendChild(makeEl('h2', { textContent: 'Devices' }));
  const devActions = makeEl('div', { className: 'hw-header-actions' });
  devActions.appendChild(actionBtn('Scan USB', '#888', scanUSBDevices));
  devHeader.appendChild(devActions);
  devSection.appendChild(devHeader);
  devSection.appendChild(makeEl('div', { id: 'hw-device-cards', className: 'hw-device-grid' }));
  scroll.appendChild(devSection);

  // ── SECTION 2: PLACEMENT + SETUP STATUS ─────────────────
  const placeSection = makeEl('div', { className: 'panel', id: 'hw-placement-panel' });
  const placeGrid = makeEl('div', { className: 'hw-place-grid' });

  // Left: Placement map
  const mapCol = makeEl('div', { className: 'hw-map-col' });
  mapCol.appendChild(makeEl('h2', { textContent: 'Node Placement' }));
  const mapCanvas = document.createElement('canvas');
  mapCanvas.id = 'hw-placement-canvas';
  mapCanvas.width = 360;
  mapCanvas.height = 280;
  mapCanvas.style.cssText = 'width:100%;max-width:360px;border:1px solid #333;border-radius:4px;background:#080808';
  mapCol.appendChild(mapCanvas);
  placeGrid.appendChild(mapCol);

  // Right: Setup status
  const statusCol = makeEl('div', { className: 'hw-status-col' });
  statusCol.appendChild(makeEl('h2', { textContent: 'Setup Status' }));
  const statusBox = makeEl('div', { id: 'hw-setup-status', className: 'hw-status-box' });
  statusCol.appendChild(statusBox);
  placeGrid.appendChild(statusCol);

  placeSection.appendChild(placeGrid);
  scroll.appendChild(placeSection);

  // ── SECTION 3: FIRMWARE & OTA ───────────────────────────
  const fwSection = makeEl('div', { className: 'panel', id: 'hw-firmware-panel' });
  const fwHeader = makeEl('div', { className: 'hw-section-header' });
  fwHeader.appendChild(makeEl('h2', { textContent: 'Firmware & OTA' }));
  const fwActions = makeEl('div', { className: 'hw-header-actions' });
  fwActions.appendChild(actionBtn('Build New Firmware', '#f80', buildFirmware));
  fwHeader.appendChild(fwActions);
  fwSection.appendChild(fwHeader);
  fwSection.appendChild(makeEl('div', { id: 'hw-firmware-grid', className: 'hw-fw-grid' }));
  scroll.appendChild(fwSection);

  container.appendChild(scroll);
}

// ══════════════════════════════════════════════════════════
// DEVICE CARDS
// ══════════════════════════════════════════════════════════
function renderDeviceCards() {
  const container = document.getElementById('hw-device-cards');
  if (!container) return;
  container.textContent = '';

  // Fetch live node status from server
  fetch('/api/status').then(r => r.json()).then(status => {
    const ps = status.pipeline_status || {};
    const nodes = status.nodes || {};

    // Render detected USB devices
    detectedDevices.forEach((dev, i) => {
      const nodeId = dev._nodeId || (i + 1);
      const nodeInfo = nodes[String(nodeId)];
      const online = !!nodeInfo;
      const card = makeEl('div', { className: 'hw-device-card' + (online ? ' online' : '') });

      // Header row
      const header = makeEl('div', { className: 'hw-device-header' });
      header.appendChild(statusDot(online));
      header.appendChild(makeEl('b', { textContent: dev.name || 'ESP32', style: 'color:#0af;font-size:13px' }));
      header.appendChild(makeEl('span', { textContent: ' ' + dev.port, style: 'color:#555;font-size:11px' }));
      const statusBadge = online ? badge('Online', '#0f0', '#001a00') : badge('USB Only', '#888', '#1a1a1a');
      statusBadge.style.marginLeft = 'auto';
      header.appendChild(statusBadge);
      card.appendChild(header);

      // Info row
      const info = makeEl('div', { className: 'hw-device-info' });
      info.appendChild(makeEl('span', { textContent: 'Node ' + nodeId, style: 'color:#0af' }));
      if (nodeInfo) {
        info.appendChild(makeEl('span', { textContent: 'RSSI ' + nodeInfo.rssi + ' dBm' }));
        info.appendChild(makeEl('span', { textContent: 'Seq ' + nodeInfo.last_seq }));
      }
      if (dev.chip) info.appendChild(makeEl('span', { textContent: dev.chip.toUpperCase() }));
      card.appendChild(info);

      // Actions
      const actions = makeEl('div', { className: 'hw-device-actions' });
      actions.appendChild(actionBtn('Re-flash', '#0af', () => startWizardForDevice(dev, nodeId)));
      if (online) actions.appendChild(actionBtn('OTA Update', '#f80', () => pushOTAToNode(nodeId)));
      card.appendChild(actions);

      container.appendChild(card);
    });

    // Show nodes that are online but not in USB list (connected via WiFi only)
    Object.keys(nodes).forEach(nid => {
      const alreadyShown = detectedDevices.some((d, i) => String(d._nodeId || i + 1) === nid);
      if (!alreadyShown) {
        const nodeInfo = nodes[nid];
        const card = makeEl('div', { className: 'hw-device-card online' });
        const header = makeEl('div', { className: 'hw-device-header' });
        header.appendChild(statusDot(true));
        header.appendChild(makeEl('b', { textContent: 'Node ' + nid, style: 'color:#0af;font-size:13px' }));
        header.appendChild(makeEl('span', { textContent: ' (WiFi)', style: 'color:#555;font-size:11px' }));
        const sb = badge('Online', '#0f0', '#001a00');
        sb.style.marginLeft = 'auto';
        header.appendChild(sb);
        card.appendChild(header);

        const info = makeEl('div', { className: 'hw-device-info' });
        info.appendChild(makeEl('span', { textContent: 'RSSI ' + nodeInfo.rssi + ' dBm' }));
        info.appendChild(makeEl('span', { textContent: 'Seq ' + nodeInfo.last_seq }));
        card.appendChild(info);

        const actions = makeEl('div', { className: 'hw-device-actions' });
        actions.appendChild(actionBtn('OTA Update', '#f80', () => pushOTAToNode(parseInt(nid))));
        card.appendChild(actions);

        container.appendChild(card);
      }
    });

    // "+ Add Device" card
    const addCard = makeEl('div', { className: 'hw-device-card hw-add-card' });
    addCard.addEventListener('click', () => startWizardForDevice(null, detectedDevices.length + 1));
    addCard.appendChild(makeEl('div', { textContent: '+', style: 'font-size:28px;color:#555' }));
    addCard.appendChild(makeEl('div', { textContent: 'Add Device', style: 'color:#888;font-size:12px;margin-top:4px' }));
    addCard.appendChild(makeEl('div', { textContent: 'Plug in USB and click', style: 'color:#555;font-size:10px' }));
    container.appendChild(addCard);

    // If no devices at all, show hint
    if (detectedDevices.length === 0 && Object.keys(nodes).length === 0) {
      const hint = makeEl('div', { style: 'grid-column:1/-1;text-align:center;padding:16px;color:#888;font-size:12px' });
      hint.textContent = 'No devices detected. Plug in an ESP32 via USB or wait for WiFi nodes to connect.';
      container.insertBefore(hint, addCard);
    }
  }).catch(() => {
    // Server offline — show add card only
    const addCard = makeEl('div', { className: 'hw-device-card hw-add-card' });
    addCard.appendChild(makeEl('div', { textContent: '+', style: 'font-size:28px;color:#555' }));
    addCard.appendChild(makeEl('div', { textContent: 'Add Device', style: 'color:#888;font-size:12px' }));
    container.appendChild(addCard);
  });
}

// ══════════════════════════════════════════════════════════
// ADD DEVICE WIZARD (inline)
// ══════════════════════════════════════════════════════════
function startWizardForDevice(dev, nodeId) {
  wizardOpen = true;
  wizardStep = 0;
  flashingDevice = dev;

  const container = document.getElementById('hw-device-cards');
  if (!container) return;

  // Remove existing wizard
  const existing = document.getElementById('hw-wizard');
  if (existing) existing.remove();

  const wizard = makeEl('div', { id: 'hw-wizard', className: 'hw-wizard' });
  wizard.style.gridColumn = '1 / -1';

  // Step bar
  const stepBar = makeEl('div', { className: 'hw-wizard-steps' });
  const stepNames = ['Detect', 'Configure', 'Flash', 'Verify'];
  stepNames.forEach((name, i) => {
    const step = makeEl('div', { className: 'hw-wizard-step' + (i === 0 ? ' active' : ''), textContent: name });
    step.dataset.step = i;
    stepBar.appendChild(step);
  });
  wizard.appendChild(stepBar);

  // Content area
  const content = makeEl('div', { id: 'hw-wizard-content', className: 'hw-wizard-content' });
  wizard.appendChild(content);

  // Close button
  const closeBtn = makeEl('button', { textContent: '✕', className: 'hw-wizard-close' });
  closeBtn.addEventListener('click', () => { wizard.remove(); wizardOpen = false; });
  wizard.appendChild(closeBtn);

  // Insert before add card
  const addCard = container.querySelector('.hw-add-card');
  container.insertBefore(wizard, addCard);

  // Start wizard
  if (dev && dev.detected) {
    // Already detected — skip to configure
    showWizardStep(1, dev, nodeId);
  } else {
    showWizardStep(0, null, nodeId);
  }
}

function showWizardStep(step, dev, nodeId) {
  wizardStep = step;
  const content = document.getElementById('hw-wizard-content');
  if (!content) return;
  content.textContent = '';

  // Update step bar
  document.querySelectorAll('.hw-wizard-step').forEach(el => {
    const s = parseInt(el.dataset.step);
    el.className = 'hw-wizard-step' + (s < step ? ' done' : s === step ? ' active' : '');
  });

  if (step === 0) {
    // DETECT
    content.appendChild(makeEl('div', { textContent: 'Scanning USB ports for ESP32 devices...', style: 'color:#888;margin-bottom:12px' }));
    const spinner = makeEl('div', { textContent: '...', style: 'color:#f80' });
    content.appendChild(spinner);

    fetch('/api/firmware/detect').then(r => r.json()).then(data => {
      const devices = (data.devices || []).filter(d => d.detected);
      if (devices.length > 0) {
        detectedDevices = data.devices.filter(d => d.detected);
        spinner.textContent = devices.length + ' device(s) found';
        spinner.style.color = '#0f0';
        setTimeout(() => showWizardStep(1, devices[0], nodeId), 800);
      } else {
        spinner.textContent = 'No ESP32 found. Check USB connection.';
        spinner.style.color = '#f44';
        const retryBtn = actionBtn('Retry Scan', '#0af', () => showWizardStep(0, null, nodeId));
        retryBtn.style.marginTop = '12px';
        content.appendChild(retryBtn);
      }
    }).catch(err => {
      spinner.textContent = 'Scan failed: ' + err.message;
      spinner.style.color = '#f44';
    });

  } else if (step === 1) {
    // CONFIGURE
    if (!dev) dev = detectedDevices[0];
    flashingDevice = dev;

    const form = makeEl('div', { className: 'hw-wizard-form' });

    const rows = [
      ['Chip', dev ? (dev.name || dev.chip || 'Unknown') : 'Unknown', '#0af'],
      ['Port', dev ? dev.port : 'N/A', '#ccc'],
      ['WiFi', wifiConfig.ssid || 'Detecting...', '#0f0'],
      ['Server', wifiConfig.server_ip || '...', '#ccc'],
    ];
    rows.forEach(([label, val, color]) => {
      const row = makeEl('div', { className: 'hw-wizard-row' });
      row.appendChild(makeEl('span', { textContent: label, style: 'color:#888;width:60px;display:inline-block' }));
      row.appendChild(makeEl('span', { textContent: val, style: 'color:' + color }));
      form.appendChild(row);
    });

    // Node ID selector
    const nodeRow = makeEl('div', { className: 'hw-wizard-row' });
    nodeRow.appendChild(makeEl('span', { textContent: 'Node ID', style: 'color:#888;width:60px;display:inline-block' }));
    const nodeButtons = makeEl('span', { style: 'display:inline-flex;gap:4px' });
    for (let n = 1; n <= 6; n++) {
      const nb = makeEl('button', { textContent: n });
      nb.style.cssText = `padding:2px 10px;border-radius:3px;border:1px solid #333;cursor:pointer;font-size:12px;font-weight:bold;` +
        (n === nodeId ? 'background:#0af;color:#000;border-color:#0af' : 'background:#222;color:#888');
      nb.addEventListener('click', () => {
        nodeId = n;
        nodeButtons.querySelectorAll('button').forEach(b => {
          b.style.background = parseInt(b.textContent) === n ? '#0af' : '#222';
          b.style.color = parseInt(b.textContent) === n ? '#000' : '#888';
          b.style.borderColor = parseInt(b.textContent) === n ? '#0af' : '#333';
        });
      });
      nodeButtons.appendChild(nb);
    }
    nodeRow.appendChild(nodeButtons);
    form.appendChild(nodeRow);

    content.appendChild(form);

    const flashBtn = makeEl('button', { textContent: 'Flash Firmware →' });
    flashBtn.style.cssText = 'margin-top:16px;padding:8px 24px;background:var(--accent-green,#0f0);border:none;color:#000;cursor:pointer;font-weight:bold;font-size:13px;border-radius:3px';
    flashBtn.addEventListener('click', () => showWizardStep(2, dev, nodeId));
    content.appendChild(flashBtn);

  } else if (step === 2) {
    // FLASH
    if (!dev) dev = flashingDevice;
    const status = makeEl('div', { id: 'hw-wizard-flash-status' });
    status.appendChild(makeEl('div', { textContent: 'Flashing ' + (dev ? dev.name : 'device') + ' as Node ' + nodeId + '...', style: 'color:#f80;margin-bottom:12px' }));

    const progWrap = makeEl('div', { style: 'background:#222;border-radius:3px;height:8px;overflow:hidden;margin-bottom:8px' });
    const progFill = makeEl('div', { id: 'hw-wizard-prog', style: 'height:100%;background:var(--accent-green,#0f0);width:0%;transition:width 0.3s' });
    progWrap.appendChild(progFill);
    status.appendChild(progWrap);

    const statusText = makeEl('div', { id: 'hw-wizard-flash-text', style: 'color:#888;font-size:11px' });
    status.appendChild(statusText);
    content.appendChild(status);

    // Trigger flash
    doFlash(dev, nodeId);

  } else if (step === 3) {
    // VERIFY
    content.appendChild(makeEl('div', { textContent: '✓', style: 'font-size:32px;color:var(--accent-green,#0f0);margin-bottom:8px' }));
    content.appendChild(makeEl('div', { textContent: 'Flash complete!', style: 'color:#ccc;font-weight:bold;font-size:14px;margin-bottom:4px' }));
    content.appendChild(makeEl('div', { textContent: 'Node ' + nodeId + ' firmware flashed successfully. The device will reboot and connect to WiFi.', style: 'color:#888;font-size:12px;margin-bottom:16px' }));

    const doneBtn = makeEl('button', { textContent: 'Done' });
    doneBtn.style.cssText = 'padding:8px 24px;background:var(--accent-green,#0f0);border:none;color:#000;cursor:pointer;font-weight:bold;font-size:13px;border-radius:3px';
    doneBtn.addEventListener('click', () => {
      const wizard = document.getElementById('hw-wizard');
      if (wizard) wizard.remove();
      wizardOpen = false;
      scanUSBDevices();
    });
    content.appendChild(doneBtn);
  }
}

async function doFlash(dev, nodeId) {
  const prog = document.getElementById('hw-wizard-prog');
  const text = document.getElementById('hw-wizard-flash-text');
  if (!dev || !dev.port) { if (text) text.textContent = 'No device to flash'; return; }

  try {
    if (prog) prog.style.width = '10%';
    if (text) text.textContent = 'Starting build & flash...';

    const resp = await fetch('/api/firmware/auto?port=' + encodeURIComponent(dev.port) + '&node_id=' + nodeId, { method: 'POST' });
    const startData = await resp.json();
    if (startData.error) { if (text) { text.textContent = 'Error: ' + startData.error; text.style.color = '#f44'; } return; }

    if (prog) prog.style.width = '20%';
    if (text) text.textContent = 'Building firmware...';

    // Poll for completion
    let done = false;
    while (!done) {
      await sleep(3000);
      const sr = await fetch('/api/firmware/status');
      const s = await sr.json();
      if (s.status === 'building') {
        if (prog) prog.style.width = '50%';
      } else {
        done = true;
        if (s.status === 'complete' && s.success !== false) {
          if (prog) prog.style.width = '100%';
          if (text) { text.textContent = 'Flash complete!'; text.style.color = '#0f0'; }
          await sleep(500);
          showWizardStep(3, dev, nodeId);
        } else {
          if (text) { text.textContent = 'Failed: ' + (s.error || 'unknown'); text.style.color = '#f44'; }
        }
      }
    }
  } catch (err) {
    if (text) { text.textContent = 'Error: ' + err.message; text.style.color = '#f44'; }
  }
}

// ══════════════════════════════════════════════════════════
// SETUP STATUS
// ══════════════════════════════════════════════════════════
let _statusRendering = false;
async function renderSetupStatus() {
  const box = document.getElementById('hw-setup-status');
  if (!box || _statusRendering) return;
  _statusRendering = true;
  box.textContent = '';

  // WiFi info
  const items = [
    ['WiFi', wifiConfig.ssid || 'Not detected', wifiConfig.detected ? '#0f0' : '#f44'],
    ['Server', wifiConfig.server_ip || 'Unknown', '#ccc'],
    ['UDP Port', '5005', '#888'],
  ];
  items.forEach(([label, val, color]) => {
    const row = makeEl('div', { className: 'hw-status-row' });
    row.appendChild(makeEl('span', { textContent: label, style: 'color:#888' }));
    row.appendChild(makeEl('span', { textContent: val, style: 'color:' + color }));
    box.appendChild(row);
  });

  // WiFi mismatch warning — firmware was flashed with different WiFi
  if (wifiConfig.firmware_match === false) {
    const warn = makeEl('div', { style: 'background:#331100;border:1px solid #f80;border-radius:4px;padding:8px;margin-top:8px;font-size:11px' });
    warn.appendChild(makeEl('div', { textContent: 'WiFi changed since last flash!', style: 'color:#f80;font-weight:bold;margin-bottom:4px' }));
    warn.appendChild(makeEl('div', { textContent: 'Firmware: ' + (wifiConfig.firmware_ssid || '?') + ' → ' + (wifiConfig.firmware_ip || '?'), style: 'color:#888' }));
    warn.appendChild(makeEl('div', { textContent: 'Current:  ' + wifiConfig.ssid + ' → ' + wifiConfig.server_ip, style: 'color:#ccc' }));
    const reflashBtn = makeEl('button', { textContent: 'Re-flash All Devices with New WiFi' });
    reflashBtn.style.cssText = 'margin-top:8px;padding:6px 14px;background:#f80;border:none;color:#000;cursor:pointer;font-weight:bold;font-size:11px;border-radius:3px;width:100%';
    reflashBtn.addEventListener('click', reflashAllDevices);
    warn.appendChild(reflashBtn);
    box.appendChild(warn);
  }

  // Profile (compact)
  try {
    const profData = await fetch('/api/profiles').then(r => r.json());
    const active = (profData.profiles || []).find(p => p.active);
    if (active) {
      box.appendChild(makeEl('div', { style: 'border-top:1px solid #222;margin:6px 0' }));
      const row = makeEl('div', { className: 'hw-status-row' });
      row.appendChild(makeEl('span', { textContent: 'Profile', style: 'color:#888' }));
      row.appendChild(makeEl('span', { textContent: active.name, style: 'color:#ccc' }));
      box.appendChild(row);
      const specRow = makeEl('div', { className: 'hw-status-row' });
      specRow.appendChild(makeEl('span', { textContent: '', style: 'color:#888' }));
      specRow.appendChild(makeEl('span', { textContent: active.num_subcarriers + ' sub · ' + active.csi_sample_rate + ' Hz · ' + active.frequency_ghz + ' GHz', style: 'color:#555;font-size:10px' }));
      box.appendChild(specRow);
    }
  } catch {}

  // Calibration
  try {
    const calData = await fetch('/api/calibration/status').then(r => r.json());
    box.appendChild(makeEl('div', { style: 'border-top:1px solid #222;margin:6px 0' }));
    const calRow = makeEl('div', { className: 'hw-status-row' });
    calRow.appendChild(makeEl('span', { textContent: 'Calibration', style: 'color:#888' }));
    const calDone = calData.last_result || calData.status === 'complete';
    calRow.appendChild(makeEl('span', { textContent: calDone ? 'Done ✓' : 'Not done', style: 'color:' + (calDone ? '#0f0' : '#f80') }));
    box.appendChild(calRow);
    const calBtn = actionBtn(calDone ? 'Re-calibrate' : 'Start Calibration', calDone ? '#888' : '#f80', startCalibration);
    calBtn.style.marginTop = '6px';
    calBtn.style.width = '100%';
    box.appendChild(calBtn);
  } catch {}
  _statusRendering = false;
}

function startCalibration() {
  fetch('/api/calibration/start', { method: 'POST' }).then(r => r.json()).then(data => {
    if (data.status === 'calibrating') {
      const box = document.getElementById('hw-setup-status');
      if (!box) return;
      const prog = makeEl('div', { style: 'margin-top:8px' });
      prog.appendChild(makeEl('div', { textContent: 'Calibrating... Stand still at room centre', style: 'color:#f80;font-size:11px;margin-bottom:4px' }));
      const bar = makeEl('div', { style: 'background:#222;border-radius:3px;height:6px;overflow:hidden' });
      const fill = makeEl('div', { id: 'hw-cal-prog', style: 'height:100%;background:#f80;width:0%;transition:width 0.3s' });
      bar.appendChild(fill);
      prog.appendChild(bar);
      box.appendChild(prog);

      const timer = setInterval(() => {
        fetch('/api/calibration/status').then(r => r.json()).then(d => {
          const pct = Math.round((d.progress || 0) * 100);
          const f = document.getElementById('hw-cal-prog');
          if (f) f.style.width = pct + '%';
          if (d.status !== 'calibrating') {
            clearInterval(timer);
            fetch('/api/calibration/finish', { method: 'POST' }).then(() => renderSetupStatus());
          }
        });
      }, 500);

      setTimeout(() => {
        clearInterval(timer);
        fetch('/api/calibration/finish', { method: 'POST' }).then(() => renderSetupStatus());
      }, 6000);
    }
  }).catch(() => {});
}

// ══════════════════════════════════════════════════════════
// PLACEMENT MAP
// ══════════════════════════════════════════════════════════
function renderPlacementMap() {
  const canvas = document.getElementById('hw-placement-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const roomR = Math.min(w, h) * 0.4;

  ctx.clearRect(0, 0, w, h);

  // Room outline
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 2;
  ctx.strokeRect(cx - roomR, cy - roomR, roomR * 2, roomR * 2);

  // Grid
  ctx.strokeStyle = '#1a1a30';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 4; i++) {
    const off = (roomR * 2 * i) / 4;
    ctx.beginPath(); ctx.moveTo(cx - roomR + off, cy - roomR); ctx.lineTo(cx - roomR + off, cy + roomR); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - roomR, cy - roomR + off); ctx.lineTo(cx + roomR, cy - roomR + off); ctx.stroke();
  }

  // Person marker
  ctx.fillStyle = '#ffffff22';
  ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Person', cx, cy + 22);

  // Draw nodes from status
  fetch('/api/status').then(r => r.json()).then(status => {
    const nodes = status.nodes || {};
    const positions = status.node_positions || {};
    const nodeIds = Object.keys(nodes);

    nodeIds.forEach((nid, i) => {
      const pos = positions[parseInt(nid)] || positions[nid];
      let nx, ny;
      if (pos) {
        // pos = [x, y, z], map to canvas
        nx = cx + (pos[0] / 2.0) * roomR;
        ny = cy - (pos[2] / 2.0) * roomR;
      } else {
        // Default: distribute around perimeter
        const angle = (i / Math.max(nodeIds.length, 1)) * Math.PI * 2 - Math.PI / 2;
        nx = cx + Math.cos(angle) * roomR * 0.85;
        ny = cy + Math.sin(angle) * roomR * 0.85;
      }

      const color = NODE_COLORS[i % NODE_COLORS.length];
      // Glow
      ctx.beginPath();
      ctx.arc(nx, ny, 18, 0, Math.PI * 2);
      ctx.fillStyle = color + '22';
      ctx.fill();
      // Dot
      ctx.beginPath();
      ctx.arc(nx, ny, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('N' + nid, nx, ny - 12);
    });

    // Room dimensions label
    ctx.fillStyle = '#555';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('4m', cx, cy + roomR + 14);
  }).catch(() => {});
}

// ══════════════════════════════════════════════════════════
// FIRMWARE & OTA
// ══════════════════════════════════════════════════════════
function renderFirmware() {
  const grid = document.getElementById('hw-firmware-grid');
  if (!grid) return;
  grid.textContent = '';

  // Latest build info
  const buildCard = makeEl('div', { className: 'hw-fw-card' });
  buildCard.appendChild(makeEl('div', { textContent: 'Latest Build', style: 'color:#888;font-size:11px;margin-bottom:6px' }));

  fetch('/api/ota/firmware').then(r => r.json()).then(data => {
    const bins = data.firmware || [];
    if (bins.length > 0) {
      bins.forEach(fw => {
        const row = makeEl('div', { style: 'display:flex;justify-content:space-between;align-items:center;margin-bottom:4px' });
        row.appendChild(makeEl('span', { textContent: fw.name, style: 'color:#ccc;font-size:12px' }));
        row.appendChild(makeEl('span', { textContent: (fw.size / 1024).toFixed(0) + ' KB', style: 'color:#555;font-size:11px' }));
        buildCard.appendChild(row);
      });
    } else {
      buildCard.appendChild(makeEl('div', { textContent: 'No firmware built yet', style: 'color:#555;font-size:12px' }));
    }
  }).catch(() => {
    buildCard.appendChild(makeEl('div', { textContent: 'Server offline', style: 'color:#f44;font-size:12px' }));
  });
  grid.appendChild(buildCard);

  // OTA status
  const otaCard = makeEl('div', { className: 'hw-fw-card' });
  otaCard.appendChild(makeEl('div', { textContent: 'OTA Status', style: 'color:#888;font-size:11px;margin-bottom:6px' }));

  fetch('/api/status').then(r => r.json()).then(status => {
    const nodes = status.nodes || {};
    const ps = status.pipeline_status || {};
    const realNodes = ps.real_nodes || 0;
    if (realNodes > 0) {
      otaCard.appendChild(makeEl('div', { textContent: realNodes + ' node(s) online', style: 'color:#0f0;font-size:12px' }));
      const nodeBadges = makeEl('div', { style: 'display:flex;gap:4px;flex-wrap:wrap;margin-top:4px' });
      Object.keys(nodes).forEach(nid => {
        nodeBadges.appendChild(badge('N' + nid + ' ✓', '#0f0', '#001a00'));
      });
      otaCard.appendChild(nodeBadges);
    } else {
      otaCard.appendChild(makeEl('div', { textContent: 'No nodes online', style: 'color:#555;font-size:12px' }));
    }
  }).catch(() => {});
  grid.appendChild(otaCard);

  // Push OTA
  const pushCard = makeEl('div', { className: 'hw-fw-card' });
  pushCard.appendChild(makeEl('div', { textContent: 'Push Update', style: 'color:#888;font-size:11px;margin-bottom:6px' }));
  pushCard.appendChild(makeEl('div', { textContent: 'OTA push to all online nodes', style: 'color:#555;font-size:10px;margin-bottom:8px' }));
  const pushBtn = makeEl('button', { textContent: 'Push OTA to All' });
  pushBtn.style.cssText = 'background:#0a5;color:#fff;padding:6px 14px;border:none;border-radius:3px;font-size:11px;font-weight:bold;cursor:pointer';
  pushBtn.addEventListener('click', pushOTAToAll);
  pushCard.appendChild(pushBtn);
  pushCard.appendChild(makeEl('div', { id: 'hw-ota-status', style: 'font-size:10px;color:#888;margin-top:4px' }));
  grid.appendChild(pushCard);
}

async function pushOTAToNode(nodeId) {
  const statusEl = document.getElementById('hw-ota-status');
  try {
    const resp = await fetch('/api/ota/push?node_id=' + nodeId, { method: 'POST' });
    const data = await resp.json();
    if (statusEl) statusEl.textContent = data.message || data.error || 'Sent';
  } catch (err) {
    if (statusEl) statusEl.textContent = 'Error: ' + err.message;
  }
}

async function pushOTAToAll() {
  const statusEl = document.getElementById('hw-ota-status');
  if (statusEl) statusEl.textContent = 'Pushing OTA to all nodes...';
  try {
    const resp = await fetch('/api/ota/push', { method: 'POST' });
    const data = await resp.json();
    if (statusEl) { statusEl.textContent = data.message || 'OTA push initiated'; statusEl.style.color = '#0f0'; }
  } catch (err) {
    if (statusEl) { statusEl.textContent = 'Error: ' + err.message; statusEl.style.color = '#f44'; }
  }
}

async function buildFirmware() {
  const grid = document.getElementById('hw-firmware-grid');
  if (!grid) return;
  const statusEl = makeEl('div', { style: 'grid-column:1/-1;color:#f80;font-size:12px;padding:8px' });
  statusEl.textContent = 'Building firmware...';
  grid.appendChild(statusEl);

  try {
    const resp = await fetch('/api/firmware/build?node_ids=1', { method: 'POST' });
    const data = await resp.json();
    if (data.error) { statusEl.textContent = 'Error: ' + data.error; statusEl.style.color = '#f44'; return; }
    statusEl.textContent = 'Build started...';

    // Poll
    let done = false;
    while (!done) {
      await sleep(3000);
      const sr = await fetch('/api/firmware/status');
      const s = await sr.json();
      if (s.status !== 'building') {
        done = true;
        statusEl.textContent = s.status === 'complete' ? 'Build complete!' : 'Failed: ' + (s.error || '');
        statusEl.style.color = s.status === 'complete' ? '#0f0' : '#f44';
        if (s.status === 'complete') setTimeout(renderFirmware, 1000);
      }
    }
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
    statusEl.style.color = '#f44';
  }
}

// ══════════════════════════════════════════════════════════
// SCAN & INIT
// ══════════════════════════════════════════════════════════
let _scanInProgress = false;
function scanUSBDevices() {
  if (_scanInProgress) return;
  _scanInProgress = true;
  fetch('/api/firmware/detect').then(r => r.json()).then(data => {
    // Deduplicate by port
    const seen = new Set();
    detectedDevices = (data.devices || []).filter(d => {
      if (!d.detected || seen.has(d.port)) return false;
      seen.add(d.port);
      return true;
    });
    renderDeviceCards();
  }).catch(() => { renderDeviceCards(); }).finally(() => { _scanInProgress = false; });
}

function fetchWifiConfig() {
  fetch('/api/network/wifi').then(r => r.json()).then(data => {
    wifiConfig = data;
    renderSetupStatus();
  }).catch(() => { renderSetupStatus(); });
}

function refreshAll() {
  scanUSBDevices();
  fetchWifiConfig();
  renderPlacementMap();
  renderFirmware();
}

async function reflashAllDevices() {
  if (detectedDevices.length === 0) {
    alert('No USB devices detected. Plug in ESP32 boards first.');
    return;
  }

  for (let i = 0; i < detectedDevices.length; i++) {
    const dev = detectedDevices[i];
    const nodeId = i + 1;
    startWizardForDevice(dev, nodeId);
    // Wait for wizard to finish (poll firmware status)
    let done = false;
    while (!done) {
      await sleep(2000);
      try {
        const sr = await fetch('/api/firmware/status');
        const s = await sr.json();
        if (s.status !== 'building') done = true;
      } catch { done = true; }
    }
    await sleep(1000);
  }
}

// ══════════════════════════════════════════════════════════
// CSS (injected once)
// ══════════════════════════════════════════════════════════
function injectCSS() {
  if (document.getElementById('hw-tab-css')) return;
  const style = document.createElement('style');
  style.id = 'hw-tab-css';
  style.textContent = `
    .hw-section-header { display:flex;justify-content:space-between;align-items:center;margin-bottom:12px }
    .hw-header-actions { display:flex;gap:6px }
    .hw-device-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px }
    .hw-device-card { border:1px solid #333;padding:12px;border-radius:6px;background:#0a0a0a }
    .hw-device-card.online { border-color:#0a5 }
    .hw-device-card.hw-add-card { border:2px dashed #333;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100px;cursor:pointer;transition:border-color 0.2s }
    .hw-device-card.hw-add-card:hover { border-color:#555 }
    .hw-device-header { display:flex;align-items:center;gap:4px;margin-bottom:6px }
    .hw-device-info { display:flex;gap:12px;font-size:11px;color:#888;margin-bottom:8px }
    .hw-device-info b { color:#ccc }
    .hw-device-actions { display:flex;gap:6px }
    .hw-place-grid { display:grid;grid-template-columns:2fr 1fr;gap:16px }
    @media(max-width:768px) { .hw-place-grid { grid-template-columns:1fr } }
    .hw-map-col h2,.hw-status-col h2 { font-size:12px;color:var(--text-secondary);text-transform:uppercase;margin-bottom:8px }
    .hw-status-box { background:#111;border-radius:4px;padding:10px;font-size:12px }
    .hw-status-row { display:flex;justify-content:space-between;padding:3px 0 }
    .hw-fw-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px }
    .hw-fw-card { background:#111;padding:12px;border-radius:4px }
    .hw-wizard { border:2px solid #f80;border-radius:6px;padding:16px;background:#0a0a0a;position:relative }
    .hw-wizard-close { position:absolute;top:8px;right:10px;background:none;border:none;color:#888;font-size:16px;cursor:pointer }
    .hw-wizard-steps { display:flex;gap:4px;margin-bottom:16px }
    .hw-wizard-step { flex:1;padding:6px;text-align:center;border:1px solid #333;border-radius:4px;font-size:10px;color:#555;transition:all 0.2s }
    .hw-wizard-step.active { background:#ff02;border-color:#ff0;color:#ff0 }
    .hw-wizard-step.done { background:#0f02;border-color:#0f0;color:#0f0 }
    .hw-wizard-content { text-align:center;padding:8px }
    .hw-wizard-form { text-align:left;max-width:320px;margin:0 auto;line-height:2.2 }
    .hw-wizard-row { display:flex;align-items:center;gap:8px }
  `;
  document.head.appendChild(style);
}

// ══════════════════════════════════════════════════════════
// EXPORT
// ══════════════════════════════════════════════════════════
export default {
  id: 'hardware',
  label: 'Hardware',

  init() {
    const el = document.getElementById('tab-hardware');
    if (!el) return;
    injectCSS();
    buildDOM(el);
    refreshAll();
    console.log('Hardware tab initialized (device-centric)');
  },

  activate() {
    const el = document.getElementById('tab-hardware');
    if (el) el.style.display = 'block';
    refreshAll();
  },

  deactivate() {
    const el = document.getElementById('tab-hardware');
    if (el) el.style.display = 'none';
  },
};
