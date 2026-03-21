// dashboard/src/tabs/demo.js
/**
 * Live Demo tab — real-time signal canvas (6 CSI subcarrier channels)
 * and 2D stick-figure pose canvas.
 *
 * Subscribes to EventBus 'vitals' for breathing-rate influence on
 * signal waveforms and 'tab:changed' to start/stop its animation
 * loop only when visible.
 */
import { bus } from '../events.js';
import { observeResize } from '../utils/resize.js';

// ── State ─────────────────────────────────────────────────────────
let animId = null;
let t = 0;
let signalScrollOffset = 0;
let breathRate = 15;
let active = false;

// Canvas refs
let signalCanvas, signalCtx;
let poseCanvas, poseCtx;
let unobserveSig = null;
let unobservePose = null;
let unsubVitals = null;

// REST pose reference (24 joints) for 2D stick-figure
const REST = [
  [0.000,1.700],[0.000,1.550],[0.000,1.380],[0.000,1.120],
  [-0.200,1.400],[-0.480,1.400],[-0.700,1.400],
  [0.200,1.400],[0.480,1.400],[0.700,1.400],
  [0.000,0.950],[0.000,0.900],
  [-0.100,0.880],[-0.100,0.500],[-0.100,0.080],
  [0.100,0.880],[0.100,0.500],[0.100,0.080],
  [-0.100,0.030],[0.100,0.030],[-0.780,1.400],[0.780,1.400],
  [-0.030,1.720],[0.030,1.720],
];

const POSE_BONES_2D = [
  [0,1],[1,2],[2,3],[3,10],       // head -> spine
  [3,4],[4,5],[5,6],              // L arm
  [3,7],[7,8],[8,9],              // R arm
  [10,11],[11,12],[12,13],[13,14],// L leg
  [11,15],[15,16],[16,17],        // R leg
];

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

/**
 * Build the demo tab DOM using safe DOM API methods.
 */
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);

  const scroll = makeEl('div', { className: 'tab-scroll' });

  // Controls panel
  const controlsPanel = makeEl('div', { className: 'panel' });
  controlsPanel.appendChild(makeEl('h2', { textContent: 'Live Demonstration' }));
  const demoControls = makeEl('div', { className: 'demo-controls' });
  const startBtn = makeEl('button', { className: 'btn btn-primary', id: 'startDemo', textContent: 'Start Stream' });
  demoControls.appendChild(startBtn);
  const stopBtn = makeEl('button', { className: 'btn btn-secondary', id: 'stopDemo', textContent: 'Stop Stream' });
  stopBtn.disabled = true;
  demoControls.appendChild(stopBtn);
  demoControls.appendChild(makeEl('span', { className: 'demo-status', id: 'demoStatus', textContent: 'Ready' }));
  controlsPanel.appendChild(demoControls);
  scroll.appendChild(controlsPanel);

  // Demo grid (signal + pose)
  const demoGrid = makeEl('div', { className: 'demo-grid' });

  // Signal panel
  const sigPanel = makeEl('div', { className: 'panel' });
  sigPanel.appendChild(makeEl('h3', { textContent: 'WiFi Signal Analysis' }));
  const sigCanvas = document.createElement('canvas');
  sigCanvas.id = 'signalCanvas';
  sigCanvas.width = 400;
  sigCanvas.height = 200;
  sigPanel.appendChild(sigCanvas);
  const sigMetrics = makeEl('div', { className: 'signal-metrics' });
  const sigRow1 = makeEl('div', { className: 'metric-row' });
  sigRow1.appendChild(makeEl('span', { className: 'metric-label', textContent: 'Signal Strength:' }));
  sigRow1.appendChild(makeEl('span', { className: 'metric-val', id: 'signalStrength', textContent: '-45 dBm' }));
  sigMetrics.appendChild(sigRow1);
  const sigRow2 = makeEl('div', { className: 'metric-row' });
  sigRow2.appendChild(makeEl('span', { className: 'metric-label', textContent: 'Processing Latency:' }));
  sigRow2.appendChild(makeEl('span', { className: 'metric-val', id: 'latency', textContent: '12 ms' }));
  sigMetrics.appendChild(sigRow2);
  sigPanel.appendChild(sigMetrics);
  demoGrid.appendChild(sigPanel);

  // Pose panel
  const posePanel = makeEl('div', { className: 'panel' });
  posePanel.appendChild(makeEl('h3', { textContent: 'Pose Detection' }));
  const pCanvas = document.createElement('canvas');
  pCanvas.id = 'poseCanvas';
  pCanvas.width = 400;
  pCanvas.height = 200;
  posePanel.appendChild(pCanvas);
  const detInfo = makeEl('div', { className: 'detection-info' });
  [['Persons Detected:', 'personCount', '0'], ['Confidence:', 'confidence', '0.0%'], ['Keypoints:', 'keypoints', '0/24']].forEach(([label, id, val]) => {
    const row = makeEl('div', { className: 'metric-row' });
    row.appendChild(makeEl('span', { className: 'metric-label', textContent: label }));
    row.appendChild(makeEl('span', { className: 'metric-val', id: id, textContent: val }));
    detInfo.appendChild(row);
  });
  posePanel.appendChild(detInfo);
  demoGrid.appendChild(posePanel);

  scroll.appendChild(demoGrid);
  container.appendChild(scroll);
}

// ── Signal Canvas ─────────────────────────────────────────────────
function renderSignalCanvas() {
  if (!signalCtx) return;
  const w = signalCanvas.width;
  const h = signalCanvas.height;
  signalCtx.clearRect(0, 0, w, h);

  // Background
  signalCtx.fillStyle = '#0e0e22';
  signalCtx.fillRect(0, 0, w, h);

  // Grid
  signalCtx.strokeStyle = 'rgba(255,255,255,0.06)';
  signalCtx.lineWidth = 0.5;
  for (let gx = 0; gx < w; gx += 40) {
    signalCtx.beginPath(); signalCtx.moveTo(gx, 0); signalCtx.lineTo(gx, h); signalCtx.stroke();
  }
  for (let gy = 0; gy < h; gy += 30) {
    signalCtx.beginPath(); signalCtx.moveTo(0, gy); signalCtx.lineTo(w, gy); signalCtx.stroke();
  }

  signalScrollOffset += 1.5;

  // 6 subcarrier channels with retro-cyberpunk colours
  const colors = ['#00cc88', '#00aaff', '#ffaa00', '#ff4466', '#aa66ff', '#66ffcc'];
  const freqs  = [0.03, 0.05, 0.04, 0.025, 0.06, 0.035];
  const amps   = [0.6, 0.45, 0.55, 0.35, 0.4, 0.5];
  const phases = [0, 1.2, 2.5, 0.8, 3.7, 1.9];

  for (let ch = 0; ch < 6; ch++) {
    const baseY = h * (0.12 + ch * 0.14);
    signalCtx.strokeStyle = colors[ch];
    signalCtx.lineWidth = 1.5;
    signalCtx.globalAlpha = 0.85;
    signalCtx.beginPath();
    for (let x = 0; x < w; x++) {
      const tt = (x + signalScrollOffset) * freqs[ch];
      const val = Math.sin(tt + phases[ch]) * amps[ch]
               + Math.sin(tt * 2.3 + phases[ch] * 0.7) * amps[ch] * 0.3
               + Math.sin(tt * (breathRate / 60) * 0.15 + ch) * 0.15
               + (Math.random() - 0.5) * 0.08;
      const y = baseY + val * 22;
      if (x === 0) signalCtx.moveTo(x, y); else signalCtx.lineTo(x, y);
    }
    signalCtx.stroke();

    // Channel label
    signalCtx.globalAlpha = 0.5;
    signalCtx.fillStyle = colors[ch];
    signalCtx.font = '9px monospace';
    signalCtx.fillText('CH' + (ch + 1), 4, baseY - 12);
    signalCtx.globalAlpha = 1.0;
  }

  // Signal metrics
  const sigEl = document.getElementById('signalStrength');
  const latEl = document.getElementById('latency');
  if (sigEl) sigEl.textContent = (-42 + Math.sin(t * 0.3) * 5).toFixed(0) + ' dBm';
  if (latEl) latEl.textContent = (10 + Math.random() * 8).toFixed(0) + ' ms';
}

// ── Pose Canvas ───────────────────────────────────────────────────
function renderPoseCanvas() {
  if (!poseCtx) return;
  const w = poseCanvas.width;
  const h = poseCanvas.height;
  poseCtx.clearRect(0, 0, w, h);

  // Background
  poseCtx.fillStyle = '#0e0e22';
  poseCtx.fillRect(0, 0, w, h);

  // Compute 2D joint positions from REST pose + animation
  const pts = [];
  const cx = w * 0.5;
  const cy = h * 0.88;
  const scale = h * 0.42;
  for (let i = 0; i < 24; i++) {
    pts.push([cx + REST[i][0] * scale, cy - REST[i][1] * scale]);
  }

  // Arm swing, breathing, sway
  const armSwing = Math.sin(t * 0.6) * 8;
  const sway = Math.sin(t * 0.4) * 3;
  const breathOffset = Math.sin(t * 0.5) * 2;

  // L arm
  pts[5][1] += armSwing;    pts[5][0] -= 8;
  pts[6][1] += armSwing * 1.4; pts[6][0] -= 4;
  // R arm
  pts[8][1] -= armSwing;    pts[8][0] += 8;
  pts[9][1] -= armSwing * 1.4; pts[9][0] += 4;
  // Arms down (not T-pose)
  pts[4][1] += 6;  pts[5][1] += 45; pts[6][1] += 80;
  pts[7][1] += 6;  pts[8][1] += 45; pts[9][1] += 80;
  pts[5][0] += 3;  pts[6][0] += 5;
  pts[8][0] -= 3;  pts[9][0] -= 5;

  // Sway all
  for (let i = 0; i < 24; i++) pts[i][0] += sway;
  // Breathing (chest expansion)
  pts[2][0] += breathOffset * 0.3;
  pts[3][0] += breathOffset * 0.2;

  // Confidence heatmap circle
  const grad = poseCtx.createRadialGradient(
    cx + sway, cy - scale * 0.5, 10,
    cx + sway, cy - scale * 0.5, scale * 0.8,
  );
  grad.addColorStop(0, 'rgba(0,204,136,0.08)');
  grad.addColorStop(1, 'rgba(0,204,136,0)');
  poseCtx.fillStyle = grad;
  poseCtx.fillRect(0, 0, w, h);

  // Bones
  poseCtx.strokeStyle = '#00cc88';
  poseCtx.lineWidth = 2.5;
  poseCtx.lineCap = 'round';
  for (let bi = 0; bi < POSE_BONES_2D.length; bi++) {
    const a = POSE_BONES_2D[bi][0];
    const b = POSE_BONES_2D[bi][1];
    if (a >= 24 || b >= 24) continue;
    poseCtx.globalAlpha = 0.8;
    poseCtx.beginPath();
    poseCtx.moveTo(pts[a][0], pts[a][1]);
    poseCtx.lineTo(pts[b][0], pts[b][1]);
    poseCtx.stroke();
  }

  // Joints
  for (let i = 0; i < 18; i++) {
    const r = i === 0 ? 5 : 3.5;
    poseCtx.globalAlpha = 0.95;
    poseCtx.fillStyle = '#00ff88';
    poseCtx.beginPath();
    poseCtx.arc(pts[i][0], pts[i][1], r, 0, Math.PI * 2);
    poseCtx.fill();
    // Confidence ring
    poseCtx.globalAlpha = 0.3;
    poseCtx.strokeStyle = '#00ff88';
    poseCtx.lineWidth = 1;
    poseCtx.beginPath();
    poseCtx.arc(pts[i][0], pts[i][1], r + 3, 0, Math.PI * 2);
    poseCtx.stroke();
  }
  poseCtx.globalAlpha = 1.0;

  // Joint labels
  poseCtx.fillStyle = 'rgba(255,255,255,0.35)';
  poseCtx.font = '8px monospace';
  poseCtx.fillText('head', pts[0][0] + 8, pts[0][1] - 4);
  poseCtx.fillText('L', pts[6][0] - 14, pts[6][1]);
  poseCtx.fillText('R', pts[9][0] + 6, pts[9][1]);

  // Metrics
  const confPct = (85 + Math.sin(t * 0.2) * 8).toFixed(1);
  const personEl = document.getElementById('personCount');
  const confEl = document.getElementById('confidence');
  const kpEl = document.getElementById('keypoints');
  if (personEl) personEl.textContent = '1';
  if (confEl) confEl.textContent = confPct + '%';
  if (kpEl) kpEl.textContent = '24/24';
}

// ── Animation loop (only when tab is visible) ─────────────────────
let lastTime = 0;

function tick(time) {
  if (!active) return;
  const dt = Math.min((time - lastTime) / 1000, 0.1);
  lastTime = time;
  t += dt;

  renderSignalCanvas();
  renderPoseCanvas();

  animId = requestAnimationFrame(tick);
}

function startLoop() {
  if (animId) return;
  active = true;
  lastTime = performance.now();
  animId = requestAnimationFrame(tick);
}

function stopLoop() {
  active = false;
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

// ── Vitals listener (for breathRate influence on signal canvas) ───
function onVitals(data) {
  if (data.breathRate !== undefined) breathRate = data.breathRate;
}

function initDemoButtons() {
  const startBtn = document.getElementById('startDemo');
  const stopBtn = document.getElementById('stopDemo');
  const demoSt = document.getElementById('demoStatus');

  if (startBtn) startBtn.addEventListener('click', () => {
    startBtn.disabled = true; stopBtn.disabled = false;
    demoSt.textContent = 'Streaming...';
    demoSt.className = 'demo-status streaming';
  });
  if (stopBtn) stopBtn.addEventListener('click', () => {
    startBtn.disabled = false; stopBtn.disabled = true;
    demoSt.textContent = 'Stopped';
    demoSt.className = 'demo-status';
  });
}

// ── Public API ────────────────────────────────────────────────────
export default {
  id: 'demo',
  label: 'Live Demo',

  init() {
    const container = document.getElementById('tab-demo');
    if (!container) return;

    // Build the DOM structure first
    buildDOM(container);

    // Initialize demo buttons
    initDemoButtons();

    signalCanvas = document.getElementById('signalCanvas');
    poseCanvas   = document.getElementById('poseCanvas');
    signalCtx = signalCanvas ? signalCanvas.getContext('2d') : null;
    poseCtx   = poseCanvas   ? poseCanvas.getContext('2d')   : null;

    if (signalCanvas) {
      unobserveSig = observeResize(signalCanvas.parentElement || signalCanvas, (w, h) => {
        signalCanvas.width = w;
        signalCanvas.height = h;
      });
    }
    if (poseCanvas) {
      unobservePose = observeResize(poseCanvas.parentElement || poseCanvas, (w, h) => {
        poseCanvas.width = w;
        poseCanvas.height = h;
      });
    }

    const handler = (data) => onVitals(data);
    bus.on('vitals', handler);
    unsubVitals = () => bus.off('vitals', handler);

    console.log('Demo tab initialized (signal + pose canvases)');
  },

  activate() {
    const el = document.getElementById('tab-demo');
    if (el) el.style.display = 'block';
    startLoop();
  },

  deactivate() {
    const el = document.getElementById('tab-demo');
    if (el) el.style.display = 'none';
    stopLoop();
  },

  dispose() {
    stopLoop();
    if (unsubVitals) { unsubVitals(); unsubVitals = null; }
    if (unobserveSig) { unobserveSig(); unobserveSig = null; }
    if (unobservePose) { unobservePose(); unobservePose = null; }
    signalCanvas = poseCanvas = null;
    signalCtx = poseCtx = null;
  },
};
