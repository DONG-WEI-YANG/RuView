// dashboard/src/vitals/waveform.js
/**
 * Breathing and heartbeat waveform canvas rendering.
 *
 * Subscribes to EventBus 'vitals' events and draws two real-time
 * waveform traces (breathing green, heart red) plus the Dashboard
 * tab's extended health-metrics panel.
 *
 * Retro-cyberpunk style: dark background, neon glow strokes.
 */
import { bus } from '../events.js';
import { observeResize } from '../utils/resize.js';

let unsub = null;
let unobserveBreath = null;
let unobserveHeart = null;

// ── Canvas elements ───────────────────────────────────────────────
let breathCanvas, heartCanvas;
let breathCtx, heartCtx;

// ── Latest waveform data ──────────────────────────────────────────
let breathWave = [];
let heartWave = [];

// ── Latest extended metrics for health-metrics panel ──────────────
let latestVitals = null;

// ── Drawing ───────────────────────────────────────────────────────
function drawWaveform(ctx, w, h, data, color) {
  if (!ctx || data.length < 2) return;
  ctx.clearRect(0, 0, w, h);

  // Center grid line
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Waveform trace
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    const y = h / 2 - data[i] * h * 0.4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Glow pass
  ctx.globalAlpha = 0.3;
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.globalAlpha = 1.0;
}

// ── Health Metrics (Dashboard panel) ──────────────────────────────
function renderHealthMetrics(d) {
  let el;

  // HRV
  el = document.getElementById('hrv-rmssd-val');
  if (el) el.textContent = d.hrvRmssd.toFixed(0) + ' ms';
  el = document.getElementById('hrv-sdnn-val');
  if (el) el.textContent = d.hrvSdnn.toFixed(0) + ' ms';

  // Stress
  el = document.getElementById('stress-val');
  if (el) {
    el.textContent = d.stressIndex.toFixed(0) + '/100';
    el.style.color =
      d.stressIndex > 60 ? '#ff4466' :
      d.stressIndex > 35 ? '#ffaa00' : '#00cc88';
  }
  el = document.getElementById('stress-bar');
  if (el) el.style.width = d.stressIndex.toFixed(0) + '%';

  // Motion
  el = document.getElementById('motion-val');
  if (el) {
    el.textContent = d.motionIntensity.toFixed(0) + '/100';
    el.style.color =
      d.motionIntensity > 60 ? '#ff4466' :
      d.motionIntensity > 30 ? '#ffaa00' : '#00aaff';
  }
  el = document.getElementById('movement-val');
  if (el) {
    el.textContent = d.bodyMovement.charAt(0).toUpperCase() + d.bodyMovement.slice(1);
  }

  // Sleep
  el = document.getElementById('sleep-val');
  if (el) {
    const labels = { awake: 'Awake', light: 'Light', deep: 'Deep', rem: 'REM' };
    const colors = { awake: '#e0e0e0', light: '#66ccff', deep: '#4466ff', rem: '#aa66ff' };
    el.textContent = labels[d.sleepStage] || '--';
    el.style.color = colors[d.sleepStage] || '#e0e0e0';
  }

  // Breathing regularity
  el = document.getElementById('breath-reg-bar');
  if (el) el.style.width = (d.breathRegularity * 100).toFixed(0) + '%';
}

// ── Event handler ─────────────────────────────────────────────────
function onVitals(data) {
  breathWave = data.breathWave || breathWave;
  heartWave = data.heartWave || heartWave;
  latestVitals = data;

  // Draw canvases (only when contexts exist, i.e. tab visible)
  if (breathCtx && breathCanvas) {
    drawWaveform(breathCtx, breathCanvas.width, breathCanvas.height, breathWave, '#00cc88');
  }
  if (heartCtx && heartCanvas) {
    drawWaveform(heartCtx, heartCanvas.width, heartCanvas.height, heartWave, '#ff4466');
  }

  // Update health-metrics DOM elements
  renderHealthMetrics(data);
}

// ── Public API ────────────────────────────────────────────────────
export function init() {
  breathCanvas = document.getElementById('breath-wave');
  heartCanvas = document.getElementById('heart-wave');
  breathCtx = breathCanvas ? breathCanvas.getContext('2d') : null;
  heartCtx = heartCanvas ? heartCanvas.getContext('2d') : null;

  // Observe canvas resize so coordinates always match actual pixel size
  if (breathCanvas) {
    unobserveBreath = observeResize(breathCanvas.parentElement || breathCanvas, (w, h) => {
      breathCanvas.width = w;
      breathCanvas.height = h;
    });
  }
  if (heartCanvas) {
    unobserveHeart = observeResize(heartCanvas.parentElement || heartCanvas, (w, h) => {
      heartCanvas.width = w;
      heartCanvas.height = h;
    });
  }

  const handler = (data) => onVitals(data);
  bus.on('vitals', handler);
  unsub = () => bus.off('vitals', handler);
  console.log('Waveform renderer initialized');
}

export function dispose() {
  if (unsub) { unsub(); unsub = null; }
  if (unobserveBreath) { unobserveBreath(); unobserveBreath = null; }
  if (unobserveHeart) { unobserveHeart(); unobserveHeart = null; }
  breathCanvas = heartCanvas = null;
  breathCtx = heartCtx = null;
}
