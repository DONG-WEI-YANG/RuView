// dashboard/src/vitals/waterfall.js
/**
 * CSI subcarrier waterfall spectrogram renderer.
 *
 * Subscribes to EventBus 'csi' events.  Each event carries one row
 * of subcarrier amplitudes; rows accumulate into a scrolling
 * spectrogram rendered on a <canvas> element.
 *
 * Color map: Blue -> Cyan -> Green -> Yellow -> Red
 */
import { bus } from '../events.js';
import { observeResize } from '../utils/resize.js';

const MAX_ROWS = 100;

let canvas = null;
let ctx = null;
let unobserve = null;
let unsub = null;

// History buffer: array of amplitude rows
const csiHistory = [];

// ── Color map ─────────────────────────────────────────────────────
function waterfallColor(v) {
  v = Math.max(0, Math.min(1, v));
  let r, g, b;
  if (v < 0.25) {
    r = 0; g = Math.round(v * 4 * 180); b = 180;
  } else if (v < 0.5) {
    r = 0; g = 180; b = Math.round((1 - (v - 0.25) * 4) * 180);
  } else if (v < 0.75) {
    r = Math.round((v - 0.5) * 4 * 255); g = 180; b = 0;
  } else {
    r = 255; g = Math.round((1 - (v - 0.75) * 4) * 180); b = 0;
  }
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

// ── Rendering ─────────────────────────────────────────────────────
function render() {
  if (!ctx || csiHistory.length < 2) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const rows = csiHistory.length;
  const numSub = csiHistory[0].length;
  const cellW = w / numSub;
  const cellH = h / Math.min(rows, MAX_ROWS);

  for (let r = 0; r < rows; r++) {
    const row = csiHistory[r];
    for (let i = 0; i < numSub; i++) {
      ctx.fillStyle = waterfallColor(row[i]);
      ctx.fillRect(i * cellW, (rows - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // Axis labels
  ctx.fillStyle = '#666';
  ctx.font = '9px sans-serif';
  ctx.fillText('Subcarrier \u2192', 4, h - 4);
  ctx.save();
  ctx.translate(w - 4, h - 4);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Time \u2192', 0, 0);
  ctx.restore();
}

// ── Event handler ─────────────────────────────────────────────────
function onCSI(data) {
  const amps = data.amplitudes || data;
  if (!Array.isArray(amps)) return;
  csiHistory.push(amps);
  if (csiHistory.length > MAX_ROWS) csiHistory.shift();
  render();
}

// ── Public API ────────────────────────────────────────────────────
export function init(container) {
  canvas = (container && container.querySelector('#csi-waterfall'))
    || document.getElementById('csi-waterfall');
  if (!canvas) {
    console.warn('Waterfall: #csi-waterfall canvas not found');
    return;
  }
  ctx = canvas.getContext('2d');

  unobserve = observeResize(canvas.parentElement || canvas, (w, h) => {
    canvas.width = w;
    canvas.height = h;
    render();
  });

  const handler = (data) => onCSI(data);
  bus.on('csi', handler);
  unsub = () => bus.off('csi', handler);
  console.log('Waterfall renderer initialized');
}

export function dispose() {
  if (unsub) { unsub(); unsub = null; }
  if (unobserve) { unobserve(); unobserve = null; }
  csiHistory.length = 0;
  canvas = null;
  ctx = null;
}
