// dashboard/src/vitals/heatmap.js
/**
 * Presence heatmap renderer — room-scale spatial heat map showing
 * WiFi-sensed motion intensity.
 *
 * Subscribes to EventBus 'presence' events.
 *
 * Visual style: dark room background, transparent-blue-to-orange heat,
 * WiFi AP markers with signal rings.
 */
import { bus } from '../events.js';
import { observeResize } from '../utils/resize.js';

const GRID_W = 12;
const GRID_H = 8;

let canvas = null;
let ctx = null;
let unobserve = null;
let unsub = null;

// Latest grid snapshot
let presenceGrid = [];

// ── Color map ─────────────────────────────────────────────────────
function heatColor(v) {
  v = Math.max(0, Math.min(1, v));
  const a = Math.min(0.85, v * 1.2);
  let r, g, b;
  if (v < 0.3) {
    r = 30; g = 30; b = Math.round(100 + v * 500);
  } else if (v < 0.6) {
    const t2 = (v - 0.3) / 0.3;
    r = Math.round(100 + t2 * 155); g = 30; b = Math.round(250 - t2 * 100);
  } else {
    const t3 = (v - 0.6) / 0.4;
    r = 255; g = Math.round(30 + t3 * 200); b = Math.round(150 - t3 * 100);
  }
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a.toFixed(2) + ')';
}

// ── AP marker ─────────────────────────────────────────────────────
function drawAP(ctx, gx, gy, cw, ch, label) {
  const x = gx * cw;
  const y = gy * ch;
  ctx.fillStyle = '#00ff88';
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = '10px sans-serif';
  ctx.fillText(label, x + 7, y + 3);
  // Signal rings
  ctx.strokeStyle = 'rgba(0,255,136,0.15)';
  for (let r = 1; r <= 3; r++) {
    ctx.beginPath();
    ctx.arc(x, y, r * 18, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// ── Rendering ─────────────────────────────────────────────────────
function render() {
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Room background
  ctx.fillStyle = '#151530';
  ctx.fillRect(0, 0, w, h);

  const cellW = w / GRID_W;
  const cellH = h / GRID_H;

  // Heat cells
  for (let gy = 0; gy < GRID_H; gy++) {
    if (!presenceGrid[gy]) continue;
    for (let gx = 0; gx < GRID_W; gx++) {
      const v = presenceGrid[gy][gx];
      if (v > 0.02) {
        ctx.fillStyle = heatColor(v);
        ctx.fillRect(gx * cellW, gy * cellH, cellW, cellH);
      }
    }
  }

  // Room grid overlay
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= GRID_W; x++) {
    ctx.beginPath();
    ctx.moveTo(x * cellW, 0);
    ctx.lineTo(x * cellW, h);
    ctx.stroke();
  }
  for (let y = 0; y <= GRID_H; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * cellH);
    ctx.lineTo(w, y * cellH);
    ctx.stroke();
  }

  // WiFi AP indicators
  drawAP(ctx, 2, 0.5, cellW, cellH, 'AP1');
  drawAP(ctx, GRID_W - 3, GRID_H - 1.5, cellW, cellH, 'AP2');
}

// ── Event handler ─────────────────────────────────────────────────
function onPresence(data) {
  presenceGrid = data.grid || data;
  render();
}

// ── Public API ────────────────────────────────────────────────────
export function init(container) {
  canvas = (container && container.querySelector('#presence-heatmap'))
    || document.getElementById('presence-heatmap');
  if (!canvas) {
    console.warn('Heatmap: #presence-heatmap canvas not found');
    return;
  }
  ctx = canvas.getContext('2d');

  unobserve = observeResize(canvas.parentElement || canvas, (w, h) => {
    canvas.width = w;
    canvas.height = h;
    render();
  });

  const handler = (data) => onPresence(data);
  bus.on('presence', handler);
  unsub = () => bus.off('presence', handler);
  console.log('Heatmap renderer initialized');
}

export function dispose() {
  if (unsub) { unsub(); unsub = null; }
  if (unobserve) { unobserve(); unobserve = null; }
  presenceGrid = [];
  canvas = null;
  ctx = null;
}
