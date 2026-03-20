// dashboard/src/simulation/demo-data.js
/**
 * Client-side demo data generator.
 *
 * When the backend WebSocket is offline this module produces synthetic
 * vitals, CSI, and presence data and emits them onto the EventBus using
 * the same event names as real data so rendering modules are agnostic.
 *
 * Events emitted:
 *   'vitals'   — { breathRate, heartRate, breathConf, heartConf,
 *                   breathWave[], heartWave[], hrvRmssd, hrvSdnn,
 *                   stressIndex, motionIntensity, bodyMovement,
 *                   sleepStage, breathRegularity }
 *   'csi'      — { amplitudes: number[] }
 *   'presence' — { grid: number[][] }
 */
import { bus } from '../events.js';

// ── Internal state ────────────────────────────────────────────────
let t = 0;
let animId = null;
let lastTime = 0;
let running = false;

// Waveform history
const breathWave = [];
const heartWave = [];

// CSI history (kept here so waterfall gets a stream of rows)
const NUM_SUBCARRIERS = 30;

// Presence grid
const GRID_W = 12;
const GRID_H = 8;
let presenceGrid = [];
for (let gy = 0; gy < GRID_H; gy++) {
  presenceGrid.push(new Array(GRID_W).fill(0));
}

// ── Simulation helpers ────────────────────────────────────────────
function simulateVitals(dt) {
  t += dt;

  // Breathing: 12-20 BPM with slow drift
  const breathRate = 15 + Math.sin(t * 0.05) * 3 + Math.sin(t * 0.17) * 1;
  const breathConf = 0.75 + Math.sin(t * 0.08) * 0.15;

  // Heart: 62-78 BPM with slight variability
  const heartRate = 70 + Math.sin(t * 0.07) * 6 + Math.sin(t * 0.23) * 2;
  const heartConf = 0.82 + Math.sin(t * 0.06) * 0.12;

  // Breathing waveform (~0.25 Hz sinusoid)
  const breathFreq = breathRate / 60;
  breathWave.push(
    Math.sin(t * breathFreq * Math.PI * 2) * 0.8 +
    Math.sin(t * breathFreq * Math.PI * 4) * 0.1,
  );
  if (breathWave.length > 120) breathWave.shift();

  // Heart waveform (sharp QRS-like peaks)
  const heartFreq = heartRate / 60;
  const hp = (t * heartFreq) % 1;
  const heartVal =
    hp < 0.08 ? Math.sin((hp / 0.08) * Math.PI) * 1.0 :
    hp < 0.15 ? -0.3 * Math.sin(((hp - 0.08) / 0.07) * Math.PI) :
    hp < 0.20 ? 0.15 * Math.sin(((hp - 0.15) / 0.05) * Math.PI) :
    Math.sin(((hp - 0.20) / 0.80) * Math.PI) * 0.05;
  heartWave.push(heartVal);
  if (heartWave.length > 120) heartWave.shift();

  // Extended HRV/stress metrics
  const hrvRmssd = 42 + Math.sin(t * 0.03) * 15 + Math.sin(t * 0.11) * 5;
  const hrvSdnn = hrvRmssd * 1.3 + Math.sin(t * 0.04) * 8;
  const stressIndex = Math.max(0, Math.min(100,
    50 - (hrvRmssd - 42) * 1.5 + Math.sin(t * 0.06) * 10));

  // Motion: periodic activity bursts
  const motionBase = 12 + Math.sin(t * 0.08) * 10;
  const burst = Math.max(0, Math.sin(t * 0.02) * 40);
  const motionIntensity = Math.max(0, Math.min(100, motionBase + burst));

  const bodyMovement =
    motionIntensity < 10 ? 'still' :
    motionIntensity < 40 ? 'micro' : 'gross';

  // Breathing regularity
  const breathRegularity = Math.max(0, Math.min(1,
    0.65 + Math.sin(t * 0.04) * 0.25));

  // Sleep stage (slow cycle for demo)
  const sleepPhase = (t * 0.01) % 4;
  let sleepStage;
  if (motionIntensity > 30) sleepStage = 'awake';
  else if (sleepPhase < 1) sleepStage = 'awake';
  else if (sleepPhase < 2) sleepStage = 'light';
  else if (sleepPhase < 3) sleepStage = 'deep';
  else sleepStage = 'rem';

  bus.emit('vitals', {
    breathRate, heartRate, breathConf, heartConf,
    breathWave: breathWave.slice(),
    heartWave: heartWave.slice(),
    hrvRmssd, hrvSdnn, stressIndex,
    motionIntensity, bodyMovement,
    sleepStage, breathRegularity,
  });
}

function simulateCSI() {
  const row = [];
  for (let i = 0; i < NUM_SUBCARRIERS; i++) {
    const base = 0.4 + 0.2 * Math.sin(i * 0.3);
    const motion = 0.3 * Math.sin(t * 0.5 + i * 0.15) *
                   Math.sin(t * 0.12 + i * 0.05);
    const breathRate = 15 + Math.sin(t * 0.05) * 3;
    const breathEffect = 0.08 * Math.sin(t * (breathRate / 60) * Math.PI * 2 + i * 0.2);
    const noise = (Math.random() - 0.5) * 0.1;
    row.push(Math.max(0, Math.min(1, base + motion + breathEffect + noise)));
  }
  bus.emit('csi', { amplitudes: row });
}

function simulatePresence() {
  const px = GRID_W / 2 + Math.sin(t * 0.15) * (GRID_W * 0.3);
  const py = GRID_H / 2 + Math.cos(t * 0.11) * (GRID_H * 0.25);

  for (let gy = 0; gy < GRID_H; gy++) {
    for (let gx = 0; gx < GRID_W; gx++) {
      const dx = gx - px;
      const dy = gy - py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const signal = Math.exp(-dist * dist / 3.0);
      presenceGrid[gy][gx] = presenceGrid[gy][gx] * 0.92 + signal * 0.08;
    }
  }
  bus.emit('presence', { grid: presenceGrid.map((row) => row.slice()) });
}

// ── Animation loop ────────────────────────────────────────────────
function tick(time) {
  if (!running) return;
  const dt = Math.min((time - lastTime) / 1000, 0.1);
  lastTime = time;

  simulateVitals(dt);
  simulateCSI();
  simulatePresence();

  animId = requestAnimationFrame(tick);
}

// ── Public API ────────────────────────────────────────────────────
export function init() {
  if (running) return;
  running = true;
  lastTime = performance.now();

  // Stop demo generation when the backend connects, restart on disconnect
  bus.on('ws:connected', stop);
  bus.on('ws:disconnected', start);

  start();
  console.log('DemoData generator started');
}

function start() {
  if (animId) return;
  running = true;
  lastTime = performance.now();
  animId = requestAnimationFrame(tick);
}

function stop() {
  running = false;
  if (animId) {
    cancelAnimationFrame(animId);
    animId = null;
  }
}

export function dispose() {
  stop();
  bus.off('ws:connected', stop);
  bus.off('ws:disconnected', start);
}
