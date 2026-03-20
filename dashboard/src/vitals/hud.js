// dashboard/src/vitals/hud.js
/**
 * Vital Signs HUD overlay — renders breathing/heart rate numbers,
 * confidence bars, HRV, and stress on the 3D viewer tab.
 *
 * Subscribes to EventBus 'vitals' events.
 * Data-source agnostic: works with both real and demo data.
 */
import { bus } from '../events.js';

let unsub = null;

// ── Cached DOM refs ───────────────────────────────────────────────
let elBreathVal, elBreathConfBar;
let elHeartVal, elHeartConfBar;
let elHudHrv, elHudStress, elHudStressBar;

function cacheDom() {
  elBreathVal     = document.getElementById('breath-val');
  elBreathConfBar = document.getElementById('breath-conf-bar');
  elHeartVal      = document.getElementById('heart-val');
  elHeartConfBar  = document.getElementById('heart-conf-bar');
  elHudHrv        = document.getElementById('hud-hrv-val');
  elHudStress     = document.getElementById('hud-stress-val');
  elHudStressBar  = document.getElementById('hud-stress-bar');
}

// ── Rendering ─────────────────────────────────────────────────────
function onVitals(data) {
  // Breathing
  if (elBreathVal) {
    elBreathVal.textContent = data.breathRate.toFixed(1) + ' BPM';
  }
  if (elBreathConfBar) {
    elBreathConfBar.style.width = (data.breathConf * 100).toFixed(0) + '%';
  }

  // Heart
  if (elHeartVal) {
    elHeartVal.textContent = data.heartRate.toFixed(0) + ' BPM';
  }
  if (elHeartConfBar) {
    elHeartConfBar.style.width = (data.heartConf * 100).toFixed(0) + '%';
  }

  // HRV
  if (elHudHrv) {
    elHudHrv.textContent = data.hrvRmssd.toFixed(0) + ' ms';
  }

  // Stress
  if (elHudStress) {
    elHudStress.textContent = data.stressIndex.toFixed(0);
    elHudStress.style.color =
      data.stressIndex > 60 ? '#ff4466' :
      data.stressIndex > 35 ? '#ffaa00' : '#00cc88';
  }
  if (elHudStressBar) {
    elHudStressBar.style.width = data.stressIndex.toFixed(0) + '%';
  }
}

// ── Public API ────────────────────────────────────────────────────
export function init() {
  cacheDom();
  const handler = (data) => onVitals(data);
  bus.on('vitals', handler);
  unsub = () => bus.off('vitals', handler);
  console.log('VitalsHUD renderer initialized');
}

export function dispose() {
  if (unsub) { unsub(); unsub = null; }
}
