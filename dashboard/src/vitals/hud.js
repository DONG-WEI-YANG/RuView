// dashboard/src/vitals/hud.js
/**
 * Vital Signs HUD overlay — renders breathing/heart rate numbers,
 * confidence bars, HRV, and stress on the 3D viewer tab.
 *
 * Subscribes to EventBus 'vitals' events (single person) and
 * 'persons' events (multi-person compact vitals cards).
 * Data-source agnostic: works with both real and demo data.
 */
import { bus } from '../events.js';

let unsubVitals = null;
let unsubPersons = null;

// ── Cached DOM refs ───────────────────────────────────────────────
let elBreathVal, elBreathConfBar;
let elHeartVal, elHeartConfBar;
let elHudHrv, elHudStress, elHudStressBar;
let elMultiPersonPanel = null;

function cacheDom() {
  elBreathVal     = document.getElementById('breath-val');
  elBreathConfBar = document.getElementById('breath-conf-bar');
  elHeartVal      = document.getElementById('heart-val');
  elHeartConfBar  = document.getElementById('heart-conf-bar');
  elHudHrv        = document.getElementById('hud-hrv-val');
  elHudStress     = document.getElementById('hud-stress-val');
  elHudStressBar  = document.getElementById('hud-stress-bar');
}

// ── Normalize vitals keys (server snake_case → frontend camelCase) ──
function norm(data) {
  return {
    breathRate:    data.breathRate    ?? data.breathing_bpm ?? 0,
    breathConf:    data.breathConf    ?? data.breathing_confidence ?? 0,
    heartRate:     data.heartRate     ?? data.heart_bpm ?? 0,
    heartConf:     data.heartConf     ?? data.heart_confidence ?? 0,
    hrvRmssd:      data.hrvRmssd      ?? data.hrv_rmssd ?? 0,
    hrvSdnn:       data.hrvSdnn       ?? data.hrv_sdnn ?? 0,
    stressIndex:   data.stressIndex   ?? data.stress_index ?? 0,
    motionIntensity: data.motionIntensity ?? data.motion_intensity ?? 0,
    bodyMovement:  data.bodyMovement   ?? data.body_movement ?? 'still',
    breathRegularity: data.breathRegularity ?? data.breath_regularity ?? 0,
    sleepStage:    data.sleepStage     ?? data.sleep_stage ?? 'awake',
  };
}

// ── Single-person vitals rendering ───────────────────────────────
function onVitals(raw) {
  const data = norm(raw);

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

// ── Multi-person vitals panel ────────────────────────────────────
function ensureMultiPersonPanel() {
  if (elMultiPersonPanel) return elMultiPersonPanel;
  const hud = document.getElementById('vitals-hud');
  if (!hud) return null;

  elMultiPersonPanel = document.createElement('div');
  elMultiPersonPanel.id = 'multi-person-vitals';
  elMultiPersonPanel.style.cssText = [
    'display:none',
    'margin-top:8px',
    'padding:6px 8px',
    'background:rgba(0,0,0,0.6)',
    'border:1px solid rgba(255,255,255,0.1)',
    'border-radius:6px',
    'font-size:11px',
    'color:#ccc',
    'max-width:320px',
  ].join(';');
  hud.appendChild(elMultiPersonPanel);
  return elMultiPersonPanel;
}

function onPersons(data) {
  if (!data || !data.persons || data.count <= 1) {
    // Hide multi-person panel when single or no persons
    if (elMultiPersonPanel) elMultiPersonPanel.style.display = 'none';
    return;
  }

  const panel = ensureMultiPersonPanel();
  if (!panel) return;
  panel.style.display = 'block';

  // Clear existing cards
  while (panel.firstChild) panel.removeChild(panel.firstChild);

  // Header
  const header = document.createElement('div');
  header.style.cssText = 'font-size:10px;color:#888;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px';
  header.textContent = data.count + ' People Detected';
  panel.appendChild(header);

  // Per-person compact card
  for (const person of data.persons) {
    const card = document.createElement('div');
    card.style.cssText = [
      'display:flex',
      'align-items:center',
      'gap:8px',
      'padding:4px 0',
      'border-bottom:1px solid rgba(255,255,255,0.05)',
    ].join(';');

    // Color dot
    const dot = document.createElement('span');
    dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${person.color || '#00ff88'};flex-shrink:0`;
    card.appendChild(dot);

    // Person label
    const label = document.createElement('span');
    label.style.cssText = 'min-width:56px;color:#aaa;font-size:10px';
    label.textContent = 'Person ' + (person.id + 1);
    card.appendChild(label);

    // Heart BPM
    const vitals = person.vitals || {};
    const heartBpm = vitals.heart_bpm || vitals.heartRate || 0;
    const breathBpm = vitals.breathing_bpm || vitals.breathRate || 0;

    const heartEl = document.createElement('span');
    heartEl.style.cssText = 'color:#ff6b6b;min-width:52px;font-size:11px';
    heartEl.textContent = '\u2665 ' + (heartBpm > 0 ? heartBpm.toFixed(0) : '--');
    card.appendChild(heartEl);

    // Breathing BPM
    const breathEl = document.createElement('span');
    breathEl.style.cssText = 'color:#4ecdc4;min-width:52px;font-size:11px';
    breathEl.textContent = '~ ' + (breathBpm > 0 ? breathBpm.toFixed(1) : '--');
    card.appendChild(breathEl);

    panel.appendChild(card);
  }
}

// ── Public API ────────────────────────────────────────────────────
export function init() {
  cacheDom();
  const vitalsHandler = (data) => onVitals(data);
  const personsHandler = (data) => onPersons(data);
  bus.on('vitals', vitalsHandler);
  bus.on('persons', personsHandler);
  unsubVitals = () => bus.off('vitals', vitalsHandler);
  unsubPersons = () => bus.off('persons', personsHandler);
  console.log('VitalsHUD renderer initialized (multi-person support)');
}

export function dispose() {
  if (unsubVitals) { unsubVitals(); unsubVitals = null; }
  if (unsubPersons) { unsubPersons(); unsubPersons = null; }
  if (elMultiPersonPanel) {
    elMultiPersonPanel.remove();
    elMultiPersonPanel = null;
  }
}
