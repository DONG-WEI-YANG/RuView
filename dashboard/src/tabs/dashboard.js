// dashboard/src/tabs/dashboard.js
/**
 * Dashboard tab — health metrics overview, breathing/heart waveforms,
 * and presence heatmap.
 *
 * Delegates rendering to:
 *   vitals/waveform.js  — breathing + heart rate waveform canvases
 *                          + extended health-metrics panel
 *   vitals/heatmap.js   — room-scale presence heat map
 */
import * as waveform from '../vitals/waveform.js';
import * as heatmap from '../vitals/heatmap.js';

let initialized = false;

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

/** Helper: create a progress-bar metric row. */
function makeMetricRow(label, barId, valId, valText) {
  const row = makeEl('div', { className: 'metric-row' });
  row.appendChild(makeEl('span', { className: 'metric-label', textContent: label }));
  const bar = makeEl('div', { className: 'progress-bar' });
  const fill = makeEl('div', { className: 'progress-fill' });
  fill.id = barId;
  fill.style.width = '0%';
  bar.appendChild(fill);
  row.appendChild(bar);
  const val = makeEl('span', { className: 'metric-val', textContent: valText || '0%' });
  val.id = valId;
  row.appendChild(val);
  return row;
}

/** Helper: create a stat box. */
function makeStatBox(value, label, valueId) {
  const box = makeEl('div', { className: 'stat-box' });
  const v = makeEl('span', { className: 'stat-value', textContent: value });
  if (valueId) v.id = valueId;
  box.appendChild(v);
  box.appendChild(makeEl('span', { className: 'stat-label', textContent: label }));
  return box;
}

/**
 * Build the dashboard tab DOM using safe DOM API methods.
 */
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);

  const scroll = makeEl('div', { className: 'tab-scroll' });

  // ── System Status ──────────────────────────────────────────
  const statusPanel = makeEl('div', { className: 'panel' });
  statusPanel.appendChild(makeEl('h2', { textContent: 'System Status' }));
  const statusGrid = makeEl('div', { className: 'status-grid' });
  ['api|API Server', 'hardware|Hardware', 'inference|Inference', 'streaming|Streaming'].forEach(item => {
    const [comp, name] = item.split('|');
    const row = makeEl('div', { className: 'component-status' });
    row.dataset.component = comp;
    row.appendChild(makeEl('span', { className: 'comp-name', textContent: name }));
    row.appendChild(makeEl('span', { className: 'comp-status status-unknown', textContent: '--' }));
    statusGrid.appendChild(row);
  });
  statusPanel.appendChild(statusGrid);
  scroll.appendChild(statusPanel);

  // ── System Metrics ─────────────────────────────────────────
  const metricsPanel = makeEl('div', { className: 'panel' });
  metricsPanel.appendChild(makeEl('h2', { textContent: 'System Metrics' }));
  const metricsGrid = makeEl('div', { className: 'metrics-grid' });
  metricsGrid.appendChild(makeMetricRow('CPU Usage', 'cpu-bar', 'cpu-val', '0%'));
  metricsGrid.appendChild(makeMetricRow('Memory Usage', 'mem-bar', 'mem-val', '0%'));
  metricsGrid.appendChild(makeMetricRow('Disk Usage', 'disk-bar', 'disk-val', '0%'));
  metricsPanel.appendChild(metricsGrid);
  scroll.appendChild(metricsPanel);

  // ── Live Statistics ────────────────────────────────────────
  const livePanel = makeEl('div', { className: 'panel' });
  livePanel.appendChild(makeEl('h2', { textContent: 'Live Statistics' }));
  const statsGrid = makeEl('div', { className: 'stats-grid' });
  statsGrid.appendChild(makeStatBox('0', 'Active Persons', 'stat-persons'));
  statsGrid.appendChild(makeStatBox('0%', 'Avg Confidence', 'stat-confidence'));
  statsGrid.appendChild(makeStatBox('0', 'Total Detections', 'stat-detections'));
  livePanel.appendChild(statsGrid);
  scroll.appendChild(livePanel);

  // ── Presence Heatmap ───────────────────────────────────────
  const heatmapPanel = makeEl('div', { className: 'panel' });
  heatmapPanel.appendChild(makeEl('h2', { textContent: 'Presence Heatmap' }));
  const heatCanvas = document.createElement('canvas');
  heatCanvas.id = 'presence-heatmap';
  heatCanvas.width = 480;
  heatCanvas.height = 260;
  heatmapPanel.appendChild(heatCanvas);
  scroll.appendChild(heatmapPanel);

  // ── Zone Occupancy ─────────────────────────────────────────
  const zonePanel = makeEl('div', { className: 'panel' });
  zonePanel.appendChild(makeEl('h2', { textContent: 'Zone Occupancy' }));
  const zonesSummary = makeEl('div', { className: 'zones-grid' });
  zonesSummary.id = 'zones-summary';
  zonePanel.appendChild(zonesSummary);
  scroll.appendChild(zonePanel);

  // ── Health Metrics (Wi-Mesh Enhanced) ──────────────────────
  const healthPanel = makeEl('div', { className: 'panel' });
  healthPanel.appendChild(makeEl('h2', { textContent: 'Health Metrics' }));
  const healthGrid = makeEl('div', { className: 'health-grid' });

  // HRV card
  const hrvCard = makeEl('div', { className: 'health-card' });
  hrvCard.appendChild(makeEl('div', { className: 'health-icon hrv-icon', textContent: '~' }));
  const hrvInfo = makeEl('div', { className: 'health-info' });
  hrvInfo.appendChild(makeEl('span', { className: 'health-label', textContent: 'HRV (RMSSD)' }));
  hrvInfo.appendChild(makeEl('span', { className: 'health-value', id: 'hrv-rmssd-val', textContent: '-- ms' }));
  hrvCard.appendChild(hrvInfo);
  const hrvSub = makeEl('div', { className: 'health-sub' });
  hrvSub.appendChild(makeEl('span', { className: 'health-label', textContent: 'SDNN' }));
  hrvSub.appendChild(makeEl('span', { className: 'health-value-sm', id: 'hrv-sdnn-val', textContent: '-- ms' }));
  hrvCard.appendChild(hrvSub);
  healthGrid.appendChild(hrvCard);

  // Stress card
  const stressCard = makeEl('div', { className: 'health-card' });
  stressCard.appendChild(makeEl('div', { className: 'health-icon stress-icon', textContent: '\u2696' }));
  const stressInfo = makeEl('div', { className: 'health-info' });
  stressInfo.appendChild(makeEl('span', { className: 'health-label', textContent: 'STRESS INDEX' }));
  stressInfo.appendChild(makeEl('span', { className: 'health-value', id: 'stress-val', textContent: '--' }));
  stressCard.appendChild(stressInfo);
  const stressBarWrap = makeEl('div', { className: 'stress-bar-wrap' });
  stressBarWrap.appendChild(makeEl('div', { className: 'stress-bar', id: 'stress-bar' }));
  stressCard.appendChild(stressBarWrap);
  healthGrid.appendChild(stressCard);

  // Motion card
  const motionCard = makeEl('div', { className: 'health-card' });
  motionCard.appendChild(makeEl('div', { className: 'health-icon motion-icon', textContent: '\u25B6' }));
  const motionInfo = makeEl('div', { className: 'health-info' });
  motionInfo.appendChild(makeEl('span', { className: 'health-label', textContent: 'MOTION' }));
  motionInfo.appendChild(makeEl('span', { className: 'health-value', id: 'motion-val', textContent: '--' }));
  motionCard.appendChild(motionInfo);
  const motionSub = makeEl('div', { className: 'health-sub' });
  motionSub.appendChild(makeEl('span', { className: 'health-label', textContent: 'Movement' }));
  motionSub.appendChild(makeEl('span', { className: 'health-value-sm', id: 'movement-val', textContent: '--' }));
  motionCard.appendChild(motionSub);
  healthGrid.appendChild(motionCard);

  // Sleep card
  const sleepCard = makeEl('div', { className: 'health-card' });
  sleepCard.appendChild(makeEl('div', { className: 'health-icon sleep-icon', textContent: '\u263E' }));
  const sleepInfo = makeEl('div', { className: 'health-info' });
  sleepInfo.appendChild(makeEl('span', { className: 'health-label', textContent: 'SLEEP STAGE' }));
  sleepInfo.appendChild(makeEl('span', { className: 'health-value', id: 'sleep-val', textContent: '--' }));
  sleepCard.appendChild(sleepInfo);
  const sleepSub = makeEl('div', { className: 'health-sub' });
  sleepSub.appendChild(makeEl('span', { className: 'health-label', textContent: 'Breath Regularity' }));
  const miniBar = makeEl('div', { className: 'mini-bar' });
  miniBar.appendChild(makeEl('div', { className: 'mini-fill', id: 'breath-reg-bar' }));
  sleepSub.appendChild(miniBar);
  sleepCard.appendChild(sleepSub);
  healthGrid.appendChild(sleepCard);

  healthPanel.appendChild(healthGrid);
  scroll.appendChild(healthPanel);

  // ── System Specs ───────────────────────────────────────────
  const specsPanel = makeEl('div', { className: 'panel' });
  specsPanel.appendChild(makeEl('h2', { textContent: 'System Specs' }));
  const specsGrid = makeEl('div', { className: 'stats-grid' });
  specsGrid.appendChild(makeStatBox('24', 'Body Regions'));
  specsGrid.appendChild(makeStatBox('100Hz', 'Sampling Rate'));
  specsGrid.appendChild(makeStatBox('87.2%', 'Accuracy (AP@50)'));
  specsGrid.appendChild(makeStatBox('$30', 'Hardware Cost'));
  specsPanel.appendChild(specsGrid);
  scroll.appendChild(specsPanel);

  container.appendChild(scroll);
}

export default {
  id: 'dashboard',
  label: 'Dashboard',

  init() {
    const container = document.getElementById('tab-dashboard');
    if (!container) return;

    // Build the DOM structure first
    buildDOM(container);

    // Waveform renderer (breathing + heart canvases + health metrics DOM)
    waveform.init();

    // Presence heatmap (room-scale motion map)
    heatmap.init(container);

    initialized = true;
    console.log('Dashboard tab initialized (waveform + heatmap)');
  },

  activate() {
    const el = document.getElementById('tab-dashboard');
    if (el) el.style.display = 'block';
    // Trigger resize so canvases pick up new dimensions
    window.dispatchEvent(new Event('resize'));
  },

  deactivate() {
    const el = document.getElementById('tab-dashboard');
    if (el) el.style.display = 'none';
  },

  dispose() {
    if (initialized) {
      waveform.dispose();
      heatmap.dispose();
      initialized = false;
    }
  },
};
