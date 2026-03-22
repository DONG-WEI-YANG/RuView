// dashboard/src/tabs/sensing.js
/**
 * Sensing tab — CSI waterfall spectrogram and signal analysis.
 *
 * Delegates rendering to:
 *   vitals/waterfall.js — CSI subcarrier waterfall canvas
 */
import * as waterfall from '../vitals/waterfall.js';

let initialized = false;

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

/** Helper: create a progress-bar meter row. */
function makeMeterRow(label, fillId) {
  const row = makeEl('div', { className: 'meter-row' });
  row.appendChild(makeEl('span', { textContent: label }));
  const bar = makeEl('div', { className: 'progress-bar' });
  const fill = makeEl('div', { className: 'progress-fill', id: fillId });
  fill.style.width = '0%';
  bar.appendChild(fill);
  row.appendChild(bar);
  return row;
}

/**
 * Build the sensing tab DOM using safe DOM API methods.
 */
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);

  const scroll = makeEl('div', { className: 'tab-scroll' });
  const grid = makeEl('div', { className: 'sensing-grid' });

  // ── CSI Waterfall ──────────────────────────────────────────
  const waterfallPanel = makeEl('div', { className: 'panel' });
  waterfallPanel.style.gridColumn = '1 / -1';
  waterfallPanel.appendChild(makeEl('h2', { textContent: 'CSI Subcarrier Waterfall' }));
  const wfCanvas = document.createElement('canvas');
  wfCanvas.id = 'csi-waterfall';
  wfCanvas.width = 600;
  wfCanvas.height = 240;
  waterfallPanel.appendChild(wfCanvas);
  grid.appendChild(waterfallPanel);

  // ── WiFi Signal Sensing ────────────────────────────────────
  const sensingPanel = makeEl('div', { className: 'panel' });
  sensingPanel.appendChild(makeEl('h2', { textContent: 'WiFi Signal Sensing' }));
  const sensingStatus = makeEl('div', { className: 'sensing-status' });
  sensingStatus.appendChild(makeEl('span', { className: 'status-dot offline', id: 'sensing-conn', textContent: 'Disconnected' }));
  const rssiSpan = document.createElement('span');
  rssiSpan.textContent = 'RSSI: ';
  rssiSpan.appendChild(makeEl('b', { id: 'sensing-rssi', textContent: '-- dBm' }));
  sensingStatus.appendChild(rssiSpan);
  sensingPanel.appendChild(sensingStatus);
  const rssiCanvas = document.createElement('canvas');
  rssiCanvas.id = 'rssiSparkline';
  rssiCanvas.width = 300;
  rssiCanvas.height = 60;
  sensingPanel.appendChild(rssiCanvas);
  grid.appendChild(sensingPanel);

  // ── Signal Features ────────────────────────────────────────
  const featPanel = makeEl('div', { className: 'panel' });
  featPanel.appendChild(makeEl('h2', { textContent: 'Signal Features' }));
  const featMeters = makeEl('div', { className: 'feature-meters' });
  featMeters.appendChild(makeMeterRow('Variance', 'feat-variance'));
  featMeters.appendChild(makeMeterRow('Motion Band', 'feat-motion'));
  featMeters.appendChild(makeMeterRow('Breathing Band', 'feat-breathing'));
  featMeters.appendChild(makeMeterRow('Spectral Power', 'feat-spectral'));
  featPanel.appendChild(featMeters);
  grid.appendChild(featPanel);

  // ── Classification ─────────────────────────────────────────
  const classPanel = makeEl('div', { className: 'panel' });
  classPanel.appendChild(makeEl('h2', { textContent: 'Classification' }));
  const classDisplay = makeEl('div', { className: 'classification-display' });
  classDisplay.appendChild(makeEl('div', { className: 'class-label', id: 'motion-level', textContent: 'ABSENT' }));
  const classConf = makeEl('div', { className: 'class-conf' });
  classConf.appendChild(makeEl('span', { textContent: 'Confidence:' }));
  const confBar = makeEl('div', { className: 'progress-bar' });
  const confFill = makeEl('div', { className: 'progress-fill', id: 'class-conf-bar' });
  confFill.style.width = '0%';
  confBar.appendChild(confFill);
  classConf.appendChild(confBar);
  classConf.appendChild(makeEl('span', { id: 'class-conf-val', textContent: '0%' }));
  classDisplay.appendChild(classConf);
  classPanel.appendChild(classDisplay);

  const detailsGrid = makeEl('div', { className: 'details-grid' });
  [['Dominant Freq', 'dom-freq', '0 Hz'], ['Change Points', 'change-pts', '0'], ['Sample Rate', 'sample-rate', '--']].forEach(([label, id, val]) => {
    const item = makeEl('div', { className: 'detail-item' });
    item.appendChild(makeEl('span', { className: 'detail-label', textContent: label }));
    item.appendChild(makeEl('span', { className: 'detail-value', id: id, textContent: val }));
    detailsGrid.appendChild(item);
  });
  classPanel.appendChild(detailsGrid);
  grid.appendChild(classPanel);

  scroll.appendChild(grid);
  container.appendChild(scroll);
}

export default {
  id: 'sensing',
  label: 'Sensing',

  init() {
    const container = document.getElementById('tab-sensing');
    if (!container) return;

    // Build the DOM structure first
    buildDOM(container);

    // CSI waterfall spectrogram
    waterfall.init(container);

    initialized = true;
    console.log('Sensing tab initialized (waterfall)');
  },

  activate() {
    window.dispatchEvent(new Event('resize'));
  },

  deactivate() {},

  dispose() {
    if (initialized) {
      waterfall.dispose();
      initialized = false;
    }
  },
};
