// dashboard/src/tabs/performance.js
/**
 * Performance tab — benchmark results, latency metrics, system stats.
 */

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

/** Helper: create a metric row with label and value. */
function makeMetricRow(label, value, highlight) {
  const row = makeEl('div', { className: 'metric-row' });
  row.appendChild(makeEl('span', { className: 'metric-label', textContent: label }));
  row.appendChild(makeEl('span', { className: 'metric-val' + (highlight ? ' highlight' : ''), textContent: value }));
  return row;
}

/**
 * Build the performance tab DOM using safe DOM API methods.
 */
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);

  const scroll = makeEl('div', { className: 'tab-scroll' });

  // ── Comparison Grid ────────────────────────────────────────
  const perfGrid = makeEl('div', { className: 'perf-grid' });

  // WiFi-based
  const wifiPanel = makeEl('div', { className: 'panel' });
  wifiPanel.appendChild(makeEl('h2', { textContent: 'WiFi-based (Same Layout)' }));
  const wifiMetrics = makeEl('div', { className: 'metric-list' });
  wifiMetrics.appendChild(makeMetricRow('Average Precision:', '43.5%', false));
  wifiMetrics.appendChild(makeMetricRow('AP@50:', '87.2%', true));
  wifiMetrics.appendChild(makeMetricRow('AP@75:', '44.6%', false));
  wifiPanel.appendChild(wifiMetrics);
  perfGrid.appendChild(wifiPanel);

  // Image-based
  const imgPanel = makeEl('div', { className: 'panel' });
  imgPanel.appendChild(makeEl('h2', { textContent: 'Image-based (Reference)' }));
  const imgMetrics = makeEl('div', { className: 'metric-list' });
  imgMetrics.appendChild(makeMetricRow('Average Precision:', '84.7%', true));
  imgMetrics.appendChild(makeMetricRow('AP@50:', '94.4%', true));
  imgMetrics.appendChild(makeMetricRow('AP@75:', '77.1%', true));
  imgPanel.appendChild(imgMetrics);
  perfGrid.appendChild(imgPanel);

  scroll.appendChild(perfGrid);

  // ── Advantages & Limitations ───────────────────────────────
  const prosConsPanel = makeEl('div', { className: 'panel' });
  prosConsPanel.appendChild(makeEl('h2', { textContent: 'Advantages & Limitations' }));
  const prosCons = makeEl('div', { className: 'pros-cons' });

  // Pros
  const prosDiv = makeEl('div', { className: 'pros' });
  prosDiv.appendChild(makeEl('h3', { textContent: 'Advantages' }));
  const prosList = document.createElement('ul');
  ['Through-wall detection', 'Privacy preserving', 'Lighting independent', 'Low cost hardware', 'Uses existing WiFi'].forEach(text => {
    prosList.appendChild(makeEl('li', { textContent: text }));
  });
  prosDiv.appendChild(prosList);
  prosCons.appendChild(prosDiv);

  // Cons
  const consDiv = makeEl('div', { className: 'cons' });
  consDiv.appendChild(makeEl('h3', { textContent: 'Limitations' }));
  const consList = document.createElement('ul');
  ['Performance drops in different layouts', 'Requires WiFi-compatible devices', 'Training requires synchronized data'].forEach(text => {
    consList.appendChild(makeEl('li', { textContent: text }));
  });
  consDiv.appendChild(consList);
  prosCons.appendChild(consDiv);

  prosConsPanel.appendChild(prosCons);
  scroll.appendChild(prosConsPanel);

  container.appendChild(scroll);
}

export default {
  id: 'performance',
  label: 'Performance',

  init() {
    const el = document.getElementById('tab-performance');
    if (!el) return;

    // Build the DOM structure
    buildDOM(el);

    console.log('Performance tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-performance');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-performance');
    if (el) el.style.display = 'none';
  },
};
