// dashboard/src/tabs/architecture.js
/**
 * Architecture tab — system diagram, pipeline status, technical honesty panel.
 */

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

/**
 * Build a step-card element for the flow diagrams.
 */
function makeStepCard(num, title, desc) {
  const card = makeEl('div', { className: 'step-card' });
  card.appendChild(makeEl('div', { className: 'step-num', textContent: num }));
  card.appendChild(makeEl('h3', { textContent: title }));
  card.appendChild(makeEl('p', { textContent: desc }));
  return card;
}

/**
 * Build a benefit card.
 */
function makeBenefitCard(title, desc) {
  const card = makeEl('div', { className: 'benefit-card' });
  card.appendChild(makeEl('h3', { textContent: title }));
  card.appendChild(makeEl('p', { textContent: desc }));
  return card;
}

/**
 * Build the architecture tab DOM using safe DOM API methods.
 */
function buildDOM(container) {
  while (container.firstChild) container.removeChild(container.firstChild);

  const scroll = makeEl('div', { className: 'tab-scroll' });

  // ── Pipeline Transparency ──────────────────────────────────
  const pipePanel = makeEl('div', { className: 'panel transparency-panel' });
  pipePanel.appendChild(makeEl('h2', { textContent: 'Pipeline Status' }));
  pipePanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Real-time transparency: what\'s real hardware vs simulation' }));
  const pipeGrid = makeEl('div', { className: 'status-grid' });

  const pipeComponents = [
    ['CSI Receiver', 'pipe-csi', 'UDP :5005', 'status-healthy', 'Real async UDP listener for ESP32 binary CSI frames'],
    ['Signal Processing', 'pipe-signal', 'SOS Bandpass', 'status-healthy', 'Real SOS butterworth filter + Z-score normalization'],
    ['Pose Model', 'pipe-model', 'No Weights', 'status-unknown', 'Conv1D+Attention architecture. Train with: python -m server.train'],
    ['Vital Signs', 'pipe-vitals', 'Active', 'status-healthy', 'Real FFT-based extraction: breathing, heart, HRV, stress, motion'],
  ];
  pipeComponents.forEach(([name, id, statusText, statusClass, helpText]) => {
    const row = makeEl('div', { className: 'component-status' });
    row.appendChild(makeEl('span', { className: 'comp-name', textContent: name }));
    row.appendChild(makeEl('span', { className: 'comp-status ' + statusClass, id: id, textContent: statusText }));
    row.appendChild(makeEl('span', { className: 'help-text', textContent: helpText }));
    pipeGrid.appendChild(row);
  });
  pipePanel.appendChild(pipeGrid);

  const transpNote = makeEl('div', { className: 'transparency-note' });
  const strong = document.createElement('strong');
  strong.textContent = 'Honest disclosure:';
  transpNote.appendChild(strong);
  transpNote.appendChild(document.createTextNode(' Dashboard runs in DEMO mode with simulated vitals when backend is offline. When ESP32 hardware is connected, CSI data feeds real signal processing and vital signs extraction. Pose inference requires trained model weights (not included \u2014 train with your own data).'));
  pipePanel.appendChild(transpNote);
  scroll.appendChild(pipePanel);

  // ── System Architecture ────────────────────────────────────
  const archPanel = makeEl('div', { className: 'panel' });
  archPanel.appendChild(makeEl('h2', { textContent: 'System Architecture' }));
  const archSteps = makeEl('div', { className: 'flow-steps' });
  archSteps.appendChild(makeStepCard('1', 'CSI Capture', 'ESP32 antenna array (3TX x 6RX) captures Channel State Information at 20-100Hz via UDP'));
  archSteps.appendChild(makeStepCard('2', 'Phase Sanitization', 'Remove hardware phase offset, carrier frequency offset, and sampling frequency offset (linear fit removal)'));
  archSteps.appendChild(makeStepCard('3', 'Modality Translation', '1D Conv encoder maps sanitized CSI amplitude/phase to pose-correlated feature space'));
  archSteps.appendChild(makeStepCard('4', 'Temporal Attention', 'Attention mechanism weights important time steps, pooling 60-frame windows into pose estimates'));
  archSteps.appendChild(makeStepCard('5', 'Joint Decoder', 'FC decoder outputs 24 joints x 3 coordinates (xyz). Parallel vital signs extraction via FFT'));
  archPanel.appendChild(archSteps);
  scroll.appendChild(archPanel);

  // ── Vital Signs Pipeline ───────────────────────────────────
  const vitalsPanel = makeEl('div', { className: 'panel' });
  vitalsPanel.appendChild(makeEl('h2', { textContent: 'Vital Signs Extraction Pipeline' }));
  const vitalsSteps = makeEl('div', { className: 'flow-steps' });
  vitalsSteps.appendChild(makeStepCard('A', 'CSI Buffer', '30-second rolling buffer of subcarrier amplitudes (mean across subcarriers)'));
  vitalsSteps.appendChild(makeStepCard('B', 'Bandpass Filter', 'SOS Butterworth: 0.1-0.5Hz (breathing), 0.8-2.0Hz (heart), 1-8Hz (motion)'));
  vitalsSteps.appendChild(makeStepCard('C', 'FFT Analysis', 'Peak detection in frequency domain for BPM estimation + SNR confidence'));
  vitalsSteps.appendChild(makeStepCard('D', 'HRV/Stress', 'Inter-beat interval analysis: RMSSD, SDNN from heart signal peaks'));
  vitalsSteps.appendChild(makeStepCard('E', 'Sleep/Motion', 'Breath regularity + motion intensity classify sleep stage and activity level'));
  vitalsPanel.appendChild(vitalsSteps);
  scroll.appendChild(vitalsPanel);

  // ── Advantages ─────────────────────────────────────────────
  const advPanel = makeEl('div', { className: 'panel' });
  advPanel.appendChild(makeEl('h2', { textContent: 'Advantages' }));
  const advGrid = makeEl('div', { className: 'benefits-grid' });
  advGrid.appendChild(makeBenefitCard('Through Walls', 'WiFi CSI penetrates solid barriers \u2014 no line of sight required (Wi-Mesh: 2.4cm joint error)'));
  advGrid.appendChild(makeBenefitCard('Privacy-Preserving', 'No cameras or visual recording \u2014 only RF signal amplitude/phase analysis'));
  advGrid.appendChild(makeBenefitCard('Real-Time', 'Maps 24 body regions in real-time. Vital signs extracted from same CSI stream'));
  advGrid.appendChild(makeBenefitCard('Low Cost', 'ESP32 ($5) + antenna array. Total hardware cost under $30'));
  advPanel.appendChild(advGrid);
  scroll.appendChild(advPanel);

  // ── Technical Honesty ──────────────────────────────────────
  const honPanel = makeEl('div', { className: 'panel' });
  honPanel.appendChild(makeEl('h2', { textContent: 'Technical Honesty' }));
  honPanel.appendChild(makeEl('p', { className: 'help-text', textContent: 'Addressing common WiFi DensePose criticisms (UDN/GitHub community feedback)' }));
  const honGrid = makeEl('div', { className: 'benefits-grid' });
  honGrid.appendChild(makeBenefitCard('Real CSI Code', 'Binary frame parser (magic 0xC5110001), I/Q demodulation, async UDP receiver \u2014 not stubs or mock hardware'));
  honGrid.appendChild(makeBenefitCard('No Hidden Mocks', 'DEMO badge visible when simulating. LIVE mode activates only with real backend WebSocket connection'));
  honGrid.appendChild(makeBenefitCard('Model Weights Required', 'Architecture is real PyTorch (Conv1D+Attn). Weights not included \u2014 must train on your CSI data'));
  honGrid.appendChild(makeBenefitCard('Vitals Work Independently', 'Breathing, heart rate, HRV extraction from real CSI does NOT require pose model. Works with raw CSI input'));
  honPanel.appendChild(honGrid);
  scroll.appendChild(honPanel);

  container.appendChild(scroll);
}

export default {
  id: 'architecture',
  label: 'Architecture',

  init() {
    const el = document.getElementById('tab-architecture');
    if (!el) return;

    // Build the DOM structure
    buildDOM(el);

    console.log('Architecture tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-architecture');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-architecture');
    if (el) el.style.display = 'none';
  },
};
