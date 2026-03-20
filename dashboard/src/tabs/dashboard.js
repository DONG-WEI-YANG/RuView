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

export default {
  id: 'dashboard',
  label: 'Dashboard',

  init() {
    const container = document.getElementById('tab-dashboard');

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
