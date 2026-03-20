// dashboard/src/tabs/sensing.js
/**
 * Sensing tab — CSI waterfall spectrogram and signal analysis.
 *
 * Delegates rendering to:
 *   vitals/waterfall.js — CSI subcarrier waterfall canvas
 */
import * as waterfall from '../vitals/waterfall.js';

let initialized = false;

export default {
  id: 'sensing',
  label: 'Sensing',

  init() {
    const container = document.getElementById('tab-sensing');

    // CSI waterfall spectrogram
    waterfall.init(container);

    initialized = true;
    console.log('Sensing tab initialized (waterfall)');
  },

  activate() {
    const el = document.getElementById('tab-sensing');
    if (el) el.style.display = 'block';
    // Trigger resize so canvas recalculates dimensions
    window.dispatchEvent(new Event('resize'));
  },

  deactivate() {
    const el = document.getElementById('tab-sensing');
    if (el) el.style.display = 'none';
  },

  dispose() {
    if (initialized) {
      waterfall.dispose();
      initialized = false;
    }
  },
};
