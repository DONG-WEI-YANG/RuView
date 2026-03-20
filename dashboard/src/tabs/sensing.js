// dashboard/src/tabs/sensing.js
/**
 * Sensing tab — CSI waterfall, presence heatmap, waveform displays.
 */
export default {
  id: 'sensing',
  label: 'Sensing',

  init() {
    console.log('Sensing tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-sensing');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-sensing');
    if (el) el.style.display = 'none';
  },
};
