// dashboard/src/tabs/hardware.js
/**
 * Hardware tab — ESP32 device management, AP placement guide, flash tool.
 */
export default {
  id: 'hardware',
  label: 'Hardware',

  init() {
    console.log('Hardware tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-hardware');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-hardware');
    if (el) el.style.display = 'none';
  },
};
