// dashboard/src/tabs/demo.js
/**
 * Live Demo tab — simulated or live CSI data walkthrough.
 */
export default {
  id: 'demo',
  label: 'Live Demo',

  init() {
    console.log('Demo tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-demo');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-demo');
    if (el) el.style.display = 'none';
  },
};
