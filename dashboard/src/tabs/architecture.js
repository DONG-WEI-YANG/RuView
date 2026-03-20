// dashboard/src/tabs/architecture.js
/**
 * Architecture tab — system diagram, pipeline status, technical honesty panel.
 */
export default {
  id: 'architecture',
  label: 'Architecture',

  init() {
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
