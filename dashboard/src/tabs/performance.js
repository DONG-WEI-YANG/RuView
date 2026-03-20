// dashboard/src/tabs/performance.js
/**
 * Performance tab — benchmark results, latency metrics, system stats.
 */
export default {
  id: 'performance',
  label: 'Performance',

  init() {
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
