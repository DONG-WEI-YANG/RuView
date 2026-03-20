// dashboard/src/tabs/dashboard.js
/**
 * Dashboard tab — vital signs HUD and health metrics overview.
 * Full implementation will be migrated from vitals-hud.js in a later step.
 */
export default {
  id: 'dashboard',
  label: 'Dashboard',

  init() {
    console.log('Dashboard tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-dashboard');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-dashboard');
    if (el) el.style.display = 'none';
  },
};
