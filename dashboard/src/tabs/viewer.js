// dashboard/src/tabs/viewer.js
/**
 * 3D Viewer tab — Three.js scene with skeleton + body mesh.
 * Full implementation will be migrated from skeleton3d.js in a later step.
 */
export default {
  id: 'viewer',
  label: '3D Viewer',

  init() {
    // Will initialize Three.js scene
    console.log('Viewer tab initialized');
  },

  activate() {
    const el = document.getElementById('tab-viewer');
    if (el) el.style.display = 'block';
  },

  deactivate() {
    const el = document.getElementById('tab-viewer');
    if (el) el.style.display = 'none';
  },
};
