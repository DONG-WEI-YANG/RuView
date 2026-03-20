// dashboard/src/tabs/viewer.js
/**
 * 3D Viewer tab — wires together Three.js scene modules:
 *   three-setup  (renderer, camera, lights, animation loop)
 *   skeleton     (24-joint dot+line overlay)
 *   body-mesh    (SMPL wireframe surface)
 *   room         (floor, grid, walls, node markers, receiver)
 *
 * Also manages render-mode cycling (mesh / skeleton / both)
 * and hosts the HUD overlay for vital signs on the 3D view.
 */
import { bus } from '../events.js';
import { initScene } from '../scene/three-setup.js';
import { createSkeleton } from '../scene/skeleton.js';
import { createBodyMesh } from '../scene/body-mesh.js';
import { createRoom } from '../scene/room.js';
import * as hud from '../vitals/hud.js';

let ctx = null;       // SceneContext from three-setup
let skeleton = null;
let bodyMesh = null;
let room = null;
let renderMode = 'mesh'; // "mesh" | "skeleton" | "both"

function applyRenderMode() {
  const showMesh = renderMode === 'mesh' || renderMode === 'both';
  const showSkeleton = renderMode === 'skeleton' || renderMode === 'both';

  if (bodyMesh) bodyMesh.group.visible = showMesh;
  if (skeleton) skeleton.group.visible = showSkeleton;

  const btn = document.getElementById('render-mode-btn');
  if (btn) {
    const labels = { mesh: 'Mesh', skeleton: 'Skeleton', both: 'Both' };
    btn.textContent = labels[renderMode] || renderMode;
  }
}

function cycleRenderMode() {
  const modes = ['mesh', 'skeleton', 'both'];
  const idx = modes.indexOf(renderMode);
  renderMode = modes[(idx + 1) % modes.length];
  applyRenderMode();
}

export default {
  id: 'viewer',
  label: '3D Viewer',

  init() {
    const container = document.getElementById('skeleton-canvas-container')
      || document.getElementById('tab-viewer');
    if (!container) {
      console.warn('Viewer tab: no container element found');
      return;
    }

    // Initialize Three.js scene
    ctx = initScene(container);

    // Add room geometry (floor, grid, walls, nodes)
    room = createRoom(ctx.scene);

    // Add SMPL body mesh (with demo idle animation)
    bodyMesh = createBodyMesh(ctx.scene);

    // Add skeleton overlay (joint dots + bone lines)
    skeleton = createSkeleton(ctx.scene);

    // Apply initial render mode
    applyRenderMode();

    // Render-mode button
    const modeBtn = document.getElementById('render-mode-btn');
    if (modeBtn) modeBtn.addEventListener('click', cycleRenderMode);

    // Keyboard shortcut: M key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'm' || e.key === 'M') cycleRenderMode();
    });

    // Initialize vital-signs HUD overlay (renders on top of 3D view)
    hud.init();

    console.log('Viewer tab initialized (ES6 scene modules + HUD)');
  },

  activate() {
    const el = document.getElementById('tab-viewer');
    if (el) el.style.display = 'block';
    // Trigger a resize so the renderer picks up the correct dimensions
    // after the tab becomes visible.
    if (ctx && ctx.renderer) {
      window.dispatchEvent(new Event('resize'));
    }
  },

  deactivate() {
    const el = document.getElementById('tab-viewer');
    if (el) el.style.display = 'none';
  },

  dispose() {
    hud.dispose();
    if (skeleton) { skeleton.dispose(); skeleton = null; }
    if (bodyMesh) { bodyMesh.dispose(); bodyMesh = null; }
    if (room) { room.dispose(); room = null; }
    if (ctx) { ctx.dispose(); ctx = null; }
  },
};
