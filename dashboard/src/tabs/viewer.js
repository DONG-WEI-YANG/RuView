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

/**
 * Build the inner DOM for the viewer tab, including the canvas container,
 * HUD overlay, and the side info panel.
 * Uses DOM API methods only — no innerHTML — to avoid XSS surface.
 */
function buildDOM(el) {
  // Clear existing content
  while (el.firstChild) el.removeChild(el.firstChild);

  const main = document.createElement('main');

  // ── viewer-section ────────────────────────────────────────
  const section = document.createElement('section');
  section.id = 'viewer-section';

  const canvasContainer = document.createElement('div');
  canvasContainer.id = 'skeleton-canvas-container';
  section.appendChild(canvasContainer);

  // ── Vitals HUD overlay ────────────────────────────────────
  const hudDiv = document.createElement('div');
  hudDiv.id = 'vitals-hud';

  // Breath card
  const breathCard = document.createElement('div');
  breathCard.className = 'hud-card hud-breath';
  breathCard.appendChild(makeEl('div', { className: 'hud-icon', textContent: '~' }));
  const breathInfo = document.createElement('div');
  breathInfo.className = 'hud-info';
  breathInfo.appendChild(makeEl('span', { className: 'hud-label', textContent: 'BREATHING' }));
  breathInfo.appendChild(makeEl('span', { className: 'hud-value', id: 'breath-val', textContent: '-- BPM' }));
  const breathConf = makeEl('div', { className: 'hud-conf' });
  breathConf.appendChild(makeEl('div', { className: 'hud-conf-fill', id: 'breath-conf-bar' }));
  breathInfo.appendChild(breathConf);
  breathCard.appendChild(breathInfo);
  const breathWaveCanvas = document.createElement('canvas');
  breathWaveCanvas.id = 'breath-wave';
  breathWaveCanvas.width = 120;
  breathWaveCanvas.height = 36;
  breathCard.appendChild(breathWaveCanvas);
  hudDiv.appendChild(breathCard);

  // Heart card
  const heartCard = document.createElement('div');
  heartCard.className = 'hud-card hud-heart';
  heartCard.appendChild(makeEl('div', { className: 'hud-icon', textContent: '\u2665' }));
  const heartInfo = document.createElement('div');
  heartInfo.className = 'hud-info';
  heartInfo.appendChild(makeEl('span', { className: 'hud-label', textContent: 'HEART RATE' }));
  heartInfo.appendChild(makeEl('span', { className: 'hud-value', id: 'heart-val', textContent: '-- BPM' }));
  const heartConf = makeEl('div', { className: 'hud-conf' });
  heartConf.appendChild(makeEl('div', { className: 'hud-conf-fill heart', id: 'heart-conf-bar' }));
  heartInfo.appendChild(heartConf);
  heartCard.appendChild(heartInfo);
  const heartWaveCanvas = document.createElement('canvas');
  heartWaveCanvas.id = 'heart-wave';
  heartWaveCanvas.width = 120;
  heartWaveCanvas.height = 36;
  heartCard.appendChild(heartWaveCanvas);
  hudDiv.appendChild(heartCard);

  // HRV / Stress card
  const hrvCard = document.createElement('div');
  hrvCard.className = 'hud-card hud-hrv';
  hrvCard.appendChild(makeEl('div', { className: 'hud-icon', textContent: '~' }));
  const hrvInfo = document.createElement('div');
  hrvInfo.className = 'hud-info';
  hrvInfo.appendChild(makeEl('span', { className: 'hud-label', textContent: 'HRV' }));
  hrvInfo.appendChild(makeEl('span', { className: 'hud-value', id: 'hud-hrv-val', textContent: '-- ms' }));
  hrvCard.appendChild(hrvInfo);
  const stressInfo = document.createElement('div');
  stressInfo.className = 'hud-info';
  stressInfo.style.minWidth = '60px';
  stressInfo.appendChild(makeEl('span', { className: 'hud-label', textContent: 'STRESS' }));
  stressInfo.appendChild(makeEl('span', { className: 'hud-value', id: 'hud-stress-val', textContent: '--' }));
  const stressConf = makeEl('div', { className: 'hud-conf' });
  stressConf.appendChild(makeEl('div', { className: 'hud-conf-fill stress', id: 'hud-stress-bar' }));
  stressInfo.appendChild(stressConf);
  hrvCard.appendChild(stressInfo);
  hudDiv.appendChild(hrvCard);

  section.appendChild(hudDiv);
  main.appendChild(section);

  // ── Side info panel ───────────────────────────────────────
  const aside = document.createElement('aside');
  aside.id = 'info-panel';

  const activityCard = document.createElement('div');
  activityCard.id = 'activity-card';
  activityCard.className = 'card';
  activityCard.appendChild(makeEl('h3', { textContent: 'Activity' }));
  activityCard.appendChild(makeEl('p', { id: 'activity-type', textContent: '--' }));
  aside.appendChild(activityCard);

  const alertCard = document.createElement('div');
  alertCard.id = 'alert-card';
  alertCard.className = 'card';
  alertCard.appendChild(makeEl('h3', { textContent: 'Fall Alert' }));
  alertCard.appendChild(makeEl('p', { id: 'fall-status', textContent: 'Normal' }));
  aside.appendChild(alertCard);

  const nodesCard = document.createElement('div');
  nodesCard.id = 'nodes-card';
  nodesCard.className = 'card';
  nodesCard.appendChild(makeEl('h3', { textContent: 'Nodes' }));
  const nodeList = document.createElement('ul');
  nodeList.id = 'node-list';
  nodesCard.appendChild(nodeList);
  aside.appendChild(nodesCard);

  // ── Scene mode toggle ────────────────────────────────────
  const sceneCard = document.createElement('div');
  sceneCard.id = 'scene-card';
  sceneCard.className = 'card';
  sceneCard.appendChild(makeEl('h3', { textContent: 'Scene Mode' }));

  const toggleWrap = document.createElement('div');
  toggleWrap.style.cssText = 'display:flex;gap:4px;margin-top:6px';

  const btnSafety = document.createElement('button');
  btnSafety.id = 'scene-btn-safety';
  btnSafety.textContent = '🛡 Safety';
  btnSafety.className = 'scene-toggle active';
  btnSafety.onclick = () => switchScene('safety');

  const btnFitness = document.createElement('button');
  btnFitness.id = 'scene-btn-fitness';
  btnFitness.textContent = '🏃 Fitness';
  btnFitness.className = 'scene-toggle';
  btnFitness.onclick = () => switchScene('fitness');

  toggleWrap.appendChild(btnSafety);
  toggleWrap.appendChild(btnFitness);
  sceneCard.appendChild(toggleWrap);

  const sceneDesc = makeEl('p', { id: 'scene-desc', textContent: 'Fall detection, apnea monitoring' });
  sceneDesc.style.cssText = 'font-size:10px;color:var(--text-secondary,#888);margin-top:4px';
  sceneCard.appendChild(sceneDesc);
  aside.appendChild(sceneCard);

  main.appendChild(aside);
  el.appendChild(main);
}

/** Switch scene mode via API and update toggle UI. */
function switchScene(mode) {
  fetch(`/api/system/scene?scene=${mode}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      const btnS = document.getElementById('scene-btn-safety');
      const btnF = document.getElementById('scene-btn-fitness');
      const desc = document.getElementById('scene-desc');
      if (btnS) btnS.className = 'scene-toggle' + (mode === 'safety' ? ' active' : '');
      if (btnF) btnF.className = 'scene-toggle' + (mode === 'fitness' ? ' active' : '');
      if (desc) desc.textContent = data.description || '';
    })
    .catch(() => {});
}

/** Helper: create an element with optional property overrides. */
function makeEl(tag, props) {
  const el = document.createElement(tag);
  if (props) Object.assign(el, props);
  return el;
}

export default {
  id: 'viewer',
  label: '3D Viewer',

  init() {
    const el = document.getElementById('tab-viewer');
    if (!el) {
      console.warn('Viewer tab: no #tab-viewer element found');
      return;
    }

    // Build DOM content first (must exist before scene/hud modules query it)
    buildDOM(el);

    // Now find the container for Three.js
    const container = document.getElementById('skeleton-canvas-container')
      || document.getElementById('viewer-section');
    if (!container) {
      console.warn('Viewer tab: no container element found after buildDOM');
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
