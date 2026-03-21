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

  // ── Setup Wizard Overlay (guides user through hardware setup) ──
  const waitOverlay = document.createElement('div');
  waitOverlay.id = 'waiting-overlay';
  waitOverlay.style.cssText = 'position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:10;pointer-events:auto;background:rgba(0,0,0,0.85)';

  const wizard = makeEl('div', { id: 'setup-wizard' });
  wizard.style.cssText = 'max-width:420px;text-align:center;padding:24px';

  // Step indicator
  const stepBar = makeEl('div', { id: 'wizard-steps' });
  stepBar.style.cssText = 'display:flex;justify-content:center;gap:8px;margin-bottom:20px';
  const stepLabels = ['Connect', 'Place', 'Calibrate', 'Go'];
  stepLabels.forEach((label, i) => {
    const dot = makeEl('div', { textContent: label, className: 'wizard-step' + (i === 0 ? ' active' : '') });
    dot.dataset.step = i;
    dot.style.cssText = 'padding:4px 12px;border:1px solid #333;font-size:10px;color:#666;border-radius:2px';
    stepBar.appendChild(dot);
  });
  wizard.appendChild(stepBar);

  // Step content area
  const stepContent = makeEl('div', { id: 'wizard-content' });
  wizard.appendChild(stepContent);

  // -- Step 0: Connect --
  const s0 = makeEl('div', { className: 'wizard-page', id: 'wizard-page-0' });
  s0.appendChild(makeEl('div', { textContent: '📡', style: 'font-size:48px;opacity:0.5;margin-bottom:12px' }));
  s0.appendChild(makeEl('div', { textContent: 'Connect ESP32 Nodes', style: 'font-size:16px;color:#ccc;margin-bottom:8px;font-weight:bold' }));
  s0.appendChild(makeEl('div', { textContent: 'Power on your ESP32-S3 boards. They will auto-connect to WiFi and stream CSI data.', style: 'font-size:12px;color:#888;line-height:1.6;margin-bottom:12px' }));
  const nodeCount = makeEl('div', { id: 'wizard-node-count', textContent: 'Detected: 0 nodes', style: 'font-size:14px;color:var(--accent-green,#0f0);margin-bottom:12px' });
  s0.appendChild(nodeCount);
  const skipBtn = makeEl('button', { textContent: 'Skip (use simulation)', id: 'wizard-skip' });
  skipBtn.style.cssText = 'padding:6px 16px;background:transparent;border:1px solid #555;color:#888;cursor:pointer;font-size:11px;margin-top:8px';
  s0.appendChild(skipBtn);
  stepContent.appendChild(s0);

  // -- Step 1: Place --
  const s1 = makeEl('div', { className: 'wizard-page', id: 'wizard-page-1', style: 'display:none' });
  s1.appendChild(makeEl('div', { textContent: '📐', style: 'font-size:48px;opacity:0.5;margin-bottom:12px' }));
  s1.appendChild(makeEl('div', { textContent: 'Place Your Nodes', style: 'font-size:16px;color:#ccc;margin-bottom:8px;font-weight:bold' }));
  const placeAdvice = makeEl('div', { id: 'wizard-place-advice', style: 'font-size:12px;color:#888;line-height:1.8;text-align:left;margin-bottom:16px' });
  s1.appendChild(placeAdvice);
  const nextBtn1 = makeEl('button', { textContent: 'Nodes are placed →', id: 'wizard-next-1' });
  nextBtn1.style.cssText = 'padding:8px 20px;background:var(--accent-green,#0f0);border:none;color:#000;cursor:pointer;font-weight:bold;font-size:12px';
  s1.appendChild(nextBtn1);
  stepContent.appendChild(s1);

  // -- Step 2: Calibrate --
  const s2 = makeEl('div', { className: 'wizard-page', id: 'wizard-page-2', style: 'display:none' });
  s2.appendChild(makeEl('div', { textContent: '⚖', style: 'font-size:48px;opacity:0.5;margin-bottom:12px' }));
  s2.appendChild(makeEl('div', { textContent: 'Background Calibration', style: 'font-size:16px;color:#ccc;margin-bottom:8px;font-weight:bold' }));
  s2.appendChild(makeEl('div', { textContent: 'Leave the room empty for 5 seconds. This captures the static environment so we can subtract it from the signal.', style: 'font-size:12px;color:#888;line-height:1.6;margin-bottom:16px' }));
  const calBtn = makeEl('button', { textContent: 'Start Calibration', id: 'wizard-cal-btn' });
  calBtn.style.cssText = 'padding:8px 20px;background:var(--accent-green,#0f0);border:none;color:#000;cursor:pointer;font-weight:bold;font-size:12px';
  s2.appendChild(calBtn);
  const calStatus = makeEl('div', { id: 'wizard-cal-status', style: 'font-size:11px;color:#888;margin-top:8px' });
  s2.appendChild(calStatus);
  stepContent.appendChild(s2);

  // -- Step 3: Go --
  const s3 = makeEl('div', { className: 'wizard-page', id: 'wizard-page-3', style: 'display:none' });
  s3.appendChild(makeEl('div', { textContent: '✓', style: 'font-size:48px;color:var(--accent-green,#0f0);margin-bottom:12px' }));
  s3.appendChild(makeEl('div', { textContent: 'Ready!', style: 'font-size:16px;color:#ccc;margin-bottom:8px;font-weight:bold' }));
  const goSummary = makeEl('div', { id: 'wizard-summary', style: 'font-size:12px;color:#888;line-height:1.6;margin-bottom:16px' });
  s3.appendChild(goSummary);
  const goBtn = makeEl('button', { textContent: 'Start Monitoring', id: 'wizard-go' });
  goBtn.style.cssText = 'padding:10px 28px;background:var(--accent-green,#0f0);border:none;color:#000;cursor:pointer;font-weight:bold;font-size:14px';
  s3.appendChild(goBtn);
  stepContent.appendChild(s3);

  waitOverlay.appendChild(wizard);
  section.appendChild(waitOverlay);

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

    // ── Setup Wizard Logic ──────────────────────────────
    let wizardStep = 0;
    let detectedNodes = 0;
    let pollTimer = null;

    function showWizardStep(step) {
      wizardStep = step;
      document.querySelectorAll('.wizard-page').forEach(p => p.style.display = 'none');
      const page = document.getElementById('wizard-page-' + step);
      if (page) page.style.display = 'block';
      document.querySelectorAll('.wizard-step').forEach(d => {
        const s = parseInt(d.dataset.step);
        d.style.borderColor = s <= step ? 'var(--accent-green,#0f0)' : '#333';
        d.style.color = s <= step ? 'var(--accent-green,#0f0)' : '#666';
        d.style.background = s === step ? 'rgba(0,255,0,0.1)' : 'transparent';
      });
    }

    function updatePlacementAdvice(n) {
      const advice = document.getElementById('wizard-place-advice');
      if (!advice) return;
      const tips = {
        1: '• Place your single node 2-3m from the subject\n• Height: ~1.2m (chest level)\n• Best for: presence detection + vital signs',
        2: '• Place nodes on opposite sides of the room\n• Vary heights: one at 1.5m, one at 0.8m\n• The person should be between the two nodes\n• Best for: basic pose + vital signs',
        3: '• Triangle formation around the subject\n• Heights: 1.5m / 1.0m / 0.6m\n• 2-3m from centre, antennas facing inward\n• Best for: 3D pose estimation',
        4: '• Square formation at room corners\n• Heights: 1.5m / 1.0m / 0.5m / 1.0m\n• Provides full surround coverage\n• Best for: accurate pose + gait analysis',
      };
      const key = Math.min(n, 4);
      advice.textContent = tips[key] || tips[2];
      advice.style.whiteSpace = 'pre-line';
    }

    // Poll server for detected nodes
    function pollNodes() {
      fetch('/api/status').then(r => r.json()).then(data => {
        const ps = data.pipeline_status || {};
        detectedNodes = ps.detected_nodes || Object.keys(data.nodes || {}).length;
        const countEl = document.getElementById('wizard-node-count');
        if (countEl) {
          countEl.textContent = 'Detected: ' + detectedNodes + ' node' + (detectedNodes !== 1 ? 's' : '');
          countEl.style.color = detectedNodes > 0 ? 'var(--accent-green,#0f0)' : '#888';
        }
        // Auto-advance to step 1 when nodes detected
        if (detectedNodes > 0 && wizardStep === 0) {
          updatePlacementAdvice(detectedNodes);
          showWizardStep(1);
        }
        // Update summary
        const summary = document.getElementById('wizard-summary');
        if (summary) {
          const strategy = ps.strategy_description || '';
          summary.textContent = detectedNodes + ' nodes connected\nStrategy: ' + (ps.strategy || 'auto') + '\n' + strategy;
          summary.style.whiteSpace = 'pre-line';
        }
      }).catch(() => {});
    }
    pollTimer = setInterval(pollNodes, 2000);
    pollNodes();

    // Skip button → start simulation
    const skipEl = document.getElementById('wizard-skip');
    if (skipEl) skipEl.onclick = () => {
      fetch('/api/system/mode?mode=simulation', { method: 'POST' }).then(() => {
        const ov = document.getElementById('waiting-overlay');
        if (ov) ov.style.display = 'none';
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      }).catch(() => {});
    };

    // Step 1 → 2
    const next1El = document.getElementById('wizard-next-1');
    if (next1El) next1El.onclick = () => showWizardStep(2);

    // Step 2: calibration
    const calBtnEl = document.getElementById('wizard-cal-btn');
    if (calBtnEl) calBtnEl.onclick = () => {
      const calSt = document.getElementById('wizard-cal-status');
      calBtnEl.disabled = true;
      calBtnEl.textContent = 'Calibrating...';
      if (calSt) calSt.textContent = 'Leave the room empty...';
      fetch('/api/calibration/start?mode=background', { method: 'POST' })
        .then(r => r.json())
        .then(() => {
          // Wait for calibration to finish (5s)
          setTimeout(() => {
            fetch('/api/calibration/finish', { method: 'POST' })
              .then(r => r.json())
              .then(data => {
                if (calSt) calSt.textContent = 'Calibration complete!';
                calBtnEl.textContent = 'Done';
                showWizardStep(3);
              });
          }, 5500);
        })
        .catch(() => {
          if (calSt) calSt.textContent = 'Calibration failed — you can skip this step';
          calBtnEl.textContent = 'Retry';
          calBtnEl.disabled = false;
        });
    };

    // Step 3: Go!
    const goEl = document.getElementById('wizard-go');
    if (goEl) goEl.onclick = () => {
      const ov = document.getElementById('waiting-overlay');
      if (ov) ov.style.display = 'none';
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    };

    // Also hide overlay when pose data arrives (real or simulated)
    bus.on('pose', function hideWait() {
      const ov = document.getElementById('waiting-overlay');
      if (ov && ov.style.display !== 'none') ov.style.display = 'none';
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      bus.off('pose', hideWait);
    });

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
