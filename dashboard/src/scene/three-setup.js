// dashboard/src/scene/three-setup.js
/**
 * Three.js scene initialization — camera, renderer, OrbitControls, lighting,
 * and animation loop.  Uses observeResize() instead of per-frame resize hack.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { observeResize } from '../utils/resize.js';

/** @typedef {{ scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer, controls: OrbitControls }} SceneContext */

/**
 * Initialize the Three.js scene inside the given container element.
 * @param {HTMLElement} container — parent element (must already be in the DOM)
 * @returns {SceneContext}
 */
export function initScene(container) {
  // ── Scene ──────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  // ── Camera ─────────────────────────────────────────────────
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 600;
  const camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 100);
  camera.position.set(1.5, 1.0, 4.0);
  camera.lookAt(0, 0.80, 0);

  // ── Renderer ───────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // ── OrbitControls ──────────────────────────────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.85, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.0;
  controls.maxDistance = 8.0;
  controls.update();

  // ── Lighting ───────────────────────────────────────────────
  // Ambient fill
  scene.add(new THREE.AmbientLight(0x889abb, 0.7));

  // Key light (warm directional from upper-right-front)
  const keyLight = new THREE.DirectionalLight(0xffeedd, 1.1);
  keyLight.position.set(3, 6, 4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 1024;
  keyLight.shadow.mapSize.height = 1024;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 20;
  keyLight.shadow.camera.left = -3;
  keyLight.shadow.camera.right = 3;
  keyLight.shadow.camera.top = 3;
  keyLight.shadow.camera.bottom = -1;
  scene.add(keyLight);

  // Fill light (cool from left)
  const fillLight = new THREE.DirectionalLight(0x99aadd, 0.5);
  fillLight.position.set(-4, 3, 2);
  scene.add(fillLight);

  // Hemisphere sky/ground
  scene.add(new THREE.HemisphereLight(0x8899bb, 0x333344, 0.3));

  // Rim/back light for depth separation
  const rimLight = new THREE.DirectionalLight(0xaaccff, 0.25);
  rimLight.position.set(0, 2, -5);
  scene.add(rimLight);

  // ── Resize via ResizeObserver ──────────────────────────────
  const stopResize = observeResize(container, (cw, ch) => {
    camera.aspect = cw / ch;
    camera.updateProjectionMatrix();
    renderer.setSize(cw, ch);
  });

  // ── Animation loop ─────────────────────────────────────────
  let lastFrameTime = 0;
  let frameCount = 0;
  let _fpsEl = null;

  function animate(time) {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);

    // FPS counter
    frameCount++;
    if (time - lastFrameTime > 1000) {
      if (!_fpsEl) _fpsEl = document.getElementById('fps-counter');
      if (_fpsEl) _fpsEl.textContent = frameCount + ' FPS';
      frameCount = 0;
      lastFrameTime = time;
    }
  }
  requestAnimationFrame(animate);

  return { scene, camera, renderer, controls, dispose: stopResize };
}
