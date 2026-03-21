// dashboard/src/scene/skeleton.js
/**
 * 24-joint skeleton overlay — dots and bone lines.
 * Subscribes to bus.on('pose') to update joint positions.
 */
import * as THREE from 'three';
import { bus } from '../events.js';

// ── Joint connectivity (24-joint skeleton) ──────────────────
const BONES = [
  [0, 1], [1, 2], [2, 3],
  [3, 4], [4, 5], [5, 6],
  [3, 7], [7, 8], [8, 9],
  [3, 10], [10, 11],
  [11, 12], [12, 13], [13, 14],
  [11, 15], [15, 16], [16, 17],
  [14, 18], [17, 19],
  [6, 20], [9, 21],
  [0, 22], [0, 23],
];

/**
 * Create the skeleton overlay group and wire up the bus listener.
 * @param {THREE.Scene} scene — the Three.js scene to add the skeleton to
 * @returns {{ group: THREE.Group, update: (joints: number[][]) => void, dispose: () => void }}
 */
export function createSkeleton(scene) {
  const group = new THREE.Group();
  group.name = 'skeleton-overlay';
  scene.add(group);

  // ── Joint spheres ──────────────────────────────────────────
  const jointMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
  const joints = [];
  for (let i = 0; i < 24; i++) {
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.025, 8, 8),
      jointMaterial,
    );
    sphere.visible = false;
    group.add(sphere);
    joints.push(sphere);
  }

  // ── Bone lines ─────────────────────────────────────────────
  const boneMaterial = new THREE.LineBasicMaterial({
    color: 0x00cc66,
    linewidth: 2,
    transparent: true,
    opacity: 0.8,
  });

  const boneLines = [];
  for (let bi = 0; bi < BONES.length; bi++) {
    const [a, b] = BONES[bi];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3),
    );
    const line = new THREE.Line(geometry, boneMaterial);
    line.visible = false;
    group.add(line);
    boneLines.push({ line, a, b });
  }

  // ── Update function ────────────────────────────────────────
  function update(jointData) {
    if (!jointData || jointData.length !== 24) return;

    for (let i = 0; i < 24; i++) {
      const j = jointData[i];
      joints[i].position.set(j[0], j[1], j[2]);
      joints[i].visible = true;
    }

    for (let i = 0; i < boneLines.length; i++) {
      const bl = boneLines[i];
      const positions = bl.line.geometry.attributes.position.array;
      positions[0] = jointData[bl.a][0];
      positions[1] = jointData[bl.a][1];
      positions[2] = jointData[bl.a][2];
      positions[3] = jointData[bl.b][0];
      positions[4] = jointData[bl.b][1];
      positions[5] = jointData[bl.b][2];
      bl.line.geometry.attributes.position.needsUpdate = true;
      bl.line.visible = true;
    }
  }

  // Start hidden — show only when real pose data arrives
  group.visible = false;

  // ── Subscribe to pose events ───────────────────────────────
  function onPose(data) {
    if (data && data.joints) {
      if (!group.visible) group.visible = true;
      update(data.joints);
    }
  }
  bus.on('pose', onPose);

  // ── Dispose ────────────────────────────────────────────────
  function dispose() {
    bus.off('pose', onPose);
    scene.remove(group);
    group.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
  }

  return { group, update, dispose };
}
