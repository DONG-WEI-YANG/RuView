// dashboard/src/scene/skeleton.js
/**
 * 24-joint skeleton overlay — dots and bone lines.
 * Subscribes to bus.on('pose') for single-person and
 * bus.on('persons') for multi-person skeleton rendering.
 * Manages a pool of up to 4 skeleton overlays.
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

const MAX_SKELETON_POOL = 4;
const SKELETON_TIMEOUT_MS = 3000;

// Confidence -> color: green (high) -> yellow -> red (low)
function confidenceColor(c) {
  const r = c < 0.5 ? 1.0 : 1.0 - (c - 0.5) * 2;
  const g = c > 0.5 ? 1.0 : c * 2;
  return new THREE.Color(r, g, 0.1);
}

/**
 * Build a single skeleton group with 24 joint spheres and bone lines.
 * @param {THREE.Color|number} baseColor — default joint/bone color
 * @returns {{ group, joints, boneLines, update }}
 */
function buildSkeletonGroup(baseColor = 0x00ff88) {
  const group = new THREE.Group();
  group.name = 'skeleton-overlay';
  group.visible = false;

  const joints = [];
  for (let i = 0; i < 24; i++) {
    const mat = new THREE.MeshPhongMaterial({ color: baseColor });
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.025, 8, 8),
      mat,
    );
    sphere.visible = false;
    group.add(sphere);
    joints.push(sphere);
  }

  const boneMaterial = new THREE.LineBasicMaterial({
    color: baseColor,
    linewidth: 2,
    transparent: true,
    opacity: 0.8,
  });

  const boneLines = [];
  for (let bi = 0; bi < BONES.length; bi++) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3),
    );
    const line = new THREE.Line(geometry, boneMaterial.clone());
    line.visible = false;
    group.add(line);
    boneLines.push({ line, a: BONES[bi][0], b: BONES[bi][1] });
  }

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

  function setColor(hexColor) {
    const color = new THREE.Color(hexColor);
    for (const j of joints) j.material.color.copy(color);
    for (const bl of boneLines) bl.line.material.color.copy(color);
  }

  return { group, joints, boneLines, update, setColor };
}

/**
 * Create skeleton overlay pool and wire up bus listeners.
 * @param {THREE.Scene} scene
 * @returns {{ group: THREE.Group, update: (joints: number[][]) => void, dispose: () => void }}
 */
export function createSkeleton(scene) {
  // ── Pool of skeleton groups ────────────────────────────────
  const pool = [];
  const personIdToSlot = new Map();
  const lastSeen = new Array(MAX_SKELETON_POOL).fill(0);

  for (let i = 0; i < MAX_SKELETON_POOL; i++) {
    const skel = buildSkeletonGroup(0x00ff88);
    skel.group.name = `skeleton-overlay-${i}`;
    scene.add(skel.group);
    pool.push(skel);
  }

  // Primary skeleton is pool[0] for backward compat
  const primary = pool[0];

  // ── Single-person pose handler ─────────────────────────────
  function onPose(data) {
    if (data && data.joints) {
      if (!primary.group.visible) primary.group.visible = true;
      primary.update(data.joints);
      lastSeen[0] = Date.now();

      const jc = data.joint_confidence;
      if (jc && jc.length === 24) {
        for (let i = 0; i < 24; i++) {
          primary.joints[i].material.color.copy(confidenceColor(jc[i]));
        }
      }
    }
  }
  bus.on('pose', onPose);

  // ── Multi-person handler ───────────────────────────────────
  function onPersons(data) {
    if (!data || !data.persons || !Array.isArray(data.persons)) return;
    const now = Date.now();
    const seenSlots = new Set();

    for (const person of data.persons) {
      const pid = person.id;

      let slot = personIdToSlot.get(pid);
      if (slot === undefined) {
        for (let i = 0; i < MAX_SKELETON_POOL; i++) {
          if (!Array.from(personIdToSlot.values()).includes(i)) {
            slot = i;
            break;
          }
        }
        if (slot === undefined) continue;
        personIdToSlot.set(pid, slot);
      }
      seenSlots.add(slot);

      const skel = pool[slot];
      if (!skel) continue;

      skel.group.visible = true;
      lastSeen[slot] = now;

      if (person.joints && person.joints.length === 24) {
        skel.update(person.joints);
      }

      if (person.color) {
        skel.setColor(person.color);
      }

      const jc = person.joint_confidence;
      if (jc && jc.length === 24) {
        for (let i = 0; i < 24; i++) {
          skel.joints[i].material.color.copy(confidenceColor(jc[i]));
        }
      }
    }

    // Hide stale skeletons
    for (let i = 0; i < MAX_SKELETON_POOL; i++) {
      if (!seenSlots.has(i) && pool[i].group.visible) {
        if (now - lastSeen[i] > SKELETON_TIMEOUT_MS) {
          pool[i].group.visible = false;
          for (const [pid, s] of personIdToSlot.entries()) {
            if (s === i) { personIdToSlot.delete(pid); break; }
          }
        }
      }
    }
  }
  bus.on('persons', onPersons);

  // ── Dispose ────────────────────────────────────────────────
  function dispose() {
    bus.off('pose', onPose);
    bus.off('persons', onPersons);
    for (const skel of pool) {
      scene.remove(skel.group);
      skel.group.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
    }
  }

  return { group: primary.group, update: primary.update, pool, dispose };
}
