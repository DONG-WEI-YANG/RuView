// dashboard/src/scene/room.js
/**
 * Room geometry — floor, grid, wireframe walls, node markers,
 * coverage/Fresnel zone overlay, and receiver object.
 * Subscribes to bus.on('status') for node positions and room dimensions.
 */
import * as THREE from 'three';
import { bus } from '../events.js';

/**
 * Create room geometry and add it to the scene.
 * @param {THREE.Scene} scene
 * @returns {{ dispose: () => void }}
 */
export function createRoom(scene) {
  // ── Grid + floor ───────────────────────────────────────────
  let gridHelper = new THREE.GridHelper(4, 4, 0x444466, 0x33334a);
  scene.add(gridHelper);

  const floorGeom = new THREE.PlaneGeometry(4, 4);
  const floorMat = new THREE.MeshPhongMaterial({
    color: 0x1a1a2e, depthWrite: false, transparent: true, opacity: 0.6,
  });
  const floor = new THREE.Mesh(floorGeom, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.005;
  floor.receiveShadow = true;
  scene.add(floor);

  // ── Room wireframe box ─────────────────────────────────────
  const roomGroup = new THREE.Group();
  scene.add(roomGroup);

  function updateRoom(dims) {
    if (!dims) return;
    const w = dims.width || 4.0;
    const d = dims.depth || 4.0;
    const h = dims.height || 2.8;

    // Rebuild grid
    scene.remove(gridHelper);
    gridHelper = new THREE.GridHelper(Math.max(w, d), Math.max(w, d), 0x444466, 0x33334a);
    scene.add(gridHelper);

    // Rebuild floor
    floor.geometry.dispose();
    floor.geometry = new THREE.PlaneGeometry(w, d);

    // Rebuild walls
    while (roomGroup.children.length > 0) {
      roomGroup.remove(roomGroup.children[0]);
    }
    const boxGeo = new THREE.BoxGeometry(w, h, d);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x333344 }),
    );
    line.position.set(0, h / 2, 0);
    roomGroup.add(line);
  }

  // ── Coverage / Fresnel zone overlay ────────────────────────
  const coverageGroup = new THREE.Group();
  coverageGroup.name = 'coverage-overlay';
  scene.add(coverageGroup);

  function renderCoverage(positions) {
    while (coverageGroup.children.length > 0) {
      coverageGroup.remove(coverageGroup.children[0]);
    }
    const nodeKeys = Object.keys(positions);
    if (nodeKeys.length < 2) return;

    const material = new THREE.MeshBasicMaterial({
      color: 0x00ff00, transparent: true, opacity: 0.05,
      side: THREE.DoubleSide, depthWrite: false,
    });

    for (let i = 0; i < nodeKeys.length; i++) {
      for (let j = i + 1; j < nodeKeys.length; j++) {
        const p1 = positions[nodeKeys[i]];
        const p2 = positions[nodeKeys[j]];
        const v1 = new THREE.Vector3(p1[0], p1[1], p1[2]);
        const v2 = new THREE.Vector3(p2[0], p2[1], p2[2]);

        const distance = v1.distanceTo(v2);
        const geometry = new THREE.CylinderGeometry(0.1, 0.1, distance, 8, 1, true);
        const mesh = new THREE.Mesh(geometry, material);

        const mid = v1.clone().add(v2).multiplyScalar(0.5);
        mesh.position.copy(mid);
        mesh.lookAt(v2);
        mesh.rotateX(Math.PI / 2);

        coverageGroup.add(mesh);
      }
    }
  }

  // ── Node markers ───────────────────────────────────────────
  const nodesGroup = new THREE.Group();
  nodesGroup.name = 'nodes-overlay';
  scene.add(nodesGroup);
  let nodesRendered = false;

  function renderNodes(positions) {
    if (nodesRendered || !positions) return;

    const nodeMat = new THREE.MeshPhongMaterial({
      color: 0x00aaff, emissive: 0x004466,
      specular: 0xffffff, shininess: 30,
    });
    const antennaMat = new THREE.LineBasicMaterial({ color: 0xffffff });

    for (const nid in positions) {
      const pos = positions[nid];
      const x = pos[0], y = pos[1], z = pos[2];
      const group = new THREE.Group();
      group.position.set(x, y, z);

      // Box body
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.08, 0.1), nodeMat);
      group.add(box);

      // Antenna
      const antGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0.06, 0.04, 0),
        new THREE.Vector3(0.06, 0.15, 0),
      ]);
      group.add(new THREE.Line(antGeo, antennaMat));

      // Label sprite
      const labelCanvas = document.createElement('canvas');
      const ctx = labelCanvas.getContext('2d');
      labelCanvas.width = 64;
      labelCanvas.height = 32;
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(0, 0, 64, 32);
      ctx.strokeStyle = '#00aaff';
      ctx.strokeRect(0, 0, 64, 32);
      ctx.fillStyle = '#00aaff';
      ctx.font = 'bold 20px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('N' + nid, 32, 22);

      const texture = new THREE.CanvasTexture(labelCanvas);
      const spriteMat = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.set(0, 0.15, 0);
      sprite.scale.set(0.3, 0.15, 1);
      group.add(sprite);

      nodesGroup.add(group);
    }
    nodesRendered = true;
  }

  // ── Receiver / Laptop object ───────────────────────────────
  const receiverGroup = new THREE.Group();
  receiverGroup.position.set(0, 0.5, -2.5);
  scene.add(receiverGroup);

  const lapBase = new THREE.Mesh(
    new THREE.BoxGeometry(0.4, 0.02, 0.3),
    new THREE.MeshPhongMaterial({ color: 0x333333 }),
  );
  receiverGroup.add(lapBase);

  const lapScreen = new THREE.Mesh(
    new THREE.BoxGeometry(0.4, 0.25, 0.02),
    new THREE.MeshPhongMaterial({ color: 0x111111 }),
  );
  lapScreen.position.set(0, 0.125, -0.15);
  receiverGroup.add(lapScreen);

  const rLabel = document.createElement('canvas');
  const rctx = rLabel.getContext('2d');
  rLabel.width = 128;
  rLabel.height = 32;
  rctx.fillStyle = '#ffffff';
  rctx.font = 'bold 20px monospace';
  rctx.textAlign = 'center';
  rctx.fillText('Receiver PC', 64, 22);
  const rTex = new THREE.CanvasTexture(rLabel);
  const rSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: rTex }));
  rSprite.position.set(0, 0.4, 0);
  rSprite.scale.set(0.6, 0.15, 1);
  receiverGroup.add(rSprite);

  // ── Subscribe to status events for node positions / room dims ──
  function onStatus(data) {
    if (!data) return;
    if (data.node_positions) {
      renderNodes(data.node_positions);
      renderCoverage(data.node_positions);
    }
    if (data.room_dimensions) {
      updateRoom(data.room_dimensions);
    }
  }
  bus.on('status', onStatus);

  // ── Dispose ────────────────────────────────────────────────
  function dispose() {
    bus.off('status', onStatus);

    const groups = [roomGroup, coverageGroup, nodesGroup, receiverGroup];
    for (const g of groups) {
      scene.remove(g);
      g.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (child.material.map) child.material.map.dispose();
          child.material.dispose();
        }
      });
    }
    scene.remove(gridHelper);
    scene.remove(floor);
    floor.geometry.dispose();
    floorMat.dispose();
  }

  return { updateRoom, renderNodes, renderCoverage, dispose };
}
