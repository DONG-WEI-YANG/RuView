// dashboard/src/scene/body-mesh.js
/**
 * SMPL-style body mesh rendering — procedural wireframe with glow.
 * Subscribes to bus.on('pose') to drive pose deformation.
 *
 * Ported from the monolithic smpl-mesh.js + skeleton3d.js body-mesh section.
 * Self-contained: all chain definitions, vector math, and vertex skinning
 * are encapsulated here.
 */
import * as THREE from 'three';
import { bus } from '../events.js';

// ── Precomputed radial trig tables ──────────────────────────
const RAD = 20;
const COS = [];
const SIN = [];
for (let i = 0; i <= RAD; i++) {
  const a = (i / RAD) * Math.PI * 2;
  COS.push(Math.cos(a));
  SIN.push(Math.sin(a));
}

// ── T-pose rest positions (24 joints, Y-up, meters) ─────────
const REST = [
  [0.000, 1.700, 0.000],   //  0 head
  [0.000, 1.550, 0.000],   //  1 neck
  [0.000, 1.380, 0.000],   //  2 chest
  [0.000, 1.120, 0.000],   //  3 spine
  [-0.200, 1.400, 0.000],  //  4 L shoulder
  [-0.480, 1.400, 0.000],  //  5 L elbow
  [-0.700, 1.400, 0.000],  //  6 L wrist
  [0.200, 1.400, 0.000],   //  7 R shoulder
  [0.480, 1.400, 0.000],   //  8 R elbow
  [0.700, 1.400, 0.000],   //  9 R wrist
  [0.000, 0.950, 0.000],   // 10 pelvis
  [0.000, 0.900, 0.000],   // 11 hip center
  [-0.100, 0.880, 0.000],  // 12 L hip
  [-0.100, 0.500, 0.000],  // 13 L knee
  [-0.100, 0.080, 0.000],  // 14 L ankle
  [0.100, 0.880, 0.000],   // 15 R hip
  [0.100, 0.500, 0.000],   // 16 R knee
  [0.100, 0.080, 0.000],   // 17 R ankle
  [-0.100, 0.030, 0.080],  // 18 L foot
  [0.100, 0.030, 0.080],   // 19 R foot
  [-0.780, 1.400, 0.000],  // 20 L hand
  [0.780, 1.400, 0.000],   // 21 R hand
  [-0.030, 1.720, 0.060],  // 22 L eye
  [0.030, 1.720, 0.060],   // 23 R eye
];

// ── Body chain definitions ──────────────────────────────────
const CHAINS = [
  {
    name: 'torso',
    joints: [11, 10, 3, 2, 1],
    jointT: [0, 0.18, 0.52, 0.76, 1.0],
    up: [0, 0, 1],
    rings: [
      { t: 0.000, rx: 0.148, rz: 0.105 }, { t: 0.060, rx: 0.142, rz: 0.100 },
      { t: 0.120, rx: 0.132, rz: 0.095 }, { t: 0.180, rx: 0.120, rz: 0.088 },
      { t: 0.250, rx: 0.108, rz: 0.078 }, { t: 0.320, rx: 0.112, rz: 0.082 },
      { t: 0.400, rx: 0.128, rz: 0.090 }, { t: 0.480, rx: 0.142, rz: 0.098 },
      { t: 0.520, rx: 0.150, rz: 0.100 }, { t: 0.580, rx: 0.158, rz: 0.108 },
      { t: 0.640, rx: 0.162, rz: 0.110 }, { t: 0.700, rx: 0.160, rz: 0.108 },
      { t: 0.760, rx: 0.155, rz: 0.100 }, { t: 0.800, rx: 0.138, rz: 0.088 },
      { t: 0.830, rx: 0.110, rz: 0.072 }, { t: 0.860, rx: 0.078, rz: 0.055 },
      { t: 0.900, rx: 0.054, rz: 0.044 }, { t: 0.950, rx: 0.048, rz: 0.040 },
      { t: 1.000, rx: 0.044, rz: 0.038 },
    ],
  },
  {
    name: 'l_arm',
    joints: [4, 5, 6],
    jointT: [0, 0.50, 1.0],
    up: [0, 0, 1],
    rings: [
      { t: 0.00, rx: 0.052, rz: 0.048 }, { t: 0.10, rx: 0.050, rz: 0.046 },
      { t: 0.25, rx: 0.046, rz: 0.042 }, { t: 0.40, rx: 0.042, rz: 0.038 },
      { t: 0.50, rx: 0.040, rz: 0.036 }, { t: 0.55, rx: 0.038, rz: 0.034 },
      { t: 0.65, rx: 0.035, rz: 0.031 }, { t: 0.80, rx: 0.030, rz: 0.027 },
      { t: 0.90, rx: 0.027, rz: 0.024 }, { t: 1.00, rx: 0.025, rz: 0.021 },
    ],
  },
  {
    name: 'r_arm',
    joints: [7, 8, 9],
    jointT: [0, 0.50, 1.0],
    up: [0, 0, 1],
    rings: [
      { t: 0.00, rx: 0.052, rz: 0.048 }, { t: 0.10, rx: 0.050, rz: 0.046 },
      { t: 0.25, rx: 0.046, rz: 0.042 }, { t: 0.40, rx: 0.042, rz: 0.038 },
      { t: 0.50, rx: 0.040, rz: 0.036 }, { t: 0.55, rx: 0.038, rz: 0.034 },
      { t: 0.65, rx: 0.035, rz: 0.031 }, { t: 0.80, rx: 0.030, rz: 0.027 },
      { t: 0.90, rx: 0.027, rz: 0.024 }, { t: 1.00, rx: 0.025, rz: 0.021 },
    ],
  },
  {
    name: 'l_leg',
    joints: [12, 13, 14],
    jointT: [0, 0.52, 1.0],
    up: [0, 0, 1],
    rings: [
      { t: 0.00, rx: 0.072, rz: 0.068 }, { t: 0.08, rx: 0.074, rz: 0.070 },
      { t: 0.18, rx: 0.072, rz: 0.066 }, { t: 0.30, rx: 0.065, rz: 0.058 },
      { t: 0.42, rx: 0.055, rz: 0.050 }, { t: 0.52, rx: 0.048, rz: 0.044 },
      { t: 0.58, rx: 0.046, rz: 0.042 }, { t: 0.70, rx: 0.042, rz: 0.038 },
      { t: 0.82, rx: 0.038, rz: 0.034 }, { t: 0.92, rx: 0.036, rz: 0.032 },
      { t: 1.00, rx: 0.034, rz: 0.030 },
    ],
  },
  {
    name: 'r_leg',
    joints: [15, 16, 17],
    jointT: [0, 0.52, 1.0],
    up: [0, 0, 1],
    rings: [
      { t: 0.00, rx: 0.072, rz: 0.068 }, { t: 0.08, rx: 0.074, rz: 0.070 },
      { t: 0.18, rx: 0.072, rz: 0.066 }, { t: 0.30, rx: 0.065, rz: 0.058 },
      { t: 0.42, rx: 0.055, rz: 0.050 }, { t: 0.52, rx: 0.048, rz: 0.044 },
      { t: 0.58, rx: 0.046, rz: 0.042 }, { t: 0.70, rx: 0.042, rz: 0.038 },
      { t: 0.82, rx: 0.038, rz: 0.034 }, { t: 0.92, rx: 0.036, rz: 0.032 },
      { t: 1.00, rx: 0.034, rz: 0.030 },
    ],
  },
];

// Joint spheres for smooth transitions
const JSPHERES = [
  { idx: 1, r: 0.046 }, { idx: 2, r: 0.055 }, { idx: 3, r: 0.068 },
  { idx: 4, r: 0.050 }, { idx: 5, r: 0.038 }, { idx: 7, r: 0.050 },
  { idx: 8, r: 0.038 }, { idx: 10, r: 0.068 }, { idx: 11, r: 0.075 },
  { idx: 12, r: 0.068 }, { idx: 13, r: 0.046 }, { idx: 15, r: 0.068 },
  { idx: 16, r: 0.046 },
];

// ── Vector math helpers ─────────────────────────────────────
function v3sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function v3lerp(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
function v3cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function v3len(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
function v3norm(v) {
  const l = v3len(v);
  return l > 1e-7 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 1, 0];
}

// ── Path interpolation along a joint chain ──────────────────
function chainPos(jts, jt, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < jt.length - 1; i++) {
    if (t <= jt[i + 1] + 1e-6) {
      const s = jt[i + 1] - jt[i];
      const lt = s > 1e-6 ? (t - jt[i]) / s : 0;
      return v3lerp(jts[i], jts[i + 1], Math.max(0, Math.min(1, lt)));
    }
  }
  return jts[jts.length - 1];
}

function chainTangent(jts, jt, t) {
  const eps = 0.008;
  return v3norm(v3sub(
    chainPos(jts, jt, Math.min(1, t + eps)),
    chainPos(jts, jt, Math.max(0, t - eps)),
  ));
}

function localFrame(tangent, prefUp) {
  let bn = v3cross(tangent, prefUp);
  if (v3len(bn) < 0.01) {
    bn = v3cross(tangent, [1, 0, 0]);
    if (v3len(bn) < 0.01) bn = v3cross(tangent, [0, 0, 1]);
  }
  bn = v3norm(bn);
  const nm = v3norm(v3cross(bn, tangent));
  return { b: bn, n: nm };
}

function fillVerts(posArr, norArr, rings, jts, jt, up) {
  let vi = 0, ni = 0;
  for (let r = 0; r < rings.length; r++) {
    const ring = rings[r];
    const center = chainPos(jts, jt, ring.t);
    const tan = chainTangent(jts, jt, ring.t);
    const fr = localFrame(tan, up);
    const bn = fr.b, nm = fr.n;

    for (let i = 0; i <= RAD; i++) {
      const c = COS[i], s = SIN[i];
      posArr[vi] = center[0] + c * ring.rx * bn[0] + s * ring.rz * nm[0];
      posArr[vi + 1] = center[1] + c * ring.rx * bn[1] + s * ring.rz * nm[1];
      posArr[vi + 2] = center[2] + c * ring.rx * bn[2] + s * ring.rz * nm[2];

      const enx = c / (ring.rx > 0.001 ? ring.rx : 0.001);
      const enz = s / (ring.rz > 0.001 ? ring.rz : 0.001);
      const el = Math.sqrt(enx * enx + enz * enz) || 1;
      norArr[ni] = (enx * bn[0] + enz * nm[0]) / el;
      norArr[ni + 1] = (enx * bn[1] + enz * nm[1]) / el;
      norArr[ni + 2] = (enx * bn[2] + enz * nm[2]) / el;

      vi += 3;
      ni += 3;
    }
  }
}

function buildChainGeom(chain, jointPositions) {
  const jts = chain.joints.map((i) => jointPositions[i]);
  const rings = chain.rings;
  const nR = rings.length;
  const nV = nR * (RAD + 1);

  const pos = new Float32Array(nV * 3);
  const nor = new Float32Array(nV * 3);
  const uv = new Float32Array(nV * 2);

  fillVerts(pos, nor, rings, jts, chain.jointT, chain.up);

  let ui = 0;
  for (let r = 0; r < nR; r++) {
    for (let i = 0; i <= RAD; i++) {
      uv[ui++] = i / RAD;
      uv[ui++] = rings[r].t;
    }
  }

  const nFaces = (nR - 1) * RAD * 2;
  const idx = new Uint16Array(nFaces * 3);
  let ii = 0;
  for (let r = 0; r < nR - 1; r++) {
    for (let i = 0; i < RAD; i++) {
      const a = r * (RAD + 1) + i;
      const b = a + 1;
      const cc = a + (RAD + 1);
      const d = cc + 1;
      idx[ii++] = a; idx[ii++] = cc; idx[ii++] = b;
      idx[ii++] = b; idx[ii++] = cc; idx[ii++] = d;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geom.setIndex(new THREE.BufferAttribute(idx, 1));
  return geom;
}

// ── Internal renderer class ─────────────────────────────────
class SMPLMeshRenderer {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'smpl-body';
    this._chainData = [];
    this._spheres = [];
    this._demoRaf = 0;
    this.headMesh = null;
    this._headGlow = null;
    this._build();
  }

  _build() {
    // ── Solid body material (mannequin look) ──────────────────
    // Neutral skin-like tone, semi-transparent, smooth Phong shading
    const solidMat = new THREE.MeshPhongMaterial({
      color: 0xd4a574,          // warm neutral skin tone
      specular: 0x443322,
      shininess: 15,
      transparent: true,
      opacity: 0.75,
      side: THREE.DoubleSide,
      depthWrite: false,        // let wireframe overlay show through
    });

    // ── Wireframe overlay (subtle structural lines) ───────────
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00ffaa, wireframe: true, transparent: true, opacity: 0.35,
    });
    // Glow layer (outer rim)
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88, wireframe: true, transparent: true, opacity: 0.08,
    });

    // ── Render style: 'mannequin' | 'wireframe' | 'xray' ─────
    this._renderStyle = 'mannequin';

    for (let ci = 0; ci < CHAINS.length; ci++) {
      const ch = CHAINS[ci];

      // Solid body mesh
      const solidGeom = buildChainGeom(ch, REST);
      const solidMesh = new THREE.Mesh(solidGeom, solidMat.clone());
      solidMesh.frustumCulled = false;
      solidMesh.renderOrder = 0;
      this.group.add(solidMesh);

      // Wireframe overlay
      const geom = buildChainGeom(ch, REST);
      const mesh = new THREE.Mesh(geom, wireMat.clone());
      mesh.frustumCulled = false;
      mesh.renderOrder = 1;
      this.group.add(mesh);

      // Glow
      const glowGeom = buildChainGeom(ch, REST);
      const glowMesh = new THREE.Mesh(glowGeom, glowMat.clone());
      glowMesh.scale.set(1.03, 1.0, 1.03);
      glowMesh.frustumCulled = false;
      glowMesh.renderOrder = 2;
      this.group.add(glowMesh);

      this._chainData.push({
        def: ch,
        solidMesh, solidGeom,
        solidPosAttr: solidGeom.getAttribute('position'),
        solidNorAttr: solidGeom.getAttribute('normal'),
        mesh, geom,
        posAttr: geom.getAttribute('position'),
        norAttr: geom.getAttribute('normal'),
        glowMesh, glowGeom,
        glowPosAttr: glowGeom.getAttribute('position'),
        glowNorAttr: glowGeom.getAttribute('normal'),
      });
    }

    // Head ellipsoid — solid + wireframe
    const headGeom = new THREE.SphereGeometry(0.098, 16, 12);
    this._headSolid = new THREE.Mesh(headGeom, solidMat.clone());
    this._headSolid.scale.set(1.0, 1.18, 0.92);
    this._headSolid.renderOrder = 0;
    this.group.add(this._headSolid);

    this.headMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.098, 16, 12),
      wireMat.clone(),
    );
    this.headMesh.scale.set(1.0, 1.18, 0.92);
    this.headMesh.renderOrder = 1;
    this.group.add(this.headMesh);

    const headGlow = new THREE.Mesh(
      new THREE.SphereGeometry(0.104, 16, 12),
      glowMat.clone(),
    );
    headGlow.scale.set(1.0, 1.18, 0.92);
    headGlow.renderOrder = 2;
    this.group.add(headGlow);
    this._headGlow = headGlow;

    // Joint spheres (solid)
    const sphGeom = new THREE.SphereGeometry(1, 8, 6);
    for (let si = 0; si < JSPHERES.length; si++) {
      const jd = JSPHERES[si];
      const sm = new THREE.Mesh(sphGeom, solidMat.clone());
      sm.scale.set(jd.r, jd.r, jd.r);
      this.group.add(sm);
      this._spheres.push({ mesh: sm, idx: jd.idx });
    }

    // End caps (wrists, ankles)
    const caps = [
      { j: 6, r: 0.024 }, { j: 9, r: 0.024 },
      { j: 14, r: 0.033 }, { j: 17, r: 0.033 },
    ];
    for (const cd of caps) {
      const cm = new THREE.Mesh(sphGeom, solidMat.clone());
      cm.scale.set(cd.r, cd.r * 0.7, cd.r);
      this.group.add(cm);
      this._spheres.push({ mesh: cm, idx: cd.j });
    }

    this.group.visible = false;
  }

  update(joints) {
    if (!joints || joints.length < 18) return;
    this.group.visible = true;

    for (let ci = 0; ci < this._chainData.length; ci++) {
      const cd = this._chainData[ci];
      const ch = cd.def;
      const jts = ch.joints.map((ji) => joints[ji]);

      // Solid body
      if (cd.solidPosAttr) {
        fillVerts(cd.solidPosAttr.array, cd.solidNorAttr.array, ch.rings, jts, ch.jointT, ch.up);
        cd.solidPosAttr.needsUpdate = true;
        cd.solidNorAttr.needsUpdate = true;
        cd.solidGeom.boundingSphere = null;
      }

      // Wireframe overlay
      fillVerts(cd.posAttr.array, cd.norAttr.array, ch.rings, jts, ch.jointT, ch.up);
      cd.posAttr.needsUpdate = true;
      cd.norAttr.needsUpdate = true;
      cd.geom.boundingSphere = null;

      // Glow
      if (cd.glowPosAttr) {
        fillVerts(cd.glowPosAttr.array, cd.glowNorAttr.array, ch.rings, jts, ch.jointT, ch.up);
        cd.glowPosAttr.needsUpdate = true;
        cd.glowNorAttr.needsUpdate = true;
        cd.glowGeom.boundingSphere = null;
      }
    }

    const h = joints[0];
    if (this._headSolid) this._headSolid.position.set(h[0], h[1], h[2]);
    this.headMesh.position.set(h[0], h[1], h[2]);
    if (this._headGlow) this._headGlow.position.set(h[0], h[1], h[2]);

    for (const sp of this._spheres) {
      const jp = joints[sp.idx];
      sp.mesh.position.set(jp[0], jp[1], jp[2]);
    }
  }

  showDemo() {
    let t = 0;
    const tick = () => {
      t += 0.016;
      const joints = REST.map((r) => [r[0], r[1], r[2]]);

      const armSwing = Math.sin(t * 0.6) * 0.03;
      joints[4] = [-0.18, 1.36, 0];
      joints[5] = [-0.22, 1.12 + armSwing, 0.02];
      joints[6] = [-0.20, 0.90 + armSwing * 1.4, 0.04];
      joints[20] = [-0.19, 0.82 + armSwing * 1.6, 0.05];
      joints[7] = [0.18, 1.36, 0];
      joints[8] = [0.22, 1.12 - armSwing, 0.02];
      joints[9] = [0.20, 0.90 - armSwing * 1.4, 0.04];
      joints[21] = [0.19, 0.82 - armSwing * 1.6, 0.05];

      const sway = Math.sin(t * 0.4) * 0.008;
      for (let i = 0; i < 24; i++) {
        joints[i][0] += sway;
        joints[i][2] += Math.sin(t * 0.3 + i * 0.1) * 0.002;
      }

      const breath = Math.sin(t * 0.5) * 0.008;
      joints[2][2] += breath;
      joints[3][2] += breath * 0.5;

      const kneeBend = Math.sin(t * 0.35) * 0.008;
      joints[13][1] += kneeBend;
      joints[16][1] += kneeBend;

      this.update(joints);
      this._demoRaf = requestAnimationFrame(tick);
    };
    tick();
  }

  stopDemo() {
    if (this._demoRaf) {
      cancelAnimationFrame(this._demoRaf);
      this._demoRaf = 0;
    }
  }

  /**
   * Color each chain by the average confidence of its joints.
   * Solid: skin tone modulated by confidence (brighter = higher)
   * Wire: green→yellow→red
   * @param {number[]} jointConf - 24 confidence values (0-1)
   */
  applyConfidenceColors(jointConf) {
    for (const cd of this._chainData) {
      const chainJoints = cd.def.joints;
      const avgConf = chainJoints.reduce((s, j) => s + (jointConf[j] || 0.5), 0) / chainJoints.length;

      // Wire overlay: green→yellow→red
      const wr = avgConf < 0.5 ? 1.0 : 1.0 - (avgConf - 0.5) * 2;
      const wg = avgConf > 0.5 ? 1.0 : avgConf * 2;
      const wireColor = new THREE.Color(wr, wg, 0.15);
      cd.mesh.material.color.copy(wireColor);
      cd.glowMesh.material.color.copy(wireColor);

      // Solid body: skin tone * confidence brightness
      if (cd.solidMesh) {
        const brightness = 0.5 + avgConf * 0.5; // 0.5-1.0
        cd.solidMesh.material.color.set(
          new THREE.Color(0.83 * brightness, 0.65 * brightness, 0.46 * brightness)
        );
        cd.solidMesh.material.opacity = 0.5 + avgConf * 0.35;
      }
    }
    // Head
    const hc = jointConf[0] !== undefined ? jointConf[0] : 0.5;
    const wr = hc < 0.5 ? 1.0 : 1.0 - (hc - 0.5) * 2;
    const wg = hc > 0.5 ? 1.0 : hc * 2;
    if (this.headMesh) this.headMesh.material.color.set(new THREE.Color(wr, wg, 0.15));
    if (this._headGlow) this._headGlow.material.color.set(new THREE.Color(wr, wg, 0.15));
    if (this._headSolid) {
      const b = 0.5 + hc * 0.5;
      this._headSolid.material.color.set(new THREE.Color(0.83 * b, 0.65 * b, 0.46 * b));
      this._headSolid.material.opacity = 0.5 + hc * 0.35;
    }
  }

  /**
   * Switch render style: 'mannequin' | 'wireframe' | 'xray'
   */
  setRenderStyle(style) {
    this._renderStyle = style;
    const showSolid = style === 'mannequin' || style === 'xray';
    const showWire = style === 'wireframe' || style === 'xray';
    const solidOpacity = style === 'xray' ? 0.3 : 0.75;

    for (const cd of this._chainData) {
      if (cd.solidMesh) {
        cd.solidMesh.visible = showSolid;
        cd.solidMesh.material.opacity = solidOpacity;
      }
      cd.mesh.visible = showWire;
      cd.glowMesh.visible = showWire;
    }
    if (this._headSolid) {
      this._headSolid.visible = showSolid;
      this._headSolid.material.opacity = solidOpacity;
    }
    if (this.headMesh) this.headMesh.visible = showWire;
    if (this._headGlow) this._headGlow.visible = showWire;
    for (const sp of this._spheres) {
      sp.mesh.material.wireframe = !showSolid;
      sp.mesh.material.opacity = showSolid ? 0.9 : 0.85;
    }
  }

  dispose() {
    this.stopDemo();
    this.group.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    });
  }
}

// ── Public API ───────────────────────────────────────────────

/**
 * Create the SMPL body mesh, add it to the scene, and wire up the bus listener.
 * @param {THREE.Scene} scene
 * @returns {{ group: THREE.Group, renderer: SMPLMeshRenderer, dispose: () => void }}
 */
export function createBodyMesh(scene) {
  const meshRenderer = new SMPLMeshRenderer();
  scene.add(meshRenderer.group);

  // Start hidden — only show when real data or explicit demo mode
  meshRenderer.group.visible = false;
  let demoActive = false;

  function onPose(data) {
    if (!data || !data.joints) return;
    // Show mesh on first real data
    if (!meshRenderer.group.visible) {
      meshRenderer.group.visible = true;
    }
    if (demoActive) {
      meshRenderer.stopDemo();
      demoActive = false;
    }
    meshRenderer.update(data.joints);

    // Color each body chain by its average joint confidence
    const jc = data.joint_confidence;
    if (jc && jc.length === 24) {
      meshRenderer.applyConfidenceColors(jc);
    }
  }
  bus.on('pose', onPose);

  function dispose() {
    bus.off('pose', onPose);
    scene.remove(meshRenderer.group);
    meshRenderer.dispose();
  }

  return { group: meshRenderer.group, renderer: meshRenderer, dispose };
}
