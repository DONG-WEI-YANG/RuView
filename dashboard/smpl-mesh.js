/**
 * smpl-mesh.js — SMPL-style 3D Human Body Surface Mesh
 *
 * Generates a smooth, continuous body surface from 24-joint skeleton data.
 * Each body part is built from elliptical cross-section rings along the
 * skeleton path. Vertices are recomputed each frame for pose-driven
 * deformation (software Linear Blend Skinning).
 */
(function () {
    "use strict";

    var RAD = 20;
    var COS = [], SIN = [];
    for (var i = 0; i <= RAD; i++) {
        var a = (i / RAD) * Math.PI * 2;
        COS.push(Math.cos(a));
        SIN.push(Math.sin(a));
    }

    // ═══ T-pose joint positions (meters, Y-up) ═══════════════
    // Matches BONES tree in skeleton3d.js:
    //   0:head 1:neck 2:chest 3:spine
    //   4:Lshoulder 5:Lelbow 6:Lwrist
    //   7:Rshoulder 8:Relbow 9:Rwrist
    //   10:pelvis 11:hipCenter
    //   12:Lhip 13:Lknee 14:Lankle
    //   15:Rhip 16:Rknee 17:Rankle
    //   18:Lfoot 19:Rfoot 20:Lhand 21:Rhand 22:Leye 23:Reye
    var REST = [
        [ 0.000, 1.700, 0.000],  //  0 head
        [ 0.000, 1.550, 0.000],  //  1 neck
        [ 0.000, 1.380, 0.000],  //  2 chest
        [ 0.000, 1.120, 0.000],  //  3 spine
        [-0.200, 1.400, 0.000],  //  4 L shoulder
        [-0.480, 1.400, 0.000],  //  5 L elbow
        [-0.700, 1.400, 0.000],  //  6 L wrist
        [ 0.200, 1.400, 0.000],  //  7 R shoulder
        [ 0.480, 1.400, 0.000],  //  8 R elbow
        [ 0.700, 1.400, 0.000],  //  9 R wrist
        [ 0.000, 0.950, 0.000],  // 10 pelvis
        [ 0.000, 0.900, 0.000],  // 11 hip center
        [-0.100, 0.880, 0.000],  // 12 L hip
        [-0.100, 0.500, 0.000],  // 13 L knee
        [-0.100, 0.080, 0.000],  // 14 L ankle
        [ 0.100, 0.880, 0.000],  // 15 R hip
        [ 0.100, 0.500, 0.000],  // 16 R knee
        [ 0.100, 0.080, 0.000],  // 17 R ankle
        [-0.100, 0.030, 0.080],  // 18 L foot
        [ 0.100, 0.030, 0.080],  // 19 R foot
        [-0.780, 1.400, 0.000],  // 20 L hand
        [ 0.780, 1.400, 0.000],  // 21 R hand
        [-0.030, 1.720, 0.060],  // 22 L eye
        [ 0.030, 1.720, 0.060],  // 23 R eye
    ];

    // ═══ Body chain definitions ══════════════════════════════
    // joints:  indices into the 24-joint array (path order)
    // jointT:  parameter t at each joint along the chain [0..1]
    // up:      preferred "up" vector for computing the local frame
    // rings:   cross-section profiles {t, rx, rz}
    var CHAINS = [
        // ── Torso ────────────────────────────────────────────
        {
            name: "torso",
            joints: [11, 10, 3, 2, 1],
            jointT: [0, 0.18, 0.52, 0.76, 1.0],
            up: [0, 0, 1],
            rings: [
                { t: 0.000, rx: 0.148, rz: 0.105 },
                { t: 0.060, rx: 0.142, rz: 0.100 },
                { t: 0.120, rx: 0.132, rz: 0.095 },
                { t: 0.180, rx: 0.120, rz: 0.088 },
                { t: 0.250, rx: 0.108, rz: 0.078 },
                { t: 0.320, rx: 0.112, rz: 0.082 },
                { t: 0.400, rx: 0.128, rz: 0.090 },
                { t: 0.480, rx: 0.142, rz: 0.098 },
                { t: 0.520, rx: 0.150, rz: 0.100 },
                { t: 0.580, rx: 0.158, rz: 0.108 },
                { t: 0.640, rx: 0.162, rz: 0.110 },
                { t: 0.700, rx: 0.160, rz: 0.108 },
                { t: 0.760, rx: 0.155, rz: 0.100 },
                { t: 0.800, rx: 0.138, rz: 0.088 },
                { t: 0.830, rx: 0.110, rz: 0.072 },
                { t: 0.860, rx: 0.078, rz: 0.055 },
                { t: 0.900, rx: 0.054, rz: 0.044 },
                { t: 0.950, rx: 0.048, rz: 0.040 },
                { t: 1.000, rx: 0.044, rz: 0.038 },
            ],
        },
        // ── Left arm ─────────────────────────────────────────
        {
            name: "l_arm",
            joints: [4, 5, 6],
            jointT: [0, 0.50, 1.0],
            up: [0, 0, 1],
            rings: [
                { t: 0.00, rx: 0.052, rz: 0.048 },
                { t: 0.10, rx: 0.050, rz: 0.046 },
                { t: 0.25, rx: 0.046, rz: 0.042 },
                { t: 0.40, rx: 0.042, rz: 0.038 },
                { t: 0.50, rx: 0.040, rz: 0.036 },
                { t: 0.55, rx: 0.038, rz: 0.034 },
                { t: 0.65, rx: 0.035, rz: 0.031 },
                { t: 0.80, rx: 0.030, rz: 0.027 },
                { t: 0.90, rx: 0.027, rz: 0.024 },
                { t: 1.00, rx: 0.025, rz: 0.021 },
            ],
        },
        // ── Right arm ────────────────────────────────────────
        {
            name: "r_arm",
            joints: [7, 8, 9],
            jointT: [0, 0.50, 1.0],
            up: [0, 0, 1],
            rings: [
                { t: 0.00, rx: 0.052, rz: 0.048 },
                { t: 0.10, rx: 0.050, rz: 0.046 },
                { t: 0.25, rx: 0.046, rz: 0.042 },
                { t: 0.40, rx: 0.042, rz: 0.038 },
                { t: 0.50, rx: 0.040, rz: 0.036 },
                { t: 0.55, rx: 0.038, rz: 0.034 },
                { t: 0.65, rx: 0.035, rz: 0.031 },
                { t: 0.80, rx: 0.030, rz: 0.027 },
                { t: 0.90, rx: 0.027, rz: 0.024 },
                { t: 1.00, rx: 0.025, rz: 0.021 },
            ],
        },
        // ── Left leg ─────────────────────────────────────────
        {
            name: "l_leg",
            joints: [12, 13, 14],
            jointT: [0, 0.52, 1.0],
            up: [0, 0, 1],
            rings: [
                { t: 0.00, rx: 0.072, rz: 0.068 },
                { t: 0.08, rx: 0.074, rz: 0.070 },
                { t: 0.18, rx: 0.072, rz: 0.066 },
                { t: 0.30, rx: 0.065, rz: 0.058 },
                { t: 0.42, rx: 0.055, rz: 0.050 },
                { t: 0.52, rx: 0.048, rz: 0.044 },
                { t: 0.58, rx: 0.046, rz: 0.042 },
                { t: 0.70, rx: 0.042, rz: 0.038 },
                { t: 0.82, rx: 0.038, rz: 0.034 },
                { t: 0.92, rx: 0.036, rz: 0.032 },
                { t: 1.00, rx: 0.034, rz: 0.030 },
            ],
        },
        // ── Right leg ────────────────────────────────────────
        {
            name: "r_leg",
            joints: [15, 16, 17],
            jointT: [0, 0.52, 1.0],
            up: [0, 0, 1],
            rings: [
                { t: 0.00, rx: 0.072, rz: 0.068 },
                { t: 0.08, rx: 0.074, rz: 0.070 },
                { t: 0.18, rx: 0.072, rz: 0.066 },
                { t: 0.30, rx: 0.065, rz: 0.058 },
                { t: 0.42, rx: 0.055, rz: 0.050 },
                { t: 0.52, rx: 0.048, rz: 0.044 },
                { t: 0.58, rx: 0.046, rz: 0.042 },
                { t: 0.70, rx: 0.042, rz: 0.038 },
                { t: 0.82, rx: 0.038, rz: 0.034 },
                { t: 0.92, rx: 0.036, rz: 0.032 },
                { t: 1.00, rx: 0.034, rz: 0.030 },
            ],
        },
        // Connectors removed — joint spheres handle torso-limb transitions
    ];

    // Joint spheres for smooth transitions
    var JSPHERES = [
        { idx:  1, r: 0.046 },
        { idx:  2, r: 0.055 },
        { idx:  3, r: 0.068 },
        { idx:  4, r: 0.050 },
        { idx:  5, r: 0.038 },
        { idx:  7, r: 0.050 },
        { idx:  8, r: 0.038 },
        { idx: 10, r: 0.068 },
        { idx: 11, r: 0.075 },
        { idx: 12, r: 0.068 },
        { idx: 13, r: 0.046 },
        { idx: 15, r: 0.068 },
        { idx: 16, r: 0.046 },
    ];

    // ═══ Vector math ═════════════════════════════════════════
    function v3sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
    function v3lerp(a, b, t) {
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    }
    function v3cross(a, b) {
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    }
    function v3len(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
    function v3norm(v) {
        var l = v3len(v);
        return l > 1e-7 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 1, 0];
    }

    // ═══ Path interpolation along a joint chain ══════════════
    function chainPos(jts, jt, t) {
        t = Math.max(0, Math.min(1, t));
        for (var i = 0; i < jt.length - 1; i++) {
            if (t <= jt[i + 1] + 1e-6) {
                var s = jt[i + 1] - jt[i];
                var lt = s > 1e-6 ? (t - jt[i]) / s : 0;
                return v3lerp(jts[i], jts[i + 1], Math.max(0, Math.min(1, lt)));
            }
        }
        return jts[jts.length - 1];
    }

    function chainTangent(jts, jt, t) {
        var eps = 0.008;
        return v3norm(v3sub(
            chainPos(jts, jt, Math.min(1, t + eps)),
            chainPos(jts, jt, Math.max(0, t - eps))
        ));
    }

    // ═══ Local coordinate frame at a point on the chain ══════
    function localFrame(tangent, prefUp) {
        var bn = v3cross(tangent, prefUp);
        if (v3len(bn) < 0.01) {
            bn = v3cross(tangent, [1, 0, 0]);
            if (v3len(bn) < 0.01) bn = v3cross(tangent, [0, 0, 1]);
        }
        bn = v3norm(bn);
        var nm = v3norm(v3cross(bn, tangent));
        return { b: bn, n: nm };
    }

    // ═══ Fill vertex buffers for a chain ═════════════════════
    function fillVerts(posArr, norArr, rings, jts, jt, up) {
        var vi = 0, ni = 0;
        for (var r = 0; r < rings.length; r++) {
            var ring = rings[r];
            var center = chainPos(jts, jt, ring.t);
            var tan = chainTangent(jts, jt, ring.t);
            var fr = localFrame(tan, up);
            var bn = fr.b, nm = fr.n;

            for (var i = 0; i <= RAD; i++) {
                var c = COS[i], s = SIN[i];
                posArr[vi]     = center[0] + c * ring.rx * bn[0] + s * ring.rz * nm[0];
                posArr[vi + 1] = center[1] + c * ring.rx * bn[1] + s * ring.rz * nm[1];
                posArr[vi + 2] = center[2] + c * ring.rx * bn[2] + s * ring.rz * nm[2];

                var enx = c / (ring.rx > 0.001 ? ring.rx : 0.001);
                var enz = s / (ring.rz > 0.001 ? ring.rz : 0.001);
                var el = Math.sqrt(enx * enx + enz * enz) || 1;
                norArr[ni]     = (enx * bn[0] + enz * nm[0]) / el;
                norArr[ni + 1] = (enx * bn[1] + enz * nm[1]) / el;
                norArr[ni + 2] = (enx * bn[2] + enz * nm[2]) / el;

                vi += 3;
                ni += 3;
            }
        }
    }

    // ═══ Build geometry for one chain ════════════════════════
    function buildChainGeom(chain, jointPositions) {
        var jts = chain.joints.map(function (i) { return jointPositions[i]; });
        var rings = chain.rings;
        var nR = rings.length;
        var nV = nR * (RAD + 1);

        var pos = new Float32Array(nV * 3);
        var nor = new Float32Array(nV * 3);
        var uv  = new Float32Array(nV * 2);

        fillVerts(pos, nor, rings, jts, chain.jointT, chain.up);

        // UVs (static — don't change per frame)
        var ui = 0;
        for (var r = 0; r < nR; r++) {
            for (var i = 0; i <= RAD; i++) {
                uv[ui++] = i / RAD;
                uv[ui++] = rings[r].t;
            }
        }

        // Indices
        var nFaces = (nR - 1) * RAD * 2;
        var idx = new Uint16Array(nFaces * 3);
        var ii = 0;
        for (var r = 0; r < nR - 1; r++) {
            for (var i = 0; i < RAD; i++) {
                var a = r * (RAD + 1) + i;
                var b = a + 1;
                var cc = a + (RAD + 1);
                var d = cc + 1;
                idx[ii++] = a; idx[ii++] = cc; idx[ii++] = b;
                idx[ii++] = b; idx[ii++] = cc; idx[ii++] = d;
            }
        }

        var geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        geom.setAttribute("normal",   new THREE.BufferAttribute(nor, 3));
        geom.setAttribute("uv",       new THREE.BufferAttribute(uv, 2));
        geom.setIndex(new THREE.BufferAttribute(idx, 1));
        return geom;
    }

    // ═══ Procedural skin texture ═════════════════════════════
    function createBodyTexture() {
        var sz = 512, c = document.createElement("canvas");
        c.width = sz; c.height = sz;
        var ctx = c.getContext("2d");

        // Base skin gradient
        var g = ctx.createLinearGradient(0, 0, 0, sz);
        g.addColorStop(0.00, "#e8c4a8");
        g.addColorStop(0.25, "#ddb69a");
        g.addColorStop(0.50, "#d0a58a");
        g.addColorStop(0.75, "#c4957c");
        g.addColorStop(1.00, "#b28570");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, sz, sz);

        // Subtle horizontal bands (muscle definition hint)
        ctx.globalAlpha = 0.04;
        for (var y = 0; y < sz; y += 16) {
            ctx.fillStyle = y % 32 === 0 ? "#000" : "#fff";
            ctx.fillRect(0, y, sz, 8);
        }
        ctx.globalAlpha = 1.0;

        // Per-pixel noise
        var img = ctx.getImageData(0, 0, sz, sz);
        var d = img.data;
        for (var i = 0; i < d.length; i += 4) {
            var n = (Math.random() - 0.5) * 16;
            d[i]     = clamp8(d[i]     + n);
            d[i + 1] = clamp8(d[i + 1] + n * 0.7);
            d[i + 2] = clamp8(d[i + 2] + n * 0.5);
        }
        ctx.putImageData(img, 0, 0);

        var tex = new THREE.CanvasTexture(c);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        return tex;
    }

    function clamp8(v) { return v < 0 ? 0 : v > 255 ? 255 : v | 0; }

    // ═══ SMPLMeshRenderer ════════════════════════════════════
    function SMPLMeshRenderer() {
        this.group = new THREE.Group();
        this.group.name = "smpl-body";
        this._chainData = [];
        this._spheres = [];
        this._demoRaf = 0;
        this.headMesh = null;
        this._glbMesh = null;
        this._build();
    }

    SMPLMeshRenderer.prototype._build = function () {
        // ── Wireframe mesh style (matching Wi-Mesh / 3D activity detection) ──
        // Primary wireframe: bright cyan/green edges
        var wireMat = new THREE.MeshBasicMaterial({
            color: 0x00ffaa,
            wireframe: true,
            transparent: true,
            opacity: 0.85,
        });
        // Glow layer: slightly larger, softer edges for halo effect
        var glowMat = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            wireframe: true,
            transparent: true,
            opacity: 0.15,
        });
        this._material = wireMat;

        // Build chain meshes (wireframe + glow double layer)
        for (var ci = 0; ci < CHAINS.length; ci++) {
            var ch = CHAINS[ci];
            var geom = buildChainGeom(ch, REST);

            // Primary wireframe mesh
            var mesh = new THREE.Mesh(geom, wireMat.clone());
            mesh.frustumCulled = false;
            this.group.add(mesh);

            // Glow halo mesh (slightly scaled up)
            var glowGeom = buildChainGeom(ch, REST);
            var glowMesh = new THREE.Mesh(glowGeom, glowMat.clone());
            glowMesh.scale.set(1.03, 1.0, 1.03);
            glowMesh.frustumCulled = false;
            this.group.add(glowMesh);

            this._chainData.push({
                def: ch,
                mesh: mesh,
                geom: geom,
                posAttr: geom.getAttribute("position"),
                norAttr: geom.getAttribute("normal"),
                glowMesh: glowMesh,
                glowGeom: glowGeom,
                glowPosAttr: glowGeom.getAttribute("position"),
                glowNorAttr: glowGeom.getAttribute("normal"),
            });
        }

        // Head ellipsoid (wireframe sphere)
        var headGeom = new THREE.SphereGeometry(0.098, 16, 12);
        this.headMesh = new THREE.Mesh(headGeom, wireMat.clone());
        this.headMesh.scale.set(1.0, 1.18, 0.92);
        this.group.add(this.headMesh);

        // Head glow
        var headGlow = new THREE.Mesh(
            new THREE.SphereGeometry(0.104, 16, 12),
            glowMat.clone()
        );
        headGlow.scale.set(1.0, 1.18, 0.92);
        this.group.add(headGlow);
        this._headGlow = headGlow;

        // Joint spheres (wireframe)
        var sphGeom = new THREE.SphereGeometry(1, 8, 6);
        for (var si = 0; si < JSPHERES.length; si++) {
            var jd = JSPHERES[si];
            var sm = new THREE.Mesh(sphGeom, wireMat.clone());
            sm.scale.set(jd.r, jd.r, jd.r);
            this.group.add(sm);
            this._spheres.push({ mesh: sm, idx: jd.idx });
        }

        // End caps (wrists, ankles)
        var caps = [
            { j: 6,  r: 0.024 }, { j: 9,  r: 0.024 },
            { j: 14, r: 0.033 }, { j: 17, r: 0.033 },
        ];
        for (var ci2 = 0; ci2 < caps.length; ci2++) {
            var cd = caps[ci2];
            var cm = new THREE.Mesh(sphGeom, wireMat.clone());
            cm.scale.set(cd.r, cd.r * 0.7, cd.r);
            this.group.add(cm);
            this._spheres.push({ mesh: cm, idx: cd.j });
        }

        this.group.visible = false;
    };

    // ═══ Per-frame update ════════════════════════════════════
    SMPLMeshRenderer.prototype.update = function (joints) {
        if (!joints || joints.length < 18) return;
        this.group.visible = true;

        // Update chain meshes (software skinning: recompute vertex positions)
        for (var ci = 0; ci < this._chainData.length; ci++) {
            var cd = this._chainData[ci];
            var ch = cd.def;
            var jts = [];
            for (var ji = 0; ji < ch.joints.length; ji++) {
                jts.push(joints[ch.joints[ji]]);
            }
            // Primary wireframe
            fillVerts(cd.posAttr.array, cd.norAttr.array, ch.rings, jts, ch.jointT, ch.up);
            cd.posAttr.needsUpdate = true;
            cd.norAttr.needsUpdate = true;
            cd.geom.boundingSphere = null;
            // Glow layer
            if (cd.glowPosAttr) {
                fillVerts(cd.glowPosAttr.array, cd.glowNorAttr.array, ch.rings, jts, ch.jointT, ch.up);
                cd.glowPosAttr.needsUpdate = true;
                cd.glowNorAttr.needsUpdate = true;
                cd.glowGeom.boundingSphere = null;
            }
        }

        // Head
        var h = joints[0];
        this.headMesh.position.set(h[0], h[1], h[2]);
        if (this._headGlow) this._headGlow.position.set(h[0], h[1], h[2]);

        // Joint spheres + caps
        for (var si = 0; si < this._spheres.length; si++) {
            var sp = this._spheres[si];
            var jp = joints[sp.idx];
            sp.mesh.position.set(jp[0], jp[1], jp[2]);
        }
    };

    // ═══ Demo mode: animated T-pose ══════════════════════════
    SMPLMeshRenderer.prototype.showDemo = function () {
        var self = this;
        var t = 0;

        function tick() {
            t += 0.016;
            var joints = [];
            for (var i = 0; i < 24; i++) {
                var r = REST[i];
                joints.push([r[0], r[1], r[2]]);
            }

            // Arms relaxed at sides (not T-pose) — much more natural
            var armSwing = Math.sin(t * 0.6) * 0.03;
            // L arm: shoulder(4), elbow(5), wrist(6)
            joints[4]  = [-0.18, 1.36, 0];
            joints[5]  = [-0.22, 1.12 + armSwing, 0.02];
            joints[6]  = [-0.20, 0.90 + armSwing * 1.4, 0.04];
            joints[20] = [-0.19, 0.82 + armSwing * 1.6, 0.05];
            // R arm: shoulder(7), elbow(8), wrist(9)
            joints[7]  = [0.18, 1.36, 0];
            joints[8]  = [0.22, 1.12 - armSwing, 0.02];
            joints[9]  = [0.20, 0.90 - armSwing * 1.4, 0.04];
            joints[21] = [0.19, 0.82 - armSwing * 1.6, 0.05];

            // Subtle idle sway
            var sway = Math.sin(t * 0.4) * 0.008;
            for (var i = 0; i < 24; i++) {
                joints[i][0] += sway;
                joints[i][2] += Math.sin(t * 0.3 + i * 0.1) * 0.002;
            }

            // Breathing
            var breath = Math.sin(t * 0.5) * 0.008;
            joints[2][2] += breath;
            joints[3][2] += breath * 0.5;

            // Slight knee bend
            var kneeBend = Math.sin(t * 0.35) * 0.008;
            joints[13][1] += kneeBend;
            joints[16][1] += kneeBend;

            self.update(joints);
            self._demoRaf = requestAnimationFrame(tick);
        }

        tick();
    };

    SMPLMeshRenderer.prototype.stopDemo = function () {
        if (this._demoRaf) {
            cancelAnimationFrame(this._demoRaf);
            this._demoRaf = 0;
        }
    };

    // ═══ GLB model loader (optional upgrade) ═════════════════
    // If a .glb body model exists, load it and replace procedural mesh.
    // The GLB must have a SkinnedMesh with bones matching the 24 joints.
    SMPLMeshRenderer.prototype.loadGLB = function (url, onDone) {
        if (typeof THREE.GLTFLoader === "undefined") {
            console.warn("GLTFLoader not available; using procedural mesh.");
            if (onDone) onDone(false);
            return;
        }
        var self = this;
        var loader = new THREE.GLTFLoader();
        loader.load(
            url,
            function (gltf) {
                // Hide procedural mesh
                self._chainData.forEach(function (cd) { cd.mesh.visible = false; });
                self._spheres.forEach(function (sp) { sp.mesh.visible = false; });
                self.headMesh.visible = false;

                // Add loaded model
                self._glbMesh = gltf.scene;
                self._glbMesh.traverse(function (child) {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                self.group.add(self._glbMesh);

                // Store skeleton reference for animation
                gltf.scene.traverse(function (child) {
                    if (child.isSkinnedMesh) {
                        self._skinnedMesh = child;
                        self._skeleton = child.skeleton;
                    }
                });

                if (onDone) onDone(true);
            },
            undefined,
            function () {
                console.warn("GLB load failed; using procedural mesh.");
                if (onDone) onDone(false);
            }
        );
    };

    SMPLMeshRenderer.prototype.setVisible = function (v) {
        this.group.visible = v;
    };

    SMPLMeshRenderer.prototype.dispose = function () {
        this.stopDemo();
        this.group.traverse(function (child) {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
    };

    window.SMPLMeshRenderer = SMPLMeshRenderer;
})();
