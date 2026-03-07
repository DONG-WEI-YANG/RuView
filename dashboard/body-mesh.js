/**
 * body-mesh.js — 3D Human Body Surface Mesh with UV Textures
 *
 * Generates a smooth, SMPL-like body mesh from 24-joint skeleton data.
 * Each body segment is an elliptical tube with tapered radii, connected
 * by sphere joints for seamless transitions. A procedural skin texture
 * is UV-mapped across the entire surface.
 */
(function () {
    "use strict";

    // ─── Geometry resolution ──────────────────────────────────
    var RADIAL = 16;   // vertices around each cross-section ring
    var AXIAL  = 6;    // rings along each segment

    // ─── Body segment definitions ─────────────────────────────
    // f/t  = from-joint / to-joint index
    // r0   = [radiusX, radiusZ] at the "from" end (elliptical cross-section)
    // r1   = [radiusX, radiusZ] at the "to" end
    //
    // The skeleton tree (from BONES in skeleton3d.js):
    //   Head chain : 0→1→2→3       (head→neck→chest→spine)
    //   L arm      : 3→4→5→6→20   (spine→Lshoulder→Lelbow→Lwrist→Lhand)
    //   R arm      : 3→7→8→9→21
    //   Core       : 3→10→11      (spine→pelvis→hip-center)
    //   L leg      : 11→12→13→14→18
    //   R leg      : 11→15→16→17→19
    var SEGS = [
        // ── Torso ──────────────────────────────────────────────
        { f: 1,  t: 0,  r0: [0.050, 0.040], r1: [0.042, 0.036] },  // neck
        { f: 2,  t: 1,  r0: [0.155, 0.095], r1: [0.058, 0.048] },  // upper chest
        { f: 3,  t: 2,  r0: [0.125, 0.085], r1: [0.150, 0.095] },  // lower chest
        { f: 10, t: 3,  r0: [0.115, 0.078], r1: [0.125, 0.085] },  // abdomen
        { f: 11, t: 10, r0: [0.138, 0.095], r1: [0.115, 0.078] },  // pelvis

        // ── Left arm ───────────────────────────────────────────
        { f: 3, t: 4,  r0: [0.055, 0.050], r1: [0.048, 0.045] },   // L clavicle
        { f: 4, t: 5,  r0: [0.048, 0.042], r1: [0.036, 0.032] },   // L upper arm
        { f: 5, t: 6,  r0: [0.036, 0.032], r1: [0.026, 0.022] },   // L forearm

        // ── Right arm ──────────────────────────────────────────
        { f: 3, t: 7,  r0: [0.055, 0.050], r1: [0.048, 0.045] },   // R clavicle
        { f: 7, t: 8,  r0: [0.048, 0.042], r1: [0.036, 0.032] },   // R upper arm
        { f: 8, t: 9,  r0: [0.036, 0.032], r1: [0.026, 0.022] },   // R forearm

        // ── Left leg ───────────────────────────────────────────
        { f: 11, t: 12, r0: [0.062, 0.058], r1: [0.068, 0.062] },  // L hip conn
        { f: 12, t: 13, r0: [0.072, 0.065], r1: [0.048, 0.042] },  // L thigh
        { f: 13, t: 14, r0: [0.048, 0.040], r1: [0.036, 0.030] },  // L shin

        // ── Right leg ──────────────────────────────────────────
        { f: 11, t: 15, r0: [0.062, 0.058], r1: [0.068, 0.062] },  // R hip conn
        { f: 15, t: 16, r0: [0.072, 0.065], r1: [0.048, 0.042] },  // R thigh
        { f: 16, t: 17, r0: [0.048, 0.040], r1: [0.036, 0.030] },  // R shin
    ];

    // Joint spheres: index → radius (for smooth transitions between tubes)
    var JOINT_SPHERES = {
         1: 0.050,  // neck
         2: 0.085,  // chest (shoulder line)
         3: 0.075,  // spine
         4: 0.046,  // L shoulder
         5: 0.036,  // L elbow
         7: 0.046,  // R shoulder
         8: 0.036,  // R elbow
        10: 0.072,  // pelvis
        11: 0.080,  // hip center
        12: 0.063,  // L hip
        13: 0.046,  // L knee
        15: 0.063,  // R hip
        16: 0.046,  // R knee
    };

    // ─── Procedural skin texture ──────────────────────────────
    function createSkinTexture() {
        var sz = 256;
        var c  = document.createElement("canvas");
        c.width = sz; c.height = sz;
        var ctx = c.getContext("2d");

        // Warm skin gradient
        var g = ctx.createLinearGradient(0, 0, 0, sz);
        g.addColorStop(0.0, "#e2b59b");
        g.addColorStop(0.3, "#d4a088");
        g.addColorStop(0.6, "#c4917a");
        g.addColorStop(1.0, "#b07a66");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, sz, sz);

        // Subtle cross-hatched variation for pore-like texture
        var img = ctx.getImageData(0, 0, sz, sz);
        var d = img.data;
        for (var i = 0; i < d.length; i += 4) {
            var n = (Math.random() - 0.5) * 14;
            d[i]     = clamp8(d[i]     + n);
            d[i + 1] = clamp8(d[i + 1] + n * 0.75);
            d[i + 2] = clamp8(d[i + 2] + n * 0.55);
        }
        ctx.putImageData(img, 0, 0);

        var tex = new THREE.CanvasTexture(c);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        return tex;
    }

    function clamp8(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

    // ─── Elliptical tube geometry ─────────────────────────────
    // Creates a unit-height (y: 0→1) tube with elliptical cross-sections
    // tapering from (r0x, r0z) at y=0 to (r1x, r1z) at y=1.
    function createTubeGeometry(r0x, r0z, r1x, r1z) {
        var positions = [];
        var normals   = [];
        var uvs       = [];
        var indices   = [];

        for (var j = 0; j <= AXIAL; j++) {
            var t  = j / AXIAL;
            var rx = r0x + (r1x - r0x) * t;
            var rz = r0z + (r1z - r0z) * t;

            for (var i = 0; i <= RADIAL; i++) {
                var a   = (i / RADIAL) * Math.PI * 2;
                var cos = Math.cos(a);
                var sin = Math.sin(a);

                positions.push(cos * rx, t, sin * rz);

                // Ellipse outward normal: (cos/rx, 0, sin/rz) normalised
                var nx = rx > 0.0001 ? cos / rx : 0;
                var nz = rz > 0.0001 ? sin / rz : 0;
                var nl = Math.sqrt(nx * nx + nz * nz) || 1;
                normals.push(nx / nl, 0, nz / nl);

                uvs.push(i / RADIAL, t);
            }
        }

        for (var j = 0; j < AXIAL; j++) {
            for (var i = 0; i < RADIAL; i++) {
                var a = j * (RADIAL + 1) + i;
                var b = a + 1;
                var c = a + (RADIAL + 1);
                var d = c + 1;
                indices.push(a, c, b,  b, c, d);
            }
        }

        var geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        geom.setAttribute("normal",   new THREE.Float32BufferAttribute(normals, 3));
        geom.setAttribute("uv",       new THREE.Float32BufferAttribute(uvs, 2));
        geom.setIndex(indices);
        return geom;
    }

    // ─── Orientation helper ───────────────────────────────────
    var _up   = new THREE.Vector3(0, 1, 0);
    var _dir  = new THREE.Vector3();
    var _quat = new THREE.Quaternion();

    function orientBetween(mesh, fromPos, toPos) {
        _dir.set(
            toPos[0] - fromPos[0],
            toPos[1] - fromPos[1],
            toPos[2] - fromPos[2]
        );
        var len = _dir.length();
        if (len < 0.001) { mesh.visible = false; return; }
        mesh.visible = true;
        _dir.divideScalar(len);

        mesh.position.set(
            (fromPos[0] + toPos[0]) * 0.5,
            (fromPos[1] + toPos[1]) * 0.5,
            (fromPos[2] + toPos[2]) * 0.5
        );

        if (Math.abs(_dir.dot(_up)) < 0.9999) {
            _quat.setFromUnitVectors(_up, _dir);
            mesh.quaternion.copy(_quat);
        } else {
            mesh.quaternion.set(0, 0, 0, 1);
        }

        mesh.scale.y = len;
    }

    // ─── Hand / foot cap geometry ─────────────────────────────
    function createCapGeom(rx, rz) {
        var geom = new THREE.SphereGeometry(1, 10, 8);
        geom.scale(rx, rx * 0.7, rz);
        return geom;
    }

    // ─── BodyMeshRenderer ─────────────────────────────────────
    function BodyMeshRenderer() {
        this.group    = new THREE.Group();
        this.group.name = "body-mesh";
        this.segments = [];
        this.spheres  = [];
        this.headMesh = null;
        this.caps     = [];

        var tex = createSkinTexture();
        this._baseMaterial = new THREE.MeshPhongMaterial({
            map: tex,
            color: 0xddaa88,
            emissive: 0x221100,
            emissiveIntensity: 0.08,
            shininess: 25,
            side: THREE.DoubleSide,
        });

        this._build();
        this.group.visible = false;
    }

    BodyMeshRenderer.prototype._build = function () {
        var mat = this._baseMaterial;
        var i, s, geom, mesh;

        // Tube segments
        for (i = 0; i < SEGS.length; i++) {
            s = SEGS[i];
            geom = createTubeGeometry(s.r0[0], s.r0[1], s.r1[0], s.r1[1]);
            mesh = new THREE.Mesh(geom, mat.clone());
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.group.add(mesh);
            this.segments.push({ mesh: mesh, f: s.f, t: s.t });
        }

        // Head ellipsoid
        var headGeom = new THREE.SphereGeometry(0.095, 20, 16);
        this.headMesh = new THREE.Mesh(headGeom, mat.clone());
        this.headMesh.scale.set(1.0, 1.2, 0.92);
        this.headMesh.castShadow = true;
        this.group.add(this.headMesh);

        // Joint spheres
        var sphereGeom8  = new THREE.SphereGeometry(1, 12, 10);
        var keys = Object.keys(JOINT_SPHERES);
        for (i = 0; i < keys.length; i++) {
            var ji = parseInt(keys[i], 10);
            var r  = JOINT_SPHERES[ji];
            mesh = new THREE.Mesh(sphereGeom8, mat.clone());
            mesh.scale.set(r, r, r);
            mesh.castShadow = true;
            this.group.add(mesh);
            this.spheres.push({ mesh: mesh, joint: ji });
        }

        // End caps: hands (6, 9 / 18, 19) and feet (14, 17 / 20, 21)
        var capDefs = [
            { j: 6,  rx: 0.028, rz: 0.018 },   // L wrist cap
            { j: 9,  rx: 0.028, rz: 0.018 },   // R wrist cap
            { j: 14, rx: 0.038, rz: 0.032 },   // L ankle cap
            { j: 17, rx: 0.038, rz: 0.032 },   // R ankle cap
        ];
        for (i = 0; i < capDefs.length; i++) {
            var cd = capDefs[i];
            mesh = new THREE.Mesh(createCapGeom(cd.rx, cd.rz), mat.clone());
            mesh.castShadow = true;
            this.group.add(mesh);
            this.caps.push({ mesh: mesh, joint: cd.j });
        }
    };

    BodyMeshRenderer.prototype.update = function (joints) {
        if (!joints || joints.length < 18) return;
        this.group.visible = true;

        // Head
        var h = joints[0];
        this.headMesh.position.set(h[0], h[1], h[2]);

        // Tube segments
        for (var i = 0; i < this.segments.length; i++) {
            var seg = this.segments[i];
            orientBetween(seg.mesh, joints[seg.f], joints[seg.t]);
        }

        // Joint spheres
        for (var i = 0; i < this.spheres.length; i++) {
            var sp = this.spheres[i];
            var jp = joints[sp.joint];
            sp.mesh.position.set(jp[0], jp[1], jp[2]);
        }

        // End caps
        for (var i = 0; i < this.caps.length; i++) {
            var cp = this.caps[i];
            var jp = joints[cp.joint];
            cp.mesh.position.set(jp[0], jp[1], jp[2]);
        }
    };

    BodyMeshRenderer.prototype.setVisible = function (v) {
        this.group.visible = v;
    };

    BodyMeshRenderer.prototype.dispose = function () {
        this.group.traverse(function (child) {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
    };

    window.BodyMeshRenderer = BodyMeshRenderer;
})();
