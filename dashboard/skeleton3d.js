(function () {
    "use strict";

    // ─── Joint connectivity (24-joint skeleton) ───────────────
    var BONES = [
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

    // ─── Render mode: "mesh" | "skeleton" | "both" ────────────
    var renderMode = "mesh";

    // ─── Scene ────────────────────────────────────────────────
    var canvas = document.getElementById("skeleton-canvas");
    var scene  = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    var camera = new THREE.PerspectiveCamera(
        55, canvas.clientWidth / canvas.clientHeight, 0.1, 100
    );
    camera.position.set(1.5, 1.0, 4.0);
    camera.lookAt(0, 0.80, 0);

    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // ─── OrbitControls ────────────────────────────────────────
    var controls = null;
    if (typeof THREE.OrbitControls === "function") {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.85, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 1.0;
        controls.maxDistance = 8.0;
        controls.update();
    }

    // ─── Lighting ─────────────────────────────────────────────
    // Ambient fill
    scene.add(new THREE.AmbientLight(0x889abb, 0.7));

    // Key light (warm directional from upper-right-front)
    var keyLight = new THREE.DirectionalLight(0xffeedd, 1.1);
    keyLight.position.set(3, 6, 4);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width  = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far  = 20;
    keyLight.shadow.camera.left   = -3;
    keyLight.shadow.camera.right  =  3;
    keyLight.shadow.camera.top    =  3;
    keyLight.shadow.camera.bottom = -1;
    scene.add(keyLight);

    // Fill light (cool from left)
    var fillLight = new THREE.DirectionalLight(0x99aadd, 0.5);
    fillLight.position.set(-4, 3, 2);
    scene.add(fillLight);

    // Hemisphere sky/ground
    scene.add(new THREE.HemisphereLight(0x8899bb, 0x333344, 0.3));

    // Rim/back light for depth separation
    var rimLight = new THREE.DirectionalLight(0xaaccff, 0.25);
    rimLight.position.set(0, 2, -5);
    scene.add(rimLight);

    // ─── Ground ───────────────────────────────────────────────
    var grid = new THREE.GridHelper(6, 12, 0x444466, 0x33334a);
    scene.add(grid);

    // Shadow-receiving floor plane
    var floorGeom = new THREE.PlaneGeometry(6, 6);
    var floorMat  = new THREE.MeshPhongMaterial({
        color: 0x1a1a2e, depthWrite: false, transparent: true, opacity: 0.6,
    });
    var floor = new THREE.Mesh(floorGeom, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.005;
    floor.receiveShadow = true;
    scene.add(floor);

    // ─── Skeleton overlay (dots + lines) ──────────────────────
    var skeletonGroup = new THREE.Group();
    skeletonGroup.name = "skeleton-overlay";
    scene.add(skeletonGroup);

    var jointMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
    var joints = [];
    for (var i = 0; i < 24; i++) {
        var sphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.025, 8, 8), jointMaterial
        );
        sphere.visible = false;
        skeletonGroup.add(sphere);
        joints.push(sphere);
    }

    var boneMaterial = new THREE.LineBasicMaterial({
        color: 0x00cc66, linewidth: 2, transparent: true, opacity: 0.8,
    });
    var boneLines = [];
    for (var bi = 0; bi < BONES.length; bi++) {
        var pair = BONES[bi];
        var geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3)
        );
        var line = new THREE.Line(geometry, boneMaterial);
        line.visible = false;
        skeletonGroup.add(line);
        boneLines.push({ line: line, a: pair[0], b: pair[1] });
    }

    // ─── SMPL body mesh (from smpl-mesh.js) ─────────────────
    var bodyMesh = null;
    var demoActive = false;

    if (typeof SMPLMeshRenderer === "function") {
        bodyMesh = new SMPLMeshRenderer();
        scene.add(bodyMesh.group);

        // GLB loading disabled — procedural mesh supports pose animation
        // bodyMesh.loadGLB("models/body.glb", function (ok) {
        //     if (!ok) console.log("Using procedural SMPL mesh (no GLB found).");
        // });

        // Show demo T-pose immediately so the mesh is visible on load
        bodyMesh.showDemo();
        demoActive = true;
    } else if (typeof BodyMeshRenderer === "function") {
        bodyMesh = new BodyMeshRenderer();
        scene.add(bodyMesh.group);
    }

    // ─── Render mode management ───────────────────────────────
    function applyRenderMode() {
        var showMesh     = renderMode === "mesh" || renderMode === "both";
        var showSkeleton = renderMode === "skeleton" || renderMode === "both";

        if (bodyMesh) bodyMesh.group.visible = showMesh;
        skeletonGroup.visible = showSkeleton;

        var btn = document.getElementById("render-mode-btn");
        if (btn) {
            var labels = { mesh: "Mesh", skeleton: "Skeleton", both: "Both" };
            btn.textContent = labels[renderMode] || renderMode;
        }
    }

    function cycleRenderMode() {
        var modes = ["mesh", "skeleton", "both"];
        var idx = modes.indexOf(renderMode);
        renderMode = modes[(idx + 1) % modes.length];
        applyRenderMode();
    }

    // Wire up button
    var modeBtn = document.getElementById("render-mode-btn");
    if (modeBtn) modeBtn.addEventListener("click", cycleRenderMode);

    // Keyboard shortcut: M key
    document.addEventListener("keydown", function (e) {
        if (e.key === "m" || e.key === "M") cycleRenderMode();
    });

    applyRenderMode();

    // ─── Update pose data ─────────────────────────────────────
    function updateSkeleton(jointData) {
        if (!jointData || jointData.length !== 24) return;

        // Stop demo animation once real data arrives
        if (demoActive && bodyMesh && bodyMesh.stopDemo) {
            bodyMesh.stopDemo();
            demoActive = false;
        }

        // Body mesh
        if (bodyMesh) bodyMesh.update(jointData);

        // Skeleton overlay
        for (var i = 0; i < 24; i++) {
            var j = jointData[i];
            joints[i].position.set(j[0], j[1], j[2]);
            joints[i].visible = true;
        }
        for (var i = 0; i < boneLines.length; i++) {
            var bl = boneLines[i];
            var positions = bl.line.geometry.attributes.position.array;
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

    // ─── Animation loop ──────────────────────────────────────
    var lastFrameTime = 0, frameCount = 0;
    function animate(time) {
        requestAnimationFrame(animate);
        if (controls) controls.update();
        renderer.render(scene, camera);
        frameCount++;
        if (time - lastFrameTime > 1000) {
            document.getElementById("fps-counter").textContent = frameCount + " FPS";
            frameCount = 0;
            lastFrameTime = time;
        }
    }
    animate(0);

    // ─── Resize ───────────────────────────────────────────────
    window.addEventListener("resize", function () {
        var w = canvas.clientWidth, h = canvas.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    // ─── WebSocket ────────────────────────────────────────────
    var wsUrl = "ws://" + window.location.hostname + ":8000/ws/pose";
    var ws = null;
    var statusEl = document.getElementById("connection-status");

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.onopen = function () {
            statusEl.textContent = "Online";
            statusEl.className = "status-dot online";
        };
        ws.onmessage = function (event) {
            var data = JSON.parse(event.data);
            if (data.joints) updateSkeleton(data.joints);
            // Forward vitals + CSI to HUD if available
            if (data.vitals && window.VitalsHUD) {
                var v = data.vitals;
                window.VitalsHUD.setVitals(
                    v.breathing_bpm, v.heart_bpm,
                    v.breathing_confidence, v.heart_confidence
                );
            }
            if (data.csi_amplitudes && window.VitalsHUD) {
                window.VitalsHUD.pushCSI(data.csi_amplitudes);
            }
        };
        ws.onclose = function () {
            statusEl.textContent = "Offline";
            statusEl.className = "status-dot offline";
            setTimeout(connect, 3000);
        };
    }

    // ─── Status polling ───────────────────────────────────────
    function clearChildren(el) {
        while (el.firstChild) el.removeChild(el.firstChild);
    }

    async function pollStatus() {
        try {
            var resp = await fetch("/api/status");
            var data = await resp.json();
            var nodeList = document.getElementById("node-list");
            clearChildren(nodeList);
            for (var nid in data.nodes || {}) {
                var info = data.nodes[nid];
                var li = document.createElement("li");
                li.textContent = "Node " + nid + ": RSSI " + info.rssi + " dBm";
                nodeList.appendChild(li);
            }
            document.getElementById("node-count").textContent =
                Object.keys(data.nodes || {}).length + " nodes";
            document.getElementById("activity-type").textContent =
                data.current_activity || "--";
            var fallEl = document.getElementById("fall-status");
            if (data.is_fallen) {
                fallEl.textContent = "FALL DETECTED";
                fallEl.className = "alert-danger";
            } else {
                fallEl.textContent = "Normal";
                fallEl.className = "";
            }
        } catch (e) {}
    }

    connect();
    setInterval(pollStatus, 2000);
})();
