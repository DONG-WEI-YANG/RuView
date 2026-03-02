(function () {
    "use strict";

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

    const canvas = document.getElementById("skeleton-canvas");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
    camera.position.set(0, 1, 4);
    camera.lookAt(0, 0.8, 0);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);

    scene.add(new THREE.AmbientLight(0x404040));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    const grid = new THREE.GridHelper(6, 12, 0x444444, 0x333333);
    scene.add(grid);

    const jointMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
    const joints = [];
    for (let i = 0; i < 24; i++) {
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 8), jointMaterial);
        sphere.visible = false;
        scene.add(sphere);
        joints.push(sphere);
    }

    const boneMaterial = new THREE.LineBasicMaterial({ color: 0x00cc66, linewidth: 2 });
    const boneLines = [];
    for (const [a, b] of BONES) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute([0,0,0, 0,0,0], 3));
        const line = new THREE.Line(geometry, boneMaterial);
        line.visible = false;
        scene.add(line);
        boneLines.push({ line, a, b });
    }

    function updateSkeleton(jointData) {
        if (!jointData || jointData.length !== 24) return;
        for (let i = 0; i < 24; i++) {
            const [x, y, z] = jointData[i];
            joints[i].position.set(x, y, z);
            joints[i].visible = true;
        }
        for (const { line, a, b } of boneLines) {
            const positions = line.geometry.attributes.position.array;
            positions[0] = jointData[a][0]; positions[1] = jointData[a][1]; positions[2] = jointData[a][2];
            positions[3] = jointData[b][0]; positions[4] = jointData[b][1]; positions[5] = jointData[b][2];
            line.geometry.attributes.position.needsUpdate = true;
            line.visible = true;
        }
    }

    let lastFrameTime = 0, frameCount = 0;
    function animate(time) {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
        frameCount++;
        if (time - lastFrameTime > 1000) {
            document.getElementById("fps-counter").textContent = frameCount + " FPS";
            frameCount = 0;
            lastFrameTime = time;
        }
    }
    animate(0);

    window.addEventListener("resize", () => {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    const wsUrl = "ws://" + window.location.hostname + ":8000/ws/pose";
    let ws = null;
    const statusEl = document.getElementById("connection-status");

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => { statusEl.textContent = "Online"; statusEl.className = "status-dot online"; };
        ws.onmessage = (event) => { const data = JSON.parse(event.data); if (data.joints) updateSkeleton(data.joints); };
        ws.onclose = () => { statusEl.textContent = "Offline"; statusEl.className = "status-dot offline"; setTimeout(connect, 3000); };
    }

    function clearChildren(element) {
        while (element.firstChild) {
            element.removeChild(element.firstChild);
        }
    }

    async function pollStatus() {
        try {
            const resp = await fetch("/api/status");
            const data = await resp.json();
            const nodeList = document.getElementById("node-list");
            clearChildren(nodeList);
            for (const [nid, info] of Object.entries(data.nodes || {})) {
                const li = document.createElement("li");
                li.textContent = "Node " + nid + ": RSSI " + info.rssi + " dBm";
                nodeList.appendChild(li);
            }
            document.getElementById("node-count").textContent = Object.keys(data.nodes || {}).length + " nodes";
            document.getElementById("activity-type").textContent = data.current_activity || "--";
            const fallEl = document.getElementById("fall-status");
            if (data.is_fallen) { fallEl.textContent = "FALL DETECTED"; fallEl.className = "alert-danger"; }
            else { fallEl.textContent = "Normal"; fallEl.className = ""; }
        } catch (e) {}
    }

    connect();
    setInterval(pollStatus, 2000);
})();
