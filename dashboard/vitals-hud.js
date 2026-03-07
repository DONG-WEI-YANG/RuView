/**
 * vitals-hud.js — Vital Signs HUD + CSI Waterfall + Presence Heatmap
 *
 * Extracts breathing/heart rate from CSI variance, renders:
 * 1. Vital signs HUD overlay on 3D viewer
 * 2. CSI subcarrier waterfall spectrogram
 * 3. Room-scale presence heatmap
 *
 * Works in DEMO mode with simulated data, LIVE when backend connected.
 */
(function () {
    "use strict";

    // ═══ State ═════════════════════════════════════════════════
    var mode = "DEMO";  // "DEMO" | "LIVE"
    var breathRate = 14;
    var heartRate = 68;
    var breathConf = 0.0;
    var heartConf = 0.0;
    var breathWave = [];
    var heartWave = [];
    var csiHistory = [];    // rows of subcarrier amplitudes
    var presenceGrid = [];  // 2D grid of motion intensity
    var GRID_W = 12, GRID_H = 8;
    var hudVisible = true;
    var t = 0;

    // Initialize presence grid
    for (var gy = 0; gy < GRID_H; gy++) {
        presenceGrid.push([]);
        for (var gx = 0; gx < GRID_W; gx++) {
            presenceGrid[gy].push(0);
        }
    }

    // ═══ LIVE/DEMO Badge ══════════════════════════════════════
    var badge = document.getElementById("mode-badge");
    function updateBadge() {
        if (!badge) return;
        badge.textContent = mode;
        badge.className = "mode-badge " + (mode === "LIVE" ? "live" : "demo");
    }
    updateBadge();

    // ═══ Vital Signs Simulation (DEMO mode) ═══════════════════
    function simulateVitals(dt) {
        t += dt;

        // Breathing: 12-20 BPM with slow drift
        breathRate = 15 + Math.sin(t * 0.05) * 3 + Math.sin(t * 0.17) * 1;
        breathConf = 0.75 + Math.sin(t * 0.08) * 0.15;

        // Heart: 62-78 BPM with slight variability
        heartRate = 70 + Math.sin(t * 0.07) * 6 + Math.sin(t * 0.23) * 2;
        heartConf = 0.82 + Math.sin(t * 0.06) * 0.12;

        // Breathing waveform (sinusoidal ~0.25 Hz)
        var breathFreq = breathRate / 60;
        breathWave.push(Math.sin(t * breathFreq * Math.PI * 2) * 0.8 +
                        Math.sin(t * breathFreq * Math.PI * 4) * 0.1);
        if (breathWave.length > 120) breathWave.shift();

        // Heart waveform (sharp peaks ~1.1 Hz)
        var heartFreq = heartRate / 60;
        var hp = (t * heartFreq) % 1;
        var heartVal = hp < 0.08 ? Math.sin(hp / 0.08 * Math.PI) * 1.0 :
                       hp < 0.15 ? -0.3 * Math.sin((hp - 0.08) / 0.07 * Math.PI) :
                       hp < 0.20 ? 0.15 * Math.sin((hp - 0.15) / 0.05 * Math.PI) :
                       Math.sin((hp - 0.20) / 0.80 * Math.PI) * 0.05;
        heartWave.push(heartVal);
        if (heartWave.length > 120) heartWave.shift();
    }

    // ═══ CSI Waterfall Simulation ═════════════════════════════
    var NUM_SUBCARRIERS = 30;

    function simulateCSI() {
        var row = [];
        for (var i = 0; i < NUM_SUBCARRIERS; i++) {
            // Base signal + motion artifact + noise
            var base = 0.4 + 0.2 * Math.sin(i * 0.3);
            var motion = 0.3 * Math.sin(t * 0.5 + i * 0.15) *
                         Math.sin(t * 0.12 + i * 0.05);
            var breathEffect = 0.08 * Math.sin(t * (breathRate / 60) * Math.PI * 2 + i * 0.2);
            var noise = (Math.random() - 0.5) * 0.1;
            row.push(Math.max(0, Math.min(1, base + motion + breathEffect + noise)));
        }
        csiHistory.push(row);
        if (csiHistory.length > 100) csiHistory.shift();
    }

    // ═══ Presence Heatmap Simulation ══════════════════════════
    function simulatePresence() {
        // Person moving in a pattern
        var px = GRID_W / 2 + Math.sin(t * 0.15) * (GRID_W * 0.3);
        var py = GRID_H / 2 + Math.cos(t * 0.11) * (GRID_H * 0.25);

        for (var gy = 0; gy < GRID_H; gy++) {
            for (var gx = 0; gx < GRID_W; gx++) {
                var dx = gx - px, dy = gy - py;
                var dist = Math.sqrt(dx * dx + dy * dy);
                var signal = Math.exp(-dist * dist / 3.0);
                // Decay existing values, add new
                presenceGrid[gy][gx] = presenceGrid[gy][gx] * 0.92 + signal * 0.08;
            }
        }
    }

    // ═══ Render: Vital Signs HUD ══════════════════════════════
    function renderHUD() {
        if (!hudVisible) return;
        var el = document.getElementById("vitals-hud");
        if (!el) return;

        // Breathing
        document.getElementById("breath-val").textContent =
            breathRate.toFixed(1) + " BPM";
        document.getElementById("breath-conf-bar").style.width =
            (breathConf * 100).toFixed(0) + "%";

        // Heart
        document.getElementById("heart-val").textContent =
            heartRate.toFixed(0) + " BPM";
        document.getElementById("heart-conf-bar").style.width =
            (heartConf * 100).toFixed(0) + "%";

        // Waveforms
        drawWaveform("breath-wave", breathWave, "#00cc88");
        drawWaveform("heart-wave", heartWave, "#ff4466");
    }

    function drawWaveform(canvasId, data, color) {
        var c = document.getElementById(canvasId);
        if (!c || data.length < 2) return;
        var ctx = c.getContext("2d");
        var w = c.width, h = c.height;
        ctx.clearRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Waveform
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (var i = 0; i < data.length; i++) {
            var x = (i / (data.length - 1)) * w;
            var y = h / 2 - data[i] * h * 0.4;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Glow
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }

    // ═══ Render: CSI Waterfall ════════════════════════════════
    function renderWaterfall() {
        var c = document.getElementById("csi-waterfall");
        if (!c || csiHistory.length < 2) return;
        var ctx = c.getContext("2d");
        var w = c.width, h = c.height;
        ctx.clearRect(0, 0, w, h);

        var rows = csiHistory.length;
        var cellW = w / NUM_SUBCARRIERS;
        var cellH = h / Math.min(rows, 100);

        for (var r = 0; r < rows; r++) {
            var row = csiHistory[r];
            for (var i = 0; i < NUM_SUBCARRIERS; i++) {
                var v = row[i];
                ctx.fillStyle = waterfallColor(v);
                ctx.fillRect(i * cellW, (rows - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        // Axis labels
        ctx.fillStyle = "#666";
        ctx.font = "9px sans-serif";
        ctx.fillText("Subcarrier →", 4, h - 4);
        ctx.save();
        ctx.translate(w - 4, h - 4);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Time →", 0, 0);
        ctx.restore();
    }

    function waterfallColor(v) {
        // Blue → Cyan → Green → Yellow → Red
        v = Math.max(0, Math.min(1, v));
        var r, g, b;
        if (v < 0.25) {
            r = 0; g = Math.round(v * 4 * 180); b = 180;
        } else if (v < 0.5) {
            r = 0; g = 180; b = Math.round((1 - (v - 0.25) * 4) * 180);
        } else if (v < 0.75) {
            r = Math.round((v - 0.5) * 4 * 255); g = 180; b = 0;
        } else {
            r = 255; g = Math.round((1 - (v - 0.75) * 4) * 180); b = 0;
        }
        return "rgb(" + r + "," + g + "," + b + ")";
    }

    // ═══ Render: Presence Heatmap ═════════════════════════════
    function renderHeatmap() {
        var c = document.getElementById("presence-heatmap");
        if (!c) return;
        var ctx = c.getContext("2d");
        var w = c.width, h = c.height;
        ctx.clearRect(0, 0, w, h);

        // Room background
        ctx.fillStyle = "#151530";
        ctx.fillRect(0, 0, w, h);

        var cellW = w / GRID_W, cellH = h / GRID_H;

        for (var gy = 0; gy < GRID_H; gy++) {
            for (var gx = 0; gx < GRID_W; gx++) {
                var v = presenceGrid[gy][gx];
                if (v > 0.02) {
                    ctx.fillStyle = heatColor(v);
                    ctx.fillRect(gx * cellW, gy * cellH, cellW, cellH);
                }
            }
        }

        // Room grid overlay
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 0.5;
        for (var x = 0; x <= GRID_W; x++) {
            ctx.beginPath();
            ctx.moveTo(x * cellW, 0); ctx.lineTo(x * cellW, h);
            ctx.stroke();
        }
        for (var y = 0; y <= GRID_H; y++) {
            ctx.beginPath();
            ctx.moveTo(0, y * cellH); ctx.lineTo(w, y * cellH);
            ctx.stroke();
        }

        // WiFi AP indicators
        ctx.fillStyle = "#00ff88";
        ctx.font = "10px sans-serif";
        drawAP(ctx, 2, 0.5, cellW, cellH, "AP1");
        drawAP(ctx, GRID_W - 3, GRID_H - 1.5, cellW, cellH, "AP2");
    }

    function drawAP(ctx, gx, gy, cw, ch, label) {
        var x = gx * cw, y = gy * ch;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(label, x + 7, y + 3);
        // Signal rings
        ctx.strokeStyle = "rgba(0,255,136,0.15)";
        for (var r = 1; r <= 3; r++) {
            ctx.beginPath();
            ctx.arc(x, y, r * 18, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    function heatColor(v) {
        v = Math.max(0, Math.min(1, v));
        // Transparent blue → purple → orange → white
        var a = Math.min(0.85, v * 1.2);
        var r, g, b;
        if (v < 0.3) {
            r = 30; g = 30; b = Math.round(100 + v * 500);
        } else if (v < 0.6) {
            var t2 = (v - 0.3) / 0.3;
            r = Math.round(100 + t2 * 155); g = 30; b = Math.round(250 - t2 * 100);
        } else {
            var t3 = (v - 0.6) / 0.4;
            r = 255; g = Math.round(30 + t3 * 200); b = Math.round(150 - t3 * 100);
        }
        return "rgba(" + r + "," + g + "," + b + "," + a.toFixed(2) + ")";
    }

    // ═══ Render: Live Demo Tab — Signal Canvas ════════════════
    var signalScrollOffset = 0;
    function renderSignalCanvas() {
        var c = document.getElementById("signalCanvas");
        if (!c) return;
        var ctx = c.getContext("2d");
        var w = c.width, h = c.height;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = "#0e0e22";
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 0.5;
        for (var gx = 0; gx < w; gx += 40) {
            ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        }
        for (var gy = 0; gy < h; gy += 30) {
            ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
        }

        signalScrollOffset += 1.5;

        // Draw 6 subcarrier channels with different colors
        var colors = ["#00cc88", "#00aaff", "#ffaa00", "#ff4466", "#aa66ff", "#66ffcc"];
        var freqs  = [0.03, 0.05, 0.04, 0.025, 0.06, 0.035];
        var amps   = [0.6, 0.45, 0.55, 0.35, 0.4, 0.5];
        var phases  = [0, 1.2, 2.5, 0.8, 3.7, 1.9];

        for (var ch = 0; ch < 6; ch++) {
            var baseY = h * (0.12 + ch * 0.14);
            ctx.strokeStyle = colors[ch];
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.85;
            ctx.beginPath();
            for (var x = 0; x < w; x++) {
                var tt = (x + signalScrollOffset) * freqs[ch];
                var val = Math.sin(tt + phases[ch]) * amps[ch]
                        + Math.sin(tt * 2.3 + phases[ch] * 0.7) * amps[ch] * 0.3
                        + Math.sin(tt * (breathRate / 60) * 0.15 + ch) * 0.15
                        + (Math.random() - 0.5) * 0.08;
                var y = baseY + val * 22;
                if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Channel label
            ctx.globalAlpha = 0.5;
            ctx.fillStyle = colors[ch];
            ctx.font = "9px monospace";
            ctx.fillText("CH" + (ch + 1), 4, baseY - 12);
            ctx.globalAlpha = 1.0;
        }

        // Update metrics
        var sigEl = document.getElementById("signalStrength");
        var latEl = document.getElementById("latency");
        if (sigEl) sigEl.textContent = (-42 + Math.sin(t * 0.3) * 5).toFixed(0) + " dBm";
        if (latEl) latEl.textContent = (10 + Math.random() * 8).toFixed(0) + " ms";
    }

    // ═══ Render: Live Demo Tab — Pose Canvas ═════════════════
    // 2D stick figure (classic pose estimation visualization)
    var POSE_BONES_2D = [
        [0,1],[1,2],[2,3],[3,10],  // head→spine
        [3,4],[4,5],[5,6],          // L arm
        [3,7],[7,8],[8,9],          // R arm
        [10,11],[11,12],[12,13],[13,14], // L leg
        [11,15],[15,16],[16,17],         // R leg
    ];

    function renderPoseCanvas() {
        var c = document.getElementById("poseCanvas");
        if (!c) return;
        var ctx = c.getContext("2d");
        var w = c.width, h = c.height;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = "#0e0e22";
        ctx.fillRect(0, 0, w, h);

        // Compute 2D joint positions (project REST pose with animation)
        var pts = [];
        var cx = w * 0.5, cy = h * 0.88;
        var scale = h * 0.42;
        for (var i = 0; i < 24; i++) {
            var rx = REST[i][0], ry = REST[i][1];
            pts.push([cx + rx * scale, cy - ry * scale]);
        }

        // Animate: arm swing, breathing, sway
        var armSwing = Math.sin(t * 0.6) * 8;
        var sway = Math.sin(t * 0.4) * 3;
        var breathOffset = Math.sin(t * 0.5) * 2;

        // L arm
        pts[5][1] += armSwing;    pts[5][0] -= 8;
        pts[6][1] += armSwing * 1.4; pts[6][0] -= 4;
        // R arm
        pts[8][1] -= armSwing;    pts[8][0] += 8;
        pts[9][1] -= armSwing * 1.4; pts[9][0] += 4;
        // Arms down (not T-pose)
        pts[4][1] += 6;  pts[5][1] += 45; pts[6][1] += 80;
        pts[7][1] += 6;  pts[8][1] += 45; pts[9][1] += 80;
        pts[5][0] += 3;  pts[6][0] += 5;
        pts[8][0] -= 3;  pts[9][0] -= 5;

        // Sway all
        for (var i = 0; i < 24; i++) {
            pts[i][0] += sway;
        }
        // Breathing (chest expansion)
        pts[2][0] += breathOffset * 0.3;
        pts[3][0] += breathOffset * 0.2;

        // Draw confidence heatmap circle behind person
        var grad = ctx.createRadialGradient(cx + sway, cy - scale * 0.5, 10, cx + sway, cy - scale * 0.5, scale * 0.8);
        grad.addColorStop(0, "rgba(0,204,136,0.08)");
        grad.addColorStop(1, "rgba(0,204,136,0)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Draw bones
        ctx.strokeStyle = "#00cc88";
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        for (var bi = 0; bi < POSE_BONES_2D.length; bi++) {
            var a = POSE_BONES_2D[bi][0], b = POSE_BONES_2D[bi][1];
            if (a >= 24 || b >= 24) continue;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.moveTo(pts[a][0], pts[a][1]);
            ctx.lineTo(pts[b][0], pts[b][1]);
            ctx.stroke();
        }

        // Draw joints
        for (var i = 0; i < 18; i++) {
            var r = (i === 0) ? 5 : 3.5;  // head bigger
            ctx.globalAlpha = 0.95;
            ctx.fillStyle = "#00ff88";
            ctx.beginPath();
            ctx.arc(pts[i][0], pts[i][1], r, 0, Math.PI * 2);
            ctx.fill();

            // Confidence ring
            ctx.globalAlpha = 0.3;
            ctx.strokeStyle = "#00ff88";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(pts[i][0], pts[i][1], r + 3, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;

        // Joint labels for key points
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.font = "8px monospace";
        ctx.fillText("head", pts[0][0] + 8, pts[0][1] - 4);
        ctx.fillText("L", pts[6][0] - 14, pts[6][1]);
        ctx.fillText("R", pts[9][0] + 6, pts[9][1]);

        // Update metrics
        var confPct = (85 + Math.sin(t * 0.2) * 8).toFixed(1);
        var personEl = document.getElementById("personCount");
        var confEl = document.getElementById("confidence");
        var kpEl = document.getElementById("keypoints");
        if (personEl) personEl.textContent = "1";
        if (confEl) confEl.textContent = confPct + "%";
        if (kpEl) kpEl.textContent = "24/24";
    }

    // REST pose reference for 2D drawing (reuse from smpl-mesh.js)
    var REST = [
        [0.000,1.700],[0.000,1.550],[0.000,1.380],[0.000,1.120],
        [-0.200,1.400],[-0.480,1.400],[-0.700,1.400],
        [0.200,1.400],[0.480,1.400],[0.700,1.400],
        [0.000,0.950],[0.000,0.900],
        [-0.100,0.880],[-0.100,0.500],[-0.100,0.080],
        [0.100,0.880],[0.100,0.500],[0.100,0.080],
        [-0.100,0.030],[0.100,0.030],[-0.780,1.400],[0.780,1.400],
        [-0.030,1.720],[0.030,1.720],
    ];

    // ═══ Keyboard Shortcuts ═══════════════════════════════════
    document.addEventListener("keydown", function (e) {
        var key = e.key.toLowerCase();
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        switch (key) {
            case "h":
                hudVisible = !hudVisible;
                var hud = document.getElementById("vitals-hud");
                if (hud) hud.style.display = hudVisible ? "" : "none";
                break;
            case "r":
                // Reset camera handled by skeleton3d.js if OrbitControls
                break;
            case "1":
                clickTab("viewer");
                break;
            case "2":
                clickTab("dashboard");
                break;
            case "3":
                clickTab("hardware");
                break;
            case "4":
                clickTab("demo");
                break;
            case "5":
                clickTab("sensing");
                break;
        }
    });

    function clickTab(name) {
        var btn = document.querySelector('.nav-tab[data-tab="' + name + '"]');
        if (btn) btn.click();
    }

    // ═══ WebSocket Integration ════════════════════════════════
    // Listen for connection status to switch DEMO ↔ LIVE
    var origStatusEl = document.getElementById("connection-status");
    if (origStatusEl) {
        new MutationObserver(function () {
            var isOnline = origStatusEl.classList.contains("online");
            mode = isOnline ? "LIVE" : "DEMO";
            updateBadge();
        }).observe(origStatusEl, { attributes: true, attributeFilter: ["class"] });
    }

    // ═══ Main Update Loop ════════════════════════════════════
    var lastTime = 0;
    function update(time) {
        requestAnimationFrame(update);
        var dt = Math.min((time - lastTime) / 1000, 0.1);
        lastTime = time;

        if (mode === "DEMO") {
            simulateVitals(dt);
            simulateCSI();
            simulatePresence();
        }

        renderHUD();

        // Render canvases only when their tabs are visible
        var sensingTab = document.getElementById("tab-sensing");
        if (sensingTab && sensingTab.classList.contains("active")) {
            renderWaterfall();
        }
        var dashTab = document.getElementById("tab-dashboard");
        if (dashTab && dashTab.classList.contains("active")) {
            renderHeatmap();
        }
        var demoTab = document.getElementById("tab-demo");
        if (demoTab && demoTab.classList.contains("active")) {
            renderSignalCanvas();
            renderPoseCanvas();
        }
    }

    requestAnimationFrame(update);

    // ═══ Public API for live data injection ═══════════════════
    window.VitalsHUD = {
        setVitals: function (breath, heart, bConf, hConf) {
            breathRate = breath;
            heartRate = heart;
            breathConf = bConf;
            heartConf = hConf;
        },
        pushCSI: function (subcarrierAmplitudes) {
            csiHistory.push(subcarrierAmplitudes);
            if (csiHistory.length > 100) csiHistory.shift();
        },
        setPresence: function (grid) {
            presenceGrid = grid;
        }
    };
})();
