const $ = (sel) => document.querySelector(sel);

// === Frame WebSocket + requestAnimationFrame rendering ===
const canvas = $("#game-canvas");
const ctx = canvas.getContext("2d");
canvas.width = 256;
canvas.height = 240;
let frameWs;
let latestBitmap = null;
let agentBitmap = null;
let newFrameReady = false;
let liveScroll = 0;
let liveEpisode = 0;
let frameCount = 0;
let lastFpsTime = performance.now();
let displayFps = 0;

// Render loop — runs at monitor refresh rate (60fps), always smooth
function renderLoop() {
    if (newFrameReady && latestBitmap) {
        ctx.drawImage(latestBitmap, 0, 0);

        // Agent view in bottom-right corner
        if (agentBitmap) {
            const aw = 80, ah = 80;
            const ax = canvas.width - aw - 2, ay = canvas.height - ah - 2;
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.fillRect(ax - 1, ay - 1, aw + 2, ah + 2);
            ctx.drawImage(agentBitmap, ax, ay, aw, ah);
        }

        newFrameReady = false;

        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime > 1000) {
            displayFps = frameCount;
            frameCount = 0;
            lastFpsTime = now;
            $("#stream-fps").textContent = displayFps;
        }
    }
    requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);

function connectFrameWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    frameWs = new WebSocket(`${proto}://${location.host}/ws/frames`);
    frameWs.binaryType = "blob";

    frameWs.onopen = () => {
        $("#status-badge").innerHTML = '<i data-lucide="radio" class="icon-xs"></i> LIVE';
        $("#status-badge").className = "badge live";
        $("#no-frame").style.display = "none";
        lucide.createIcons();
    };

    frameWs.onclose = () => {
        $("#status-badge").innerHTML = '<i data-lucide="wifi-off" class="icon-xs"></i> OFFLINE';
        $("#status-badge").className = "badge";
        setTimeout(connectFrameWS, 1000);
    };

    frameWs.onmessage = async (event) => {
        const buf = await event.data.arrayBuffer();
        // Header: scroll(u16) + episode(u16) + main_size(u32) + agent_size(u32) = 12 bytes
        const header = new DataView(buf, 0, 12);
        liveScroll = header.getUint16(0, true);
        liveEpisode = header.getUint16(2, true);
        const mainSize = header.getUint32(4, true);
        const agentSize = header.getUint32(8, true);

        // Main frame (color)
        const mainBlob = new Blob([buf.slice(12, 12 + mainSize)], { type: "image/jpeg" });
        const mainBmp = await createImageBitmap(mainBlob);
        if (latestBitmap) latestBitmap.close();
        latestBitmap = mainBmp;

        // Agent view (grayscale with overlay)
        if (agentSize > 0) {
            const agentBlob = new Blob([buf.slice(12 + mainSize)], { type: "image/jpeg" });
            const agentBmp = await createImageBitmap(agentBlob);
            if (agentBitmap) agentBitmap.close();
            agentBitmap = agentBmp;
        }

        newFrameReady = true;
    };
}

// === Stats WebSocket (2fps JSON) ===
let statsWs;
let statsReconnectDelay = 1000;

function connectStatsWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    statsWs = new WebSocket(`${proto}://${location.host}/ws/stats`);

    statsWs.onopen = () => { statsReconnectDelay = 1000; };
    statsWs.onclose = () => {
        setTimeout(connectStatsWS, statsReconnectDelay);
        statsReconnectDelay = Math.min(statsReconnectDelay * 2, 10000);
    };
    statsWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.stats) updateStats(msg.stats);
    };
}

let lastSeenEpisode = -1;
let currentEpisode = 0;

// === Stats rendering ===
function formatTime(seconds) {
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
}

function formatNumber(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toString();
}

function updateStats(s) {
    currentEpisode = s.env0_episode || s.episode;
    $("#episode").textContent = s.episode.toLocaleString();
    $("#training-time").textContent = formatTime(s.training_time);
    $("#timesteps").textContent = `${formatNumber(s.timesteps)} / ${formatNumber(s.total_timesteps)}`;
    $("#fps").textContent = s.fps.toFixed(0);
    if (s.timeout_pct !== undefined) $("#timeout-pct").textContent = s.timeout_pct + "%";
    if (s.rollback_count !== undefined) {
        $("#rollback-count").textContent = s.rollback_count;
        if (s.last_rollback_ago > 0) {
            const row = $("#rollback-ago-row");
            row.style.display = "";
            const min = Math.floor(s.last_rollback_ago / 60);
            const sec = s.last_rollback_ago % 60;
            $("#rollback-ago").textContent = min > 0 ? `${min}m ${sec}s ago` : `${sec}s ago`;
        }
    }
    if (s.last_autosave_ago > 0) {
        const min = Math.floor(s.last_autosave_ago / 60);
        const sec = s.last_autosave_ago % 60;
        $("#autosave-ago").textContent = min > 0 ? `${min}m ${sec}s ago` : `${sec}s ago`;
    }


    lastSeenEpisode = s.episode;
}

// === Chart ===
let rewardChart;
const maxChartPoints = 5000;

function initChart() {
    const chartCtx = $("#reward-chart").getContext("2d");
    rewardChart = new Chart(chartCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Episode Reward",
                    data: [],
                    borderColor: "rgba(255, 68, 68, 0.3)",
                    backgroundColor: "rgba(255, 68, 68, 0.05)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1,
                },
                {
                    label: "Avg Reward (50 ep)",
                    data: [],
                    borderColor: "#ffcc00",
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4,
                },
                {
                    label: "Avg Survival (50 ep)",
                    data: [],
                    borderColor: "#4CAF50",
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4,
                    yAxisID: "y2",
                },
                {
                    label: "Episode limit",
                    data: [],
                    borderColor: "rgba(255, 255, 255, 0.3)",
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    yAxisID: "y2",
                }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: "Episode", color: "#666" },
                    ticks: { color: "#555", maxTicksLimit: 8 },
                    grid: { color: "#1a1a1a" },
                },
                y: {
                    display: true,
                    position: "left",
                    title: { display: true, text: "Reward", color: "#ffcc00" },
                    ticks: { color: "#ffcc00" },
                    grid: { color: "#1a1a1a" },
                },
                y2: {
                    display: true,
                    position: "right",
                    title: { display: true, text: "Survival (steps)", color: "#4CAF50" },
                    ticks: { color: "#4CAF50" },
                    grid: { drawOnChartArea: false },
                    min: 0,
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

let rollingWindow = 50;
const avgSlider = $("#avg-window");
const avgVal = $("#avg-window-val");
avgSlider.addEventListener("input", () => {
    rollingWindow = parseInt(avgSlider.value);
    avgVal.textContent = rollingWindow;
    updateChart();
});
let allRewards = [];
let allSurvival = [];

function computeRollingAvg(arr) {
    const avg = [];
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - rollingWindow + 1);
        const window = arr.slice(start, i + 1);
        avg.push(window.reduce((a, b) => a + b, 0) / window.length);
    }
    return avg;
}

function updateChart() {
    const visible = allRewards.slice(-maxChartPoints);
    const visibleSurv = allSurvival.slice(-maxChartPoints);
    const data = rewardChart.data;
    data.labels = visible.map((_, i) => i);
    data.datasets[0].data = visible;
    data.datasets[1].data = computeRollingAvg(visible);
    data.datasets[2].data = computeRollingAvg(visibleSurv);
    // Set Y2 max to episode limit so the line always fills to the top
    const limitSteps = parseInt(episodeSlider.value) * 15;
    rewardChart.options.scales.y2.max = limitSteps;
    data.datasets[3].data = visible.map(() => limitSteps);
    rewardChart.update();
}

function addRewardPoint(episode, reward) {
    allRewards.push(reward);
    allSurvival.push(0); // placeholder, sync fills real data
    updateChart();
}

// === Top Runs ===
async function loadHistory() {
    try {
        const res = await fetch("/api/history");
        const data = await res.json();
        if (data.reward_history && data.reward_history.length > 0) {
            allRewards = data.reward_history;
            allSurvival = data.survival_history || [];
            lastSeenEpisode = data.reward_history.length;
            updateChart();
        }
        if (data.top_runs) updateTopRuns(data.top_runs);
    } catch (e) { /* ignore */ }
}

function updateTopRuns(runs) {
    const tbody = $("#top-runs tbody");
    tbody.innerHTML = "";
    runs.forEach((run, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${run.episode.toLocaleString()}</td>
            <td>${run.reward.toFixed(1)}</td>
            <td>${run.level}</td>
            <td>${run.duration}s</td>
        `;
        tbody.appendChild(tr);
    });
}

// Periodic full sync — chart + top runs every 2s
setInterval(async () => {
    try {
        const res = await fetch("/api/history");
        const data = await res.json();
        if (data.top_runs) updateTopRuns(data.top_runs);
        if (data.reward_history && data.reward_history.length > 0) {
            allRewards = data.reward_history;
            allSurvival = data.survival_history || [];
            lastSeenEpisode = data.reward_history.length;
            updateChart();
        }
    } catch (e) { /* ignore */ }
}, 2000);

// === Controls ===
async function togglePause() {
    const res = await fetch("/api/pause", { method: "POST" });
    const data = await res.json();
    const btn = $("#btn-pause");
    if (data.paused) {
        btn.textContent = "Resume";
        btn.classList.add("active");
    } else {
        btn.textContent = "Pause";
        btn.classList.remove("active");
    }
}

async function restartGame() {
    await fetch("/api/restart", { method: "POST" });
    $("#btn-restart").textContent = "Restarting...";
    setTimeout(() => { $("#btn-restart").textContent = "Restart"; }, 1000);
}

async function saveCheckpoint() {
    await fetch("/api/save", { method: "POST" });
    $("#btn-save").textContent = "Saving...";
    setTimeout(() => { $("#btn-save").textContent = "Save"; }, 2000);
}

async function playBestRun() {
    $("#btn-replay").textContent = "Rendering...";
    const res = await fetch("/api/play-best", { method: "POST" });
    const data = await res.json();
    if (data.ok) {
        window.open(data.url, "_blank");
    } else {
        alert(data.message || "No replay available");
    }
    $("#btn-replay").textContent = "Top Replay";
}

// === Level Progress Bar ===
const progCanvas = $("#progress-bar");
const progCtx = progCanvas.getContext("2d");

let deathMarkers = []; // accumulate death positions as 0.0-1.0

function drawProgressBar(maxScroll, env0Scroll, deathPositions, timeSincePB) {
    const w = progCanvas.width = progCanvas.clientWidth;
    const h = 32;
    progCanvas.height = h;

    if (maxScroll <= 0) maxScroll = 1;

    // Dark background
    progCtx.fillStyle = "#0a0a0f";
    progCtx.fillRect(0, 0, w, h);

    // Death markers — each death = 1px line with low opacity, overlapping = deeper red
    for (const pos of deathPositions) {
        const x = Math.floor((pos / maxScroll) * w);
        progCtx.fillStyle = "rgba(255, 40, 40, 0.08)";
        progCtx.fillRect(x, 0, 1, h);
    }

    // Progress fill — how far env 0 is right now
    const progress = Math.min(env0Scroll / maxScroll, 1.0);
    const fillW = Math.floor(progress * w);
    progCtx.fillStyle = "rgba(76, 175, 80, 0.25)";
    progCtx.fillRect(0, 0, fillW, h);

    // Current position marker
    progCtx.fillStyle = "#4CAF50";
    progCtx.fillRect(fillW - 2, 0, 3, h);

    // PB line at right edge (100%)
    progCtx.fillStyle = "rgba(255, 204, 0, 0.5)";
    progCtx.fillRect(w - 2, 0, 2, h);
    progCtx.font = "9px monospace";
    progCtx.fillStyle = "#ffcc00";
    progCtx.fillText("PB", w - 16, 10);

    // Percentage text
    const pct = Math.floor(progress * 100);
    progCtx.font = "bold 11px monospace";
    progCtx.fillStyle = "#fff";
    progCtx.fillText(pct + "%", 4, 20);

    // PB timer
    const pbEl = $("#pb-timer");
    if (timeSincePB > 0) {
        const min = Math.floor(timeSincePB / 60);
        const sec = timeSincePB % 60;
        pbEl.textContent = min > 0 ? `Last record: ${min}m ${sec}s ago` : `Last record: ${sec}s ago`;
    } else {
        pbEl.textContent = "";
    }
}

// Poll death positions + PB timer every 3s, but use liveScroll for position
let cachedDeaths = [];
let cachedMaxScroll = 1;
let cachedTimeSincePB = 0;

setInterval(async () => {
    try {
        const res = await fetch("/api/level");
        const d = await res.json();
        cachedDeaths = d.death_positions;
        cachedMaxScroll = d.max_scroll;
        cachedTimeSincePB = d.time_since_pb;
    } catch (e) { /* ignore */ }
}, 3000);

// Redraw progress bar every frame using live scroll from frame WS
function updateProgressBar() {
    drawProgressBar(cachedMaxScroll, liveScroll, cachedDeaths, cachedTimeSincePB);
    requestAnimationFrame(updateProgressBar);
}
requestAnimationFrame(updateProgressBar);

// === Settings (episode length slider) ===
const episodeSlider = $("#episode-length");
const episodeVal = $("#episode-length-val");
let settingsTimeout = null;

// Convert seconds → steps: steps = seconds * 60fps / 2 frame_skip = seconds * 30
const secToSteps = (sec) => Math.round(sec * 30);
const stepsToSec = (steps) => Math.round(steps / 30);

episodeSlider.addEventListener("input", () => {
    episodeVal.textContent = episodeSlider.value + "s";
});

episodeSlider.addEventListener("change", () => {
    clearTimeout(settingsTimeout);
    settingsTimeout = setTimeout(async () => {
        await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ episode_length: secToSteps(parseInt(episodeSlider.value)) }),
        });
    }, 200);
});

async function loadSettings() {
    try {
        const res = await fetch("/api/settings");
        const data = await res.json();
        const sec = stepsToSec(data.episode_length);
        episodeSlider.value = sec;
        episodeVal.textContent = sec + "s";
    } catch (e) { /* ignore */ }
}

// === Admin mode (auto-detect local network) ===
fetch("/api/is-admin").then(r => r.json()).then(d => {
    if (d.admin) {
        document.querySelectorAll(".admin-only").forEach(el => el.style.removeProperty("display"));
    }
}).catch(() => {});

// === Init ===
initChart();
connectFrameWS();
connectStatsWS();
loadHistory();
loadSettings();
