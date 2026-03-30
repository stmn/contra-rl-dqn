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
let swappedView = false;

canvas.style.cursor = "pointer";
canvas.addEventListener("click", () => { swappedView = !swappedView; });

// Render loop — runs at monitor refresh rate (60fps), always smooth
function renderLoop() {
    if (newFrameReady && latestBitmap) {
        const mainBmp = swappedView ? agentBitmap : latestBitmap;
        const pipBmp = swappedView ? latestBitmap : agentBitmap;

        if (mainBmp) ctx.drawImage(mainBmp, 0, 0, canvas.width, canvas.height);

        if (pipBmp) {
            const aw = 80, ah = 80;
            const ax = canvas.width - aw - 2, ay = canvas.height - ah - 2;
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.fillRect(ax - 1, ay - 1, aw + 2, ah + 2);
            ctx.drawImage(pipBmp, ax, ay, aw, ah);
        }

        newFrameReady = false;
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
    $("#header-episode").textContent = s.episode.toLocaleString();
    $("#header-time").textContent = formatTime(s.training_time);
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

    // Practice mode indicator
    const practiceBadge = $("#practice-badge");
    if (practiceBadge) {
        practiceBadge.style.display = s.practice ? "" : "none";
    }
    const btnClear = $("#btn-clear-state");
    const btnSave = $("#btn-save-state");
    if (btnClear) btnClear.style.display = s.practice ? "" : "none";
    if (btnSave && s.practice) btnSave.classList.add("active");
    if (btnSave && !s.practice) btnSave.classList.remove("active");


    lastSeenEpisode = s.episode;

    // Update action counts for Agent Output
    if (s.action_counts) {
        for (let i = 0; i < s.action_counts.length; i++) {
            actionCounts[i] = s.action_counts[i];
        }
        totalActions = actionCounts.reduce((a, b) => a + b, 0);
    }

    // Update Agent Input features
    if (s.features && s.features.length > 0) {
        const grid = $("#features-grid");
        const section = $("#features-section");
        if (grid && section) {
            section.style.display = "";
            grid.innerHTML = "";
            s.features.forEach((val, i) => {
                const name = FEATURE_NAMES[i] || "f" + i;
                const div = document.createElement("div");
                div.className = "feature-item";
                div.innerHTML = `<span class="feature-label">${name}</span><span class="feature-value">${val.toFixed(3)}</span>`;
                grid.appendChild(div);
            });
        }
    }
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

let rollingWindow = parseInt(localStorage.getItem("contra_avg") || "50");
let dataRange = parseInt(localStorage.getItem("contra_range") || "5000");
let chartViewMode = localStorage.getItem("contra_chartView") || "chart";

const avgSlider = $("#avg-window");
const avgVal = $("#avg-window-val");
avgSlider.value = rollingWindow;
avgVal.textContent = rollingWindow;
avgSlider.addEventListener("input", () => {
    rollingWindow = parseInt(avgSlider.value);
    avgVal.textContent = rollingWindow;
    localStorage.setItem("contra_avg", rollingWindow);
    updateChart();
});

const rangeSlider = $("#data-range");
const rangeVal = $("#data-range-val");
rangeSlider.value = dataRange;
rangeVal.textContent = dataRange >= 5000 ? "All" : dataRange;
rangeSlider.addEventListener("input", () => {
    dataRange = parseInt(rangeSlider.value);
    rangeVal.textContent = dataRange >= 5000 ? "All" : dataRange;
    localStorage.setItem("contra_range", dataRange);
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
    const ranged = allRewards.slice(-dataRange);
    const rangedSurv = allSurvival.slice(-dataRange);
    const startEp = Math.max(0, allRewards.length - dataRange);

    const avgRewards = computeRollingAvg(ranged);
    const avgSurv = computeRollingAvg(rangedSurv);

    const data = rewardChart.data;
    data.labels = ranged.map((_, i) => startEp + i + 1);
    data.datasets[0].data = ranged;
    data.datasets[1].data = avgRewards;
    data.datasets[2].data = avgSurv;
    rewardChart.update();

    // Live avg values in legend
    const elAvgR = $("#live-avg-reward");
    const elAvgS = $("#live-avg-survival");
    if (elAvgR && avgRewards.length > 0) elAvgR.textContent = avgRewards[avgRewards.length - 1].toFixed(0);
    if (elAvgS && avgSurv.length > 0) elAvgS.textContent = avgSurv[avgSurv.length - 1].toFixed(0);

    updateRewardTable(ranged, rangedSurv, startEp);
}

function setChartView(view) {
    chartViewMode = view;
    localStorage.setItem("contra_chartView", view);
    $("#chart-view").style.display = view === "chart" ? "" : "none";
    $("#table-view").style.display = view === "table" ? "" : "none";
    $("#btn-chart-view").classList.toggle("active", view === "chart");
    $("#btn-table-view").classList.toggle("active", view === "table");
}

function updateRewardTable(rewards, survival, startEp) {
    const tbody = document.querySelector("#reward-table tbody");
    if (!tbody || rewards.length === 0) return;

    const bucketSize = rollingWindow;
    tbody.innerHTML = "";

    for (let i = 0; i < rewards.length; i += bucketSize) {
        const chunk = rewards.slice(i, i + bucketSize);
        const survChunk = survival.slice(i, i + bucketSize);
        const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
        const avgSurv = survChunk.length > 0 ? survChunk.reduce((a, b) => a + b, 0) / survChunk.length : 0;
        const best = Math.max(...chunk);
        const epFrom = startEp + i + 1;
        const epTo = startEp + Math.min(i + bucketSize, rewards.length);

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${epFrom}–${epTo}</td>
            <td>${avg.toFixed(1)}</td>
            <td>${avgSurv.toFixed(0)}</td>
            <td>${best.toFixed(1)}</td>
            <td>${chunk.length}</td>
        `;
        tbody.appendChild(tr);
    }
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
    setTimeout(() => { $("#btn-save").textContent = "Save Model"; }, 2000);
}

async function playBestRun() {
    $("#btn-replay").textContent = "Loading...";
    const res = await fetch("/api/play-best", { method: "POST" });
    const data = await res.json();
    if (data.ok) {
        window.open(data.url, "_blank");
    } else {
        alert(data.message || "No replay available");
    }
    $("#btn-replay").textContent = "Best Replay";
}

async function saveGameState() {
    await fetch("/api/save-state", { method: "POST" });
    $("#btn-save-state").classList.add("active");
    $("#btn-clear-state").style.display = "";
}

async function clearGameState() {
    await fetch("/api/clear-state", { method: "POST" });
    $("#btn-save-state").classList.remove("active");
    $("#btn-clear-state").style.display = "none";
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


// === Action tracking ===
const actionCounts = new Array(16).fill(0);
let totalActions = 0;

function updateActionBars() {
    const container = $("#action-bars");
    if (!container || totalActions === 0) return;

    const colors = [
        "#555", "#4CAF50", "#ff4444", "#888", "#888",
        "#2196F3", "#ff9800", "#4CAF50", "#4CAF50", "#4CAF50",
        "#ff4444", "#ff4444", "#ff9800", "#ff9800",
        "#4CAF50", "#4CAF50"
    ];

    container.innerHTML = "";
    ACTIONS.forEach((name, i) => {
        const pct = ((actionCounts[i] / totalActions) * 100);
        const row = document.createElement("div");
        row.className = "action-bar-row";
        row.innerHTML = `
            <span class="action-bar-label">${name}</span>
            <div class="action-bar-track">
                <div class="action-bar-fill" style="width:${pct}%;background:${colors[i]}"></div>
            </div>
            <span class="action-bar-pct">${pct.toFixed(1)}%</span>
        `;
        container.appendChild(row);
    });
}

// Poll action stats every 3s
setInterval(updateActionBars, 3000);

// === Tabs ===
function switchTab(tabName) {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    const btn = document.querySelector(`.tab[data-tab="${tabName}"]`);
    if (btn) btn.classList.add("active");
    const content = document.getElementById("tab-" + tabName);
    if (content) content.classList.add("active");
    localStorage.setItem("contra_tab", tabName);
}

document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

const savedTab = localStorage.getItem("contra_tab");
if (savedTab) switchTab(savedTab);

// === Agent Input tab ===
const ACTIONS = [
    "NOOP", "Right", "Left", "Up", "Down", "Jump",
    "Shoot", "Right+Jump", "Right+Shoot", "Right+Jump+Shoot",
    "Left+Jump", "Left+Shoot", "Up+Shoot", "Down+Shoot",
    "Right+Up+Shoot", "Right+Down+Shoot"
];

const agentCanvas = $("#agent-input-canvas");
const agentCtx = agentCanvas ? agentCanvas.getContext("2d") : null;

const FEATURE_NAMES = [
    "Player X", "Player Y", "Alive", "In air",
    "Enemy1 dX", "Enemy1 dY", "Enemy1 velX", "Enemy1 atkDelay",
    "Enemy2 dX", "Enemy2 dY", "Enemy2 velX", "Enemy2 atkDelay",
    "Enemy3 dX", "Enemy3 dY", "Enemy3 velX", "Enemy3 atkDelay",
    "Bullet1 dX", "Bullet1 dY",
    "Bullet2 dX", "Bullet2 dY",
    "Bullet3 dX", "Bullet3 dY",
    "Enemies count", "Nearest dist", "Attack flag",
    "Edge fall", "Jump status", "Y velocity"
];

// Populate actions list
const actionsList = $("#actions-list");
if (actionsList) {
    ACTIONS.forEach((name, i) => {
        const div = document.createElement("div");
        div.className = "action-item";
        div.id = "action-" + i;
        div.textContent = i + ": " + name;
        actionsList.appendChild(div);
    });
}

// Update agent view in tab (use the same agentBitmap from frame WS)
function updateAgentInput() {
    if (agentCtx && agentBitmap) {
        agentCtx.drawImage(agentBitmap, 0, 0, 128, 128);
    }
    requestAnimationFrame(updateAgentInput);
}
requestAnimationFrame(updateAgentInput);

// === Admin mode (auto-detect local network) ===
fetch("/api/is-admin").then(r => r.json()).then(d => {
    if (d.admin) {
        document.querySelectorAll(".admin-only").forEach(el => el.style.removeProperty("display"));
    }
}).catch(() => {});

// === Config tab ===
const CONFIG_TIPS = {
    hybrid_observation: "Agent receives both pixel frames AND 28 numerical RAM features (enemy positions, distances). Helps react to things hard to see in pixels.",
    prioritised_replay: "Experience replay samples surprising transitions more often (high TD-error). Learns faster from rare events like deaths and kills.",
    overlay_sprites: "Draws shape markers on game frames showing enemy/bullet positions. Helps the CNN recognize threats.",
    death_penalty: "Negative reward the agent gets each time it dies. Teaches it to avoid danger.",
    progress_scale: "Multiplier for scroll-based reward. Higher = stronger incentive to move right.",
    device: "Hardware used for neural network computation. MPS = Apple GPU, CPU = processor only.",
    total_timesteps: "Total frames to process before training ends.",
    lr: "Learning rate — how fast the neural network updates its weights. Too high = unstable, too low = slow learning.",
    gamma: "Discount factor — how much the agent values future rewards vs immediate ones. 0.99 = very forward-looking.",
    batch_size: "Number of experiences sampled from memory for each learning step.",
    buffer_size: "How many past experiences the agent remembers. Larger = more diverse learning.",
    train_freq: "Learn from replay buffer every N steps.",
    target_update_freq: "Sync target network every N steps. Stabilizes learning.",
    epsilon_start: "Initial exploration rate. 1.0 = fully random actions at the start.",
    epsilon_end: "Final exploration rate. 0.05 = 5% random actions once fully trained.",
    epsilon_decay: "Steps over which exploration decreases from start to end.",
    n_actions: "Number of different button combinations the agent can choose from.",
    hybrid: "Whether hybrid observation (pixels + RAM features) is active.",
    per: "Whether Prioritised Experience Replay is active.",
    frame_skip: "Agent makes a decision every N frames. Reduces computation and gives actions time to take effect.",
    max_episode_steps: "Safety limit — episode ends after this many steps even if agent is still alive.",
};

async function loadConfig() {
    const el = $("#config-content");
    if (!el) return;
    try {
        const res = await fetch("/api/config");
        if (!res.ok) throw new Error(res.status);
        const cfg = await res.json();

        const sections = [
            { title: "Feature Flags", items: cfg.features, format: "bool" },
            { title: "Rewards", items: cfg.rewards, format: "num" },
            { title: "Training", items: cfg.training, format: "num" },
            { title: "DQN Hyperparameters", items: cfg.dqn, format: "num" },
        ];

        el.innerHTML = sections.filter(s => s.items && Object.keys(s.items).length > 0).map(s => `
            <h3 style="color:#666;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px">${s.title}</h3>
            ${Object.entries(s.items).map(([k, v]) => {
                const tip = CONFIG_TIPS[k] ? ` data-tippy-content="${CONFIG_TIPS[k].replace(/"/g, '&quot;')}"` : '';
                const info = CONFIG_TIPS[k] ? ' <span style="color:#555;font-size:10px">(?)</span>' : '';
                return `
                <div class="stat-row">
                    <span class="stat-label"${tip}>${k.replace(/_/g, ' ')}${info}</span>
                    <span class="stat-value">${s.format === "bool"
                        ? `<span style="color:${v ? '#4CAF50' : '#ff4444'}">${v ? 'ON' : 'OFF'}</span>`
                        : typeof v === 'number' ? v.toLocaleString() : v
                    }</span>
                </div>`;
            }).join("")}
        `).join("");
        // Init tooltips for dynamically rendered config
        if (typeof tippy !== 'undefined') {
            tippy('#config-content [data-tippy-content]', {
                theme: 'contra', placement: 'top', arrow: true, delay: [200, 0], maxWidth: 280,
            });
        }
    } catch (e) {
        el.innerHTML = '<div class="stat-row"><span class="stat-label">Config unavailable — deploy new code to server</span></div>';
    }
}

// === Init ===
initChart();
setChartView(chartViewMode);
connectFrameWS();
connectStatsWS();
loadHistory();
loadConfig();
