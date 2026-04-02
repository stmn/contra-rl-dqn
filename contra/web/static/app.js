const $ = (sel) => document.querySelector(sel);
const WEAPONS = ["Default","Machine Gun","Fireball","Spread","Laser","Barrier"];

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
let highlightCoord = null; // {x, y} in NES coords (256x240)

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

        // Highlight coordinate on hover
        if (highlightCoord) {
            const hx = highlightCoord.x, hy = highlightCoord.y;
            // Crosshair
            ctx.strokeStyle = "#ff0";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(hx - 12, hy); ctx.lineTo(hx + 12, hy);
            ctx.moveTo(hx, hy - 12); ctx.lineTo(hx, hy + 12);
            ctx.stroke();
            // Circle
            ctx.beginPath();
            ctx.arc(hx, hy, 8, 0, Math.PI * 2);
            ctx.stroke();
            // Label
            ctx.fillStyle = "#ff0";
            ctx.font = "10px monospace";
            ctx.fillText(`(${hx},${hy})`, hx + 10, hy - 10);
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
    if (s.buffer_size !== undefined) {
        $("#header-buffer").textContent = `${formatNumber(s.buffer_size)} / ${formatNumber(s.buffer_capacity)}`;
    }
    if (s.ram_current_mb !== undefined) {
        const cur = s.ram_current_mb >= 1024 ? (s.ram_current_mb / 1024).toFixed(1) + 'GB' : s.ram_current_mb.toFixed(0) + 'MB';
        const peak = s.ram_peak_mb >= 1024 ? (s.ram_peak_mb / 1024).toFixed(1) + 'GB' : s.ram_peak_mb.toFixed(0) + 'MB';
        $("#header-ram").textContent = `${cur} / ${peak}`;
    }
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

    // Sync pause button state
    const btnPause = $("#btn-pause");
    if (btnPause && s.paused !== undefined) {
        btnPause.textContent = s.paused ? "Resume" : "Pause";
        btnPause.classList.toggle("active", s.paused);
    }

    // Sync auto-restart button
    const btnAR = $("#btn-auto-restart");
    if (btnAR && s.auto_restart !== undefined) {
        btnAR.classList.toggle("active", s.auto_restart);
    }

    // Show "Waiting..." on restart button when waiting for manual restart
    const btnRestart = $("#btn-restart");
    if (btnRestart && s.waiting_restart) {
        btnRestart.classList.add("active");
    } else if (btnRestart) {
        btnRestart.classList.remove("active");
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

    // Update Run Log
    if (s.run_log) {
        const r = s.run_log;
        const bd = $("#run-log-breakdown");
        if (bd) {
            bd.innerHTML = `
                <div class="stat-row"><span class="stat-label">Step</span><span class="stat-value">${r.step || 0}</span></div>
                <div class="stat-row"><span class="stat-label">Total reward</span><span class="stat-value">${(r.total_reward || 0).toFixed(0)}</span></div>
                <div class="stat-row"><span class="stat-label">Scroll</span><span class="stat-value">${r.scroll || 0}</span></div>
                <div class="stat-row"><span class="stat-label">Deaths</span><span class="stat-value">${r.deaths || 0}</span></div>
                <div class="stat-row" style="margin-top:8px;border-top:1px solid #1a1a24;padding-top:8px">
                    <span class="stat-label" style="color:#4CAF50">Scroll reward</span>
                    <span class="stat-value" style="color:#4CAF50">+${(r.reward_scroll || 0).toFixed(0)}</span>
                </div>
                <div class="stat-row"><span class="stat-label" style="color:#ff9800">Kill reward</span><span class="stat-value" style="color:#ff9800">+${(r.reward_kills || 0).toFixed(0)}</span></div>
                <div class="stat-row"><span class="stat-label" style="color:#2196F3">Turret/Boss hits</span><span class="stat-value" style="color:#2196F3">+${(r.reward_turret || 0).toFixed(0)} (${r.turret_hits || 0} hits)</span></div>
                <div class="stat-row"><span class="stat-label" style="color:#e040fb">Weapon upgrade</span><span class="stat-value" style="color:#e040fb">+${(r.reward_weapon || 0).toFixed(0)}</span></div>
                <div class="stat-row"><span class="stat-label" style="color:#ff4444">Death penalty</span><span class="stat-value" style="color:#ff4444">${(r.reward_death || 0).toFixed(0)}</span></div>
            `;
        }
        const ev = $("#run-log-events");
        if (ev && r.events) {
            ev.innerHTML = r.events.map(e => {
                const color = e[1].startsWith('Death') ? '#ff4444' : e[1].startsWith('Turret') ? '#2196F3' : e[1].startsWith('Weapon') ? '#e040fb' : e[1].startsWith('Unknown') ? '#555' : e[1].startsWith('Kill') ? '#ff9800' : '#555';
                // Replace pos=(X,Y) with hoverable coord spans
                const raw = `Step ${e[0]} ${e[1]}`;
                const text = e[1].replace(/pos=\((\d+),(\d+)\)/g, (m, x, y) => coordSpan(x, y, `pos=(${x},${y})`));
                return `<div style="padding:2px 0;color:${color};display:flex;align-items:center;gap:6px"><span class="copy-btn" onclick="copyEvent(this)" data-text="${raw.replace(/"/g,'&quot;')}" style="cursor:pointer;opacity:0.3;font-size:14px" title="Copy">&#x2398;</span><span style="flex:1"><span style="color:#555">Step ${e[0]}</span> ${text}</span></div>`;
            }).reverse().join("");
        }
        const av = $("#agent-view-text");
        if (av) {
            const p = r.player || {x:0, y:0, weapon:0};
            const weaponName = WEAPONS[p.weapon] || "Unknown";
            let html = `<div style="color:#4CAF50;margin-bottom:4px">Player: ${coordSpan(p.x, p.y)}</div>`;
            html += `<div style="color:#4CAF50;margin-bottom:8px">Weapon: ${weaponName}</div>`;

            const ENEMY_TYPES = {1:"Bullet",2:"Weapon Box",3:"Flying Bonus",4:"Rotating Gun",5:"Soldier",6:"Sniper",7:"Red Turret",8:"Wall Cannon",11:"Mortar",12:"Scuba Diver",14:"Turret Man",15:"Turret Bullet",16:"Boss Turret",17:"Boss Door",18:"Bridge"};
            const enemies = r.enemies || [];
            const known = enemies.filter(e => ENEMY_TYPES[e.type]);
            const unknown = enemies.filter(e => !ENEMY_TYPES[e.type]);
            html += `<div style="color:#ff9800;margin-bottom:4px">Enemies visible: ${enemies.length}</div>`;
            known.forEach((e, i) => {
                const name = ENEMY_TYPES[e.type];
                const hpStr = e.hp > 1 ? ` hp=${e.hp}` : '';
                html += `<div style="padding:2px 0 2px 12px;color:#ccc">${name} ${coordSpan(e.x, e.y)} dist=${e.dist}${hpStr}</div>`;
            });
            if (unknown.length > 0) {
                html += `<div style="color:#ff6600;margin:4px 0 4px 0">Unknown types: ${unknown.length}</div>`;
                unknown.forEach((e) => {
                    const hpStr = e.hp > 1 ? ` hp=${e.hp}` : '';
                    html += `<div style="padding:2px 0 2px 12px;color:#ff6600">type=${e.type} ${coordSpan(e.x, e.y)} dist=${e.dist}${hpStr}</div>`;
                });
            }

            const bullets = r.bullets || [];
            html += `<div style="color:#ff4444;margin:8px 0 4px">Bullets visible: ${bullets.length}</div>`;
            bullets.forEach((b, i) => {
                html += `<div style="padding:2px 0 2px 12px;color:#ccc">#${i+1} ${coordSpan(b.x, b.y)} dist=${b.dist}</div>`;
            });

            const OTHER_TYPES = {2:"Bridge Boom",3:"Flying Bonus",18:"Bridge"};
            const other = r.other || [];
            if (other.length > 0) {
                html += `<div style="color:#888;margin:8px 0 4px">Other visible: ${other.length}</div>`;
                other.forEach((o) => {
                    let name = OTHER_TYPES[o.type] || `type=${o.type}`;
                    if (o.weapon !== undefined) name += ` (${WEAPONS[o.weapon] || '?'})`;
                    html += `<div style="padding:2px 0 2px 12px;color:#888">${name} ${coordSpan(o.x, o.y)} dist=${o.dist}</div>`;
                });
            }

            if (enemies.length === 0 && bullets.length === 0 && other.length === 0) {
                html += `<div style="color:#555;margin-top:8px">No entities on screen</div>`;
            }
            av.innerHTML = html;
        }
    }

    // Practice mode: swap chart data to practice rewards
    if (s.practice && s.practice_rewards && s.practice_rewards.length > 0) {
        allPracticeRewards = s.practice_rewards;
        if (!practiceMode) {
            practiceMode = true;
            updateChart();
        }
    } else if (practiceMode) {
        practiceMode = false;
        allPracticeRewards = [];
        updateChart();
    }

    // Update Agent Input features
    if (s.features && s.features.length > 0) {
        const grid = $("#features-grid");
        const section = $("#features-section");
        if (grid && section) {
            section.style.display = "";
            grid.innerHTML = "";
            const px = s.run_log ? (s.run_log.player || {}).x || 0 : 0;
            const py = s.run_log ? (s.run_log.player || {}).y || 0 : 0;
            s.features.forEach((val, i) => {
                const name = FEATURE_NAMES[i] || "f" + i;
                const div = document.createElement("div");
                div.className = "feature-item";
                let coord = '';
                // Player position
                if (i === 0) coord = coordSpan(Math.round(val * 256), py, '⊕');
                if (i === 1) coord = coordSpan(px, Math.round(val * 240), '⊕');
                // Enemy dx/dy → absolute screen position
                if ([4,8,12].includes(i)) {
                    const ex = Math.round((val * 256) - 128 + px);
                    const ey_val = s.features[i + 1];
                    const ey = Math.round((ey_val * 240) - 120 + py);
                    if (val > 0.01 || val < -0.01) coord = coordSpan(ex, ey, '⊕');
                }
                // Bullet dx/dy
                if ([16,18,20].includes(i)) {
                    const bx = Math.round((val * 256) - 128 + px);
                    const by_val = s.features[i + 1];
                    const by = Math.round((by_val * 240) - 120 + py);
                    if (val > 0.01 || val < -0.01) coord = coordSpan(bx, by, '⊕');
                }
                div.innerHTML = `<span class="feature-label">${name} ${coord}</span><span class="feature-value">${val.toFixed(3)}</span>`;
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
                    hidden: true,
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
                    display: false,
                    position: "right",
                    title: { display: true, text: "Survival (steps)", color: "#4CAF50" },
                    ticks: { color: "#4CAF50" },
                    grid: { drawOnChartArea: false },
                    min: 0,
                },
                y3: {
                    display: true,
                    position: "right",
                    title: { display: true, text: "Boss reach %", color: "#e040fb" },
                    ticks: { color: "#e040fb", callback: v => v + '%' },
                    grid: { drawOnChartArea: false },
                    min: 0, max: 100,
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(ctx) {
                            if (ctx.dataset.hidden) return null;
                            const v = ctx.parsed.y;
                            if (ctx.dataset.yAxisID === 'y3') return `${ctx.dataset.label}: ${v.toFixed(0)}%`;
                            if (ctx.dataset.yAxisID === 'y2') return `${ctx.dataset.label}: ${v.toFixed(0)} steps`;
                            return `${ctx.dataset.label}: ${v.toFixed(0)}`;
                        }
                    }
                },
                crosshair: {
                    line: { color: '#555', width: 1, dashPattern: [4, 4] }
                }
            },
            interaction: { mode: 'index', intersect: false },
            hover: { mode: 'index', intersect: false }
        }
    });

    // Crosshair plugin (vertical line at cursor)
    const crosshairPlugin = {
        id: 'crosshair',
        afterDraw(chart) {
            if (chart.tooltip?._active?.length) {
                const x = chart.tooltip._active[0].element.x;
                const ctx = chart.ctx;
                const top = chart.chartArea.top;
                const bottom = chart.chartArea.bottom;
                ctx.save();
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = '#555';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, top);
                ctx.lineTo(x, bottom);
                ctx.stroke();
                ctx.restore();
            }
        }
    };
    Chart.register(crosshairPlugin);
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
let allBoss = [];
let allPracticeRewards = [];
let practiceMode = false;

function computeRollingAvg(arr) {
    const avg = [];
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - rollingWindow + 1);
        const window = arr.slice(start, i + 1);
        avg.push(window.reduce((a, b) => a + b, 0) / window.length);
    }
    return avg;
}

function computeRollingPct(arr) {
    const pct = [];
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - rollingWindow + 1);
        const window = arr.slice(start, i + 1);
        const count = window.filter(v => v).length;
        pct.push(100 * count / window.length);
    }
    return pct;
}

function toggleDataset(checkbox) {
    const idx = parseInt(checkbox.dataset.dataset);
    rewardChart.data.datasets[idx].hidden = !checkbox.checked;
    // Show/hide axes based on visible datasets
    const ds = rewardChart.data.datasets;
    const yVisible = !ds[0].hidden || !ds[1].hidden;  // raw or avg reward
    const y2Visible = !ds[2].hidden;  // survival
    const y3Visible = ds.slice(3).some(d => !d.hidden);  // any boss %
    rewardChart.options.scales.y.display = yVisible;
    rewardChart.options.scales.y2.display = y2Visible;
    rewardChart.options.scales.y3.display = y3Visible;
    rewardChart.update();
}

const BOSS_COLORS = ['#e040fb','#00bcd4','#ff9800','#4CAF50','#ff5722','#9c27b0','#2196F3','#cddc39'];
let knownBossLevels = [];

function updateChart() {
    const source = practiceMode ? allPracticeRewards : allRewards;
    const ranged = source.slice(-dataRange);
    const rangedSurv = practiceMode ? [] : allSurvival.slice(-dataRange);
    if (!practiceMode && allBoss.length < allRewards.length) {
        allBoss = new Array(allRewards.length - allBoss.length).fill(-1).concat(allBoss);
    }
    const rangedBoss = practiceMode ? [] : allBoss.slice(-dataRange);
    const startEp = Math.max(0, source.length - dataRange);

    // Update chart title to show mode
    const chartTitle = document.querySelector(".chart-container h2");
    if (chartTitle) {
        chartTitle.innerHTML = practiceMode
            ? '<i data-lucide="trending-up" class="icon-heading"></i> Practice Rewards'
            : '<i data-lucide="trending-up" class="icon-heading"></i> Reward History';
    }

    const avgRewards = computeRollingAvg(ranged);
    const avgSurv = computeRollingAvg(rangedSurv);

    // Find which boss levels have been reached
    const seenLevels = [...new Set(allBoss.filter(v => typeof v === 'number' && v >= 0))].sort();

    // Add/update boss datasets dynamically (start at index 3)
    if (seenLevels.join() !== knownBossLevels.join()) {
        knownBossLevels = seenLevels;
        // Remove old boss datasets
        rewardChart.data.datasets.splice(3);
        // Add new ones
        seenLevels.forEach((lvl, i) => {
            rewardChart.data.datasets.push({
                label: `L${lvl+1} Boss reach %`,
                data: [],
                borderColor: BOSS_COLORS[lvl % BOSS_COLORS.length],
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.4,
                yAxisID: "y3",
            });
        });
        // Update checkboxes
        const container = $("#boss-checkboxes");
        if (container) {
            container.innerHTML = seenLevels.map((lvl, i) => {
                const idx = 3 + i;
                const color = BOSS_COLORS[lvl % BOSS_COLORS.length];
                return `<label class="legend-item"><input type="checkbox" checked data-dataset="${idx}" onchange="toggleDataset(this)"> <span class="dot" style="background:${color}"></span> L${lvl+1} Boss reach %: <strong id="live-boss-pct-${lvl}" style="color:${color}">—</strong></label>`;
            }).join("");
        }
        // Update table headers
        const thContainer = $("#boss-table-headers");
        if (thContainer) {
            thContainer.outerHTML = seenLevels.map(lvl => `<th>L${lvl+1} Boss reach %</th>`).join("");
        }
    }

    const data = rewardChart.data;
    data.labels = ranged.map((_, i) => startEp + i + 1);
    data.datasets[0].data = ranged;
    data.datasets[1].data = avgRewards;
    data.datasets[2].data = avgSurv;

    // Boss reach % per level
    seenLevels.forEach((lvl, i) => {
        const bossForLevel = rangedBoss.map(v => v === lvl);  // strict ===, false !== 0
        const pct = computeRollingPct(bossForLevel);
        data.datasets[3 + i].data = pct;
        const el = $(`#live-boss-pct-${lvl}`);
        if (el && pct.length > 0) el.textContent = pct[pct.length - 1].toFixed(0) + '%';
    });

    rewardChart.update();

    const elAvgR = $("#live-avg-reward");
    const elAvgS = $("#live-avg-survival");
    if (elAvgR && avgRewards.length > 0) elAvgR.textContent = avgRewards[avgRewards.length - 1].toFixed(0);
    if (elAvgS && avgSurv.length > 0) elAvgS.textContent = avgSurv[avgSurv.length - 1].toFixed(0);

    updateRewardTable(ranged, rangedSurv, rangedBoss, startEp, seenLevels);
}

function setChartView(view) {
    chartViewMode = view;
    localStorage.setItem("contra_chartView", view);
    $("#chart-view").style.display = view === "chart" ? "" : "none";
    $("#table-view").style.display = view === "table" ? "" : "none";
    $("#btn-chart-view").classList.toggle("active", view === "chart");
    $("#btn-table-view").classList.toggle("active", view === "table");
}

function updateRewardTable(rewards, survival, boss, startEp, seenLevels) {
    const tbody = document.querySelector("#reward-table tbody");
    if (!tbody || rewards.length === 0) return;

    const bucketSize = rollingWindow;
    const rows = [];
    const levels = seenLevels || [];
    for (let i = 0; i < rewards.length; i += bucketSize) {
        const chunk = rewards.slice(i, i + bucketSize);
        const survChunk = survival.slice(i, i + bucketSize);
        const bossChunk = boss.slice(i, i + bucketSize);
        const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
        const avgSurv = survChunk.length > 0 ? survChunk.reduce((a, b) => a + b, 0) / survChunk.length : 0;
        const best = Math.max(...chunk);
        const bossCols = levels.map(lvl => {
            const pct = bossChunk.length > 0 ? (100 * bossChunk.filter(v => v === lvl).length / bossChunk.length).toFixed(0) : 0;
            return `<td>${pct}%</td>`;
        }).join("");
        const epFrom = startEp + i + 1;
        const epTo = startEp + Math.min(i + bucketSize, rewards.length);
        rows.push(`<tr><td>${epFrom}–${epTo}</td><td>${avg.toFixed(1)}</td><td>${avgSurv.toFixed(0)}</td><td>${best.toFixed(1)}</td>${bossCols}<td>${chunk.length}</td></tr>`);
    }
    tbody.innerHTML = rows.reverse().join("");
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
            allBoss = data.boss_history || [];
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
            allBoss = data.boss_history || [];
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
    setTimeout(() => { $("#btn-save").textContent = "Checkpoint"; }, 2000);
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
    $("#btn-replay").textContent = "Best Run";
}

async function toggleAutoRestart() {
    const res = await fetch("/api/auto-restart", { method: "POST" });
    const data = await res.json();
    const btn = $("#btn-auto-restart");
    btn.classList.toggle("active", data.auto_restart);
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
    "Player X", "Player Y", "Weapon", "In air",
    "Enemy1 dX", "Enemy1 dY", "Enemy1 velX", "Enemy1 HP",
    "Enemy2 dX", "Enemy2 dY", "Enemy2 velX", "Enemy2 HP",
    "Enemy3 dX", "Enemy3 dY", "Enemy3 velX", "Enemy3 HP",
    "Bullet1 dX", "Bullet1 dY",
    "Bullet2 dX", "Bullet2 dY",
    "Bullet3 dX", "Bullet3 dY",
    "Enemies count", "Nearest dist", "Invincible",
    "Edge fall", "Y velocity",
    "Level progress"
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

// === Coordinate highlight on hover ===
function coordSpan(x, y, text) {
    return `<span class="coord-hover" data-cx="${x}" data-cy="${y}" style="text-decoration:underline dotted;cursor:crosshair">${text || `(${x},${y})`}</span>`;
}

document.addEventListener("mouseover", (e) => {
    const t = e.target.closest(".coord-hover");
    if (t) highlightCoord = { x: parseInt(t.dataset.cx), y: parseInt(t.dataset.cy) };
});
document.addEventListener("mouseout", (e) => {
    const t = e.target.closest(".coord-hover");
    if (t) highlightCoord = null;
});

function copyEvent(el) {
    const text = el.dataset.text;
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
    } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
    }
}

// === Sub-tabs (Run Log) ===
function switchSubTab(btn) {
    const parent = btn.closest(".tab-content");
    parent.querySelectorAll(".subtab").forEach(t => t.classList.remove("active"));
    parent.querySelectorAll(".subtab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("subtab-" + btn.dataset.subtab).classList.add("active");
}

// === Keyboard shortcuts (admin only) ===
document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea")) return;
    if (e.code === "Space") {
        e.preventDefault();
        togglePause();
    } else if (e.code === "ArrowRight") {
        e.preventDefault();
        fetch("/api/step", { method: "POST" });
    }
});

// === Init ===
initChart();
setChartView(chartViewMode);
connectFrameWS();
connectStatsWS();
loadHistory();
loadConfig();
