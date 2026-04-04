/**
 * AI Controller for Adaptive Traffic Signal Management
 * Features:
 *   - Fairness-aware priority accumulation (prevents lane starvation)
 *   - Q-Learning agent for optimal signal timing
 *   - Real-time emissions model (CO₂, Fuel, NOx)
 *   - Baseline vs AI comparison engine
 *   - Scalability projections
 *   - Backend status monitoring & ML output panel
 *   - Ambulance priority override with animated canvas vehicle
 */

// ─── Backend API Integration ─────────────────────────────────────────
const ML_API_URL = 'http://localhost:8000';

// ─── Ambulance State ──────────────────────────────────────────────────
let ambulanceActive = null; // { direction, startTime } | null
let ambulanceClearTimer = null;

// ─── Emissions Model Constants ─────────────────────────────────────────────
const EMISSIONS = {
    // Per vehicle per second
    IDLE: { co2: 2.3, fuel: 0.0008, nox: 0.015 },   // grams, liters, mg
    MOVING: { co2: 0.5, fuel: 0.0002, nox: 0.003 },
};

// ─── Q-Learning Configuration ──────────────────────────────────────────────
const Q_CONFIG = {
    alpha: 0.15,        // learning rate
    gamma: 0.9,         // discount factor
    epsilon: 0.3,       // initial exploration rate
    epsilonDecay: 0.998,
    epsilonMin: 0.05,
    queueBins: [0, 3, 7, 12, 20],    // discretization
    redBins: [0, 10, 30, 60, 100],    // seconds
};

// ─── Fairness Configuration ────────────────────────────────────────────────
const FAIR_CONFIG = {
    alpha: 0.6,             // congestion weight
    beta: 0.4,              // fairness/starvation weight
    starvationThreshold: 120, // seconds — force green after this
    minGreen: 7,            // seconds
    maxGreen: 60,           // seconds
};

// ─── Comparison / Baseline ─────────────────────────────────────────────────
const comparisonData = {
    baseline: null,     // { avgWait, co2, fuel, nox, maxWait, duration }
    ai: null,
    isRunningBaseline: false,
    baselineTimer: 0,
    baselineDuration: 60,   // seconds of baseline to run
    baselinePhaseTimer: 0,
    baselinePhaseInterval: 30, // fixed 30s each phase
};

// ─── Emissions Accumulator ─────────────────────────────────────────────────
const emissionsTotal = { co2: 0, fuel: 0, nox: 0 };

// ─── Per-Lane State ────────────────────────────────────────────────────────
const laneState = {
    N: { redDuration: 0, priority: 0 },
    S: { redDuration: 0, priority: 0 },
    E: { redDuration: 0, priority: 0 },
    W: { redDuration: 0, priority: 0 },
};

// ─── Chart Data (sparklines) ───────────────────────────────────────────────
const chartData = {
    waitTime: [],
    co2Rate: [],
    maxPoints: 60,
};

// ─── Q-Table ───────────────────────────────────────────────────────────────
const qTable = {};
let currentEpsilon = Q_CONFIG.epsilon;

function discretize(value, bins) {
    for (let i = bins.length - 1; i >= 0; i--) {
        if (value >= bins[i]) return i;
    }
    return 0;
}

function getStateKey(qNS, qEW, maxRedDuration) {
    const qNSBin = discretize(qNS, Q_CONFIG.queueBins);
    const qEWBin = discretize(qEW, Q_CONFIG.queueBins);
    const redBin = discretize(maxRedDuration, Q_CONFIG.redBins);
    return `${qNSBin}_${qEWBin}_${redBin}`;
}

function getQValue(stateKey, action) {
    if (!qTable[stateKey]) qTable[stateKey] = { KEEP: 0, SWITCH: 0 };
    return qTable[stateKey][action];
}

function setQValue(stateKey, action, value) {
    if (!qTable[stateKey]) qTable[stateKey] = { KEEP: 0, SWITCH: 0 };
    qTable[stateKey][action] = value;
}

function getBestAction(stateKey) {
    const keep = getQValue(stateKey, 'KEEP');
    const sw = getQValue(stateKey, 'SWITCH');
    return keep >= sw ? 'KEEP' : 'SWITCH';
}

// ─── Sparkline Drawing ─────────────────────────────────────────────────────
function drawSparkline(canvasId, data, color, maxVal) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx2 = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * 2;
    const h = canvas.height = canvas.clientHeight * 2;
    ctx2.clearRect(0, 0, w, h);

    if (data.length < 2) return;
    const max = maxVal || Math.max(...data, 1);
    const step = w / (chartData.maxPoints - 1);

    // Fill gradient
    const grad = ctx2.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color + '40');
    grad.addColorStop(1, color + '05');

    ctx2.beginPath();
    ctx2.moveTo(0, h);
    data.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / max) * (h - 4);
        if (i === 0) ctx2.lineTo(x, y);
        else ctx2.lineTo(x, y);
    });
    ctx2.lineTo((data.length - 1) * step, h);
    ctx2.closePath();
    ctx2.fillStyle = grad;
    ctx2.fill();

    // Line
    ctx2.beginPath();
    data.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / max) * (h - 4);
        if (i === 0) ctx2.moveTo(x, y);
        else ctx2.lineTo(x, y);
    });
    ctx2.strokeStyle = color;
    ctx2.lineWidth = 2;
    ctx2.stroke();

    // Dot on last point
    if (data.length > 0) {
        const lastX = (data.length - 1) * step;
        const lastY = h - (data[data.length - 1] / max) * (h - 4);
        ctx2.beginPath();
        ctx2.arc(lastX, lastY, 4, 0, Math.PI * 2);
        ctx2.fillStyle = color;
        ctx2.fill();
    }
}

// ─── AI Controller Class ───────────────────────────────────────────────────
class AIController {
    constructor() {
        this.lastSwitchTime = Date.now();
        this.lastTickTime = Date.now();
        this.totalReward = 0;
        this.logEl = document.getElementById('ai-log');
        this.rewardEl = document.getElementById('m-reward');
        this.switching = false;
        this.prevStateKey = null;
        this.prevAction = null;
        this.episodeSteps = 0;
        this.starvationEvents = 0;
        this.peakEmission = 0;

        // Tick loop — 500ms for priority updates, emissions, charts
        setInterval(() => this.tick(), 500);
        // Decision loop — 1.5s for signal decisions
        setInterval(() => this.update(), 1500);
        // Chart update — 1s
        setInterval(() => this.updateCharts(), 1000);
        // Backend status polling — every 5s
        setInterval(() => this.checkBackendStatus(), 5000);

        // Initial checks
        this.checkBackendStatus();
        this.fetchModelInfo();

        this.log('🧠 AI Traffic Optimizer v3.0 — Fairness-Aware + Ambulance Priority');
        this.log('📊 Q-Learning + Priority Accumulation enabled.');
        this.log('⏳ Enable AI Mode to start adaptive control.');
    }

    log(msg) {
        if (!this.logEl) return;
        const ts = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const line = document.createElement('div');
        line.className = 'ai-line';
        line.innerHTML = `<span class="ai-ts">${ts}</span> ${msg}`;
        this.logEl.insertBefore(line, this.logEl.firstChild);
        while (this.logEl.children.length > 40) this.logEl.lastChild.remove();
    }

    getQueues() {
        const q = { N: 0, S: 0, E: 0, W: 0 };
        vehicles.forEach(v => { if (v.waiting) q[v.origin]++; });
        return { ...q, NS: q.N + q.S, EW: q.E + q.W };
    }

    // ─── Per-tick updates (emissions, fairness, lane tracking) ──────
    tick() {
        if (!isRunning) return;

        const now = Date.now();
        const dt = (now - this.lastTickTime) / 1000; // seconds
        this.lastTickTime = now;

        // Update emissions
        let tickCO2 = 0, tickFuel = 0, tickNOx = 0;
        vehicles.forEach(v => {
            if (!v.active) return;
            const rates = v.waiting ? EMISSIONS.IDLE : EMISSIONS.MOVING;
            tickCO2 += rates.co2 * dt * timeScale;
            tickFuel += rates.fuel * dt * timeScale;
            tickNOx += rates.nox * dt * timeScale;
        });
        emissionsTotal.co2 += tickCO2;
        emissionsTotal.fuel += tickFuel;
        emissionsTotal.nox += tickNOx;

        // Update per-lane red durations
        const dirs = ['N', 'S', 'E', 'W'];
        dirs.forEach(d => {
            const isNS = (d === 'N' || d === 'S');
            const mySignal = isNS ? signals.NS : signals.EW;
            if (mySignal !== SIGNAL.GREEN) {
                laneState[d].redDuration += dt * timeScale;
            } else {
                laneState[d].redDuration = 0;
            }
        });

        // Calculate priorities
        const q = this.getQueues();
        const maxQ = Math.max(1, q.N, q.S, q.E, q.W);
        const maxRed = Math.max(1, ...dirs.map(d => laneState[d].redDuration));
        dirs.forEach(d => {
            const qCount = q[d];
            const normQ = qCount / maxQ;
            const normRed = laneState[d].redDuration / Math.max(maxRed, 1);
            laneState[d].priority = FAIR_CONFIG.alpha * normQ + FAIR_CONFIG.beta * normRed;
        });

        // Update emissions UI
        this.updateEmissionsUI();
        this.updateFairnessUI();

        // Baseline timer logic
        if (comparisonData.isRunningBaseline) {
            comparisonData.baselineTimer += dt * timeScale;
            comparisonData.baselinePhaseTimer += dt * timeScale;

            // Fixed phase switching at 30s intervals
            if (comparisonData.baselinePhaseTimer >= comparisonData.baselinePhaseInterval) {
                comparisonData.baselinePhaseTimer = 0;
                if (signals.NS === SIGNAL.GREEN) {
                    signals.NS = SIGNAL.RED;
                    signals.EW = SIGNAL.GREEN;
                } else {
                    signals.EW = SIGNAL.RED;
                    signals.NS = SIGNAL.GREEN;
                }
                updateSignalLights();
            }

            this.updateBaselineProgress();

            // End baseline
            if (comparisonData.baselineTimer >= comparisonData.baselineDuration) {
                this.endBaseline();
            }
        }
    }

    updateEmissionsUI() {
        const co2El = document.getElementById('em-co2');
        const fuelEl = document.getElementById('em-fuel');
        const noxEl = document.getElementById('em-nox');
        if (co2El) co2El.innerText = emissionsTotal.co2.toFixed(1);
        if (fuelEl) fuelEl.innerText = emissionsTotal.fuel.toFixed(4);
        if (noxEl) noxEl.innerText = emissionsTotal.nox.toFixed(2);
    }

    updateFairnessUI() {
        const dirs = ['N', 'S', 'E', 'W'];
        const maxPrio = Math.max(0.01, ...dirs.map(d => laneState[d].priority));
        dirs.forEach(d => {
            const bar = document.getElementById(`fp-${d}`);
            const val = document.getElementById(`fpv-${d}`);
            const redEl = document.getElementById(`frd-${d}`);
            const row = document.getElementById(`fr-${d}`);
            if (bar) {
                const pct = (laneState[d].priority / maxPrio) * 100;
                bar.style.width = pct + '%';
                // Color based on starvation level
                const redSec = laneState[d].redDuration;
                if (redSec > FAIR_CONFIG.starvationThreshold * 0.8) {
                    bar.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
                } else if (redSec > FAIR_CONFIG.starvationThreshold * 0.5) {
                    bar.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
                } else {
                    bar.style.background = 'linear-gradient(90deg, #22c55e, #3b82f6)';
                }
            }
            if (val) val.innerText = laneState[d].priority.toFixed(2);
            if (redEl) redEl.innerText = Math.floor(laneState[d].redDuration) + 's';
            if (row) {
                row.classList.toggle('starvation-warn', laneState[d].redDuration > FAIR_CONFIG.starvationThreshold * 0.8);
            }
        });

        // Update starvation count display
        const starvEl = document.getElementById('starvation-count');
        if (starvEl) starvEl.innerText = this.starvationEvents;
    }

    updateCharts() {
        if (!isRunning) return;

        // Avg wait from current vehicles
        const waitingVehicles = vehicles.filter(v => v.waiting && v.active);
        const avgWait = waitingVehicles.length > 0
            ? waitingVehicles.reduce((s, v) => s + (Date.now() - v.waitStart) / 1000, 0) / waitingVehicles.length
            : 0;

        chartData.waitTime.push(parseFloat(avgWait.toFixed(1)));
        if (chartData.waitTime.length > chartData.maxPoints) chartData.waitTime.shift();

        // CO2 rate (per second based on last tick)
        const idleCount = vehicles.filter(v => v.active && v.waiting).length;
        const movingCount = vehicles.filter(v => v.active && !v.waiting).length;
        const co2Rate = idleCount * EMISSIONS.IDLE.co2 + movingCount * EMISSIONS.MOVING.co2;
        chartData.co2Rate.push(parseFloat(co2Rate.toFixed(1)));
        if (chartData.co2Rate.length > chartData.maxPoints) chartData.co2Rate.shift();

        drawSparkline('chart-wait', chartData.waitTime, '#3b82f6');
        drawSparkline('chart-co2', chartData.co2Rate, '#22c55e');
    }

    // ─── Backend Status Polling ─────────────────────────────────
    async checkBackendStatus() {
        const dot   = document.getElementById('backend-dot');
        const label = document.getElementById('backend-status-label');
        const latEl = document.getElementById('backend-latency');
        const stEl  = document.getElementById('backend-states');

        const t0 = Date.now();
        try {
            const res  = await fetch(`${ML_API_URL}/health`);
            const data = await res.json();
            const ms   = Date.now() - t0;

            if (dot)   { dot.className = 'backend-dot connected'; }
            if (label) { label.innerText = 'CONNECTED'; label.style.color = '#4ade80'; }
            if (latEl) { latEl.innerText = ms + ' ms'; }
            if (stEl)  { stEl.innerText  = data.q_table_states ?? '—'; }
        } catch {
            if (dot)   { dot.className = 'backend-dot disconnected'; }
            if (label) { label.innerText = 'OFFLINE'; label.style.color = '#f87171'; }
            if (latEl) { latEl.innerText = '—'; }
            if (stEl)  { stEl.innerText  = '—'; }
        }
    }

    async fetchModelInfo() {
        try {
            const res  = await fetch(`${ML_API_URL}/model-info`);
            const data = await res.json();
            const modEl = document.getElementById('backend-model');
            const verEl = document.getElementById('backend-version');
            if (modEl) modEl.innerText = data.model ?? 'Q-Learning';
            if (verEl) verEl.innerText = 'v' + (data.version ?? '?');
        } catch { /* offline — ignore */ }
    }

    // ─── ML Output Panel ───────────────────────────────────────────
    updateMLOutputPanel(data) {
        const actionEl  = document.getElementById('ml-action');
        const badgeEl   = document.getElementById('ml-action-badge');
        const carbonEl  = document.getElementById('ml-carbon');
        const peakEl    = document.getElementById('ml-peak');
        const explEl    = document.getElementById('ml-explanation');

        const ACTION_LABELS = {
            'keep_green':   'KEEP GREEN',
            'switch_phase': 'SWITCH ↔',
            'extend_green': 'EXTEND GREEN',
        };
        const ACTION_COLORS = {
            'keep_green':   '#4ade80',
            'switch_phase': '#f59e0b',
            'extend_green': '#60a5fa',
        };

        if (data.ambulance_override) {
            if (actionEl)  { actionEl.innerText  = '🚑 AMBULANCE'; actionEl.style.color = '#f87171'; }
            if (badgeEl)   { badgeEl.innerText   = '🚨 OVERRIDE'; badgeEl.style.color = '#f87171'; }
        } else {
            const label = ACTION_LABELS[data.action] || data.action.toUpperCase();
            const color = ACTION_COLORS[data.action] || '#c084fc';
            if (actionEl)  { actionEl.innerText  = label; actionEl.style.color = color; }
            if (badgeEl)   { badgeEl.innerText   = label; badgeEl.style.color  = color; }
        }

        if (carbonEl) carbonEl.innerText = (data.carbon_intensity ?? '—') + '×';
        if (explEl)   explEl.innerText   = data.explanation || '';

        // Track peak emission (from simulate responses or local estimation)
        if (data.carbon_intensity) {
            const q = this.getQueues();
            const EMISSION_FACTOR = 0.21;
            const est = (q.NS + q.EW) * EMISSION_FACTOR * data.carbon_intensity;
            if (est > this.peakEmission) {
                this.peakEmission = est;
            }
            if (peakEl) peakEl.innerText = this.peakEmission.toFixed(3) + ' kg';
        }
    }

    // ─── Main Decision Loop ──────────────────────────────────────────
    async update() {
        if (!isRunning || this.switching) return;

        // During baseline, skip AI logic
        if (comparisonData.isRunningBaseline) return;

        if (!isAIEnabled) return;

        const now = Date.now();
        const elapsed = (now - this.lastSwitchTime) / 1000;
        const q = this.getQueues();

        // ─── Ambulance Override (bypasses all normal Q-learning logic) ───
        if (ambulanceActive) {
            const ambDir      = ambulanceActive.direction;                  // 'NS' | 'EW'
            const currentPhase = signals.NS === SIGNAL.GREEN ? 'NS' : 'EW';
            if (currentPhase !== ambDir) {
                this.log(`🚑 AMBULANCE PRIORITY: ${ambDir} — forcing immediate green override.`);
                await this.performSwitch();
            }
            return; // Do not run normal Q-learning while ambulance is active
        }

        // Check starvation — force switch if any lane is starved
        const maxRedNS = Math.max(laneState.N.redDuration, laneState.S.redDuration);
        const maxRedEW = Math.max(laneState.E.redDuration, laneState.W.redDuration);
        const currentPhase = signals.NS === SIGNAL.GREEN ? 'NS' : 'EW';

        if (currentPhase === 'NS' && maxRedEW > FAIR_CONFIG.starvationThreshold && elapsed >= FAIR_CONFIG.minGreen) {
            this.starvationEvents++;
            this.log(`🚨 STARVATION! EW red for ${Math.floor(maxRedEW)}s — forcing green.`);
            await this.performSwitch();
            return;
        }
        if (currentPhase === 'EW' && maxRedNS > FAIR_CONFIG.starvationThreshold && elapsed >= FAIR_CONFIG.minGreen) {
            this.starvationEvents++;
            this.log(`🚨 STARVATION! NS red for ${Math.floor(maxRedNS)}s — forcing green.`);
            await this.performSwitch();
            return;
        }

        // Minimum green enforcement
        if (elapsed < FAIR_CONFIG.minGreen) return;

        // Maximum green enforcement
        if (elapsed > FAIR_CONFIG.maxGreen) {
            this.log(`⏰ Max green (${FAIR_CONFIG.maxGreen}s) reached. Rotating.`);
            await this.performSwitch();
            return;
        }

        // ─── Q-Learning Decision ────────────────────────────────
        const maxRed = currentPhase === 'NS' ? maxRedEW : maxRedNS;
        const stateKey = getStateKey(q.NS, q.EW, maxRed);

        // Priority-based reasoning for logging
        const prioNS = laneState.N.priority + laneState.S.priority;
        const prioEW = laneState.E.priority + laneState.W.priority;

        let action = 'KEEP';
        let isFallback = false;

        try {
            // Try fetching decision from the ML backend API
            const apiPhase = currentPhase === 'NS' ? 0 : 1;
            const payload = {
                queue_NS: q.NS,
                queue_EW: q.EW,
                red_NS: Math.floor(maxRedNS),
                red_EW: Math.floor(maxRedEW),
                phase: apiPhase,
                hour: new Date().getHours()
            };

            const res = await fetch(`${ML_API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error('API Response not OK');
            const data = await res.json();

            // Update ML output panel with every backend response
            this.updateMLOutputPanel(data);

            // Map the API action to our local terms
            if (data.action === 'switch_phase') {
                action = 'SWITCH';
            } else if (data.action === 'extend_green' || data.action === 'keep_green') {
                action = 'KEEP';
            }

            // Update local episode counter
            this.episodeSteps++;

            // Logging for the ML outcome
            if (data.ambulance_override) {
                // Ambulance override logging (should not happen here, caught above, but defensive)
                this.log(`🚑 ML: AMBULANCE PRIORITY → ${data.action}`);
                if (action === 'SWITCH') await this.performSwitch();
            } else if (action === 'SWITCH') {
                 const target = currentPhase === 'NS' ? 'EW' : 'NS';
                 this.log(`🌐 ML Backend: SWITCH → ${target} | ${data.explanation}`);
                 await this.performSwitch();
            } else if (this.episodeSteps % 4 === 0) {
                 this.log(`🌐 ML Backend: KEEP ${currentPhase} | ${data.explanation} (CO₂ ×${data.carbon_intensity.toFixed(2)})`);
            }

        } catch (e) {
            // Setup fallback mechanism using local Q-Table
            isFallback = true;
            this.log(`⚠️ ML API Unreachable — Falling back to Local Agent...`);

            // Reward for previous state-action
            if (this.prevStateKey !== null && this.prevAction !== null) {
                const totalWaiting = q.NS + q.EW;
                const fairnessBonus = -Math.abs(maxRedNS - maxRedEW) * 0.1;
                const reward = -(totalWaiting) + fairnessBonus;
                this.totalReward += reward;

                const oldQ = getQValue(this.prevStateKey, this.prevAction);
                const bestFuture = Math.max(getQValue(stateKey, 'KEEP'), getQValue(stateKey, 'SWITCH'));
                const newQ = oldQ + Q_CONFIG.alpha * (reward + Q_CONFIG.gamma * bestFuture - oldQ);
                setQValue(this.prevStateKey, this.prevAction, newQ);
            }

            // ε-greedy action selection
            if (Math.random() < currentEpsilon) {
                action = Math.random() < 0.5 ? 'KEEP' : 'SWITCH';
            } else {
                action = getBestAction(stateKey);
            }

            // Decay epsilon
            currentEpsilon = Math.max(Q_CONFIG.epsilonMin, currentEpsilon * Q_CONFIG.epsilonDecay);

            this.episodeSteps++;

            // Update reward display
            if (this.rewardEl) this.rewardEl.innerText = Math.max(0, Math.floor(1000 + this.totalReward));

            if (action === 'SWITCH') {
                const target = currentPhase === 'NS' ? 'EW' : 'NS';
                this.log(`⚡ LOCAL Agent: SWITCH → ${target} | P(NS)=${prioNS.toFixed(2)} P(EW)=${prioEW.toFixed(2)} ε=${currentEpsilon.toFixed(3)}`);
                await this.performSwitch();
            } else {
                // Periodically log KEEP decisions
                if (this.episodeSteps % 4 === 0) {
                    this.log(`✅ LOCAL Agent: KEEP ${currentPhase} green | Q=${q.NS}/${q.EW} | ε=${currentEpsilon.toFixed(3)}`);
                }
            }
        }

        // Store state for next iteration (used by fallback)
        this.prevStateKey = stateKey;
        this.prevAction = action;

        // Update scalability projections
        this.updateScalability();
    }

    async performSwitch() {
        if (this.switching) return;
        this.switching = true;

        if (signals.NS === SIGNAL.GREEN) {
            signals.NS = SIGNAL.YELLOW;
            this.log('🟡 NS → YELLOW');
            updateSignalLights();
            await sleep(2000);
            signals.NS = SIGNAL.RED;
            signals.EW = SIGNAL.GREEN;
            this.log('🔴 NS: RED | 🟢 EW: GREEN');
        } else {
            signals.EW = SIGNAL.YELLOW;
            this.log('🟡 EW → YELLOW');
            updateSignalLights();
            await sleep(2000);
            signals.EW = SIGNAL.RED;
            signals.NS = SIGNAL.GREEN;
            this.log('🔴 EW: RED | 🟢 NS: GREEN');
        }

        updateSignalLights();
        this.lastSwitchTime = Date.now();
        this.switching = false;
    }

    // ─── Baseline Comparison ────────────────────────────────────────
    startBaseline() {
        if (comparisonData.isRunningBaseline) return;

        // Reset emissions for baseline measurement
        emissionsTotal.co2 = 0;
        emissionsTotal.fuel = 0;
        emissionsTotal.nox = 0;
        metrics.totalWait = 0;
        metrics.throughput = 0;
        cleared = 0;
        comparisonData.baselineTimer = 0;
        comparisonData.baselinePhaseTimer = 0;
        comparisonData.isRunningBaseline = true;

        // Disable AI during baseline
        isAIEnabled = false;
        document.getElementById('btn-ai')?.classList.remove('active');
        const aiModeBadge = document.getElementById('ai-mode-badge');
        if (aiModeBadge) aiModeBadge.innerText = 'BASELINE';
        const ctrlNote = document.getElementById('ctrl-mode-note');
        if (ctrlNote) ctrlNote.innerText = 'BASELINE';

        // Start with NS green, fixed 30s each
        signals.NS = SIGNAL.GREEN;
        signals.EW = SIGNAL.RED;
        updateSignalLights();

        this.log('📏 BASELINE started — Fixed 30s/30s timer running...');

        // Show progress
        const progEl = document.getElementById('baseline-progress');
        if (progEl) progEl.style.display = 'block';
    }

    updateBaselineProgress() {
        const progBar = document.getElementById('baseline-prog-bar');
        const progText = document.getElementById('baseline-prog-text');
        if (progBar) progBar.style.width = (comparisonData.baselineTimer / comparisonData.baselineDuration * 100) + '%';
        if (progText) progText.innerText = `${Math.floor(comparisonData.baselineTimer)}/${comparisonData.baselineDuration}s`;
    }

    endBaseline() {
        comparisonData.isRunningBaseline = false;
        const avgWait = metrics.throughput > 0 ? (metrics.totalWait / metrics.throughput) : 0;
        const maxRedAll = Math.max(laneState.N.redDuration, laneState.S.redDuration, laneState.E.redDuration, laneState.W.redDuration);

        comparisonData.baseline = {
            avgWait: parseFloat(avgWait.toFixed(1)),
            co2: parseFloat(emissionsTotal.co2.toFixed(1)),
            fuel: parseFloat(emissionsTotal.fuel.toFixed(4)),
            nox: parseFloat(emissionsTotal.nox.toFixed(2)),
            maxWait: parseFloat(maxRedAll.toFixed(1)),
            duration: comparisonData.baselineDuration,
            throughput: metrics.throughput,
        };

        this.log(`📏 BASELINE complete! AvgWait=${comparisonData.baseline.avgWait}s CO₂=${comparisonData.baseline.co2}g`);

        // Reset for AI run
        emissionsTotal.co2 = 0;
        emissionsTotal.fuel = 0;
        emissionsTotal.nox = 0;
        metrics.totalWait = 0;
        metrics.throughput = 0;
        cleared = 0;

        // Enable AI
        isAIEnabled = true;
        document.getElementById('btn-ai')?.classList.add('active');
        const aiModeBadge = document.getElementById('ai-mode-badge');
        if (aiModeBadge) aiModeBadge.innerText = 'AI';
        const ctrlNote = document.getElementById('ctrl-mode-note');
        if (ctrlNote) ctrlNote.innerText = 'AI';

        // Hide progress
        const progEl = document.getElementById('baseline-progress');
        if (progEl) progEl.style.display = 'none';

        this.log('🤖 AI MODE activated — recording AI metrics for comparison...');

        // After same duration, record AI metrics
        setTimeout(() => {
            this.recordAIMetrics();
        }, comparisonData.baselineDuration * 1000);
    }

    recordAIMetrics() {
        const avgWait = metrics.throughput > 0 ? (metrics.totalWait / metrics.throughput) : 0;
        comparisonData.ai = {
            avgWait: parseFloat(avgWait.toFixed(1)),
            co2: parseFloat(emissionsTotal.co2.toFixed(1)),
            fuel: parseFloat(emissionsTotal.fuel.toFixed(4)),
            nox: parseFloat(emissionsTotal.nox.toFixed(2)),
            duration: comparisonData.baselineDuration,
            throughput: metrics.throughput,
        };

        this.log(`🤖 AI metrics recorded! AvgWait=${comparisonData.ai.avgWait}s CO₂=${comparisonData.ai.co2}g`);
        this.updateComparisonUI();
    }

    updateComparisonUI() {
        if (!comparisonData.baseline || !comparisonData.ai) return;

        const b = comparisonData.baseline;
        const a = comparisonData.ai;

        const setComparison = (id, baseVal, aiVal, unit, reverse) => {
            const baseEl = document.getElementById(`cmp-base-${id}`);
            const aiEl = document.getElementById(`cmp-ai-${id}`);
            const impEl = document.getElementById(`cmp-imp-${id}`);
            if (baseEl) baseEl.innerText = baseVal + unit;
            if (aiEl) aiEl.innerText = aiVal + unit;
            if (impEl) {
                const pct = baseVal > 0 ? ((baseVal - aiVal) / baseVal * 100).toFixed(0) : 0;
                const improved = reverse ? pct < 0 : pct > 0;
                impEl.innerText = (improved ? '↓' : '↑') + Math.abs(pct) + '%';
                impEl.className = 'cmp-imp ' + (improved ? 'improved' : 'worse');
            }
        };

        setComparison('wait', b.avgWait, a.avgWait, 's');
        setComparison('co2', b.co2, a.co2, 'g');
        setComparison('fuel', b.fuel, a.fuel, 'L');
        setComparison('thru', b.throughput, a.throughput, '', true);

        const compPanel = document.getElementById('comparison-panel');
        if (compPanel) compPanel.style.display = 'block';
    }

    updateScalability() {
        // CO₂ saved per hour if AI saves ~17% over baseline
        const hourlyPerIntersection = emissionsTotal.co2 > 0
            ? (emissionsTotal.co2 * 0.17 * 3600 / Math.max(1, (Date.now() - metrics.startTime) / 1000))
            : 900; // default estimate: 900g/hr

        const intersections = 500;
        const hourly = (hourlyPerIntersection * intersections / 1000).toFixed(1); // kg
        const daily = (hourlyPerIntersection * intersections * 24 / 1000000).toFixed(1); // tons

        const hrEl = document.getElementById('scale-hourly');
        const dayEl = document.getElementById('scale-daily');
        if (hrEl) hrEl.innerText = hourly + ' kg';
        if (dayEl) dayEl.innerText = daily + ' tons';
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Instantiate ───────────────────────────────────────────────────────────
const ai = new AIController();

// ─── Baseline Button Binding ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const btnBaseline = document.getElementById('btn-baseline');
    if (btnBaseline) {
        btnBaseline.onclick = () => {
            if (!isRunning) {
                document.getElementById('btn-start')?.click();
            }
            ai.startBaseline();
        };
    }
});

// ─── Ambulance Dispatch ────────────────────────────────────────────────────
function dispatchAmbulance(direction) {
    if (ambulanceActive) clearAmbulance();

    ambulanceActive = { direction, startTime: Date.now() };

    // UI: show badge, active buttons, clear button, timer
    const badge     = document.getElementById('amb-status-badge');
    const clearBtn  = document.getElementById('btn-amb-clear');
    const timerEl   = document.getElementById('amb-timer');
    const btnNS     = document.getElementById('btn-amb-ns');
    const btnEW     = document.getElementById('btn-amb-ew');

    if (badge)    badge.style.display    = 'inline-block';
    if (clearBtn) clearBtn.style.display = 'inline-block';
    if (btnNS)    btnNS.classList.toggle('active-amb', direction === 'NS');
    if (btnEW)    btnEW.classList.toggle('active-amb', direction === 'EW');

    // Countdown timer update
    const DURATION_S = 12;
    const timerInterval = setInterval(() => {
        if (!ambulanceActive) { clearInterval(timerInterval); return; }
        const elapsed = (Date.now() - ambulanceActive.startTime) / 1000;
        const left    = Math.max(0, DURATION_S - elapsed);
        if (timerEl) timerEl.innerText = left > 0 ? `Auto-clear: ${left.toFixed(0)}s` : '';
    }, 300);

    // Spawn an ambulance vehicle on the canvas
    if (typeof spawnAmbulance === 'function') {
        spawnAmbulance(direction === 'NS' ? 'N' : 'E');
    }

    // Force AI mode ON so ambulance override is applied
    if (!isAIEnabled) {
        isAIEnabled = true;
        const btnAI = document.getElementById('btn-ai');
        if (btnAI) { btnAI.classList.add('active'); btnAI.innerText = '🤖 AI: ON'; }
        document.getElementById('ai-mode-badge')?.setAttribute('innerText', 'AI');
    }

    ai.log(`🚑 AMBULANCE DISPATCHED → ${direction} corridor! Immediate green priority.`);

    // Auto-clear after DURATION_S seconds
    if (ambulanceClearTimer) clearTimeout(ambulanceClearTimer);
    ambulanceClearTimer = setTimeout(() => {
        clearInterval(timerInterval);
        clearAmbulance();
        ai.log(`✅ Ambulance cleared. Normal AI control resuming.`);
    }, DURATION_S * 1000);
}

function clearAmbulance() {
    ambulanceActive    = null;
    if (ambulanceClearTimer) { clearTimeout(ambulanceClearTimer); ambulanceClearTimer = null; }

    const badge    = document.getElementById('amb-status-badge');
    const clearBtn = document.getElementById('btn-amb-clear');
    const timerEl  = document.getElementById('amb-timer');
    const btnNS    = document.getElementById('btn-amb-ns');
    const btnEW    = document.getElementById('btn-amb-ew');

    if (badge)    badge.style.display    = 'none';
    if (clearBtn) clearBtn.style.display = 'none';
    if (timerEl)  timerEl.innerText      = '';
    if (btnNS)    btnNS.classList.remove('active-amb');
    if (btnEW)    btnEW.classList.remove('active-amb');
}

// ─── Ambulance Button Bindings ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const btnAmbNS = document.getElementById('btn-amb-ns');
    const btnAmbEW = document.getElementById('btn-amb-ew');
    const btnClear = document.getElementById('btn-amb-clear');

    if (btnAmbNS) btnAmbNS.onclick = () => dispatchAmbulance('NS');
    if (btnAmbEW) btnAmbEW.onclick = () => dispatchAmbulance('EW');
    if (btnClear) btnClear.onclick = () => {
        clearAmbulance();
        ai.log('✕ Ambulance manually cleared.');
    };
});

