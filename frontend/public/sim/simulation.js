/**
 * Pure Web Traffic Simulation Engine
 * Handles vehicle physics, intersection rendering, and state management.
 * Features: Large 2-lane intersection, traffic pattern presets, emissions state tracking.
 */

const canvas = document.getElementById('trafficCanvas');
const ctx = canvas.getContext('2d');

// --- Configuration & Constants ---
const ROAD_WIDTH = 160;        // Wider roads for realistic 2-lane feel
const LANE_WIDTH = 70;
const VEHICLE_WIDTH = 20;
const VEHICLE_LENGTH = 36;
const STOP_DIST = 65;
const MAX_SPEED = 2.6;
const ACCEL = 0.055;
const DECEL = 0.16;
const LANE_OFFSET = 35;       // Offset between the two lanes

// Signal States
const SIGNAL = { RED: 'R', GREEN: 'G', YELLOW: 'Y' };

// --- State ---
let isRunning = false;
let isAIEnabled = false;
let timeScale = 1;
let vehicles = [];
let signals = { NS: SIGNAL.RED, EW: SIGNAL.GREEN };
let metrics = { total: 0, totalWait: 0, throughput: 0, startTime: Date.now() };
let cleared = 0;
let simElapsedTime = 0; // simulated seconds

// Traffic pattern presets
const TRAFFIC_PATTERNS = {
    balanced: { N: 0.25, S: 0.25, E: 0.25, W: 0.25, rate: 0.022 },
    rushNS:   { N: 0.40, S: 0.35, E: 0.15, W: 0.10, rate: 0.030 },
    rushEW:   { N: 0.12, S: 0.13, E: 0.40, W: 0.35, rate: 0.030 },
    burst:    { N: 0.25, S: 0.25, E: 0.25, W: 0.25, rate: 0.045 },
};
let currentPattern = 'balanced';

// --- Canvas Resizing ---
function resize() {
    const container = document.getElementById('canvas-container');
    const w = container.clientWidth;
    const h = container.clientHeight || w * 0.75;
    canvas.width = Math.max(w, 500);
    canvas.height = Math.max(h, 400);
}
window.addEventListener('resize', () => { resize(); });
resize();

// --- Vehicle Class ---
const CAR_COLORS = [
    '#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b',
    '#10b981', '#ef4444', '#f97316', '#a78bfa', '#67e8f9',
    '#14b8a6', '#e879f9', '#fb923c', '#38bdf8', '#4ade80'
];

class Vehicle {
    constructor(origin, lane) {
        this.id = Math.random().toString(36).substr(2, 9);
        this.origin = origin;
        // The road is 1 lane per direction. Ignore the random 'lane' argument.
        this.lane = 0; 
        
        this.turnDir = 'straight'; // Strictly straight to avoid turning collisions
        
        this.active = true;
        this.waiting = false;
        this.waitStart = 0;
        this.totalWait = 0;
        this.color = CAR_COLORS[Math.floor(Math.random() * CAR_COLORS.length)];
        this.speed = Math.random() * 0.5 + 1.6;
        this.setupCoordinates();
        
        this.hasTurned = false;
        this.targetAngle = this.angle;
        metrics.total++;
    }

    setupCoordinates() {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;

        // Force strictly right-hand traffic coordinates to prevent head-on collisions
        switch (this.origin) {
            case 'N':
                this.x = cx - LANE_OFFSET; // Right-hand traffic (left side of screen)
                this.y = -60 - Math.random() * 40;
                this.angle = 90;
                break;
            case 'S':
                this.x = cx + LANE_OFFSET; // Right-hand traffic (right side of screen)
                this.y = canvas.height + 60 + Math.random() * 40;
                this.angle = 270;
                break;
            case 'E':
                this.x = canvas.width + 60 + Math.random() * 40;
                this.y = cy - LANE_OFFSET; // Right-hand traffic (top side of screen)
                this.angle = 180;
                break;
            case 'W':
                this.x = -60 - Math.random() * 40;
                this.y = cy + LANE_OFFSET; // Right-hand traffic (bottom side of screen)
                this.angle = 0;
                break;
        }
    }

    distToStopLine() {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const hw = ROAD_WIDTH / 2;
        switch (this.origin) {
            case 'N': return (cy - hw) - this.y;
            case 'S': return this.y - (cy + hw);
            case 'E': return this.x - (cx + hw);
            case 'W': return (cx - hw) - this.x;
        }
    }

    isInIntersection() {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const hw = ROAD_WIDTH / 2;
        return (this.x > cx - hw && this.x < cx + hw && this.y > cy - hw && this.y < cy + hw);
    }

    update(others) {
        if (!this.active) return;

        const mySignal = (this.origin === 'N' || this.origin === 'S') ? signals.NS : signals.EW;
        let canGo = true;

        if (mySignal === SIGNAL.RED || mySignal === SIGNAL.YELLOW) {
            const dist = this.distToStopLine();
            const margin = mySignal === SIGNAL.YELLOW ? STOP_DIST * 0.5 : STOP_DIST;
            if (dist > 0 && dist < margin) canGo = false;
        }

        // Car following logic - robust euclidean distance based on spawn order
        let closestDist = 9999;
        const myIndex = others.indexOf(this);
        others.forEach((v, index) => {
            // Must be same origin to follow. No lane check needed since there is only 1 lane per origin.
            if (v.id === this.id || v.origin !== this.origin || !v.active) return;
            if (index > myIndex) return; // Ignore cars behind us
            
            const dist = Math.hypot(v.x - this.x, v.y - this.y);
            if (dist < closestDist) closestDist = dist;
        });
        if (closestDist < 50) canGo = false;

        // Speed calculation
        if (!canGo) {
            this.speed = Math.max(0, this.speed - DECEL * timeScale);
            if (this.speed === 0 && !this.waiting) {
                this.waiting = true;
                this.waitStart = Date.now();
            }
        } else {
            this.speed = Math.min(MAX_SPEED, this.speed + ACCEL * timeScale);
            if (this.waiting) {
                this.totalWait += (Date.now() - this.waitStart) / 1000;
                metrics.totalWait += this.totalWait;
                this.waiting = false;
            }
        }

        // Move vehicle (straight only, no turning collisions)
        const rad = this.angle * Math.PI / 180;
        this.x += Math.cos(rad) * this.speed * timeScale;
        this.y += Math.sin(rad) * this.speed * timeScale;

        // Deactivate when off-screen
        const pad = 150;
        if (this.x < -pad || this.x > canvas.width + pad || this.y < -pad || this.y > canvas.height + pad) {
            this.active = false;
            cleared++;
            metrics.throughput++;
        }
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle * Math.PI / 180);

        // Shadow
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.beginPath();
        ctx.roundRect(-VEHICLE_LENGTH / 2 + 2, -VEHICLE_WIDTH / 2 + 2, VEHICLE_LENGTH, VEHICLE_WIDTH, 5);
        ctx.fill();

        // Body
        const grad = ctx.createLinearGradient(-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2, VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2);
        grad.addColorStop(0, this.color);
        grad.addColorStop(1, shadeColor(this.color, -35));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2, VEHICLE_LENGTH, VEHICLE_WIDTH, 5);
        ctx.fill();

        // Windshield
        ctx.fillStyle = 'rgba(200,230,255,0.3)';
        ctx.beginPath();
        ctx.roundRect(5, -VEHICLE_WIDTH / 2 + 3, 10, VEHICLE_WIDTH - 6, 2);
        ctx.fill();

        // Headlights
        ctx.fillStyle = 'rgba(255, 240, 180, 0.9)';
        ctx.fillRect(VEHICLE_LENGTH / 2 - 4, -VEHICLE_WIDTH / 2 + 2, 3, 4);
        ctx.fillRect(VEHICLE_LENGTH / 2 - 4, VEHICLE_WIDTH / 2 - 6, 3, 4);

        // Tail lights
        ctx.fillStyle = this.waiting ? 'rgba(255, 40, 40, 1)' : 'rgba(255, 60, 60, 0.7)';
        ctx.fillRect(-VEHICLE_LENGTH / 2 + 1, -VEHICLE_WIDTH / 2 + 2, 3, 4);
        ctx.fillRect(-VEHICLE_LENGTH / 2 + 1, VEHICLE_WIDTH / 2 - 6, 3, 4);

        // Idle glow
        if (this.waiting) {
            ctx.shadowColor = 'rgba(255,0,0,0.5)';
            ctx.shadowBlur = 8;
            ctx.fillRect(-VEHICLE_LENGTH / 2 + 1, -VEHICLE_WIDTH / 2 + 2, 3, 4);
            ctx.fillRect(-VEHICLE_LENGTH / 2 + 1, VEHICLE_WIDTH / 2 - 6, 3, 4);
            ctx.shadowBlur = 0;
        }

        ctx.restore();
    }
}

function shadeColor(hex, pct) {
    let n = parseInt(hex.slice(1), 16);
    const r = Math.min(255, Math.max(0, (n >> 16) + pct));
    const g = Math.min(255, Math.max(0, ((n >> 8) & 0xff) + pct));
    const b = Math.min(255, Math.max(0, (n & 0xff) + pct));
    return `rgb(${r},${g},${b})`;
}

// --- Spawning ---
function spawnVehicle(dir) {
    const pattern = TRAFFIC_PATTERNS[currentPattern] || TRAFFIC_PATTERNS.balanced;
    let origin = dir;
    if (!origin) {
        const rand = Math.random();
        if (rand < pattern.N) origin = 'N';
        else if (rand < pattern.N + pattern.S) origin = 'S';
        else if (rand < pattern.N + pattern.S + pattern.E) origin = 'E';
        else origin = 'W';
    }
    const lane = Math.round(Math.random());
    vehicles.push(new Vehicle(origin, lane));
}

// --- Metrics Update ---
function updateMetrics() {
    document.getElementById('m-vehicles').innerText = vehicles.filter(v => v.active).length;
    const avgWait = metrics.throughput > 0 ? (metrics.totalWait / metrics.throughput).toFixed(1) : '0.0';
    document.getElementById('m-wait').innerText = avgWait + 's';
    const elapsed = (Date.now() - metrics.startTime) / 60000;
    document.getElementById('m-throughput').innerText = (cleared / Math.max(0.01, elapsed)).toFixed(0);

    // Update sim time display
    const simTimeEl = document.getElementById('sim-time');
    if (simTimeEl) simTimeEl.innerText = formatTime(simElapsedTime);

    // Queue bars
    const qCounts = { N: 0, S: 0, E: 0, W: 0 };
    vehicles.forEach(v => { if (v.waiting) qCounts[v.origin]++; });
    const maxQ = Math.max(10, ...Object.values(qCounts));
    ['N', 'S', 'E', 'W'].forEach(d => {
        const fill = document.getElementById(`q-${d}`);
        const num = document.getElementById(`qn-${d}`);
        if (fill) fill.style.width = (qCounts[d] / maxQ * 100) + '%';
        if (num) num.innerText = qCounts[d];
    });

    updateSignalLights();
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function updateSignalLights() {
    const nsColors = signalColors(signals.NS);
    const ewColors = signalColors(signals.EW);
    setLight('ns-light-r', nsColors.r);
    setLight('ns-light-y', nsColors.y);
    setLight('ns-light-g', nsColors.g);
    setLight('ew-light-r', ewColors.r);
    setLight('ew-light-y', ewColors.y);
    setLight('ew-light-g', ewColors.g);

    const nsBtn = document.getElementById('btn-ns');
    const ewBtn = document.getElementById('btn-ew');
    if (nsBtn) {
        nsBtn.innerText = signals.NS === SIGNAL.GREEN ? 'Force RED' : 'Force GREEN';
        nsBtn.className = 'btn btn-signal' + (signals.NS === SIGNAL.GREEN ? ' active-signal' : '');
    }
    if (ewBtn) {
        ewBtn.innerText = signals.EW === SIGNAL.GREEN ? 'Force RED' : 'Force GREEN';
        ewBtn.className = 'btn btn-signal' + (signals.EW === SIGNAL.GREEN ? ' active-signal' : '');
    }
}

function setLight(id, active) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('active-r', id.endsWith('-r') && active);
    el.classList.toggle('active-y', id.endsWith('-y') && active);
    el.classList.toggle('active-g', id.endsWith('-g') && active);
}

function signalColors(state) {
    return {
        r: state === SIGNAL.RED,
        y: state === SIGNAL.YELLOW,
        g: state === SIGNAL.GREEN
    };
}

function setStatus(text, running) {
    document.getElementById('sim-status-text').innerText = text;
    const pill = document.getElementById('sim-status-pill');
    pill.className = 'status-pill' + (running ? ' running' : '');
}

// --- Drawing ---
function drawBackground() {
    // Dark gradient background
    const bgGrad = ctx.createRadialGradient(canvas.width/2, canvas.height/2, 50, canvas.width/2, canvas.height/2, canvas.width * 0.8);
    bgGrad.addColorStop(0, '#0e1525');
    bgGrad.addColorStop(1, '#060a14');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle grid
    ctx.fillStyle = 'rgba(255,255,255,0.02)';
    for (let x = 0; x < canvas.width; x += 40) {
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.fillRect(x, y, 1, 1);
        }
    }

    // Building blocks (city context)
    ctx.fillStyle = 'rgba(30, 45, 70, 0.4)';
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const hw = ROAD_WIDTH / 2 + 15;
    // Top-left block
    ctx.fillRect(10, 10, cx - hw - 15, cy - hw - 15);
    // Top-right block
    ctx.fillRect(cx + hw + 5, 10, canvas.width - cx - hw - 15, cy - hw - 15);
    // Bottom-left block
    ctx.fillRect(10, cy + hw + 5, cx - hw - 15, canvas.height - cy - hw - 15);
    // Bottom-right block
    ctx.fillRect(cx + hw + 5, cy + hw + 5, canvas.width - cx - hw - 15, canvas.height - cy - hw - 15);
}

function drawRoads() {
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const hw = ROAD_WIDTH / 2;

    // Road surfaces with slight texture
    const roadGrad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    roadGrad.addColorStop(0, '#1a2640');
    roadGrad.addColorStop(0.5, '#1e2d4a');
    roadGrad.addColorStop(1, '#1a2640');
    ctx.fillStyle = roadGrad;
    ctx.fillRect(cx - hw, 0, ROAD_WIDTH, canvas.height);
    ctx.fillRect(0, cy - hw, canvas.width, ROAD_WIDTH);

    // Sidewalk/curb edges
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    // Vertical edges
    ctx.beginPath(); ctx.moveTo(cx - hw, 0); ctx.lineTo(cx - hw, cy - hw); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - hw, cy + hw); ctx.lineTo(cx - hw, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + hw, 0); ctx.lineTo(cx + hw, cy - hw); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + hw, cy + hw); ctx.lineTo(cx + hw, canvas.height); ctx.stroke();
    // Horizontal edges
    ctx.beginPath(); ctx.moveTo(0, cy - hw); ctx.lineTo(cx - hw, cy - hw); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + hw, cy - hw); ctx.lineTo(canvas.width, cy - hw); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy + hw); ctx.lineTo(cx - hw, cy + hw); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + hw, cy + hw); ctx.lineTo(canvas.width, cy + hw); ctx.stroke();

    // Center lane dividers (dashed)
    ctx.strokeStyle = 'rgba(255, 200, 50, 0.25)';
    ctx.lineWidth = 2;
    ctx.setLineDash([25, 20]);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, cy - hw); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy + hw); ctx.lineTo(cx, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(cx - hw, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + hw, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Stop lines (thick white)
    const stopLineOffset = hw + 6;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 4;
    // North stop (vehicles from N stop here)
    ctx.beginPath(); ctx.moveTo(cx - hw + 4, cy - stopLineOffset); ctx.lineTo(cx, cy - stopLineOffset); ctx.stroke();
    // South stop
    ctx.beginPath(); ctx.moveTo(cx, cy + stopLineOffset); ctx.lineTo(cx + hw - 4, cy + stopLineOffset); ctx.stroke();
    // East stop
    ctx.beginPath(); ctx.moveTo(cx + stopLineOffset, cy - hw + 4); ctx.lineTo(cx + stopLineOffset, cy); ctx.stroke();
    // West stop
    ctx.beginPath(); ctx.moveTo(cx - stopLineOffset, cy); ctx.lineTo(cx - stopLineOffset, cy + hw - 4); ctx.stroke();

    // Intersection box
    const intGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, hw * 1.2);
    intGrad.addColorStop(0, '#162240');
    intGrad.addColorStop(1, '#121c32');
    ctx.fillStyle = intGrad;
    ctx.fillRect(cx - hw, cy - hw, ROAD_WIDTH, ROAD_WIDTH);

    // Crosswalk markings
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    const cwW = 10, cwH = 22;
    const cwGap = 4;
    const numBars = Math.floor(ROAD_WIDTH / (cwW + cwGap));
    for (let i = 0; i < numBars; i++) {
        const offset = i * (cwW + cwGap);
        ctx.fillRect(cx - hw + offset, cy - hw - cwH - 6, cwW, cwH);  // North
        ctx.fillRect(cx - hw + offset, cy + hw + 6, cwW, cwH);        // South
        ctx.fillRect(cx - hw - cwH - 6, cy - hw + offset, cwH, cwW);  // West
        ctx.fillRect(cx + hw + 6, cy - hw + offset, cwH, cwW);        // East
    }

    // Lane direction arrows (inside intersection as per sketch)
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = 'bold 24px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // North lane arrows (coming from top, pointing down)
    ctx.fillText('↓', cx - LANE_OFFSET, cy - hw + 25);
    ctx.fillText('↑', cx + LANE_OFFSET, cy - hw + 25);
    
    // South lane arrows (coming from bottom, pointing up)
    ctx.fillText('↓', cx - LANE_OFFSET, cy + hw - 25);
    ctx.fillText('↑', cx + LANE_OFFSET, cy + hw - 25);
    
    // East lane arrows (coming from right, pointing left)
    ctx.fillText('←', cx + hw - 25, cy - LANE_OFFSET);
    ctx.fillText('→', cx + hw - 25, cy + LANE_OFFSET);
    
    // West lane arrows (coming from left, pointing right)
    ctx.fillText('←', cx - hw + 25, cy - LANE_OFFSET);
    ctx.fillText('→', cx - hw + 25, cy + LANE_OFFSET);
}

function getSignalColor(state) {
    if (state === SIGNAL.GREEN) return { fill: '#22c55e', glow: '#22c55e' };
    if (state === SIGNAL.YELLOW) return { fill: '#f59e0b', glow: '#f59e0b' };
    return { fill: '#ef4444', glow: '#ef4444' };
}

function drawSignalPole(x, y, signalState) {
    const { fill, glow } = getSignalColor(signalState);
    const r = 12;

    // Housing
    ctx.fillStyle = '#1a2744';
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x - r - 5, y - r - 5, (r + 5) * 2, (r + 5) * 2, 6);
    ctx.fill();
    ctx.stroke();

    // Glow
    ctx.shadowBlur = 22;
    ctx.shadowColor = glow;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Inner shine
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    ctx.arc(x - 3, y - 3, r * 0.3, 0, Math.PI * 2);
    ctx.fill();
}

function drawSignals() {
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const hw = ROAD_WIDTH / 2;
    const offset = hw + 28;

    // NS signals
    drawSignalPole(cx - LANE_OFFSET, cy - offset, signals.NS);
    drawSignalPole(cx + LANE_OFFSET, cy + offset, signals.NS);

    // EW signals
    drawSignalPole(cx - offset, cy + LANE_OFFSET, signals.EW);
    drawSignalPole(cx + offset, cy - LANE_OFFSET, signals.EW);
}

// --- Main Loop ---
let frameCount = 0;
let lastFrameTime = Date.now();

function loop() {
    const now = Date.now();
    const dt = (now - lastFrameTime) / 1000;
    lastFrameTime = now;

    drawBackground();
    drawRoads();

    if (isRunning) {
        vehicles = vehicles.filter(v => v.active);
        simElapsedTime += dt * timeScale;

        // Auto-spawn based on pattern
        const pattern = TRAFFIC_PATTERNS[currentPattern] || TRAFFIC_PATTERNS.balanced;
        if (Math.random() < pattern.rate * timeScale) spawnVehicle();

        vehicles.forEach(v => v.update(vehicles));
        frameCount++;
        if (frameCount % 5 === 0) updateMetrics();
    }

    vehicles.forEach(v => v.draw());
    drawSignals();
    requestAnimationFrame(loop);
}

// --- UI Bindings ---
const btnStart = document.getElementById('btn-start');
const btnPause = document.getElementById('btn-pause');
const btnReset = document.getElementById('btn-reset');
const btnAI = document.getElementById('btn-ai');
const btnNS = document.getElementById('btn-ns');
const btnEW = document.getElementById('btn-ew');

btnStart.onclick = () => {
    isRunning = true;
    metrics.startTime = Date.now();
    cleared = 0;
    simElapsedTime = 0;
    setStatus('RUNNING', true);
    btnStart.disabled = true;
    btnPause.disabled = false;
};

btnPause.onclick = () => {
    isRunning = !isRunning;
    if (isRunning) {
        btnPause.innerText = '⏸ PAUSE';
        setStatus('RUNNING', true);
    } else {
        btnPause.innerText = '▶ RESUME';
        setStatus('PAUSED', false);
    }
};

btnReset.onclick = () => {
    isRunning = false;
    isAIEnabled = false;
    vehicles = [];
    metrics = { total: 0, totalWait: 0, throughput: 0, startTime: Date.now() };
    cleared = 0;
    simElapsedTime = 0;
    signals = { NS: SIGNAL.RED, EW: SIGNAL.GREEN };
    updateSignalLights();
    setStatus('STANDBY', false);
    btnStart.disabled = false;
    btnPause.innerText = '⏸ PAUSE';
    btnPause.disabled = false;
    btnAI.classList.remove('active');
    btnAI.innerText = '🤖 AI MODE';
    document.getElementById('ai-mode-badge').innerText = 'MANUAL';
    document.getElementById('ctrl-mode-note').innerText = 'MANUAL';
    document.getElementById('m-vehicles').innerText = '0';
    document.getElementById('m-wait').innerText = '0s';
    document.getElementById('m-throughput').innerText = '0';
    document.getElementById('m-reward').innerText = '0';
    const simTimeEl = document.getElementById('sim-time');
    if (simTimeEl) simTimeEl.innerText = '0:00';
    ['N','S','E','W'].forEach(d => {
        const f = document.getElementById(`q-${d}`);
        const n = document.getElementById(`qn-${d}`);
        if (f) f.style.width = '0%';
        if (n) n.innerText = '0';
    });
    // Reset emissions
    if (typeof emissionsTotal !== 'undefined') {
        emissionsTotal.co2 = 0;
        emissionsTotal.fuel = 0;
        emissionsTotal.nox = 0;
    }
};

btnAI.onclick = () => {
    isAIEnabled = !isAIEnabled;
    btnAI.classList.toggle('active', isAIEnabled);
    btnAI.innerText = isAIEnabled ? '🤖 AI: ON' : '🤖 AI MODE';
    document.getElementById('ai-mode-badge').innerText = isAIEnabled ? 'AI' : 'MANUAL';
    document.getElementById('ctrl-mode-note').innerText = isAIEnabled ? 'AI' : 'MANUAL';
};

btnNS.onclick = () => {
    if (isAIEnabled) return;
    if (signals.NS !== SIGNAL.GREEN) {
        signals.EW = SIGNAL.RED;
        signals.NS = SIGNAL.GREEN;
    } else {
        signals.NS = SIGNAL.RED;
        signals.EW = SIGNAL.GREEN;
    }
    updateSignalLights();
};

btnEW.onclick = () => {
    if (isAIEnabled) return;
    if (signals.EW !== SIGNAL.GREEN) {
        signals.NS = SIGNAL.RED;
        signals.EW = SIGNAL.GREEN;
    } else {
        signals.EW = SIGNAL.RED;
        signals.NS = SIGNAL.GREEN;
    }
    updateSignalLights();
};

document.getElementById('sim-speed').oninput = (e) => {
    timeScale = parseFloat(e.target.value);
    document.getElementById('speed-val').innerText = timeScale + '×';
};

// Traffic pattern selector
const patternSelect = document.getElementById('traffic-pattern');
if (patternSelect) {
    patternSelect.onchange = (e) => {
        currentPattern = e.target.value;
    };
}

document.getElementById('btn-inject').onclick = () => {
    const activeDir = document.querySelector('.dir-btn.active')?.dataset.dir || 'N';
    const count = parseInt(document.getElementById('inject-count').value) || 1;
    if (activeDir === 'ALL') {
        ['N', 'S', 'E', 'W'].forEach(d => {
            for (let i = 0; i < count; i++) spawnVehicle(d);
        });
    } else {
        for (let i = 0; i < count; i++) spawnVehicle(activeDir);
    }
};

document.querySelectorAll('.dir-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.dir-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    };
});

// Initialize
updateSignalLights();
loop();
