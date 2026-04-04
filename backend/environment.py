"""
environment.py
Single canonical environment file. Delete environment1.py.

Key improvements over previous versions:
  - STRONG imbalance penalty (weight 3.0 vs old 0.25) — forces switch
    when one lane has far more cars than the other
  - Wrong-phase penalty — direct punishment for keeping green on a nearly
    empty lane while the other lane is heavily loaded (e.g. 3 vs 25 cars)
  - MAX_RED_TIME = 60 — all red-time bins (0–14, 15–29, 30–44, 45+) reachable
  - prev_action initialised in __init__ to prevent AttributeError
  - All imports consolidated — environment1.py no longer needed
"""

import random

# Constants
MAX_QUEUE        = 20      # Maximum vehicles in a queue
EMISSION_FACTOR  = 0.21    # kg CO2 per idle vehicle per timestep
MAX_RED_TIME     = 60      # Episode ends if any lane stays red this long (s)
ARRIVAL_RATE_NS  = 0.3     # Probability a vehicle arrives in NS per step
ARRIVAL_RATE_EW  = 0.3     # Probability a vehicle arrives in EW per step
VEHICLES_PASS    = 3       # Vehicles cleared when lane is green
PEAK_THRESHOLD   = 5.0     # kg CO2 — extra penalty when emission exceeds this

# Imbalance thresholds for wrong-phase detection
WRONG_PHASE_GREEN_MAX  = 5   # Green lane has ≤ this many cars ...
WRONG_PHASE_RED_MIN    = 10  # ... while red lane has ≥ this many → wrong phase

# Actions
ACTIONS = {
    0: "keep_green",    # Keep the current green phase unchanged
    1: "switch_phase",  # Toggle between NS-green and EW-green
    2: "extend_green",  # Clear one extra vehicle without switching
}
ACTION_LIST = list(ACTIONS.keys())


# Carbon Intensity
def get_carbon_intensity(hour: int) -> float:
    """
    Night / early morning (18:00–05:59) → 1.5 (fossil-fuel heavy).
    Daytime (06:00–17:59) → 1.0 (more renewables active).
    """
    if hour >= 18 or hour <= 5:
        return 1.5
    return 1.0


# Emission Estimation
def estimate_emission(idle_time: float, carbon_intensity: float) -> float:
    """CO2 (kg) for idle vehicles: idle_time * EMISSION_FACTOR * carbon_intensity."""
    return idle_time * EMISSION_FACTOR * carbon_intensity


# Reward Function
def compute_reward(
    queue_NS:         int,
    queue_EW:         int,
    red_NS:           int,
    red_EW:           int,
    carbon_intensity: float,
    action:           int,
    prev_action:      "int | None",
    phase:            int,
) -> float:
    """
    Reward balancing seven objectives:

    1. Waiting penalty    — penalises total red-time (core signal)
    2. Imbalance penalty  — STRONG penalty for large queue differences
                            (weight 3.0, was 0.25 — this is the main fix)
    3. Wrong-phase penalty — direct punishment when green lane is nearly empty
                             but red lane is heavily loaded (e.g. 3 vs 25 cars)
    4. Switch penalty     — small cost for unnecessary phase changes (stability)
    5. Carbon penalty     — extra cost for idling during high-carbon hours
    6. Clearing bonus     — reward when either lane fully drains
    7. Peak spike penalty — extra penalty when emission exceeds PEAK_THRESHOLD

    Args:
        queue_NS / queue_EW: current vehicle counts
        red_NS   / red_EW:   seconds each lane has been red
        carbon_intensity:    grid intensity multiplier
        action:              current action taken
        prev_action:         last action (for switch penalty)
        phase:               0 = NS green, 1 = EW green
    """
    # 1. Waiting penalty
    waiting_penalty = -(red_NS + red_EW)

    # 2. Strong imbalance penalty — weight 3.0 (was 0.25 before)
    #    With 3 vs 25 cars this gives: -3.0 * 22 = -66 penalty
    #    Strong enough to outweigh the waiting signal
    imbalance_penalty = -3.0 * abs(queue_NS - queue_EW)

    # 3. Wrong-phase penalty — punishes keeping green on an almost-empty lane
    #    when the other lane is heavily loaded
    green_queue = queue_NS if phase == 0 else queue_EW
    red_queue   = queue_EW if phase == 0 else queue_NS
    if green_queue <= WRONG_PHASE_GREEN_MAX and red_queue >= WRONG_PHASE_RED_MIN:
        wrong_phase_penalty = -15  # Strong direct signal to switch
    else:
        wrong_phase_penalty = 0

    # 4. Switch penalty — small cost to prevent jitter
    switch_penalty = -1 if (prev_action is not None and action != prev_action) else 0

    # 5. Carbon penalty — idling costs more during fossil-fuel hours
    carbon_penalty = -0.05 * carbon_intensity * (queue_NS + queue_EW)

    # 6. Clearing bonus — reward when a lane fully empties
    clear_bonus = 2 if (queue_NS == 0 or queue_EW == 0) else 0

    # 7. Emission spike penalty
    emission = estimate_emission(queue_NS + queue_EW, carbon_intensity)
    peak_penalty = -2 if emission > PEAK_THRESHOLD else 0

    return (
        waiting_penalty
        + imbalance_penalty
        + wrong_phase_penalty
        + switch_penalty
        + carbon_penalty
        + clear_bonus
        + peak_penalty
    )


# ---------------------------------------------------------------------------
# Traffic Intersection Environment
# ---------------------------------------------------------------------------
class TrafficEnv:
    """
    Simulates a single 4-way traffic intersection over discrete timesteps.

    State: (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, phase, carbon_bin)

    Queue bins : 0–4 → 0,  5–9 → 1, 10–14 → 2, 15–19 → 3, 20+ → 4
    Red bins   : 0–14s → 0, 15–29 → 1, 30–44 → 2, 45+ → 3
    Carbon bin : 0 = low (1.0), 1 = high (1.5)
    """

    def __init__(self, hour: int = 12):
        self.hour             = hour
        self.carbon_intensity = get_carbon_intensity(hour)
        self.queue_NS         = 0
        self.queue_EW         = 0
        self.red_NS           = 0
        self.red_EW           = 0
        self.phase            = 0        # 0 = NS green, 1 = EW green
        self.timestep         = 0
        self.prev_action      = None     # prevents AttributeError before first reset
        self.peak_emission    = 0.0

    def reset(self) -> tuple:
        """Resets to initial state. Refreshes carbon_intensity from self.hour."""
        self.carbon_intensity = get_carbon_intensity(self.hour)
        self.queue_NS         = random.randint(0, 3)
        self.queue_EW         = random.randint(0, 3)
        self.red_NS           = 0
        self.red_EW           = 0
        self.phase            = 0
        self.timestep         = 0
        self.prev_action      = None
        self.peak_emission    = 0.0
        return self._get_state()

    def _get_state(self) -> tuple:
        q_NS_bin   = min(self.queue_NS // 5, 4)
        q_EW_bin   = min(self.queue_EW // 5, 4)
        r_NS_bin   = min(self.red_NS   // 15, 3)
        r_EW_bin   = min(self.red_EW   // 15, 3)
        carbon_bin = 1 if self.carbon_intensity >= 1.5 else 0
        return (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, self.phase, carbon_bin)

    def _generate_arrivals(self):
        if random.random() < ARRIVAL_RATE_NS:
            self.queue_NS = min(self.queue_NS + random.randint(1, 3), MAX_QUEUE)
        if random.random() < ARRIVAL_RATE_EW:
            self.queue_EW = min(self.queue_EW + random.randint(1, 3), MAX_QUEUE)

    def _apply_action(self, action: int):
        if action == 1:  # switch_phase
            self.phase  = 1 - self.phase
            self.red_NS = 0
            self.red_EW = 0

        if self.phase == 0:          # NS green, EW red
            self.red_EW = min(self.red_EW + 1, MAX_RED_TIME)
            self.red_NS = 0
        else:                        # EW green, NS red
            self.red_NS = min(self.red_NS + 1, MAX_RED_TIME)
            self.red_EW = 0

        extra = 1 if action == 2 else 0
        clear = VEHICLES_PASS + extra

        if self.phase == 0:
            self.queue_NS = max(self.queue_NS - clear, 0)
        else:
            self.queue_EW = max(self.queue_EW - clear, 0)

    def step(self, action: int) -> tuple:
        """
        Step order:
            1. Apply action
            2. Generate arrivals
            3. Track emission peak
            4. Compute reward (passes phase for wrong-phase detection)
            5. Update prev_action
            6. Return (next_state, reward, done)
        """
        self._apply_action(action)
        self._generate_arrivals()

        emission = estimate_emission(
            self.queue_NS + self.queue_EW,
            self.carbon_intensity
        )
        self.peak_emission = max(self.peak_emission, emission)

        reward = compute_reward(
            self.queue_NS,
            self.queue_EW,
            self.red_NS,
            self.red_EW,
            self.carbon_intensity,
            action,
            self.prev_action,
            self.phase,
        )

        self.prev_action = action
        self.timestep   += 1
        done       = (self.red_NS >= MAX_RED_TIME) or (self.red_EW >= MAX_RED_TIME)
        next_state = self._get_state()
        return next_state, reward, done

    def get_info(self) -> dict:
        return {
            "timestep"        : self.timestep,
            "queue_NS"        : self.queue_NS,
            "queue_EW"        : self.queue_EW,
            "red_NS"          : self.red_NS,
            "red_EW"          : self.red_EW,
            "phase"           : "NS_green" if self.phase == 0 else "EW_green",
            "carbon_intensity": self.carbon_intensity,
            "hour"            : self.hour,
            "peak_emission"   : self.peak_emission,
        }