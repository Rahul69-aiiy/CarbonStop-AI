import random
# Basic settings
MAX_QUEUE = 20
EMISSION_FACTOR = 0.21
MAX_RED_TIME = 60

ARRIVAL_RATE_NS = 0.3
ARRIVAL_RATE_EW = 0.3
VEHICLES_PASS = 3

PEAK_THRESHOLD = 5.0

# Used to detect bad signal phase
WRONG_PHASE_GREEN_MAX = 5
WRONG_PHASE_RED_MIN = 10

# Actions the agent can take
ACTIONS = {
    0: "keep_green",
    1: "switch_phase",
    2: "extend_green",
}

ACTION_LIST = list(ACTIONS.keys())

# Returns carbon intensity based on hour
def get_carbon_intensity(hour: int) -> float:
    """
    Night / early morning (18:00–05:59) → 1.5 (fossil-fuel heavy).
    Daytime (06:00–17:59) → 1.0 (more renewables active).
    """
    if hour >= 18 or hour <= 5:
        return 1.5
    return 1.0

# Estimate CO2 from idle vehicles
def estimate_emission(idle_time: float, carbon_intensity: float) -> float:
    """CO2 (kg) for idle vehicles: idle_time * EMISSION_FACTOR * carbon_intensity."""
    return idle_time * EMISSION_FACTOR * carbon_intensity

# Reward logic for RL
def compute_reward(
    queue_NS,
    queue_EW,
    red_NS,
    red_EW,
    carbon_intensity,
    action,
    prev_action,
    phase,
):
    # Penalize waiting time
    waiting_penalty = -(red_NS + red_EW)

    # Penalize imbalance between lanes
    imbalance_penalty = -3.0 * abs(queue_NS - queue_EW)

    # Punish wrong signal phase
    green_queue = queue_NS if phase == 0 else queue_EW
    red_queue = queue_EW if phase == 0 else queue_NS

    if green_queue <= WRONG_PHASE_GREEN_MAX and red_queue >= WRONG_PHASE_RED_MIN:
        wrong_phase_penalty = -15
    else:
        wrong_phase_penalty = 0

    # Small penalty for switching too often
    switch_penalty = -1 if (prev_action is not None and action != prev_action) else 0

    # Carbon penalty
    carbon_penalty = -0.05 * carbon_intensity * (queue_NS + queue_EW)

    # Bonus if lane clears
    clear_bonus = 2 if (queue_NS == 0 or queue_EW == 0) else 0

    # Extra penalty for high emission
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

# Traffic environment class
class TrafficEnv:
    """
    Simulates a single 4-way traffic intersection over discrete timesteps.

    State: (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, phase, carbon_bin)

    Queue bins : 0–4 → 0,  5–9 → 1, 10–14 → 2, 15–19 → 3, 20+ → 4
    Red bins   : 0–14s → 0, 15–29 → 1, 30–44 → 2, 45+ → 3
    Carbon bin : 0 = low (1.0), 1 = high (1.5)
    """

    def __init__(self, hour: int = 12):
        self.hour = hour
        self.carbon_intensity = get_carbon_intensity(hour)

        self.queue_NS = 0
        self.queue_EW = 0

        self.red_NS = 0
        self.red_EW = 0

        self.phase = 0
        self.timestep = 0

        self.prev_action = None
        self.peak_emission = 0.0

    # Reset environment
    def reset(self):
        self.carbon_intensity = get_carbon_intensity(self.hour)

        self.queue_NS = random.randint(0, 3)
        self.queue_EW = random.randint(0, 3)

        self.red_NS = 0
        self.red_EW = 0

        self.phase = 0
        self.timestep = 0

        self.prev_action = None
        self.peak_emission = 0.0

        return self._get_state()

    # Convert raw values to bins
    def _get_state(self):
        q_NS_bin = min(self.queue_NS // 5, 4)
        q_EW_bin = min(self.queue_EW // 5, 4)

        r_NS_bin = min(self.red_NS // 15, 3)
        r_EW_bin = min(self.red_EW // 15, 3)

        carbon_bin = 1 if self.carbon_intensity >= 1.5 else 0

        return (
            q_NS_bin,
            q_EW_bin,
            r_NS_bin,
            r_EW_bin,
            self.phase,
            carbon_bin,
        )

    # Random vehicle arrivals
    def _generate_arrivals(self):
        if random.random() < ARRIVAL_RATE_NS:
            self.queue_NS = min(
                self.queue_NS + random.randint(1, 3),
                MAX_QUEUE,
            )

        if random.random() < ARRIVAL_RATE_EW:
            self.queue_EW = min(
                self.queue_EW + random.randint(1, 3),
                MAX_QUEUE,
            )

    # Apply selected action
    def _apply_action(self, action):

        if action == 1:
            self.phase = 1 - self.phase
            self.red_NS = 0
            self.red_EW = 0

        if self.phase == 0:
            self.red_EW = min(self.red_EW + 1, MAX_RED_TIME)
            self.red_NS = 0
        else:
            self.red_NS = min(self.red_NS + 1, MAX_RED_TIME)
            self.red_EW = 0

        extra = 1 if action == 2 else 0
        clear = VEHICLES_PASS + extra

        if self.phase == 0:
            self.queue_NS = max(self.queue_NS - clear, 0)
        else:
            self.queue_EW = max(self.queue_EW - clear, 0)

    # Main step function
    def step(self, action):

        self._apply_action(action)
        self._generate_arrivals()

        emission = estimate_emission(
            self.queue_NS + self.queue_EW,
            self.carbon_intensity,
        )

        self.peak_emission = max(
            self.peak_emission,
            emission,
        )

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
        self.timestep += 1

        done = (
            self.red_NS >= MAX_RED_TIME
            or self.red_EW >= MAX_RED_TIME
        )

        next_state = self._get_state()
        return next_state, reward, done

    # Return readable info
    def get_info(self):
        return {
            "timestep": self.timestep,
            "queue_NS": self.queue_NS,
            "queue_EW": self.queue_EW,
            "red_NS": self.red_NS,
            "red_EW": self.red_EW,
            "phase": "NS_green" if self.phase == 0 else "EW_green",
            "carbon_intensity": self.carbon_intensity,
            "hour": self.hour,
            "peak_emission": self.peak_emission,
        }
