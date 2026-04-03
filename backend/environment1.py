import random

# Maximum vehicles per queue lane
MAX_QUEUE        = 20
EMISSION_FACTOR  = 0.21      # kg CO2 per idle vehicle per timestep
MAX_RED_TIME     = 20        # seconds before an episode terminates early
ARRIVAL_RATE_NS  = 0.3       # probability a vehicle arrives in NS per step
ARRIVAL_RATE_EW  = 0.3
VEHICLES_PASS    = 3         # vehicles cleared per green step

ACTIONS = {
    0: "keep_green",    # keep the current green phase running
    1: "switch_phase",  # toggle between NS-green and EW-green
    2: "extend_green",  # grant an extra clearance step without switching
}
ACTION_LIST = list(ACTIONS.keys())


def get_carbon_intensity(hour: int) -> float:
    """
    Returns a carbon intensity multiplier based on the hour of day.
    Night / early morning (18:00–05:00) scores higher because the grid
    relies more on fossil fuels when solar and wind output is low.
    """
    if hour >= 18 or hour <= 5:
        return 1.5   # high carbon period
    return 1.0       # lower carbon — more renewables active


def estimate_emission(idle_time: float, carbon_intensity: float) -> float:
    """Estimates CO2 emission (kg) for idling vehicles over one timestep."""
    return idle_time * EMISSION_FACTOR * carbon_intensity


def compute_reward(
    queue_NS: int,
    queue_EW: int,
    red_NS: int,
    red_EW: int,
    carbon_intensity: float,
    action: int,
    prev_action: int,
) -> float:
    """
    Reward function balancing four objectives:
      - Efficiency  : minimise total waiting time
      - Fairness    : minimise queue imbalance between lanes
      - Stability   : penalise unnecessary phase switches
      - Sustainability : scale penalty by current carbon intensity
    """
    # Penalise total red time across both lanes
    waiting_penalty = -(red_NS + red_EW)

    # Discourage letting one lane starve while the other is clear
    imbalance_penalty = -0.5 * abs(queue_NS - queue_EW)

    # Small penalty for switching phase — avoids jittery behaviour
    switch_penalty = -1 if (prev_action is not None and action != prev_action) else 0

    # Scale queue penalty by carbon intensity (worse to idle during high-carbon periods)
    carbon_penalty = -0.05 * carbon_intensity * (queue_NS + queue_EW)

    # Bonus when at least one lane has fully cleared
    clear_bonus = 2 if (queue_NS == 0 or queue_EW == 0) else 0

    return (
        waiting_penalty
        + 0.5 * imbalance_penalty
        + switch_penalty
        + carbon_penalty
        + clear_bonus
    )


class TrafficEnv:
    """
    Simulates a single 4-way traffic intersection over discrete timesteps.

    State tuple returned by _get_state():
        (queue_NS, queue_EW, red_NS, red_EW, phase, carbon_bin)

        queue_NS / queue_EW : binned queue length (0–4)
        red_NS   / red_EW   : binned red-time bucket (0–3)
        phase               : 0 = NS green, 1 = EW green
        carbon_bin          : 0 = low carbon (daytime), 1 = high carbon (night)
    """

    def __init__(self, hour: int = 12):
        self.hour             = hour
        self.carbon_intensity = get_carbon_intensity(hour)
        self.queue_NS  = 0
        self.queue_EW  = 0
        self.red_NS    = 0
        self.red_EW    = 0
        self.phase     = 0    # 0 = NS green, 1 = EW green
        self.timestep  = 0
        self.prev_action = None

    def reset(self) -> tuple:
        """Resets the environment to its initial state with small random queues."""
        # Refresh carbon intensity in case env.hour was changed between episodes
        self.carbon_intensity = get_carbon_intensity(self.hour)
        self.queue_NS    = random.randint(0, 3)
        self.queue_EW    = random.randint(0, 3)
        self.red_NS      = 0
        self.red_EW      = 0
        self.phase       = 0
        self.timestep    = 0
        self.prev_action = None
        return self._get_state()

    def _get_state(self) -> tuple:
        """
        Discretises continuous state variables into bins for the Q-table.

        Queue bins  : 0–4 vehicles → 0, 5–9 → 1, 10–14 → 2, 15–19 → 3, 20+ → 4
        Red bins    : 0–14 s → 0, 15–29 → 1, 30–44 → 2, 45+ → 3
        Carbon bin  : 0 = low (1.0×), 1 = high (1.5×)
        """
        q_NS_bin   = min(self.queue_NS // 5, 4)
        q_EW_bin   = min(self.queue_EW // 5, 4)
        r_NS_bin   = min(self.red_NS   // 15, 3)
        r_EW_bin   = min(self.red_EW   // 15, 3)
        carbon_bin = 1 if self.carbon_intensity >= 1.5 else 0
        return (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, self.phase, carbon_bin)

    def _generate_arrivals(self):
        """Randomly adds vehicles to each lane based on arrival probabilities."""
        if random.random() < ARRIVAL_RATE_NS:
            self.queue_NS = min(self.queue_NS + random.randint(1, 3), MAX_QUEUE)
        if random.random() < ARRIVAL_RATE_EW:
            self.queue_EW = min(self.queue_EW + random.randint(1, 3), MAX_QUEUE)

    def _apply_action(self, action: int):
        """
        Applies the chosen action to update signal timing and queues.

        action = 0 (keep_green):   No change, green lane continues flowing.
        action = 1 (switch_phase): Toggle phase; red durations reset.
        action = 2 (extend_green): Same as keep_green but clears one extra vehicle.
        """
        if action == 1:  # switch_phase
            self.phase  = 1 - self.phase
            self.red_NS = 0
            self.red_EW = 0

        # The lane that is NOT green accumulates red time
        if self.phase == 0:          # NS is green, EW is red
            self.red_EW = min(self.red_EW + 1, MAX_RED_TIME)
            self.red_NS = 0
        else:                        # EW is green, NS is red
            self.red_NS = min(self.red_NS + 1, MAX_RED_TIME)
            self.red_EW = 0

        # extend_green clears one extra vehicle from the green lane
        extra = 1 if action == 2 else 0
        clear = VEHICLES_PASS + extra

        if self.phase == 0:
            self.queue_NS = max(self.queue_NS - clear, 0)
        else:
            self.queue_EW = max(self.queue_EW - clear, 0)

    def step(self, action: int) -> tuple:
        """
        Advances the simulation by one timestep.

        Order of operations:
            1. Apply action (signal logic + queue clearance)
            2. Generate new vehicle arrivals
            3. Compute reward
            4. Return (next_state, reward, done)

        done is True when any lane has been red for MAX_RED_TIME seconds.
        """
        self._apply_action(action)
        self._generate_arrivals()

        reward = compute_reward(
            self.queue_NS,
            self.queue_EW,
            self.red_NS,
            self.red_EW,
            self.carbon_intensity,
            action,
            self.prev_action,
        )

        self.prev_action = action
        self.timestep   += 1
        done = (self.red_NS >= MAX_RED_TIME) or (self.red_EW >= MAX_RED_TIME)
        next_state = self._get_state()
        return next_state, reward, done

    def get_info(self) -> dict:
        """Returns a snapshot of the current environment state for debugging or logging."""
        return {
            "timestep"        : self.timestep,
            "queue_NS"        : self.queue_NS,
            "queue_EW"        : self.queue_EW,
            "red_NS"          : self.red_NS,
            "red_EW"          : self.red_EW,
            "phase"           : "NS_green" if self.phase == 0 else "EW_green",
            "carbon_intensity": self.carbon_intensity,
            "hour"            : self.hour,
        }
