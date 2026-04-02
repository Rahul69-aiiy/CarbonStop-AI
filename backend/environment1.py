"""
environment.py
--------------
Traffic Intersection Simulation Environment for Climate-Aware
Signal Optimization using Q-Learning.

Models a 4-way intersection with:
  - North-South (NS) and East-West (EW) vehicle queues
  - Red-time tracking per lane
  - Signal phase management
  - Carbon intensity based on time of day
  - CO2 emission estimation
  - Reward function incorporating waiting time and emissions

Fixes applied:
  - Reward no longer double-penalises emissions (removed flat emission term)
  - reset() now correctly refreshes carbon_intensity from self.hour
"""

import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_QUEUE        = 20        # Maximum vehicles in a queue
EMISSION_FACTOR  = 0.21      # kg CO2 per idle vehicle per timestep (approx.)
MAX_RED_TIME     = 20        # Maximum allowed red duration (seconds)
ARRIVAL_RATE_NS  = 0.3       # Probability of a vehicle arriving in NS per step
ARRIVAL_RATE_EW  = 0.3       # Probability of a vehicle arriving in EW per step
VEHICLES_PASS    = 3         # Vehicles that clear the queue when phase is green

# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------
ACTIONS = {
    0: "keep_green",    # Keep the current green phase unchanged
    1: "switch_phase",  # Toggle between NS-green and EW-green
    2: "extend_green",  # Add extra green time to the current phase (no switch)
}
ACTION_LIST = list(ACTIONS.keys())


# ---------------------------------------------------------------------------
# Carbon Intensity
# ---------------------------------------------------------------------------
def get_carbon_intensity(hour: int) -> float:
    """
    Returns a carbon intensity multiplier based on the hour of day.

    High carbon periods (night / early morning) use less renewable energy.
    Low carbon periods (daytime) benefit from solar / wind.

    Args:
        hour: Integer from 0 to 23.

    Returns:
        float: Carbon intensity multiplier.
    """
    if hour >= 18 or hour <= 5:
        return 1.5   # High carbon — grid relies more on fossil fuels
    else:
        return 1.0   # Lower carbon — more renewables active


# ---------------------------------------------------------------------------
# Emission Estimation
# ---------------------------------------------------------------------------
def estimate_emission(idle_time: float, carbon_intensity: float) -> float:
    """
    Estimates CO2 emission for idling vehicles.

    Formula:
        emission = idle_time * EMISSION_FACTOR * carbon_intensity

    Args:
        idle_time:         Total combined idle time (queue_NS + queue_EW acting
                           as idle vehicle-steps for simplicity).
        carbon_intensity:  Multiplier from get_carbon_intensity().

    Returns:
        float: Estimated CO2 in kg.
    """
    return idle_time * EMISSION_FACTOR * carbon_intensity


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------
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
    Improved reward function balancing:
    - Efficiency (waiting time)
    - Fairness (imbalance)
    - Stability (switching)
    - Sustainability (carbon impact)
    """

    # -------------------------------
    # 1. Waiting penalty (core objective)
    # -------------------------------
    waiting_penalty = -(red_NS + red_EW)

    # -------------------------------
    # 2. Queue imbalance (fairness)
    # -------------------------------
    imbalance_penalty = -0.5*abs(queue_NS - queue_EW)

    # -------------------------------
    # 3. Switching penalty (stability)
    # -------------------------------
    switch_penalty = -1 if (prev_action is not None and action != prev_action) else 0

    # -------------------------------
    # 4. Carbon-aware penalty
    # -------------------------------
    carbon_penalty = -0.05 * carbon_intensity * (queue_NS + queue_EW)

    # -------------------------------
    # 5. Clearing bonus
    # -------------------------------
    clear_bonus = 2 if (queue_NS == 0 or queue_EW == 0) else 0

    # -------------------------------
    # FINAL REWARD
    # -------------------------------
    reward = (
        waiting_penalty
        + 0.5 * imbalance_penalty
        + switch_penalty
        + carbon_penalty
        + clear_bonus
    )

    return reward


# ---------------------------------------------------------------------------
# Traffic Intersection Environment
# ---------------------------------------------------------------------------
class TrafficEnv:
    """
    Simulates a single 4-way traffic intersection over discrete timesteps.

    State tuple (as returned by _get_state):
        (queue_NS, queue_EW, red_NS, red_EW, phase, carbon_bin)

    where:
        queue_NS / queue_EW : binned queue length (0–4)
        red_NS   / red_EW   : binned red time     (0–3)
        phase               : 0 = NS green, 1 = EW green
        carbon_bin          : 0 = low carbon, 1 = high carbon
    """

    def __init__(self, hour: int = 12):
        """
        Args:
            hour: Starting hour of day (0–23). Affects carbon intensity and
                  determines the initial carbon bin.
        """
        self.hour             = hour
        self.carbon_intensity = get_carbon_intensity(hour)

        # Continuous state variables
        self.queue_NS  = 0    # Number of vehicles waiting in NS lane
        self.queue_EW  = 0    # Number of vehicles waiting in EW lane
        self.red_NS    = 0    # Seconds NS lane has been red
        self.red_EW    = 0    # Seconds EW lane has been red
        self.phase     = 0    # 0 = NS green, 1 = EW green
        self.timestep  = 0    # Current simulation step

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> tuple:
        """
        Resets the environment to its initial state with small random queues.

        FIX: Now correctly refreshes carbon_intensity from self.hour so that
        when the training loop mutates env.hour between episodes, the new
        carbon intensity is actually applied during the episode.

        Returns:
            tuple: Initial discretised state.
        """
        self.carbon_intensity = get_carbon_intensity(self.hour)  # FIX
        self.queue_NS = random.randint(0, 3)
        self.queue_EW = random.randint(0, 3)
        self.red_NS   = 0
        self.red_EW   = 0
        self.phase    = 0
        self.timestep = 0
        self.prev_action = None  # Reset previous action for accurate switching penalty
        return self._get_state()

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------
    def _get_state(self) -> tuple:
        """
        Discretises continuous state variables into bins for the Q-table.

        Bins:
            queue bins  : 0–4  vehicles → 0, 5–9 → 1, 10–14 → 2, 15–19 → 3, 20+ → 4
            red bins    : 0–14 s → 0, 15–29 → 1, 30–44 → 2, 45+ → 3
            carbon_bin  : 0 = low (1.0), 1 = high (1.5)

        Returns:
            tuple: (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, phase, carbon_bin)
        """
        q_NS_bin   = min(self.queue_NS // 5, 4)
        q_EW_bin   = min(self.queue_EW // 5, 4)
        r_NS_bin   = min(self.red_NS   // 15, 3)
        r_EW_bin   = min(self.red_EW   // 15, 3)
        carbon_bin = 1 if self.carbon_intensity >= 1.5 else 0

        return (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, self.phase, carbon_bin)

    # ------------------------------------------------------------------
    # Vehicle Arrival
    # ------------------------------------------------------------------
    def _generate_arrivals(self):
        """
        Randomly adds vehicles to each lane based on arrival probabilities.
        Queues are capped at MAX_QUEUE.
        """
        if random.random() < ARRIVAL_RATE_NS:
            self.queue_NS = min(self.queue_NS + random.randint(1, 3), MAX_QUEUE)
        if random.random() < ARRIVAL_RATE_EW:
            self.queue_EW = min(self.queue_EW + random.randint(1, 3), MAX_QUEUE)

    # ------------------------------------------------------------------
    # Phase / Queue Update
    # ------------------------------------------------------------------
    def _apply_action(self, action: int):
        """
        Applies the chosen action to update signal timing and queues.

        action = 0 (keep_green):   No change, green lane continues flowing.
        action = 1 (switch_phase): Toggle phase; red durations reset.
        action = 2 (extend_green): Same as keep_green but grants an extra
                                   clearance step (clears one more vehicle).

        Args:
            action: Integer action index (0, 1, or 2).
        """
        if action == 1:  # switch_phase — toggle signal
            self.phase  = 1 - self.phase
            # Reset red timers on phase switch before re-accumulating below
            self.red_NS = 0
            self.red_EW = 0

        # Update red timers — the lane that is NOT green accumulates red time
        if self.phase == 0:          # NS is green, EW is red
            self.red_EW = min(self.red_EW + 1, MAX_RED_TIME)
            self.red_NS = 0          # NS is flowing; reset its red counter
        else:                        # EW is green, NS is red
            self.red_NS = min(self.red_NS + 1, MAX_RED_TIME)
            self.red_EW = 0

        # Vehicle clearance — green lane drains its queue
        extra = 1 if action == 2 else 0      # extend_green clears one more
        clear = VEHICLES_PASS + extra

        if self.phase == 0:
            self.queue_NS = max(self.queue_NS - clear, 0)
        else:
            self.queue_EW = max(self.queue_EW - clear, 0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple:
        """
        Advances the simulation by one timestep.

        Order of operations per step:
            1. Apply chosen action (signal logic + queue clearance)
            2. Generate new vehicle arrivals
            3. Compute reward
            4. Return (next_state, reward, done)

        Args:
            action: Integer action index.

        Returns:
            tuple: (next_state, reward, done)
                next_state — discretised state tuple
                reward     — float scalar
                done       — bool, True when MAX_RED_TIME exceeded (penalty cap)
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
        getattr(self, "prev_action", None), # type: ignore
)
        self.prev_action = action  # Store for next step's switching penalty
        self.timestep += 1

        # Episode ends early if any lane has been red for too long
        done = (self.red_NS >= MAX_RED_TIME) or (self.red_EW >= MAX_RED_TIME)

        next_state = self._get_state()
        return next_state, reward, done

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        """
        Returns a human-readable snapshot of the current environment state.
        Useful for debugging and logging.
        """
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
