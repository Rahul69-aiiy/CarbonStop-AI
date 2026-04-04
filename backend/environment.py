"""
environment.py

Traffic signal environment.
Simulates a simple 4-way junction with queues,
signal switching and emission-based reward.
"""

import random

# basic settings
MAX_QUEUE = 20        # max cars allowed in lane
EMISSION_FACTOR = 0.21  # rough CO2 per idle car
MAX_RED_TIME = 60     # stop if red stays too long
PEAK_THRESHOLD = 5.0  # emission spike threshold (kg CO2)

# car arrival probability
ARRIVAL_RATE_NS = 0.3
ARRIVAL_RATE_EW = 0.3

VEHICLES_PASS = 3     # cars that move when green


# actions the agent can take
ACTIONS = {
    0: "keep_green",     # keep same signal
    1: "switch_phase",   # change NS ↔ EW
    2: "extend_green",   # allow extra movement
}

ACTION_LIST = list(ACTIONS.keys())


def get_carbon_intensity(hour: int) -> float:
    # night time usually more fossil fuel usage
    if hour >= 18 or hour <= 5:
        return 1.5
    return 1.0


def estimate_emission(idle_time: float,
                      carbon_intensity: float) -> float:
    # more idle cars → more emission
    return idle_time * EMISSION_FACTOR * carbon_intensity


def compute_reward(
    queue_NS: int,
    queue_EW: int,
    red_NS: int,
    red_EW: int,
    carbon_intensity: float,
) -> float:

    # total waiting
    waiting_time = red_NS + red_EW

    # total idle cars
    idle_total = queue_NS + queue_EW

    # emission caused
    emission = estimate_emission(
        idle_total,
        carbon_intensity
    )

    # extra penalty for emission spikes above the peak threshold
    peak_penalty = -2 if emission > PEAK_THRESHOLD else 0

    # negative reward because we minimize both
    return -(waiting_time + emission) + peak_penalty


class TrafficEnv:

    def __init__(self, hour: int = 12):

        self.hour = hour
        self.carbon_intensity = get_carbon_intensity(hour)

        # queues
        self.queue_NS = 0
        self.queue_EW = 0

        # red timers
        self.red_NS = 0
        self.red_EW = 0

        # start with NS green
        self.phase = 0

        self.timestep = 0

        # highest emission recorded in the current episode
        self.peak_emission = 0.0


    def reset(self):

        # update carbon level
        self.carbon_intensity = get_carbon_intensity(self.hour)

        # small random starting traffic
        self.queue_NS = random.randint(0, 3)
        self.queue_EW = random.randint(0, 3)

        self.red_NS = 0
        self.red_EW = 0

        self.phase = 0
        self.timestep = 0
        self.peak_emission = 0.0

        return self._get_state()


    def _get_state(self):

        # convert raw values into bins
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


    def _generate_arrivals(self):

        # randomly add cars

        if random.random() < ARRIVAL_RATE_NS:
            self.queue_NS = min(
                self.queue_NS + random.randint(1, 3),
                MAX_QUEUE
            )

        if random.random() < ARRIVAL_RATE_EW:
            self.queue_EW = min(
                self.queue_EW + random.randint(1, 3),
                MAX_QUEUE
            )


    def _apply_action(self, action: int):

        # switch signal
        if action == 1:

            self.phase = 1 - self.phase

            # reset timers
            self.red_NS = 0
            self.red_EW = 0


        # update red timers
        if self.phase == 0:

            self.red_EW = min(
                self.red_EW + 1,
                MAX_RED_TIME
            )

            self.red_NS = 0

        else:

            self.red_NS = min(
                self.red_NS + 1,
                MAX_RED_TIME
            )

            self.red_EW = 0


        # extra clearing if extend_green
        extra = 1 if action == 2 else 0

        clear = VEHICLES_PASS + extra

        if self.phase == 0:

            self.queue_NS = max(
                self.queue_NS - clear,
                0
            )

        else:

            self.queue_EW = max(
                self.queue_EW - clear,
                0
            )


    def step(self, action: int):

        # apply action
        self._apply_action(action)

        # new cars arrive
        self._generate_arrivals()

        # track emission for this step and update episode peak
        emission = estimate_emission(
            self.queue_NS + self.queue_EW,
            self.carbon_intensity
        )
        self.peak_emission = max(self.peak_emission, emission)

        # compute reward
        reward = compute_reward(
            self.queue_NS,
            self.queue_EW,
            self.red_NS,
            self.red_EW,
            self.carbon_intensity,
        )

        self.timestep += 1

        # stop if red too long
        done = (
            self.red_NS >= MAX_RED_TIME
            or
            self.red_EW >= MAX_RED_TIME
        )

        next_state = self._get_state()

        return next_state, reward, done


    def get_info(self):

        # useful for debugging

        return {
            "timestep": self.timestep,
            "queue_NS": self.queue_NS,
            "queue_EW": self.queue_EW,
            "red_NS": self.red_NS,
            "red_EW": self.red_EW,
            "phase":
                "NS_green"
                if self.phase == 0
                else "EW_green",
            "carbon_intensity": self.carbon_intensity,
            "hour": self.hour,
            "peak_emission": self.peak_emission,
        }