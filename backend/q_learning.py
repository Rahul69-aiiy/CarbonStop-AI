"""
q_learning.py
-------------
Q-Learning Agent for Climate-Aware Traffic Signal Optimization.

Fix: now imports from 'environment' (single canonical file).
Previously imported from 'environment' which had the old weak reward —
training never used the improved imbalance/wrong-phase penalties.
"""

import ast
import random
import json
import os
from collections import defaultdict

# FIX: import from environment (canonical), not environment1
from environment import TrafficEnv, ACTION_LIST, ACTIONS, get_carbon_intensity

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ALPHA          = 0.1
GAMMA          = 0.9
EPSILON_START  = 1.0
EPSILON_MIN    = 0.1
EPSILON_DECAY  = 0.998
NUM_EPISODES   = 10000
MAX_STEPS      = 200
LOG_INTERVAL   = 500
WARMUP_RATIO   = 0.10
QTABLE_PATH    = "q_table.json"


class QLearningAgent:
    """
    Tabular Q-Learning agent backed by a defaultdict Q-table.
    { state_tuple : [Q_action0, Q_action1, Q_action2] }
    Unseen states default to [0.0, 0.0, 0.0].
    """

    def __init__(
        self,
        alpha:         float = ALPHA,
        gamma:         float = GAMMA,
        epsilon:       float = EPSILON_START,
        epsilon_min:   float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict = defaultdict(lambda: [0.0] * len(ACTION_LIST))

    def select_action(self, state: tuple, force_explore: bool = False) -> int:
        if force_explore or random.random() < self.epsilon:
            return random.choice(ACTION_LIST)
        return int(self._best_action(state))

    def _best_action(self, state: tuple) -> int:
        return ACTION_LIST[
            max(range(len(ACTION_LIST)), key=lambda a: self.q_table[state][a])
        ]

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        future_q  = 0.0 if done else max(self.q_table[next_state])
        target    = reward + self.gamma * future_q
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, state: tuple) -> str:
        return ACTIONS[self._best_action(state)]

    def save(self, path: str = QTABLE_PATH):
        serialisable = {str(k): v for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"[INFO] Q-table saved → {os.path.abspath(path)}  ({len(self.q_table)} states)")

    def load(self, path: str = QTABLE_PATH):
        if not os.path.exists(path):
            print(f"[WARN] Q-table not found: {path}. Starting fresh.")
            return
        with open(path, "r") as f:
            raw = json.load(f)
        self.q_table = defaultdict(lambda: [0.0] * len(ACTION_LIST))
        for k_str, v in raw.items():
            key = tuple(ast.literal_eval(k_str))
            self.q_table[key] = v
        print(f"[INFO] Q-table loaded ← {os.path.abspath(path)}  ({len(self.q_table)} states)")


def train(
    num_episodes: int = NUM_EPISODES,
    max_steps:    int = MAX_STEPS,
    log_interval: int = LOG_INTERVAL,
) -> QLearningAgent:
    """
    Trains the agent. First WARMUP_RATIO of episodes use pure exploration.
    Epsilon decays after warmup until it hits EPSILON_MIN.
    Delete q_table.json before running so the agent retrains with the
    improved reward function.
    """
    agent        = QLearningAgent()
    env          = TrafficEnv()
    warmup_limit = int(num_episodes * WARMUP_RATIO)

    print("=" * 65)
    print(" Climate-Aware Traffic Signal — Q-Learning Training")
    print("=" * 65)
    print(f"  Episodes  : {num_episodes}  (warmup: first {warmup_limit})")
    print(f"  Max steps : {max_steps}")
    print(f"  α={ALPHA}  γ={GAMMA}  ε_start={EPSILON_START}  "
          f"ε_min={EPSILON_MIN}  ε_decay={EPSILON_DECAY}")
    print("=" * 65)

    for episode in range(1, num_episodes + 1):
        in_warmup            = episode <= warmup_limit
        env.hour             = random.randint(0, 23)
        env.carbon_intensity = get_carbon_intensity(env.hour)

        state         = env.reset()
        total_reward  = 0.0
        total_waiting = 0.0

        for step in range(max_steps):
            action                   = agent.select_action(state, force_explore=in_warmup)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            total_reward  += reward
            total_waiting += (env.red_NS + env.red_EW)
            state          = next_state

            if done:
                break

        if not in_warmup:
            agent.decay_epsilon()

        if episode % log_interval == 0 or episode == 1:
            steps_run   = step + 1
            avg_waiting = total_waiting / steps_run
            phase_tag   = "WARMUP" if in_warmup else f"ε={agent.epsilon:.4f}"
            print(
                f"  Episode {episode:>6}/{num_episodes} | "
                f"Reward: {total_reward:>10.2f} | "
                f"Avg Wait: {avg_waiting:>5.2f}s | "
                f"States: {len(agent.q_table):>4} | "
                f"{phase_tag}"
            )

    print("=" * 65)
    print(f"  Training complete — {len(agent.q_table)} unique states covered.")
    print("=" * 65)

    agent.save()
    return agent


if __name__ == "__main__":
    trained_agent = train()