"""
q_learning.py
-------------
Q-Learning Agent for Climate-Aware Traffic Signal Optimization.

Implements:
  - Dictionary-based Q-table (infinite state space friendly)
  - Epsilon-greedy action selection with epsilon decay
  - Bellman Q-value update rule
  - Training loop over multiple episodes
  - Learning progress logging
  - Model persistence (save / load Q-table)

Fixes applied:
  - Removed __import__() hack; get_carbon_intensity imported directly
  - Q-table keys loaded safely via ast.literal_eval instead of fragile string split
  - NUM_EPISODES raised to 5000 for adequate state-space coverage
  - Warmup phase (first 10% of episodes) runs with epsilon=1.0 for broad exploration
"""

import ast
import random
import json
import os
from collections import defaultdict

from environment import TrafficEnv, ACTION_LIST, ACTIONS, get_carbon_intensity

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ALPHA          = 0.1     # Learning rate  — how much new info overrides old
GAMMA          = 0.9     # Discount factor — importance of future rewards
EPSILON_START  = 1.0     # Initial exploration rate (100% random)
EPSILON_MIN    = 0.05    # Minimum exploration rate
EPSILON_DECAY  = 0.999   # Multiplicative decay per episode (slower decay for 5k eps)
NUM_EPISODES   = 5000    # FIX: raised from 1000 — covers far more of state space
MAX_STEPS      = 200     # Maximum steps per episode
LOG_INTERVAL   = 250     # Print progress every N episodes
WARMUP_RATIO   = 0.10    # First 10% of episodes use pure exploration (epsilon=1.0)
QTABLE_PATH    = "q_table.json"  # Path to persist the Q-table


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------
class QLearningAgent:
    """
    Tabular Q-Learning agent using a defaultdict as the Q-table.

    Q-Table structure:
        { state_tuple : [Q_value_action0, Q_value_action1, Q_value_action2] }

    Any unseen state is initialised to [0.0, 0.0, 0.0] automatically.
    """

    def __init__(
        self,
        alpha:         float = ALPHA,
        gamma:         float = GAMMA,
        epsilon:       float = EPSILON_START,
        epsilon_min:   float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
    ):
        """
        Initialises the agent with hyperparameters and an empty Q-table.

        Args:
            alpha:         Learning rate.
            gamma:         Discount factor.
            epsilon:       Initial exploration probability.
            epsilon_min:   Floor for epsilon after decay.
            epsilon_decay: Per-episode multiplicative decay for epsilon.
        """
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict returns a zero-initialised list for new states
        self.q_table: dict = defaultdict(lambda: [0.0] * len(ACTION_LIST))

    # ------------------------------------------------------------------
    # Action Selection (Epsilon-Greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: tuple, force_explore: bool = False) -> int:
        """
        Selects an action using the epsilon-greedy strategy.

        With probability epsilon  → explore  (random action)
        With probability 1-epsilon → exploit (greedy best action)

        Args:
            state:         Discretised environment state tuple.
            force_explore: If True, always explore (used during warmup phase).

        Returns:
            int: Action index (0, 1, or 2).
        """
        if force_explore or random.random() < self.epsilon:
            return random.choice(ACTION_LIST)   # Explore
        return int(self._best_action(state))     # Exploit

    def _best_action(self, state: tuple) -> int:
        """Returns the action with the highest Q-value for the given state."""
        return ACTION_LIST[
            max(range(len(ACTION_LIST)), key=lambda a: self.q_table[state][a])
        ]

    # ------------------------------------------------------------------
    # Q-Value Update (Bellman Equation)
    # ------------------------------------------------------------------
    def update(
        self,
        state:      tuple,
        action:     int,
        reward:     float,
        next_state: tuple,
        done:       bool,
    ):
        """
        Updates the Q-value for the (state, action) pair using the
        Bellman equation:

            Q(s,a) ← Q(s,a) + α * [r + γ * max_a'(Q(s',a')) - Q(s,a)]

        When done=True the future reward term is 0 (terminal state).

        Args:
            state:      Current discretised state.
            action:     Action taken.
            reward:     Reward received.
            next_state: Resulting state after action.
            done:       Whether the episode has ended.
        """
        current_q = self.q_table[state][action]
        future_q  = 0.0 if done else max(self.q_table[next_state])
        target    = reward + self.gamma * future_q
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Epsilon Decay
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """
        Decays epsilon after each episode.
        Ensures exploration probability never falls below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Predict (Inference Mode)
    # ------------------------------------------------------------------
    def predict(self, state: tuple) -> str:
        """
        Returns the best action label (string) for inference / API use.
        No exploration — always chooses the greedy action.

        Args:
            state: Discretised state tuple.

        Returns:
            str: Action label, e.g. "keep_green", "switch_phase", "extend_green".
        """
        action_idx = self._best_action(state)
        return ACTIONS[action_idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = QTABLE_PATH):
        """
        Serialises the Q-table to a JSON file.
        defaultdict keys (tuples) are converted to strings for JSON.

        Args:
            path: File path to save the Q-table.
        """
        serialisable = {str(k): v for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"[INFO] Q-table saved → {os.path.abspath(path)}  ({len(self.q_table)} states)")

    def load(self, path: str = QTABLE_PATH):
        """
        Loads a previously saved Q-table from a JSON file.
        String keys are safely parsed back to tuples via ast.literal_eval.

        FIX: Replaced fragile str.strip().split() parsing with ast.literal_eval
        which correctly handles any valid Python tuple literal.

        Args:
            path: File path of the saved Q-table.
        """
        if not os.path.exists(path):
            print(f"[WARN] Q-table file not found: {path}. Starting fresh.")
            return

        with open(path, "r") as f:
            raw = json.load(f)

        # FIX: ast.literal_eval safely converts "(0, 1, 2, ...)" → tuple
        self.q_table = defaultdict(lambda: [0.0] * len(ACTION_LIST))
        for k_str, v in raw.items():
            key = tuple(ast.literal_eval(k_str))
            self.q_table[key] = v

        print(f"[INFO] Q-table loaded ← {os.path.abspath(path)}  ({len(self.q_table)} states)")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(
    num_episodes: int = NUM_EPISODES,
    max_steps:    int = MAX_STEPS,
    log_interval: int = LOG_INTERVAL,
) -> QLearningAgent:
    """
    Trains the Q-Learning agent on the TrafficEnv simulation.

    Per episode:
        1. Reset environment with a random hour [0–23]
        2. Run up to max_steps, collecting (s, a, r, s', done) tuples
        3. Update Q-table after each step
        4. Decay epsilon at episode end (skipped during warmup phase)

    Warmup phase (first WARMUP_RATIO * num_episodes episodes):
        epsilon is forced to 1.0 to ensure every region of the state space
        is visited before exploitation begins.

    Every log_interval episodes, prints:
        - Episode number
        - Total reward
        - Average waiting time (red_NS + red_EW averaged over steps)
        - Q-table size (unique states visited)
        - Current epsilon

    Args:
        num_episodes: Total training episodes.
        max_steps:    Maximum steps before episode is truncated.
        log_interval: Frequency of progress logging.

    Returns:
        QLearningAgent: Trained agent with populated Q-table.
    """
    agent        = QLearningAgent()
    env          = TrafficEnv()
    warmup_limit = int(num_episodes * WARMUP_RATIO)

    print("=" * 65)
    print(" Climate-Aware Traffic Signal — Q-Learning Training")
    print("=" * 65)
    print(f"  Episodes  : {num_episodes}  (warmup: first {warmup_limit})")
    print(f"  Max steps : {max_steps}")
    print(f"  α={ALPHA}  γ={GAMMA}  ε_start={EPSILON_START}  ε_min={EPSILON_MIN}  ε_decay={EPSILON_DECAY}")
    print("=" * 65)

    for episode in range(1, num_episodes + 1):
        in_warmup = episode <= warmup_limit

        # Randomise hour each episode for generalisation across carbon bins
        env.hour = random.randint(0, 23)
        # FIX: call imported get_carbon_intensity directly — no __import__ hack
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

        # Only decay epsilon after warmup is complete
        if not in_warmup:
            agent.decay_epsilon()

        # ---- Logging ----
        if episode % log_interval == 0 or episode == 1:
            steps_run   = step + 1
            avg_waiting = total_waiting / steps_run
            phase_tag   = "WARMUP" if in_warmup else f"ε={agent.epsilon:.4f}"
            print(
                f"  Episode {episode:>5}/{num_episodes} | "
                f"Reward: {total_reward:>10.2f} | "
                f"Avg Wait: {avg_waiting:>5.2f}s | "
                f"States: {len(agent.q_table):>4} | "
                f"{phase_tag}"
            )

    print("=" * 65)
    print(f"  Training complete. Q-table covers {len(agent.q_table)} unique states.")
    print("=" * 65)

    # Persist trained Q-table
    agent.save()
    return agent


# ---------------------------------------------------------------------------
# Entry Point — run training when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trained_agent = train()