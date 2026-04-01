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
"""

import random
import json
import os
from collections import defaultdict

from environment import TrafficEnv, ACTION_LIST, ACTIONS

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ALPHA          = 0.1     # Learning rate  — how much new info overrides old
GAMMA          = 0.9     # Discount factor — importance of future rewards
EPSILON_START  = 1.0     # Initial exploration rate (100% random)
EPSILON_MIN    = 0.05    # Minimum exploration rate
EPSILON_DECAY  = 0.995   # Multiplicative decay per episode
NUM_EPISODES   = 1000    # Total training episodes
MAX_STEPS      = 200     # Maximum steps per episode
LOG_INTERVAL   = 50      # Print progress every N episodes
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
    def select_action(self, state: tuple) -> int:
        """
        Selects an action using the epsilon-greedy strategy.

        With probability epsilon  → explore  (random action)
        With probability 1-epsilon → exploit (greedy best action)

        Args:
            state: Discretised environment state tuple.

        Returns:
            int: Action index (0, 1, or 2).
        """
        if random.random() < self.epsilon:
            return random.choice(ACTION_LIST)          # Explore
        return int(self._best_action(state))            # Exploit

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
        current_q  = self.q_table[state][action]
        future_q   = 0.0 if done else max(self.q_table[next_state])
        target     = reward + self.gamma * future_q
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
        print(f"[INFO] Q-table saved → {os.path.abspath(path)}")

    def load(self, path: str = QTABLE_PATH):
        """
        Loads a previously saved Q-table from a JSON file.
        String keys are converted back to tuples.

        Args:
            path: File path of the saved Q-table.
        """
        if not os.path.exists(path):
            print(f"[WARN] Q-table file not found: {path}. Starting fresh.")
            return

        with open(path, "r") as f:
            raw = json.load(f)

        # Convert string keys back to tuples
        self.q_table = defaultdict(lambda: [0.0] * len(ACTION_LIST))
        for k_str, v in raw.items():
            key = tuple(int(x) for x in k_str.strip("()").split(", "))
            self.q_table[key] = v

        print(f"[INFO] Q-table loaded ← {os.path.abspath(path)}")


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
        4. Decay epsilon at episode end

    Every log_interval episodes, prints:
        - Episode number
        - Total reward
        - Average waiting time (red_NS + red_EW averaged over steps)
        - Current epsilon

    Args:
        num_episodes: Total training episodes.
        max_steps:    Maximum steps before episode is truncated.
        log_interval: Frequency of progress logging.

    Returns:
        QLearningAgent: Trained agent with populated Q-table.
    """
    agent = QLearningAgent()
    env   = TrafficEnv()

    print("=" * 60)
    print(" Climate-Aware Traffic Signal — Q-Learning Training")
    print("=" * 60)
    print(f"  Episodes : {num_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  α={ALPHA}  γ={GAMMA}  ε_start={EPSILON_START}  ε_min={EPSILON_MIN}")
    print("=" * 60)

    for episode in range(1, num_episodes + 1):
        # Randomise hour each episode for generalisation
        env.hour             = random.randint(0, 23)
        env.carbon_intensity = __import__("environment").get_carbon_intensity(env.hour)

        state          = env.reset()
        total_reward   = 0.0
        total_waiting  = 0.0

        for step in range(max_steps):
            action              = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            total_reward  += reward
            total_waiting += (env.red_NS + env.red_EW)   # proxy for waiting time
            state          = next_state

            if done:
                break

        agent.decay_epsilon()

        # ---- Logging ----
        if episode % log_interval == 0 or episode == 1:
            steps_run   = step + 1
            avg_waiting = total_waiting / steps_run
            print(
                f"  Episode {episode:>5} | "
                f"Total Reward: {total_reward:>10.2f} | "
                f"Avg Waiting: {avg_waiting:>6.2f}s | "
                f"ε: {agent.epsilon:.4f}"
            )

    print("=" * 60)
    print("  Training complete.")
    print("=" * 60)

    # Persist trained Q-table
    agent.save()
    return agent


# ---------------------------------------------------------------------------
# Entry Point — run training when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trained_agent = train()
