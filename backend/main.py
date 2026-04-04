"""
main.py

FastAPI backend for the Climate-Aware Traffic Signal Optimizer.

Endpoints:
    POST /predict   — Accepts current intersection state, returns best action.
    POST /simulate  — Runs a multi-step simulation and returns full trajectory.
    GET  /health    — Health check.
    GET  /info      — Model metadata and Q-table size.

The Q-Learning agent loads from a pre-trained Q-table (q_table.json).
If no saved table is found, training runs automatically on startup.

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment1 import TrafficEnv, get_carbon_intensity, ACTIONS
from q_learning  import QLearningAgent, train, QTABLE_PATH

# Module-level agent singleton — read-only at inference time
agent = QLearningAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
      1. Try to load a pre-trained Q-table from disk.
      2. If not found, run training automatically and save the result.
    """
    global agent
    if os.path.exists(QTABLE_PATH):
        agent.load(QTABLE_PATH)
        print(f"[API] Loaded pre-trained Q-table ({len(agent.q_table)} states).")
    else:
        print("[API] No saved Q-table found — running training now...")
        agent = train()
        print(f"[API] Training complete. Q-table has {len(agent.q_table)} states.")
    yield
app = FastAPI(
    title       = "Climate-Aware Traffic Signal Optimizer",
    description = (
        "A Q-Learning-based reinforcement learning API that dynamically "
        "optimises traffic signal timing to minimise CO₂ emissions and "
        "reduce vehicle waiting time at a 4-way intersection."
    ),
    version     = "1.1.0",
    lifespan    = lifespan,
)

# Allow cross-origin requests so the JavaScript frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


class TrafficState(BaseModel):
    """
    Input payload representing the current state of the intersection.

    Fields:
        queue_NS:            Number of vehicles queued in the North-South lane (0–20).
        queue_EW:            Number of vehicles queued in the East-West lane (0–20).
        red_NS:              Seconds the North-South lane has been red (0–60).
        red_EW:              Seconds the East-West lane has been red (0–60).
        phase:               Current green phase — 0 = NS green, 1 = EW green.
        hour:                Hour of day (0–23) used to compute carbon intensity.
        ambulance_direction: If set ('NS' or 'EW'), triggers an immediate priority override
                             for that direction, bypassing the Q-table.
    """
    queue_NS:            int  = Field(..., ge=0, le=20, description="NS queue length")
    queue_EW:            int  = Field(..., ge=0, le=20, description="EW queue length")
    red_NS:              int  = Field(..., ge=0, le=60, description="NS red duration in seconds")
    red_EW:              int  = Field(..., ge=0, le=60, description="EW red duration in seconds")
    phase:               int  = Field(..., ge=0, le=1,  description="Current phase: 0=NS green, 1=EW green")
    hour:                int  = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    ambulance_direction: str | None = Field(None, description="Ambulance direction override: 'NS' or 'EW'")


class PredictResponse(BaseModel):
    """
    Response payload with the recommended action.

    Fields:
        action:              One of 'keep_green', 'switch_phase', 'extend_green'.
        carbon_intensity:    Carbon intensity multiplier for this hour.
        explanation:         Human-readable description of the chosen action.
        ambulance_override:  True when an ambulance priority override was applied.
    """
    action:              str
    carbon_intensity:    float
    explanation:         str
    ambulance_override:  bool = False


class SimulationStep(BaseModel):
    """A single timestep snapshot from a simulation run."""
    step:             int
    action:           str
    queue_NS:         int
    queue_EW:         int
    red_NS:           int
    red_EW:           int
    phase:            str
    reward:           float
    co2_kg:           float


class SimulateResponse(BaseModel):
    """Full simulation trajectory response."""
    steps:              List[SimulationStep]
    total_co2_kg:       float
    avg_wait_seconds:   float
    total_reward:       float
    peak_emission:      float
    peak_emission_step: int


def _build_state(ts: TrafficState) -> tuple:
    """
    Converts the raw API input into the same discretised state tuple
    used by the Q-table during training.

    Bins mirror those in TrafficEnv._get_state().
    """
    q_NS_bin   = min(ts.queue_NS  // 5, 4)
    q_EW_bin   = min(ts.queue_EW  // 5, 4)
    r_NS_bin   = min(ts.red_NS    // 15, 3)
    r_EW_bin   = min(ts.red_EW    // 15, 3)
    carbon     = get_carbon_intensity(ts.hour)
    carbon_bin = 1 if carbon >= 1.5 else 0
    return (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, ts.phase, carbon_bin)


ACTION_EXPLANATIONS = {
    "keep_green":   "Maintain the current green phase — queues are manageable.",
    "switch_phase": "Switch the active green phase to relieve the opposing lane.",
    "extend_green": "Extend green duration to clear a backlogged lane faster.",
}


@app.post(
    "/predict",
    response_model = PredictResponse,
    summary        = "Get the optimal traffic signal action",
    tags           = ["Signal Control"],
)
def predict(traffic_state: TrafficState):
    """
    Given the current intersection state, returns the recommended signal action.

    The agent uses the trained Q-table to look up the optimal action for the
    discretised state. Falls back to 'keep_green' if the state has never been
    seen during training.

    If ambulance_direction ('NS' or 'EW') is provided, the Q-table is bypassed
    and switch_phase is immediately returned to give green to that direction.

    Example request body:
    ```json
    {
      "queue_NS": 8,
      "queue_EW": 3,
      "red_NS": 20,
      "red_EW": 0,
      "phase": 0,
      "hour": 20,
      "ambulance_direction": "NS"
    }
    ```
    """
    try:
        carbon_intensity = get_carbon_intensity(traffic_state.hour)

        # ── Ambulance Priority Override ──────────────────────────────────────
        if traffic_state.ambulance_direction in ("NS", "EW"):
            amb_dir = traffic_state.ambulance_direction
            # Determine whether a phase switch is needed
            current_green = "NS" if traffic_state.phase == 0 else "EW"
            if current_green == amb_dir:
                action_label = "keep_green"
            else:
                action_label = "switch_phase"
            return PredictResponse(
                action             = action_label,
                carbon_intensity   = carbon_intensity,
                explanation        = f"🚑 AMBULANCE PRIORITY — immediate green for {amb_dir} corridor.",
                ambulance_override = True,
            )

        # ── Normal Q-Table Lookup ────────────────────────────────────────────
        state        = _build_state(traffic_state)
        action_label = agent.predict(state)
        explanation  = ACTION_EXPLANATIONS.get(action_label, "")

        return PredictResponse(
            action             = action_label,
            carbon_intensity   = carbon_intensity,
            explanation        = explanation,
            ambulance_override = False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/simulate",
    response_model = SimulateResponse,
    summary        = "Run a multi-step AI simulation",
    tags           = ["Signal Control"],
)
def simulate(
    initial_state: TrafficState,
    steps: int = Query(default=50, ge=1, le=500, description="Number of timesteps to simulate"),
):
    """
    Runs the trained agent on the environment for the requested number of steps,
    starting from the provided initial state.

    Returns the full trajectory, total CO₂ produced, average waiting time,
    and cumulative reward — useful for dashboard visualisations.

    Example request body:
    ```json
    {
      "queue_NS": 5,
      "queue_EW": 10,
      "red_NS": 0,
      "red_EW": 30,
      "phase": 1,
      "hour": 8
    }
    ```
    """
    try:
        from environment import estimate_emission, EMISSION_FACTOR

        env           = TrafficEnv(hour=initial_state.hour)
        # Seed environment with the provided initial state
        env.queue_NS  = initial_state.queue_NS
        env.queue_EW  = initial_state.queue_EW
        env.red_NS    = initial_state.red_NS
        env.red_EW    = initial_state.red_EW
        env.phase     = initial_state.phase

        trajectory    = []
        total_co2     = 0.0
        total_wait    = 0.0
        total_reward  = 0.0
        peak_emission      = 0.0
        peak_emission_step = 0

        state = env._get_state()

        for i in range(steps):
            action_label = agent.predict(state)
            action_idx   = next(k for k, v in ACTIONS.items() if v == action_label)

            next_state, reward, done = env.step(action_idx)

            co2_step   = estimate_emission(
                float(env.queue_NS + env.queue_EW),
                env.carbon_intensity,
            )
            wait_step  = float(env.red_NS + env.red_EW)

            total_co2    += co2_step
            total_wait   += wait_step
            total_reward += reward

            # Track peak emission across simulation steps
            if co2_step > peak_emission:
                peak_emission      = co2_step
                peak_emission_step = i + 1

            trajectory.append(SimulationStep(
                step     = i + 1,
                action   = action_label,
                queue_NS = env.queue_NS,
                queue_EW = env.queue_EW,
                red_NS   = env.red_NS,
                red_EW   = env.red_EW,
                phase    = "NS_green" if env.phase == 0 else "EW_green",
                reward   = round(reward, 4),
                co2_kg   = round(co2_step, 4),
            ))

            state = next_state
            if done:
                break

        return SimulateResponse(
            steps              = trajectory,
            total_co2_kg       = round(total_co2, 4),
            avg_wait_seconds   = round(total_wait / len(trajectory), 4),
            total_reward       = round(total_reward, 4),
            peak_emission      = round(peak_emission, 4),
            peak_emission_step = peak_emission_step,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/health",
    summary = "Health check",
    tags    = ["System"],
)
def health():
    """Returns a simple health status so the frontend can verify connectivity."""
    return {"status": "ok", "q_table_states": len(agent.q_table)}


@app.get(
    "/info",
    summary = "Model metadata",
    tags    = ["System"],
)
def info():
    """Returns hyperparameters and Q-table statistics for monitoring."""
    return {
        "model"       : "Q-Learning",
        "actions"     : ["keep_green", "switch_phase", "extend_green"],
        "alpha"       : agent.alpha,
        "gamma"       : agent.gamma,
        "epsilon"     : round(agent.epsilon, 4),
        "q_table_size": len(agent.q_table),
    }


@app.get(
    "/model-info",
    summary = "Detailed model metadata for dashboard",
    tags    = ["System"],
)
def model_info():
    """Returns rich model metadata consumed by the frontend status and ML output panels."""
    return {
        "model"           : "Q-Learning",
        "version"         : "1.1.0",
        "actions"         : ["keep_green", "switch_phase", "extend_green"],
        "alpha"           : agent.alpha,
        "gamma"           : agent.gamma,
        "epsilon"         : round(agent.epsilon, 4),
        "q_table_size"    : len(agent.q_table),
        "peak_threshold"  : 5.0,
        "ambulance_support": True,
    }
