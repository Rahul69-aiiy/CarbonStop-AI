"""
FastAPI backend for the Climate-Aware Traffic Signal Optimizer.

Endpoints:
    POST /predict   — Returns best action. Includes hard imbalance override.
    POST /simulate  — Runs multi-step simulation, returns full trajectory.
    GET  /health    — Health check.
    GET  /info      — Model metadata.
    GET  /model-info — Rich metadata for dashboard.
"""

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# FIX: single import source — environment only
from environment import (
    TrafficEnv, get_carbon_intensity, ACTIONS,
    estimate_emission, WRONG_PHASE_GREEN_MAX, WRONG_PHASE_RED_MIN
)
from q_learning import QLearningAgent, train, QTABLE_PATH

agent = QLearningAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    if os.path.exists(QTABLE_PATH):
        agent.load(QTABLE_PATH)
        print(f"[API] Loaded pre-trained Q-table ({len(agent.q_table)} states).")
    else:
        print("[API] No saved Q-table — training now (takes ~1 min)...")
        agent = train()
        print(f"[API] Training complete. {len(agent.q_table)} states.")
    yield


app = FastAPI(
    title       = "Climate-Aware Traffic Signal Optimizer",
    description = "Q-Learning API for traffic signal optimization.",
    version     = "1.2.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

class TrafficState(BaseModel):
    queue_NS:            int      = Field(..., ge=0, le=20)
    queue_EW:            int      = Field(..., ge=0, le=20)
    red_NS:              int      = Field(..., ge=0, le=60)
    red_EW:              int      = Field(..., ge=0, le=60)
    phase:               int      = Field(..., ge=0, le=1)
    hour:                int      = Field(..., ge=0, le=23)
    ambulance_direction: str | None = Field(None)


class PredictResponse(BaseModel):
    action:             str
    carbon_intensity:   float
    explanation:        str
    ambulance_override: bool = False
    imbalance_override: bool = False


class SimulationStep(BaseModel):
    step:     int
    action:   str
    queue_NS: int
    queue_EW: int
    red_NS:   int
    red_EW:   int
    phase:    str
    reward:   float
    co2_kg:   float


class SimulateResponse(BaseModel):
    steps:              List[SimulationStep]
    total_co2_kg:       float
    avg_wait_seconds:   float
    total_reward:       float
    peak_emission:      float
    peak_emission_step: int

# Helpers
def _build_state(ts: TrafficState) -> tuple:
    q_NS_bin   = min(ts.queue_NS // 5, 4)
    q_EW_bin   = min(ts.queue_EW // 5, 4)
    r_NS_bin   = min(ts.red_NS   // 15, 3)
    r_EW_bin   = min(ts.red_EW   // 15, 3)
    carbon_bin = 1 if get_carbon_intensity(ts.hour) >= 1.5 else 0
    return (q_NS_bin, q_EW_bin, r_NS_bin, r_EW_bin, ts.phase, carbon_bin)


def _check_hard_imbalance(ts: TrafficState) -> bool:
    """
    Returns True when the green lane has very few cars but the red lane
    is heavily loaded — regardless of what the Q-table says, we should switch.

    Thresholds: green ≤ WRONG_PHASE_GREEN_MAX and red ≥ WRONG_PHASE_RED_MIN
    """
    green_queue = ts.queue_NS if ts.phase == 0 else ts.queue_EW
    red_queue   = ts.queue_EW if ts.phase == 0 else ts.queue_NS
    return green_queue <= WRONG_PHASE_GREEN_MAX and red_queue >= WRONG_PHASE_RED_MIN


ACTION_EXPLANATIONS = {
    "keep_green":   "Maintain the current green phase — queues are manageable.",
    "switch_phase": "Switch the active green phase to relieve the opposing lane.",
    "extend_green": "Extend green duration to clear a backlogged lane faster.",
}


# Endpoints
@app.post("/predict", response_model=PredictResponse, tags=["Signal Control"])
def predict(traffic_state: TrafficState):
    """
    Returns the optimal signal action.

    Priority order:
      1. Ambulance override — immediate green for emergency vehicle direction
      2. Hard imbalance override — switch when green lane ≤ 5 cars and
         red lane ≥ 10 cars (safety net while model learns new reward)
      3. Q-table lookup — normal AI decision
    """
    try:
        carbon_intensity = get_carbon_intensity(traffic_state.hour)

        # 1. Ambulance override
        if traffic_state.ambulance_direction in ("NS", "EW"):
            amb_dir      = traffic_state.ambulance_direction
            current_green = "NS" if traffic_state.phase == 0 else "EW"
            action_label  = "keep_green" if current_green == amb_dir else "switch_phase"
            return PredictResponse(
                action             = action_label,
                carbon_intensity   = carbon_intensity,
                explanation        = f" AMBULANCE PRIORITY — immediate green for {amb_dir} corridor.",
                ambulance_override = True,
            )

        # 2. Hard imbalance override
        if _check_hard_imbalance(traffic_state):
            green_q = traffic_state.queue_NS if traffic_state.phase == 0 else traffic_state.queue_EW
            red_q   = traffic_state.queue_EW if traffic_state.phase == 0 else traffic_state.queue_NS
            return PredictResponse(
                action             = "switch_phase",
                carbon_intensity   = carbon_intensity,
                explanation        = (
                    f" IMBALANCE OVERRIDE — green lane has {green_q} cars "
                    f"but red lane has {red_q}. Switching to relieve congestion."
                ),
                imbalance_override = True,
            )

        # 3. Normal Q-table lookup
        state        = _build_state(traffic_state)
        action_label = agent.predict(state)

        return PredictResponse(
            action           = action_label,
            carbon_intensity = carbon_intensity,
            explanation      = ACTION_EXPLANATIONS.get(action_label, ""),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/simulate", response_model=SimulateResponse, tags=["Signal Control"])
def simulate(
    initial_state: TrafficState,
    steps: int = Query(default=50, ge=1, le=500),
):
    """
    Runs the agent forward N steps from the given initial state.
    Returns the full trajectory with CO₂, wait times, and rewards.
    """
    try:
        env          = TrafficEnv(hour=initial_state.hour)
        env.queue_NS = initial_state.queue_NS
        env.queue_EW = initial_state.queue_EW
        env.red_NS   = initial_state.red_NS
        env.red_EW   = initial_state.red_EW
        env.phase    = initial_state.phase

        trajectory         = []
        total_co2          = 0.0
        total_wait         = 0.0
        total_reward       = 0.0
        peak_emission      = 0.0
        peak_emission_step = 0

        state = env._get_state()

        for i in range(steps):
            action_label = agent.predict(state)
            action_idx   = next(k for k, v in ACTIONS.items() if v == action_label)

            next_state, reward, done = env.step(action_idx)

            # FIX: estimate_emission imported at top — no mid-function import
            co2_step  = estimate_emission(
                float(env.queue_NS + env.queue_EW),
                env.carbon_intensity,
            )
            wait_step = float(env.red_NS + env.red_EW)

            total_co2    += co2_step
            total_wait   += wait_step
            total_reward += reward

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

        steps_run = len(trajectory)
        return SimulateResponse(
            steps              = trajectory,
            total_co2_kg       = round(total_co2, 4),
            avg_wait_seconds   = round(total_wait / steps_run, 4),
            total_reward       = round(total_reward, 4),
            peak_emission      = round(peak_emission, 4),
            peak_emission_step = peak_emission_step,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "q_table_states": len(agent.q_table)}


@app.get("/info", tags=["System"])
def info():
    return {
        "model"       : "Q-Learning",
        "actions"     : ["keep_green", "switch_phase", "extend_green"],
        "alpha"       : agent.alpha,
        "gamma"       : agent.gamma,
        "epsilon"     : round(agent.epsilon, 4),
        "q_table_size": len(agent.q_table),
    }


@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "model"             : "Q-Learning",
        "version"           : "1.2.0",
        "actions"           : ["keep_green", "switch_phase", "extend_green"],
        "alpha"             : agent.alpha,
        "gamma"             : agent.gamma,
        "epsilon"           : round(agent.epsilon, 4),
        "q_table_size"      : len(agent.q_table),
        "peak_threshold"    : 5.0,
        "ambulance_support" : True,
        "imbalance_override": True,
    }
