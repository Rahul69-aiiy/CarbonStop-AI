# 🚦 CarbonStop-AI — Climate-Aware Traffic Signal Optimization

> Using reinforcement learning to reduce CO₂ emissions at traffic intersections by dynamically optimizing signal timing based on queue length, wait time, and real-time carbon grid intensity.

---

## 🧠 How It Works

A **Q-Learning agent** is trained on a simulated 4-way intersection. At every timestep it observes the current state — queue lengths, red durations, active phase, and the carbon intensity of the power grid at that hour — and selects the best signal action to minimize both vehicle waiting time and CO₂ emissions.

The trained agent is served via a **FastAPI backend** and integrated into a **React dashboard** with a live traffic simulation. When you enable AI Mode, the frontend calls the backend in real time to drive the signals.

---

## 🗂️ Project Structure

```
CarbonStop-AI/
├── backend/
│   ├── environment.py   # Traffic intersection simulation + reward function
│   ├── q_learning.py    # Q-Learning agent, training loop, model persistence
│   ├── q_table.json     # Pre-trained agent memory (auto-generated if missing)
│   └── main.py          # FastAPI server — /predict, /simulate, /health, /info
└── frontend/
    ├── src/
    └── ...              # React dashboard + embedded traffic simulation
```

---

## 🚀 Getting Started

You will need **two terminal windows** — one for the backend, one for the frontend.

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.8+ |
| Node.js & npm | LTS recommended |

---

### Terminal 1 — ML Backend (FastAPI)

```bash
# Navigate to the backend
cd backend

# Install Python dependencies
pip install fastapi uvicorn pydantic

# Start the server
# (Windows users: use python -m prefix if uvicorn isn't on PATH)
python -m uvicorn main:app --reload --port 8000
```

The API will be live at **`http://localhost:8000`**

> 💡 If no `q_table.json` exists, the server will automatically train the agent on startup. This takes about a minute.

---

### Terminal 2 — React Frontend

```bash
# Navigate to the frontend
cd frontend

# Install dependencies
npm install

# Start the app
npm start
```

The dashboard will open automatically at **`http://localhost:3000`**

> Click **"Enable AI Mode"** inside the simulation to connect to the backend and watch the Q-Learning agent control the signals in real time.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Returns the optimal signal action for a given intersection state |
| `POST` | `/simulate` | Runs the agent forward N steps and returns the full trajectory |
| `GET` | `/health` | Server status and Q-table size |
| `GET` | `/info` | Agent hyperparameters and model metadata |

**Example `/predict` request:**
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

**Response:**
```json
{
  "action": "keep_green",
  "carbon_intensity": 1.5,
  "explanation": "🚑 AMBULANCE PRIORITY — immediate green for NS corridor.",
  "ambulance_override": true
}
```

---

## ⚙️ Agent Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Tabular Q-Learning |
| State space | Queue bins × Red-time bins × Phase × Carbon bin |
| Actions | `keep_green`, `switch_phase`, `extend_green` |
| Episodes | 10,000 (with 10% warmup phase) |
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.9 |
| Epsilon decay | 0.998 → min 0.1 |

The reward function penalises waiting time, queue imbalance, unnecessary phase switches, carbon-heavy idling, and emission spikes — while rewarding fully cleared queues.

---

## 🌍 Carbon Awareness

The agent adjusts behaviour based on grid carbon intensity at the current hour:

| Time | Intensity | Meaning |
|------|-----------|---------|
| 06:00 – 17:59 | `1.0` | Renewables active — lower idle penalty |
| 18:00 – 05:59 | `1.5` | Fossil fuels dominant — higher idle penalty |

The agent is more aggressive about clearing queues during high-carbon night hours to reduce the emission cost of idling vehicles.

---

## 🚑 Ambulance Priority Override

If `ambulance_direction` is passed as `"NS"` or `"EW"` in a `/predict` request, the Q-table is bypassed entirely and the signal immediately switches green for that corridor. The response includes `"ambulance_override": true` so the frontend can display the override visually.

---

## 📊 Emission Spike Detection

Both the environment and the `/simulate` endpoint track the **peak CO₂ emission** recorded across an episode. If instantaneous emissions exceed **5.0 kg CO₂**, an additional penalty is applied to the reward, pushing the agent to avoid congestion spikes especially during high-carbon hours.
