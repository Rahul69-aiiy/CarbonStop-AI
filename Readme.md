# Traffic Signal Optimization

This project consists of two main components: a React Frontend Dashboard (which natively embeds the web simulation) and a Machine Learning FastAPI Backend.

Here is how you can get the entire project running locally. You will need two terminal windows.

### 1. ML Reinforcement Model (Backend)
This is the FastAPI server that runs the Q-Learning reinforcement model to predict optimal traffic light phases.
- **Prerequisites:** Python 3.8+
- **Open a Terminal** and navigate to the backend directory:
  ```bash
  cd backend
  ```
- **Install dependencies** (if you haven't already):
  ```bash
  pip install fastapi uvicorn pydantic
  ```
- **Run the server:**
  *(Note on Windows: Sometimes the `uvicorn` command alone is not recognized without adding it to PATH. To bypass this, we use the `python -m` prefix)*
  ```bash
  python -m uvicorn main:app --reload --port 8000
  ```
  *The API will be available at `http://localhost:8000`.*

### 2. Frontend React Dashboard & Web Simulation
This is the React application containing your dashboard and interfaces. It includes a fully functional traffic simulation embedded inside of it that communicates real-time with your API. 
- **Prerequisites:** Node.js & npm
- **Open a second Terminal** and navigate to the frontend directory:
  ```bash
  cd frontend
  ```
- **Install dependencies** (if you haven't already):
  ```bash
  npm install
  ```
- **Run the app:**
  ```bash
  npm start
  ```
  *The dashboard will automatically open in your default browser at `http://localhost:3000`. Inside the simulation portion, you can click "Enable AI Mode" to test the ML backend integration!*
