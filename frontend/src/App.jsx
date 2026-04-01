import React from "react";
import "./style.css";

function App() {
  return (
    <div className="app-wrapper">
      <iframe
        src="/sim/index.html"
        title="AI Traffic Simulation"
        className="sim-full-frame"
        allow="autoplay"
      />
    </div>
  );
}

export default App;
