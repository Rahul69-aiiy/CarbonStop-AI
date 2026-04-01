import React, { useState, useEffect } from "react";

function Controls({ fetchMetrics }) {
  const [customState, setCustomState] = useState("");
  const [autoMode, setAutoMode] = useState(false);
  const [lanes, setLanes] = useState({
    north: false,
    south: false,
    east: false,
    west: false,
  });

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch("/sim/status");
        const data = await res.json();
        setAutoMode(data.auto_mode);
      } catch (e) {}
    };
    checkStatus();
  }, []);

  const post = async (url, body) => {
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  };

  const start = () => post("/sim/start", {});
  const step = () => post("/sim/step", {}).then(fetchMetrics);
  const stop = () => {
    post("/sim/stop", {});
    setAutoMode(false);
  };
  
  const toggleAuto = async () => {
    const newAuto = !autoMode;
    if (newAuto) {
      await post("/sim/start_auto", {});
    } else {
      await post("/sim/stop_auto", {});
    }
    setAutoMode(newAuto);
  };

  const setLight = (state) => post("/sim/set_light", { state });

  const computeFromLanes = () => {
    const nsGreen = lanes.north || lanes.south;
    const ewGreen = lanes.east || lanes.west;
    if (nsGreen && !ewGreen) return "GrGr";
    if (!nsGreen && ewGreen) return "rGrG";
    if (nsGreen && ewGreen) return "GGGG";
    return "rrrr";
  };

  return (
    <div className="controls-container">
      <div className="primary-controls">
        <button className="btn-start" onClick={start}>Start SIM</button>
        <button className="btn-stop" onClick={stop}>Stop SIM</button>
        <button className={`btn-auto ${autoMode ? 'active' : ''}`} onClick={toggleAuto}>
          {autoMode ? "Disable Auto" : "Enable Auto Mode"}
        </button>
        {!autoMode && <button onClick={step}>Step</button>}
      </div>
      
      <div className="light-controls">
        <h3>Manual Override</h3>
        <div className="button-group">
          <button onClick={() => setLight("GrGr")}>NS Green</button>
          <button onClick={() => setLight("rGrG")}>EW Green</button>
        </div>
        
        <div className="custom-input-group">
          <input
            value={customState}
            onChange={(e) => setCustomState(e.target.value)}
            placeholder="GrGr or rGrG"
            maxLength={4}
          />
          <button onClick={() => setLight(customState)}>Set State</button>
        </div>

        <div className="lane-checks">
          <h4>Lane State Builder</h4>
          <div className="checkboxes">
            <label>
              <input
                type="checkbox"
                checked={lanes.north}
                onChange={() => setLanes({ ...lanes, north: !lanes.north })}
              />{" "}
              North
            </label>
            <label>
              <input
                type="checkbox"
                checked={lanes.south}
                onChange={() => setLanes({ ...lanes, south: !lanes.south })}
              />{" "}
              South
            </label>
            <label>
              <input
                type="checkbox"
                checked={lanes.east}
                onChange={() => setLanes({ ...lanes, east: !lanes.east })}
              />{" "}
              East
            </label>
            <label>
              <input
                type="checkbox"
                checked={lanes.west}
                onChange={() => setLanes({ ...lanes, west: !lanes.west })}
              />{" "}
              West
            </label>
          </div>
          <button className="btn-apply" onClick={() => setLight(computeFromLanes())}>
            Apply Lane State
          </button>
        </div>
      </div>
    </div>
  );
}

export default Controls;
