import React, { useRef, useEffect } from "react";

const LiveSimulation = ({ vehicles, lights }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Scale (SUMO coordinate 200 maps to canvas width)
    const scale = width / 200;

    // Draw Cross road layout
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 20 * scale; // Width of the road
    ctx.beginPath();
    // Horizontal road (centered at y=100 in SUMO)
    ctx.moveTo(0 * scale, 100 * scale);
    ctx.lineTo(200 * scale, 100 * scale);
    // Vertical road (centered at x=100 in SUMO)
    ctx.moveTo(100 * scale, 0 * scale);
    ctx.lineTo(100 * scale, 200 * scale);
    ctx.stroke();

    // Draw central intersection area
    ctx.fillStyle = "#333";
    ctx.beginPath();
    ctx.rect(90 * scale, 90 * scale, 20 * scale, 20 * scale);
    ctx.fill();

    // Draw traffic light indicators
    const normalizedLights = lights ? lights.toUpperCase() : "RRRR";
    const isNSGreen = normalizedLights[0] === 'G' || normalizedLights[2] === 'G';
    
    ctx.fillStyle = isNSGreen ? "#2ecc71" : "#e74c3c";
    ctx.beginPath();
    ctx.arc(100 * scale, 85 * scale, 8, 0, Math.PI * 2); // Top
    ctx.arc(100 * scale, 115 * scale, 8, 0, Math.PI * 2); // Bottom
    ctx.fill();

    const isEWGreen = normalizedLights[1] === 'G' || normalizedLights[3] === 'G';
    ctx.fillStyle = isEWGreen ? "#2ecc71" : "#e74c3c";
    ctx.beginPath();
    ctx.arc(85 * scale, 100 * scale, 8, 0, Math.PI * 2); // Left
    ctx.arc(115 * scale, 100 * scale, 8, 0, Math.PI * 2); // Right
    ctx.fill();

    // Draw vehicles
    ctx.fillStyle = "#3498db";
    vehicles.forEach(v => {
      ctx.save();
      // SUMO (0,0) is bottom-left, Canvas (0,0) is top-left
      // Mapping: Canvas_X = v.x * scale, Canvas_Y = (200 - v.y) * scale
      ctx.translate(v.x * scale, (200 - v.y) * scale);
      ctx.rotate((v.angle * Math.PI) / 180);
      ctx.fillRect(-6, -4, 12, 8);
      ctx.restore();
    });

  }, [vehicles, lights]);

  return (
    <div className="live-sim-wrapper">
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={400} 
        className="sim-canvas"
      />
      <div className="sim-meta">
        <span>Vehicles: {vehicles.length}</span>
        <span>Signal: {lights}</span>
      </div>
    </div>
  );
};

export default LiveSimulation;
