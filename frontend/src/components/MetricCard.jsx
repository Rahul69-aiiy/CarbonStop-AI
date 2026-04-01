import React from "react";

function MetricCard({ title, value }) {
  return (
    <div className="metric-card">
      <h4 className="metric-title">{title}</h4>
      <p className="metric-value">{value}</p>
    </div>
  );
}

export default MetricCard;
