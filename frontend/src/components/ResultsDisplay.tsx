import React from "react";
import { PredictionResult } from "../apiClient";
import "../static/style.css"; // Ensure the main CSS file is imported

interface ResultsDisplayProps {
  result: PredictionResult;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
  const confidencePercent = (result.confidence * 100).toFixed(1);

  return (
    <div className="results-wrapper">
      {/* --- CONFIDENCE HEADER --- */}
      <header className="results-header">
        <div className="confidence-display">
          <span className="confidence-display__label">Confidence Score</span>
          <span className="confidence-display__value">{confidencePercent}%</span>
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-bar__fill"
            style={{ width: `${confidencePercent}%` }}
            role="progressbar"
            aria-valuenow={parseFloat(confidencePercent)}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
        <div className="probabilities">
          <div className="prob-item">
            <span className="prob-item__label">Stable</span>
            <span className="prob-item__value">
              {(result.probabilities[0] * 100).toFixed(1)}%
            </span>
          </div>
          <div className="prob-item">
            <span className="prob-item__label">Weathered</span>
            <span className="prob-item__value">
              {(result.probabilities[1] * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </header>

      {/* --- GROUPED METRICS CONTAINER (THE FIX) --- */}
      <div className="results-grid-container">
        <div className="results-grid">

          {/* Performance Card */}
          <div className="results-card">
            <h4 className="results-card__title">Performance</h4>
            <ul className="results-card__list">
              <li><span>Inference</span><span>{(result.inference_time * 1000).toFixed(1)} ms</span></li>
              <li><span>Preprocessing</span><span>{(result.preprocessing_time * 1000).toFixed(1)} ms</span></li>
              <li><span>Total Time</span><span>{(result.total_time * 1000).toFixed(1)} ms</span></li>
              <li><span>Memory</span><span>{result.memory_usage_mb.toFixed(1)} MB</span></li>
            </ul>
          </div>

          {/* Model Card */}
          <div className="results-card">
            <h4 className="results-card__title">Model</h4>
            <ul className="results-card__list">
              <li><span>Name</span><span>{result.model_metadata.model_name}</span></li>
              <li><span>Parameters</span><span>{result.model_metadata.parameters_count || "N/A"}</span></li>
              <li><span>Accuracy</span><span>{((result.model_metadata.performance_metrics?.accuracy ?? 0) * 100).toFixed(1)}%</span></li>
            </ul>
          </div>

          {/* Quality Control Card */}
          <div className="results-card">
            <h4 className="results-card__title">Quality Control</h4>
            <ul className="results-card__list">
              <li><span>S/N Ratio</span><span>{result.quality_control.signal_to_noise_ratio?.toFixed(2) ?? "N/A"}</span></li>
              <li>
                <span>Cosmic Rays</span>
                <span className={result.quality_control.cosmic_ray_detected ? "qc-issue" : "qc-ok"}>
                  {result.quality_control.cosmic_ray_detected ? "Detected" : "Clean"}
                </span>
              </li>
              <li>
                <span>Saturation</span>
                <span className={result.quality_control.saturation_detected ? "qc-issue" : "qc-ok"}>
                  {result.quality_control.saturation_detected ? "Detected" : "Clean"}
                </span>
              </li>
            </ul>
          </div>

          {/* Preprocessing Card */}
          <div className="results-card">
            <h4 className="results-card__title">Preprocessing</h4>
            <ul className="results-card__list">
                <li><span>Range (cm⁻¹)</span><span>{result.preprocessing.wavenumber_range[0].toFixed(0)} - {result.preprocessing.wavenumber_range[1].toFixed(0)}</span></li>
                <li><span>Data Points</span><span>{result.preprocessing.original_length} → {result.preprocessing.target_length}</span></li>
                <li><span>Smooth Window</span><span>{result.preprocessing.smooth_window}</span></li>
            </ul>
          </div>

        </div>
      </div>

      {/* --- QUALITY ISSUES CALLOUT --- */}
      {(result.quality_control.issues?.length ?? 0) > 0 && (
        <div className="results-issues">
          <h5>Quality Issues Detected</h5>
          <ul>
            {(result.quality_control.issues ?? []).map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
