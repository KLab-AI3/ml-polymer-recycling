import React from "react";
import { PredictionResult } from "../apiClient";
import SpectrumChart from "./SpectrumChart";

interface ResultsDisplayProps {
  result: PredictionResult;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
  const predictionClass = result.prediction === 0 ? "stable" : "weathered";

  return (
    <div className="results-display">
      {/* Main Prediction Result */}
      <div className={`prediction-result ${predictionClass}`}>
        <div className="prediction-label">{result.prediction_label}</div>
        <div className="confidence-score">
          Confidence: {(result.confidence * 100).toFixed(1)}%
        </div>
      </div>

      {/* Probabilities */}
      <div className="results-section">
        <h4>Class Probabilities</h4>
        <div className="probability-bars">
          <div className="probability-item">
            <span>Stable (Unweathered)</span>
            <div className="probability-bar">
              <div
                className="probability-fill stable"
                style={{ width: `${result.probabilities[0] * 100}%` }}
              ></div>
            </div>
            <span>{(result.probabilities[0] * 100).toFixed(1)}%</span>
          </div>
          <div className="probability-item">
            <span>Weathered (Degraded)</span>
            <div className="probability-bar">
              <div
                className="probability-fill weathered"
                style={{ width: `${result.probabilities[1] * 100}%` }}
              ></div>
            </div>
            <span>{(result.probabilities[1] * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="results-section">
        <h4>Performance Metrics</h4>
        <div className="metadata-grid">
          <div className="metadata-item">
            <h4>Inference Time</h4>
            <p>{(result.inference_time * 1000).toFixed(1)} ms</p>
          </div>
          <div className="metadata-item">
            <h4>Preprocessing Time</h4>
            <p>{(result.preprocessing_time * 1000).toFixed(1)} ms</p>
          </div>
          <div className="metadata-item">
            <h4>Total Time</h4>
            <p>{(result.total_time * 1000).toFixed(1)} ms</p>
          </div>
          <div className="metadata-item">
            <h4>Memory Usage</h4>
            <p>{result.memory_usage_mb.toFixed(1)} MB</p>
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="results-section">
        <h4>Model Information</h4>
        <div className="metadata-grid">
          <div className="metadata-item">
            <h4>Model</h4>
            <p>{result.model_metadata.model_name}</p>
          </div>
          <div className="metadata-item">
            <h4>Parameters</h4>
            <p>{result.model_metadata.parameters_count || "Unknown"}</p>
          </div>
          <div className="metadata-item">
            <h4>Weights Loaded</h4>
            <p>{result.model_metadata.weights_loaded ? "✅ Yes" : "❌ No"}</p>
          </div>
          <div className="metadata-item">
            <h4>Accuracy</h4>
            <p>
              {(
                (result.model_metadata.performance_metrics?.accuracy ?? 0) * 100
              ).toFixed(1)}
              %
            </p>
          </div>
        </div>
        <div className="model-description">
          <p style={{ fontStyle: "italic", color: "#666", marginTop: "1rem" }}>
            {result.model_metadata.model_description}
          </p>
        </div>
      </div>

      {/* Preprocessing Details */}
      <div className="results-section">
        <h4>Preprocessing Details</h4>
        <div className="metadata-grid">
          <div className="metadata-item">
            <h4>Original Length</h4>
            <p>{result.preprocessing.original_length} points</p>
          </div>
          <div className="metadata-item">
            <h4>Target Length</h4>
            <p>{result.preprocessing.target_length} points</p>
          </div>
          <div className="metadata-item">
            <h4>Baseline Degree</h4>
            <p>{result.preprocessing.baseline_degree}</p>
          </div>
          <div className="metadata-item">
            <h4>Smooth Window</h4>
            <p>{result.preprocessing.smooth_window}</p>
          </div>
          <div className="metadata-item">
            <h4>Wavenumber Range</h4>
            <p>
              {result.preprocessing.wavenumber_range[0].toFixed(1)} -{" "}
              {result.preprocessing.wavenumber_range[1].toFixed(1)} cm⁻¹
            </p>
          </div>
          <div className="metadata-item">
            <h4>Modality Validated</h4>
            <p>
              {result.preprocessing.modality_validated
                ? "✅ Passed"
                : "⚠️ Issues"}
            </p>
          </div>
        </div>

        {(result.preprocessing.validation_issues?.length ?? 0) > 0 && (
          <div className="validation-issues">
            <h4>Validation Issues:</h4>
            <ul>
              {result.preprocessing.validation_issues?.map((issue, index) => (
                <li
                  key={index}
                  style={{ color: "#e74c3c", fontSize: "0.9rem" }}
                >
                  {issue}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Quality Control */}
      <div className="results-section">
        <h4>Quality Control</h4>
        <div className="metadata-grid">
          {result.quality_control.signal_to_noise_ratio && (
            <div className="metadata-item">
              <h4>Signal-to-Noise Ratio</h4>
              <p>{result.quality_control.signal_to_noise_ratio.toFixed(2)}</p>
            </div>
          )}
          {result.quality_control.baseline_stability && (
            <div className="metadata-item">
              <h4>Baseline Stability</h4>
              <p>{result.quality_control.baseline_stability.toFixed(3)}</p>
            </div>
          )}
          <div className="metadata-item">
            <h4>Cosmic Ray Detection</h4>
            <p>
              {result.quality_control.cosmic_ray_detected
                ? "⚠️ Detected"
                : "✅ Clean"}
            </p>
          </div>
          <div className="metadata-item">
            <h4>Saturation Detection</h4>
            <p>
              {result.quality_control.saturation_detected
                ? "⚠️ Detected"
                : "✅ Clean"}
            </p>
          </div>
        </div>

        {(result.quality_control.issues?.length ?? 0) > 0 && (
          <div className="qc-issues">
            <h4>Quality Issues:</h4>
            <ul>
              {result.quality_control.issues?.map((issue, index) => (
                <li
                  key={index}
                  style={{ color: "#f39c12", fontSize: "0.9rem" }}
                >
                  {issue}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Timestamp */}
      <div className="results-section">
        <div className="metadata-item">
          <h4>Analysis Timestamp</h4>
          <p>{new Date(result.timestamp).toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
