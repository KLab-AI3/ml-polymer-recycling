import React from "react";
import { PredictionResult } from "../apiClient";

{/* The misplaced header code has been removed */}

interface ResultsDisplayProps {
  result: PredictionResult;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
  const confidencePercent = (result.confidence * 100).toFixed(1);

  return (
    <section className="results-card">
      <header className="results-header">
        <div className="results-confidence-row">
          <span className="results-confidence-label">Confidence</span>
          <span className="results-confidence-value">{confidencePercent}%</span>
        </div>
        <div className="results-confidence-bar">
          <div
            className="results-confidence-bar-fill"
            aria-valuenow={parseFloat(confidencePercent)}
            aria-valuemin={0}
            aria-valuemax={100}
            role="progressbar"
            title={`Confidence: ${confidencePercent}%`}
          />
        </div>
        <div className="results-probabilities">
          <div>
            <span className="results-label">Stable</span>
            <span className="results-prob-value">
              {(result.probabilities[0] * 100).toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="results-label">Weathered</span>
            <span className="results-prob-value">
              {(result.probabilities[1] * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </header>

      <div className="results-main-grid">
        <section>
          <h4>Performance</h4>
          <ul>
            <li>
              <span>Inference</span>
              <span>{(result.inference_time * 1000).toFixed(1)} ms</span>
            </li>
            <li>
              <span>Preprocessing</span>
              <span>{(result.preprocessing_time * 1000).toFixed(1)} ms</span>
            </li>
            <li>
              <span>Total</span>
              <span>{(result.total_time * 1000).toFixed(1)} ms</span>
            </li>
            <li>
              <span>Memory</span>
              <span>{result.memory_usage_mb.toFixed(1)} MB</span>
            </li>
          </ul>
        </section>
        <section>
          <h4>Model</h4>
          <ul>
            <li>
              <span>Name</span>
              <span>{result.model_metadata.model_name}</span>
            </li>
            <li>
              <span>Parameters</span>
              <span>{result.model_metadata.parameters_count || "Unknown"}</span>
            </li>
            <li>
              <span>Weights</span>
              <span>
                {result.model_metadata.weights_loaded ? "✅ Yes" : "❌ No"}
              </span>
            </li>
            <li>
              <span>Accuracy</span>
              <span>
                {(
                  (result.model_metadata.performance_metrics?.accuracy ?? 0) *
                  100
                ).toFixed(1)}
                %
              </span>
            </li>
          </ul>
          <div className="results-description">
            {result.model_metadata.model_description}
          </div>
        </section>
        <section>
          <h4>Preprocessing</h4>
          <ul>
            <li>
              <span>Original</span>
              <span>{result.preprocessing.original_length} pts</span>
            </li>
            <li>
              <span>Target</span>
              <span>{result.preprocessing.target_length} pts</span>
            </li>
            <li>
              <span>Baseline Degree</span>
              <span>{result.preprocessing.baseline_degree}</span>
            </li>
            <li>
              <span>Smooth Window</span>
              <span>{result.preprocessing.smooth_window}</span>
            </li>
            <li>
              <span>Range</span>
              <span>
                {result.preprocessing.wavenumber_range[0].toFixed(1)} -{" "}
                {result.preprocessing.wavenumber_range[1].toFixed(1)} cm⁻¹
              </span>
            </li>
            <li>
              <span>Modality</span>
              <span>
                {result.preprocessing.modality_validated
                  ? "✅ Passed"
                  : "⚠️ Issues"}
              </span>
            </li>
          </ul>
          {(result.preprocessing.validation_issues ?? []).length > 0 && (
            <div className="results-issues">
              <h5>Validation Issues</h5>
              <ul>
                {(result.preprocessing.validation_issues ?? []).map((issue, i) => (
                  <li key={i}>{issue}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
        <section>
          <h4>Quality Control</h4>
          <ul>
            <li>
              <span>S/N Ratio</span>
              <span>
                {result.quality_control.signal_to_noise_ratio?.toFixed(2)}
              </span>
            </li>
            <li>
              <span>Baseline Stability</span>
              <span>
                {result.quality_control.baseline_stability?.toFixed(3)}
              </span>
            </li>
            <li>
              <span>Cosmic Rays</span>
              <span>
                {result.quality_control.cosmic_ray_detected
                  ? "⚠️ Detected"
                  : "✅ Clean"}
              </span>
            </li>
            <li>
              <span>Saturation</span>
              <span>
                {result.quality_control.saturation_detected
                  ? "⚠️ Detected"
                  : "✅ Clean"}
              </span>
            </li>
          </ul>
          {(result.quality_control.issues?.length ?? 0) > 0 && (
            <div className="results-issues">
              <h5>Quality Issues</h5>
              <ul>
                {(result.quality_control.issues ?? []).map((issue, i) => (
                  <li key={i}>{issue}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>
      <footer className="results-footer">
        <span>
          Timestamp: <strong>{new Date(result.timestamp).toLocaleString()}</strong>
        </span>
      </footer>
    </section>
  );
};

export default ResultsDisplay;
