/**
 * ExplainabilityPanel Component
 *
 * Displays model explanations and feature importance for polymer aging predictions.
 * Integrates with the enhanced ML service to provide interpretable AI results.
 */

import React, { useState } from "react";
import "./ExplainabilityPanel.css";
import { apiClient, SpectrumData, ExplanationResult } from "../apiClient";

interface ExplainabilityPanelProps {
  spectrumData: SpectrumData | null;
  selectedModel: string;
  modality: "raman" | "ftir";
  onExplainabilityResult: (result: ExplanationResult) => void;
}

const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  spectrumData,
  selectedModel,
  modality,
  onExplainabilityResult,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const analyzeWithExplanation = async () => {
    if (!spectrumData) {
      setError("No spectrum data available");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use the centralized API client
      const result = await apiClient.explainSpectrum({
        spectrum: spectrumData,
        model_name: selectedModel,
        modality: modality,
        include_provenance: true,
      });

      setExplanation(result);
      onExplainabilityResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const renderFeatureImportance = () => {
    if (!explanation?.feature_importance) return null;

    const { feature_importance } = explanation;
    // Fix: Add explicit types for destructuring
    const method = feature_importance.method ?? "";
    const top_features: { indices: number[]; values: number[] } = feature_importance.top_features ?? { indices: [], values: [] };
    const summary =
      typeof feature_importance.summary === "object" && feature_importance.summary !== null
        ? feature_importance.summary
        : {
            max_importance: 0,
            mean_importance: 0,
            important_region_start: 0,
            important_region_end: 0,
          };
    const importance_scores = feature_importance.importance_scores ?? [];

    return (
      <div className="feature-importance-section">
        <h3>🔍 Feature Importance Analysis</h3>

        <div className="importance-summary">
          <div className="summary-item">
            <label>Method:</label>
            <span>{method.replace("_", " ")}</span>
          </div>
          <div className="summary-item">
            <label>Max Importance:</label>
            <span>{summary.max_importance.toFixed(4)}</span>
          </div>
          <div className="summary-item">
            <label>Mean Importance:</label>
            <span>{summary.mean_importance.toFixed(4)}</span>
          </div>
          <div className="summary-item">
            <label>Key Region:</label>
            <span>
              Features {summary.important_region_start} -{" "}
              {summary.important_region_end}
            </span>
          </div>
        </div>

        <div className="top-features">
          <h4>Top Important Features</h4>
          <div className="features-grid">
            {top_features.indices.slice(-10).map((index: number, i: number) => (
              <div key={index} className="feature-item">
                <span className="feature-index">#{index}</span>
                <div className="importance-bar">
                  <div
                    className="importance-fill"
                    style={{
                      width: `${(top_features.values[i] / summary.max_importance) * 100}%`,
                    }}
                  />
                </div>
                <span className="importance-value">
                  {top_features.values[i].toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="interpretation-help">
          <h4>💡 Interpretation Guide</h4>
          <ul>
            <li>
              <strong>Feature Index:</strong> Position in the processed spectrum
              (0-{importance_scores.length - 1})
            </li>
            <li>
              <strong>Importance Score:</strong> How much this feature
              contributed to the prediction
            </li>
            <li>
              <strong>Key Region:</strong> The spectral range most relevant for
              classification
            </li>
            <li>
              <strong>High scores</strong> indicate features that strongly
              influenced the {explanation.class_labels[explanation.prediction]}{" "}
              prediction
            </li>
          </ul>
        </div>
      </div>
    );
  };

  const renderPredictionSummary = () => {
    if (!explanation) return null;

    const { prediction, confidence, probabilities, class_labels } = explanation;
    const predictedClass = class_labels[prediction];
    const confidencePercent = (confidence * 100).toFixed(1);

    return (
      <div className="prediction-summary">
        <h3>🎯 Prediction Results</h3>

        <div className="prediction-main">
          <div className={`prediction-badge ${predictedClass.toLowerCase()}`}>
            {predictedClass.toUpperCase()}
          </div>
          <div className="confidence-score">
            {confidencePercent}% confidence
          </div>
        </div>

        <div className="probability-breakdown">
          <h4>Class Probabilities</h4>
          {(Array.isArray(class_labels) ? class_labels : Object.values(class_labels)).map((label: string, index: number) => (
            <div key={label} className="probability-item">
              <span className="class-label">{label}</span>
              <div className="probability-bar">
                <div
                  className={`probability-fill ${label.toLowerCase()}`}
                  style={{ width: `${probabilities[index] * 100}%` }}
                />
              </div>
              <span className="probability-value">
                {(probabilities[index] * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>

        <div className="model-info">
          <small>
            Model: {explanation.model_used} | File:{" "}
            {explanation.spectrum_filename || "N/A"}
          </small>
        </div>
      </div>
    );
  };

  return (
    <div className="explainability-panel">
      <div className="panel-header">
        <h2>🔬 AI Explainability</h2>
        <p>Understand how the model makes its predictions</p>
      </div>

      <div className="panel-actions">
        <button
          onClick={analyzeWithExplanation}
          disabled={!spectrumData || isLoading}
          className="explain-button primary"
        >
          {isLoading ? "🔄 Analyzing..." : "🔍 Explain Prediction"}
        </button>
      </div>

      {error && (
        <div className="error-message">
          <h4>❌ Error</h4>
          <p>{error}</p>
        </div>
      )}

      {explanation && (
        <div className="explanation-results">
          {renderPredictionSummary()}
          {renderFeatureImportance()}
        </div>
      )}

      {!spectrumData && (
        <div className="no-data-message">
          <p>
            📁 Upload or paste spectrum data to enable explainability analysis
          </p>
        </div>
      )}
    </div>
  );
};

export default ExplainabilityPanel;
