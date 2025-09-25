import React, { useState } from "react";
import { apiClient, ExplanationResult, SpectrumData } from "../apiClient";
import "../static/style.css"; // Ensure this is imported

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
      setError("No spectrum data is available for analysis.");
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiClient.explainSpectrum({
        spectrum: spectrumData,
        model_name: selectedModel,
        modality: modality,
        include_provenance: true, // Added the missing property
      });
      setExplanation(result);
      onExplainabilityResult(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unknown error occurred during analysis."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const renderFeatureImportance = () => {
    if (!explanation?.feature_importance) return null;

    const { feature_importance, prediction, class_labels } = explanation;
    const {
      method = "",
      summary = {
        max_importance: 0,
        mean_importance: 0,
        important_region_start: 0,
        important_region_end: 0,
      },
      top_features = { indices: [], values: [] },
    } = feature_importance;

    // Combine indices and values, sort by importance, and take the top 10 for safe rendering
    const topFeaturesData = top_features.indices
      .map((index: number, i: number) => ({
        index,
        value: top_features.values[i],
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);

    const maxImportance =
      summary.max_importance > 0 ? summary.max_importance : 1;

    return (
      <div className="feature-importance-content">
        <div className="importance-summary">
          <div className="summary-item">
            <label>Method:</label>
            <span>{method.replace(/_/g, " ")}</span>
          </div>
          <div className="summary-item">
            <label>Max Importance:</label>
            <span>{summary.max_importance?.toFixed(4) ?? "N/A"}</span>
          </div>
          <div className="summary-item">
            <label>Mean Importance:</label>
            <span>{summary.mean_importance?.toFixed(4) ?? "N/A"}</span>
          </div>
          <div className="summary-item">
            <label>Key Region:</label>
            <span>
              {summary.important_region_start} - {summary.important_region_end}
            </span>
          </div>
        </div>

        <h4 className="top-features-title">Top 10 Most Important Features</h4>
        <div className="features-grid">
          {topFeaturesData.map(({ index, value }) => (
            <div key={index} className="feature-item">
              <span className="feature-index">#{index}</span>
              <div className="importance-bar">
                <div
                  className="importance-fill"
                  style={{ width: `${(value / maxImportance) * 100}%` }}
                />
              </div>
              <span className="importance-value">{value.toFixed(3)}</span>
            </div>
          ))}
        </div>

        <div className="interpretation-guide model-info__callout">
          <h4>Interpretation Guide</h4>
          <ul>
            <li>
              <strong>Feature Index:</strong> The position (wavenumber) in the
              processed spectrum.
            </li>
            <li>
              <strong>Importance Score:</strong> How much a feature contributed
              to the final prediction.
            </li>
            <li>
              High scores indicate features that strongly influenced the model's
              decision towards **{class_labels[prediction]}**.
            </li>
          </ul>
        </div>
      </div>
    );
  };

  const renderPredictionSummary = () => {
    if (!explanation) return null;

    const {
      prediction,
      confidence,
      probabilities,
      class_labels,
      model_used,
      spectrum_filename,
    } = explanation;
    const predictedClass = class_labels[prediction].toLowerCase();
    const confidencePercent = (confidence * 100).toFixed(1);

    return (
      <div className="prediction-summary-content">
        <div className={`prediction-badge ${predictedClass}`}>
          {predictedClass.toUpperCase()}
        </div>
        <div className="confidence-score">{confidencePercent}% Confidence</div>

        <div className="probability-breakdown">
          {Object.entries(class_labels).map(([index, label]) => {
            const numericIndex = Number(index);
            return (
              <div key={label} className="probability-item">
                <span className="class-label">{label}</span>
                <div className="probability-bar">
                  <div
                    className={`probability-fill ${label.toLowerCase()}`}
                    style={{ width: `${probabilities[numericIndex] * 100}%` }}
                  />
                </div>
                <span className="probability-value">
                  {(probabilities[numericIndex] * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
        <div className="model-info-footer">
          <p>
            Model: {model_used} | File: {spectrum_filename || "N/A"}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="explainability-panel">
      <div className="card">
        <div className="panel-header">
          <h2 className="card__title">AI Explainability</h2>
          <p className="card__subtitle">
            Understand the "why" behind the model's prediction by identifying
            which spectral features were most influential.
          </p>
        </div>
        <div className="button-group">
          <button
            onClick={analyzeWithExplanation}
            disabled={!spectrumData || isLoading}
            className="btn btn--primary"
          >
            {isLoading ? "Analyzing..." : "Explain Prediction"}
          </button>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {explanation ? (
        <div className="explanation-layout">
          <div className="card">
            <h3 className="card__title">Prediction Summary</h3>
            {renderPredictionSummary()}
          </div>
          <div className="card">
            <h3 className="card__title">Feature Importance</h3>
            {renderFeatureImportance()}
          </div>
        </div>
      ) : (
        !isLoading &&
        !error && (
          <div className="placeholder">
            <p>
              {spectrumData
                ? "Click 'Explain Prediction' to begin analysis."
                : "Upload a spectrum on the 'Standard Analysis' tab to enable explainability."}
            </p>
          </div>
        )
      )}
    </div>
  );
};

export default ExplainabilityPanel;
