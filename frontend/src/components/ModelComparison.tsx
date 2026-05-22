import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import {
  apiClient,
  SpectrumData,
  ComparisonResult,
  ModelInfo,
  PredictionResult,
} from "../apiClient";
import SpectrumChart from "./SpectrumChart";
import ResultsDisplay from "./ResultsDisplay"; // We'll reuse parts of this component's display logic
import "../static/style.css";

interface ModelComparisonProps {
  modality: "raman" | "ftir";
}

const ModelComparison: React.FC<ModelComparisonProps> = ({ modality }) => {
  const [spectrum, setSpectrum] = useState<SpectrumData | null>(null);
  const [comparisonResult, setComparisonResult] =
    useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const models = await apiClient.getModels();
        const available = models.filter((m) => m.available);
        setAvailableModels(available);
        // Default to selecting the first two models for comparison
        setSelectedModels(available.slice(0, 2).map((m) => m.name));
      } catch (err) {
        setError("Failed to fetch available models.");
      }
    };
    fetchModels();
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    setError(null);
    setLoading(true);
    try {
      const uploadedSpectrum = await apiClient.uploadSpectrum(file);
      setSpectrum(uploadedSpectrum);
      setComparisonResult(null); // Clear previous results on new upload
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload file");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleToggleModel = (modelName: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelName)
        ? prev.filter((m) => m !== modelName)
        : [...prev, modelName]
    );
  };

  const handleCompare = async () => {
    if (!spectrum) {
      setError("Please upload a spectrum file first.");
      return;
    }
    if (selectedModels.length < 2) {
      setError("Please select at least two models to compare.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const result = await apiClient.compareModels({
        spectrum: spectrum,
        modality: modality,
        model_names: selectedModels,
        include_provenance: true, // Add this property
      });
      setComparisonResult(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Comparison analysis failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="analysis-layout">
      {/* --- LEFT COLUMN (CONTROLS) --- */}
      <div className="analysis-column">
        <div className="card">
          <h3 className="card__title">1. Upload Spectrum</h3>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "dropzone--active" : ""}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-content">
              <span className="dropzone-icon">📤</span>
              <p className="dropzone__text">
                {isDragActive
                  ? "Drop file to upload"
                  : "Drag & drop, or click to select"}
              </p>
            </div>
          </div>
          {spectrum && (
            <div className="spectrum-info">
              <p>
                <strong>Loaded:</strong> {spectrum.filename || "Unknown"}
              </p>
            </div>
          )}
        </div>

        <div className="card">
          <h3 className="card__title">2. Select Models to Compare</h3>
          <div className="checkbox-grid">
            {availableModels.map((model) => (
              <label key={model.name} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.name)}
                  onChange={() => handleToggleModel(model.name)}
                />
                <span className="checkbox-custom"></span>
                {model.display_name || model.name}
              </label>
            ))}
          </div>
        </div>

        <div className="card">
          <h3 className="card__title">3. Run Comparison</h3>
          <div className="button-group">
            <button
              onClick={handleCompare}
              disabled={loading || !spectrum || selectedModels.length < 2}
              className="btn btn--primary"
            >
              {loading ? "Comparing..." : "Compare Models"}
            </button>
          </div>
          {error && (
            <div className="error-message" style={{ marginTop: "1rem" }}>
              {error}
            </div>
          )}
        </div>
      </div>

      {/* --- RIGHT COLUMN (RESULTS) --- */}
      <div className="analysis-column">
        <div className="card">
          <h3 className="card__title">Comparison Results</h3>
          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>Running comparison analysis...</span>
            </div>
          )}
          {comparisonResult ? (
            <div className="comparison-results-container">
              <div className="comparison-summary">
                <h4>Best Performing Model</h4>
                {comparisonResult.model_results && (
                  <p>
                    <strong>
                      {
                        Object.entries(comparisonResult.model_results).reduce(
                          (best, [modelName, res]) =>
                            res.confidence > best.confidence
                              ? { modelName, confidence: res.confidence }
                              : best,
                          { modelName: "Unknown", confidence: 0 }
                        ).modelName
                      }
                    </strong>{" "}
                    with a confidence of{" "}
                    <strong>
                      {(
                        Object.entries(comparisonResult.model_results).reduce(
                          (best, [_, res]) =>
                            res.confidence > best ? res.confidence : best,
                          0
                        ) * 100
                      ).toFixed(1)}
                      %
                    </strong>
                    .
                  </p>
                )}
              </div>
              <div className="comparison-grid">
                {comparisonResult.model_results &&
                  Object.entries(comparisonResult.model_results).map(
                    ([modelName, res]) => (
                      <div key={modelName} className="comparison-card">
                        <div className="results-wrapper">
                          <header className="results-header">
                            <div className="confidence-display">
                              <span className="confidence-display__label">
                                {modelName}
                              </span>
                              <span className="confidence-display__value">
                                {(res.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="confidence-bar">
                              <div
                                className="confidence-bar__fill"
                                style={{
                                  width: `${(res.confidence * 100).toFixed(1)}%`,
                                }}
                              />
                            </div>
                          </header>
                        </div>
                      </div>
                    )
                  )}
              </div>
            </div>
          ) : (
            !loading && (
              <div className="placeholder">
                <p>
                  Upload a spectrum and select models to see comparison results.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
