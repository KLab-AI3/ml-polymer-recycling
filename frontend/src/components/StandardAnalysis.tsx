import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { apiClient, SpectrumData, PredictionResult } from "../apiClient";
import SpectrumChart from "./SpectrumChart";
import ResultsDisplay from "./ResultsDisplay";
import "../static/style.css";

interface StandardAnalysisProps {
  selectedModel: string;
  modality: "raman" | "ftir";
  onSpectrumChange: (data: SpectrumData | null) => void;
}

const StandardAnalysis: React.FC<StandardAnalysisProps> = ({
  selectedModel,
  modality,
  onSpectrumChange,
}) => {
  const [spectrum, setSpectrum] = useState<SpectrumData | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      const file = acceptedFiles[0];
      setResult(null);
      setError(null);
      setLoading(true);
      try {
        const uploadedSpectrum = await apiClient.uploadSpectrum(file);
        setSpectrum(uploadedSpectrum);
        onSpectrumChange(uploadedSpectrum);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to upload file");
      } finally {
        setLoading(false);
      }
    },
    [onSpectrumChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/plain": [".txt", ".csv"], "application/json": [".json"] },
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!spectrum) return;
    setError(null);
    setLoading(true);
    try {
      const analysisResult = await apiClient.analyzeSpectrum({
        spectrum,
        model_name: selectedModel,
        modality,
        include_provenance: true,
      });
      setResult(analysisResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSpectrum(null);
    onSpectrumChange(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="analysis-layout">
      {/* --- LEFT COLUMN --- */}
      <div className="analysis-column">
        <div className="card">
          <h3 className="card__title">Upload Spectrum</h3>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "dropzone--active" : ""}`}
          >
            <input {...getInputProps()} />
            {/* Updated Dropzone Text */}
            <div className="dropzone-content">
              <p className="dropzone__text">
                {isDragActive
                  ? "Drop file to upload"
                  : "Drag & drop, or click to select"}
              </p>
              <p className="dropzone__subtext">
                Supports .txt, .csv, and .json files
              </p>
            </div>
          </div>

          {/* New Sample Data Link Section */}
          <div className="sample-data-prompt">
            <span>New to PolymerOS?</span>
            <a
              href="https://data.mendeley.com/datasets/kpygrf9fg6/1"
              target="_blank"
              rel="noopener noreferrer"
              className="sample-data-link"
            >
              Try a Sample Dataset
            </a>
          </div>

          {error && <div className="error-message">{error}</div>}
          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>Processing...</span>
            </div>
          )}

          {spectrum && !loading && (
            <div className="spectrum-info">
              <p>
                <strong>Loaded:</strong> {spectrum.filename || "Unknown"}
              </p>
              <div className="button-group">
                <button
                  className="btn btn--primary"
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  Analyze
                </button>
                <button
                  className="btn btn--secondary"
                  onClick={handleReset}
                  disabled={loading}
                >
                  Reset
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* --- RIGHT COLUMN --- */}
      <div className="analysis-column">
        <div className="card">
          <h3 className="card__title">Analysis Results</h3>
          {spectrum && (
            <div className="chart-container">
              <SpectrumChart spectrum={spectrum} />
            </div>
          )}
          {result ? (
            <ResultsDisplay result={result} />
          ) : (
            <div className="placeholder">
              <p>Upload a spectrum to view analysis results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StandardAnalysis;
