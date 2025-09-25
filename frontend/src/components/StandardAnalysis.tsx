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
      setError(null);
      setLoading(true);

      try {
        const uploadedSpectrum = await apiClient.uploadSpectrum(file);
        setSpectrum(uploadedSpectrum);
        onSpectrumChange(uploadedSpectrum); // Lift state up to parent
        setResult(null); // Clear previous results
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
    accept: {
      "text/plain": [".txt"],
      "text/csv": [".csv"],
      "application/json": [".json"],
    },
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
    onSpectrumChange(null); // Clear state in parent
    setResult(null);
    setError(null);
  };

  return (
    <div
      className="analysis-panel"
      role="region"
      aria-label="Standard Analysis"
    >

      <div className="input-column">
        <h3>Upload Spectrum</h3>

        <div
          {...getRootProps()}
          className={`file-upload ${isDragActive ? "dragover" : ""}`}
          aria-label="Upload spectrum file"
        >
          <input {...getInputProps()} aria-label="spectrum-file-input" />
          <div
            className="upload-icon"
            aria-hidden
            style={{
              width: 48,
              height: 48,
              borderRadius: 6,
              background: "#e6f3ff",
            }}
          />
          <div className="upload-text">
            {isDragActive ? (
              <p>Drop the spectrum file here...</p>
            ) : (
              <>
                <p>Drag & drop a spectrum file here, or click to select</p>
                <p>Supports .txt, .csv, .json files</p>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}

        {spectrum && (
          <div className="spectrum-info" aria-live="polite">
            <div className="success">
              ✅ Spectrum loaded: {spectrum.filename || "Unknown filename"}
              <br />
              Data points: {spectrum.x_values.length}
              <br />
              Range: {Math.min(...spectrum.x_values).toFixed(1)} -{" "}
              {Math.max(...spectrum.x_values).toFixed(1)} cm⁻¹
            </div>

            <div
              style={{
                marginTop: "1rem",
                display: "flex",
                gap: "0.5rem",
                flexWrap: "wrap",
              }}
            >
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={loading}
                aria-disabled={loading}
                aria-label="Analyze spectrum"
              >
                {loading ? (
                  <>
                    <div className="loading-spinner" aria-hidden />
                    Analyzing...
                  </>
                ) : (
                  <>Analyze Spectrum</>
                )}
              </button>

              <button
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
                aria-label="Reset spectrum"
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {loading && (
          <div className="loading" aria-live="polite">
            <div className="loading-spinner" aria-hidden></div>
            Processing spectrum...
          </div>
        )}
      </div>

      <div className="results-column">
        <h3>Analysis Results</h3>

        {spectrum && (
          <div className="chart-container" aria-hidden={!!result === false}>
            <h4>Spectrum Visualization</h4>
            <SpectrumChart spectrum={spectrum} />
          </div>
        )}

        {result ? (
          <ResultsDisplay result={result} />
        ) : (
          <div className="placeholder">
            Upload a spectrum file to see analysis results
          </div>
        )}
      </div>
    </div>
  );
};

export default StandardAnalysis;
