import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { apiClient, SpectrumData, PredictionResult } from "../apiClient";
import SpectrumChart from "./SpectrumChart";
import ResultsDisplay from "./ResultsDisplay";

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
    <div className="analysis-panel">
      <div className="input-column">
        <h3>📁 Upload Spectrum</h3>

        <div
          {...getRootProps()}
          className={`file-upload ${isDragActive ? "dragover" : ""}`}
        >
          <input {...getInputProps()} />
          <div className="upload-icon">📊</div>
          <div className="upload-text">
            {isDragActive ? (
              <p>Drop the spectrum file here...</p>
            ) : (
              <>
                <p>Drag & drop a spectrum file here, or click to select</p>
                <p style={{ fontSize: "0.8rem", color: "#888" }}>
                  Supports .txt, .csv, .json files
                </p>
              </>
            )}
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        {spectrum && (
          <div className="spectrum-info">
            <div className="success">
              ✅ Spectrum loaded: {spectrum.filename || "Unknown filename"}
              <br />
              Data points: {spectrum.x_values.length}
              <br />
              Range: {Math.min(...spectrum.x_values).toFixed(1)} -{" "}
              {Math.max(...spectrum.x_values).toFixed(1)} cm⁻¹
            </div>

            <div style={{ marginTop: "1rem" }}>
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={loading}
                style={{ marginRight: "0.5rem" }}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  <>🔬 Analyze Spectrum</>
                )}
              </button>

              <button
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                🔄 Reset
              </button>
            </div>
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="loading-spinner"></div>
            Processing spectrum...
          </div>
        )}
      </div>

      <div className="results-column">
        <h3>📊 Analysis Results</h3>

        {spectrum && (
          <div className="chart-container">
            <h4>Spectrum Visualization</h4>
            <SpectrumChart spectrum={spectrum} />
          </div>
        )}

        {result && <ResultsDisplay result={result} />}

        {!spectrum && !result && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "200px",
              color: "#888",
              fontSize: "1.1rem",
            }}
          >
            Upload a spectrum file to see analysis results
          </div>
        )}
      </div>
    </div>
  );
};

export default StandardAnalysis;
