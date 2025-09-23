import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import {
  apiClient,
  SpectrumData,
  ComparisonResult,
  ModelInfo,
} from "../apiClient";
import SpectrumChart from "./SpectrumChart";
import ResultsDisplay from "./ResultsDisplay";

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
        setSelectedModels(available.slice(0, 2).map((m) => m.name));
      } catch (err) {
        setError("Failed to fetch models");
      }
    };
    fetchModels();
  }, []);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      const file = acceptedFiles[0];
      setError(null);
      setLoading(true);
      try {
        const uploadedSpectrum = await apiClient.uploadSpectrum(file);
        setSpectrum(uploadedSpectrum);
        setComparisonResult(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to upload file");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const handleCompare = async () => {
    if (!spectrum || selectedModels.length < 2) {
      setError("Please upload a spectrum and select at least two models.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const result = await apiClient.compareModels({
        spectrum: spectrum,
        modality: modality,
        model_names: selectedModels,
        include_provenance: true,
      });
      setComparisonResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  // ... Abridged JSX for brevity ...
  return (
    <div className="analysis-panel">
      <div className="input-column">
        <h3>Upload Spectrum</h3>
        <div {...getRootProps()} className="file-upload">
          <input {...getInputProps()} />
          <p>Drag & drop a spectrum file, or click to select.</p>
        </div>
        <h3>Select Models</h3>
        <div>
          {availableModels.map((model) => (
            <label key={model.name}>
              <input
                type="checkbox"
                checked={selectedModels.includes(model.name)}
                onChange={() =>
                  setSelectedModels((prev) =>
                    prev.includes(model.name)
                      ? prev.filter((m) => m !== model.name)
                      : [...prev, model.name]
                  )
                }
              />
              {model.name}
            </label>
          ))}
        </div>
        <button onClick={handleCompare} disabled={loading}>Compare</button>
        {error && <div className="error">{error}</div>}
      </div>
      <div className="results-column">
        {/* Results would be rendered here */}
      </div>
    </div>
  );
};

export default ModelComparison;
