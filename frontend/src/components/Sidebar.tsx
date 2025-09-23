import React, { useEffect, useState } from "react";
import { apiClient, ModelInfo } from "../apiClient";

interface SidebarProps {
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  modality: "raman" | "ftir";
  setModality: (modality: "raman" | "ftir") => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  selectedModel,
  setSelectedModel,
  modality,
  setModality,
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const modelList = await apiClient.getModels();
        setModels(modelList);

        // Set first available model as default if current selection isn't available
        const availableModels = modelList.filter((m) => m.available);
        if (
          availableModels.length > 0 &&
          !availableModels.find((m) => m.name === selectedModel)
        ) {
          setSelectedModel(availableModels[0].name);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load models");
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [selectedModel, setSelectedModel]);

  const selectedModelInfo = models.find((m) => m.name === selectedModel);

  if (loading) {
    return (
      <div className="sidebar">
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading models...
        </div>
      </div>
    );
  }

  return (
    <div className="sidebar">
      <h3>Model Configuration</h3>

      {error && <div className="error">{error}</div>}

      <div className="form-group">
        <label htmlFor="model-select">Select Model</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models
            .filter((m) => m.available)
            .map((model) => (
              <option key={model.name} value={model.name}>
                {model.name} - {model.description.substring(0, 50)}...
              </option>
            ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="modality-select">Spectroscopy Modality</label>
        <select
          id="modality-select"
          value={modality}
          onChange={(e) => setModality(e.target.value as "raman" | "ftir")}
        >
          <option value="raman">Raman Spectroscopy</option>
          <option value="ftir">FTIR Spectroscopy</option>
        </select>
      </div>

      {selectedModelInfo && (
        <div className="model-info">
          <h3>Model Information</h3>
          <div className="metadata-grid">
            <div className="metadata-item">
              <h4>Accuracy</h4>
              <p>
                {((selectedModelInfo.performance?.accuracy ?? 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="metadata-item">
              <h4>F1 Score</h4>
              <p>
                {((selectedModelInfo.performance?.f1_score ?? 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="metadata-item">
              <h4>Parameters</h4>
              <p>{selectedModelInfo.parameters || "Unknown"}</p>
            </div>
            <div className="metadata-item">
              <h4>Speed</h4>
              <p>{selectedModelInfo.speed || "Unknown"}</p>
            </div>
          </div>

          <div className="model-description">
            <h4>Description</h4>
            <p>{selectedModelInfo.description}</p>
          </div>

          {selectedModelInfo.citation && (
            <div className="model-citation">
              <h4>Citation</h4>
              <p style={{ fontSize: "0.8rem", fontStyle: "italic" }}>
                {selectedModelInfo.citation}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Sidebar;
