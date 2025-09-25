import React, { useEffect, useState } from "react";
import { apiClient, ModelInfo } from "../apiClient";
import "../static/style.css";

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
      <aside className="sidebar sidebar--loading" aria-label="Model Configuration">
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading models...
        </div>
      </aside>
    );
  }

  return (
    <aside className="sidebar sidebar--responsive" aria-label="Model Configuration">
      <h2 className="sidebar__title">Model Configuration</h2>

      {error && <div className="sidebar__error">{error}</div>}

      <div className="form-group">
        <label htmlFor="model-select" className="form-label">
          Select Model
        </label>
        <select
          id="model-select"
          aria-label="Select Model"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="form-select"
        >
          {models
            .filter((m) => m.available)
            .map((model) => (
              <option key={model.name} value={model.name}>
                {model.display_name || model.name}
              </option>
            ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="modality-select" className="form-label">
          Spectroscopy Modality
        </label>
        <select
          id="modality-select"
          aria-label="Spectroscopy Modality"
          value={modality}
          onChange={(e) => setModality(e.target.value as "raman" | "ftir")}
          className="form-select"
        >
          <option value="raman">Raman Spectroscopy</option>
          <option value="ftir">FTIR Spectroscopy</option>
        </select>
      </div>

      {selectedModelInfo && (
        <section className="model-info model-info--responsive" aria-label="Model Information">
          <h3 className="model-info__title">{selectedModelInfo.display_name || selectedModelInfo.name}</h3>
          <div className="model-info__meta-grid">
            <div className="meta-item" title="Classification accuracy on test set">
              <span className="meta-label">Accuracy: </span>
              <span className="meta-value">
                {((selectedModelInfo.performance?.accuracy ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="meta-item" title="F1 score (harmonic mean of precision and recall)">
              <span className="meta-label">F1 Score: </span>
              <span className="meta-value">
                {((selectedModelInfo.performance?.f1_score ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="meta-item" title="Number of trainable parameters">
              <span className="meta-label">Parameters: </span>
              <span className="meta-value">
                {selectedModelInfo.parameters || "Unknown"}
              </span>
            </div>
            <div className="meta-item" title="Average inference speed (samples/sec)">
              <span className="meta-label">Speed: </span>
              <span className="meta-value">
                {selectedModelInfo.speed || "Unknown"}
              </span>
            </div>
          </div>

          <div className="model-info__description" title="Model description">
            <h4>Description</h4>
            <p>
              {selectedModelInfo.description.length > 300
                ? (
                  <span>
                    {selectedModelInfo.description.substring(0, 300)}...
                    <span className="tooltip" aria-label={selectedModelInfo.description}>
                      <i>Full description available</i>
                    </span>
                  </span>
                )
                : selectedModelInfo.description}
            </p>
          </div>

          {selectedModelInfo.citation && (
            <div className="model-info__citation" title="Academic citation">
              <h4>Citation</h4>
              <p style={{ fontSize: "0.9rem", fontStyle: "italic", wordBreak: "break-word" }}>
                {selectedModelInfo.citation}
              </p>
              {selectedModelInfo.publication_url && (
                <a
                  href={selectedModelInfo.publication_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="model-info__publication-link"
                  aria-label="View publication"
                >
                  View Publication
                </a>
              )}
            </div>
          )}
        </section>
      )}
    </aside>
  );
};

export default Sidebar;
