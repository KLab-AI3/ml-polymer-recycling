import React, { useEffect, useState } from "react";
import { apiClient, ModelInfo } from "../apiClient";
import "../App.css"; // Ensure the main CSS file is imported

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

  // --- Effect 1: Fetch models only once on component mount ---
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        setError(null);
        const modelList = await apiClient.getModels();
        setModels(modelList);
      } catch (err) {
        console.error("Failed to fetch models:", err);
        setError("Failed to load models.");
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []); // Empty dependency array means this runs only ONCE.

  // --- Effect 2: Set a default model after models have loaded ---
  useEffect(() => {
    // Don't do anything until models are loaded
    if (models.length === 0) return;

    const availableModels = models.filter((m) => m.available);
    const currentModelIsAvailable = availableModels.some((m) => m.name === selectedModel);

    // If there are available models but the currently selected one isn't in that list,
    // then update the selection to the first available model.
    if (availableModels.length > 0 && !currentModelIsAvailable) {
      setSelectedModel(availableModels[0].name);
    }
  }, [models, selectedModel, setSelectedModel]); // This effect correctly depends on these values.


  const selectedModelInfo = models.find((m) => m.name === selectedModel);

  return (
    <aside className="sidebar">
      {/* --- CONFIGURATION CARD --- */}
      <div className="card">
        <h2 className="card__title">Model Configuration</h2>
        {error && <div className="error-message">{error}</div>}
        <div className="form-group">
          <label htmlFor="model-select" className="form-label">Select Model</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="form-select"
            disabled={loading || models.length === 0}
          >
            {loading && <option>Loading...</option>}
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
          <label htmlFor="modality-select" className="form-label">Spectroscopy Modality</label>
          <select
            id="modality-select"
            value={modality}
            onChange={(e) => setModality(e.target.value as "raman" | "ftir")}
            className="form-select"
          >
            <option value="raman">Raman Spectroscopy</option>
            <option value="ftir">FTIR Spectroscopy</option>
          </select>
        </div>
      </div>

      {/* --- MODEL INFORMATION CARD --- */}
      <div className="card">
        <h2 className="card__title">Model Information</h2>
        {loading ? (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Loading model details...</span>
          </div>
        ) : selectedModelInfo ? (
          <div>
            <div className="model-info__meta-grid">
              <div className="meta-item"><span className="meta-item__label">Accuracy</span><span className="meta-item__value">{((selectedModelInfo.performance?.accuracy ?? 0) * 100).toFixed(1)}%</span></div>
              <div className="meta-item"><span className="meta-item__label">F1 Score</span><span className="meta-item__value">{((selectedModelInfo.performance?.f1_score ?? 0) * 100).toFixed(1)}%</span></div>
              <div className="meta-item"><span className="meta-item__label">Parameters</span><span className="meta-item__value">{selectedModelInfo.parameters || "N/A"}</span></div>
              <div className="meta-item"><span className="meta-item__label">Speed</span><span className="meta-item__value">{selectedModelInfo.speed || "N/A"}</span></div>
            </div>
            <div className="model-info__callout">
              <h4>Description</h4>
              <p>{selectedModelInfo.description}</p>
            </div>
            {selectedModelInfo.citation && (
              <div className="model-info__callout">
                <h4>Citation</h4>
                <p>{selectedModelInfo.citation}</p>
                {selectedModelInfo.publication_url && (<a href={selectedModelInfo.publication_url} target="_blank" rel="noopener noreferrer" className="publication-link">View Publication</a>)}
              </div>
            )}
          </div>
        ) : (
          <div className="placeholder-text">
            <p>Select a model to see its details.</p>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
