import React, { useEffect, useState } from "react";
import { apiClient, ModelInfo } from "../apiClient";
import "../static/style.css"; // Ensure the main CSS file is imported

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
  const [aboutExpanded, setAboutExpanded] = useState(false);
  const [helpExpanded, setHelpExpanded] = useState(false);

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
    const currentModelIsAvailable = availableModels.some(
      (m) => m.name === selectedModel
    );

    // If there are available models but the currently selected one isn't in that list,
    // then update the selection to the first available model.
    if (availableModels.length > 0 && !currentModelIsAvailable) {
      setSelectedModel(availableModels[0].name);
    }
  }, [models, selectedModel, setSelectedModel]); // This effect correctly depends on these values.

  const selectedModelInfo = models.find((m) => m.name === selectedModel);

  return (
    <>
      <aside className="sidebar">
        {/* --- ABOUT THIS APP EXPANDER --- */}
        <div className="sidebar-expander">
          <button
            className="sidebar-expander__toggle"
            onClick={() => setAboutExpanded((prev) => !prev)}
            aria-expanded={aboutExpanded}
            aria-controls="about-app-panel"
          >
            <span>About This App</span>
            <span
              className={`sidebar-expander__icon${aboutExpanded ? " expanded" : ""}`}
            >
              {aboutExpanded ? "▲" : "▼"}
            </span>
          </button>
          {aboutExpanded && (
            <div className="sidebar-expander__panel" id="about-app-panel">
              <p>
                <span className="sidebar-title">
                  {" "}
                  <b>AI-Driven Polymer Analysis Platform</b>
                </span>{" "}
                <br />
                <br />
                <span className="sidebar-content">
                  {" "}
                  <b>Purpose:</b> Classify, analyze, and understand polymer
                  degradation using explainable AI.
                  <br />
                </span>
                <span className="sidebar-content">
                  <b>Input:</b> Raman & FTIR spectra in .txt, .csv, or .json
                  formats.
                  <br />
                  <br />
                </span>
                <span className="sidebar-content">
                  <b>Contributors:</b>
                  <br />
                  Dr. Sanmukh Kuppannagari
                  <br />
                  Dr. Metin Karailyan <br />
                  Jaser Hasan <br />
                </span>
              </p>
            </div>
          )}
        </div>
        {/* --- CONFIGURATION CARD --- */}
        <div className="card">
          <h2 className="card__title">Model Configuration</h2>
          {error && <div className="error-message">{error}</div>}
          <div className="form-group">
            <label htmlFor="model-select" className="form-label">
              Select Model
            </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-select"
              disabled={loading || models.length === 0}
              title="Choose a model for polymer analysis"
            >
              {loading && <option>Loading...</option>}
              {models
                .filter((m) => m.available)
                .map((model) => (
                  <option
                    key={model.name}
                    value={model.name}
                    title={model.description || "No description available"}
                  >
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
              value={modality}
              onChange={(e) => setModality(e.target.value as "raman" | "ftir")}
              className="form-select"
              title="Choose the spectroscopy modality for analysis"
            >
              {models
                .filter((m) => m.available)
                .map((model) => (
                  <option
                    key={model.name}
                    value={model.name}
                    title={model.description || "No description available"}
                  >
                    {model.display_name || model.name}
                  </option>
                ))}
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
                <div className="meta-item">
                  <span className="meta-item__label">Accuracy</span>
                  <span className="meta-item__value">
                    {(
                      (selectedModelInfo.performance?.accuracy ?? 0) * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-item__label">F1 Score</span>
                  <span className="meta-item__value">
                    {(
                      (selectedModelInfo.performance?.f1_score ?? 0) * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-item__label">Parameters</span>
                  <span className="meta-item__value">
                    {selectedModelInfo.parameters || "N/A"}
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-item__label">Speed</span>
                  <span className="meta-item__value">
                    {selectedModelInfo.speed || "N/A"}
                  </span>
                </div>
              </div>
              <div className="model-info__callout">
                <h4>Description</h4>
                <p>{selectedModelInfo.description}</p>
              </div>
              {selectedModelInfo.citation && (
                <div className="model-info__callout">
                  <h4>Citation</h4>
                  <p>{selectedModelInfo.citation}</p>
                  {selectedModelInfo.publication_url && (
                    <a
                      href={selectedModelInfo.publication_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="publication-link"
                    >
                      View Publication
                    </a>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="placeholder-text">
              <p>Select a model to see its details.</p>
            </div>
          )}
        </div>

        {/* --- HELP & SUPPORT EXPANDER --- */}
        {/* <div className="sidebar-expander">
          <button
            className="sidebar-expander__toggle"
            onClick={() => setHelpExpanded((prev) => !prev)}
            aria-expanded={helpExpanded}
            aria-controls="help-panel"
          >
            <span>How to Get Started & Supported Formats</span>
            <span
              className={`sidebar-expander__icon${helpExpanded ? " expanded" : ""}`}
            >
              {helpExpanded ? "▲" : "▼"}
            </span>
          </button>
          {helpExpanded && (
            <div className="sidebar-expander__panel" id="help-panel">
              <h4>How to Get Started</h4>
              <ol>
                <li>
                  <strong>Select an AI Model:</strong> Use the sidebar dropdown
                  to choose a model trained for Raman or FTIR spectra.
                </li>
                <li>
                  <strong>Choose Input Mode:</strong>
                  <ul>
                    <li>
                      <b>Upload File:</b> Analyze a single spectrum file.
                    </li>
                    <li>
                      <b>Batch Upload:</b> Process multiple spectra at once.
                    </li>
                    <li>
                      <b>Sample Data:</b> Try built-in example spectra for quick
                      testing.
                    </li>
                  </ul>
                </li>
                <li>
                  <strong>Set Modality:</strong> Select Raman or FTIR to match
                  your data type.
                </li>
                <li>
                  <strong>Run Analysis:</strong> Click <b>Run Analysis</b> to
                  classify and view results, including confidence,
                  probabilities, and model explainability.
                </li>
              </ol>
              <h4>Supported Data Format</h4>
              <ul>
                <li>
                  <b>File Types:</b> .txt, .csv, .json
                </li>
                <li>
                  <b>Required Columns:</b> Wavenumber and Intensity (two
                  columns, header optional).
                </li>
                <li>
                  <b>Separators:</b> Space or comma separated values.
                </li>
                <li>
                  <b>Preprocessing:</b> Spectra are automatically resampled to
                  500 points and baseline-corrected for model input.
                </li>
                <li>
                  <b>Examples:</b> Use <b>Sample Data</b> mode or visit{" "}
                  <a
                    href="https://openspecy.org/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Open Specy
                  </a>{" "}
                  for public datasets.
                </li>
              </ul>
              <h4>Features</h4>
              <ul>
                <li>AI-powered polymer classification and aging prediction</li>
                <li>
                  Model explainability: see which spectral regions influenced
                  predictions
                </li>
                <li>Performance metrics and quality control checks</li>
                <li>Supports both Raman and FTIR modalities</li>
                <li>Batch analysis for multiple spectra</li>
              </ul>
            </div>
          )}
        </div> */}
      </aside>
    </>
  );
};

export default Sidebar;
