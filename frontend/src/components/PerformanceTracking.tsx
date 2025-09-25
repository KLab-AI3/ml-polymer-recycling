import React, { useEffect, useState } from "react";
import { apiClient, SystemInfo } from "../apiClient";
import "../App.css"; // Ensure the main CSS file is imported

const PerformanceTracking: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        setLoading(true);
        setError(null);
        const info = await apiClient.getSystemInfo();
        setSystemInfo(info);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load system info"
        );
      } finally {
        setLoading(false);
      }
    };

    fetchSystemInfo();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="loading-indicator">
          <div className="spinner"></div>
          <span>Loading Performance Data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  if (!systemInfo) {
    return (
      <div className="placeholder">
        <p>No system information could be loaded.</p>
      </div>
    );
  }

  return (
    <div className="performance-panel">
      {/* --- System Health Card --- */}
      <div className="card">
        <h2 className="card__title">System Health</h2>
        <div className="info-grid">
          <div className="info-item">
            <label>API Version</label>
            <span>{systemInfo.version}</span>
          </div>
          <div className="info-item">
            <label>Models Loaded</label>
            <span>
              {systemInfo.system_health.models_loaded} /{" "}
              {systemInfo.system_health.total_models}
            </span>
          </div>
          <div className="info-item">
            <label>Memory Usage</label>
            <span>
              {systemInfo.system_health.memory_usage_mb.toFixed(1)} MB
            </span>
          </div>
          <div className="info-item">
            <label>PyTorch Version</label>
            <span>{systemInfo.system_health.torch_version}</span>
          </div>
          <div className="info-item">
            <label>CUDA Available</label>
            <span
              className={
                systemInfo.system_health.cuda_available
                  ? "status-ok"
                  : "status-bad"
              }
            >
              {systemInfo.system_health.cuda_available ? "Yes" : "No"}
            </span>
          </div>
          <div className="info-item">
            <label>Max Batch Size</label>
            <span>{systemInfo.max_batch_size}</span>
          </div>
        </div>
      </div>

      {/* --- Model Performance Card --- */}
      <div className="card">
        <h2 className="card__title">Available Models</h2>
        <div className="model-grid">
          {(systemInfo.available_models || []).map((model) => (
            <div key={model.name} className="model-card">
              <h4 className="model-card__title">
                {model.display_name || model.name}
              </h4>
              <div className="model-card__metrics">
                <div>
                  <span>Accuracy</span>
                  <span>
                    {((model.performance?.accuracy ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span>F1 Score</span>
                  <span>
                    {((model.performance?.f1_score ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span>Params</span>
                  <span>{model.parameters || "N/A"}</span>
                </div>
                <div>
                  <span>Status</span>
                  <span
                    className={model.available ? "status-ok" : "status-bad"}
                  >
                    {model.available ? "Available" : "Unavailable"}
                  </span>
                </div>
              </div>
              <div className="model-info__callout">
                <p>{model.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* --- Supported Features Card --- */}
      <div className="card">
        <h2 className="card__title">Supported Features</h2>
        <div className="info-grid">
          <div className="info-item">
            <label>Modalities</label>
            <span>{systemInfo.supported_modalities?.join(", ") ?? "N/A"}</span>
          </div>
          <div className="info-item">
            <label>Target Length</label>
            <span>{systemInfo.target_spectrum_length} points</span>
          </div>
          <div className="info-item">
            <label>File Formats</label>
            <span>.txt, .csv, .json</span>
          </div>
          <div className="info-item">
            <label>Analysis Types</label>
            <span>Single, Batch, Comparison</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceTracking;
