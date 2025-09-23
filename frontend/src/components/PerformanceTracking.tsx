import React, { useEffect, useState } from "react";
import { apiClient, SystemInfo } from "../apiClient";

const PerformanceTracking: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        setLoading(true);
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
      <div className="loading">
        <div className="loading-spinner"></div>
        Loading performance data...
      </div>
    );
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="performance-panel">
      <h2>📈 Performance Tracking</h2>
      <p>System health and model performance metrics</p>

      {systemInfo && (
        <>
          {/* System Health */}
          <div className="results-section">
            <h3>System Health</h3>
            <div className="metadata-grid">
              <div className="metadata-item">
                <h4>API Version</h4>
                <p>{systemInfo.version}</p>
              </div>
              <div className="metadata-item">
                <h4>Models Loaded</h4>
                <p>
                  {systemInfo.system_health.models_loaded} /{" "}
                  {systemInfo.system_health.total_models}
                </p>
              </div>
              <div className="metadata-item">
                <h4>Memory Usage</h4>
                <p>{systemInfo.system_health.memory_usage_mb.toFixed(1)} MB</p>
              </div>
              <div className="metadata-item">
                <h4>PyTorch Version</h4>
                <p>{systemInfo.system_health.torch_version}</p>
              </div>
              <div className="metadata-item">
                <h4>CUDA Available</h4>
                <p>
                  {systemInfo.system_health.cuda_available ? "✅ Yes" : "❌ No"}
                </p>
              </div>
              <div className="metadata-item">
                <h4>Max Batch Size</h4>
                <p>{systemInfo.max_batch_size}</p>
              </div>
            </div>
          </div>

          {/* Model Performance */}
          <div className="results-section">
            <h3>Model Performance Summary</h3>
            <div className="model-performance-grid">
              {(systemInfo.available_models || []).map((model) => (
                <div key={model.name} className="model-performance-card">
                  <h4>{model.name}</h4>
                  <div className="performance-metrics">
                    <div className="metric">
                      <span className="metric-label">Accuracy</span>
                      <span className="metric-value">
                        {((model.performance?.accuracy ?? 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">F1 Score</span>
                      <span className="metric-value">
                        {((model.performance?.f1_score ?? 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Parameters</span>
                      <span className="metric-value">
                        {model.parameters || "Unknown"}
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Speed</span>
                      <span className="metric-value">
                        {model.speed || "Unknown"}
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Status</span>
                      <span
                        className={`metric-value ${
                          model.available ? "available" : "unavailable"
                        }`}
                      >
                        {model.available ? "✅ Available" : "❌ Unavailable"}
                      </span>
                    </div>
                  </div>
                  <div className="model-description">
                    <p>{model.description}</p>
                  </div>
                  {model.citation && (
                    <div className="model-citation">
                      <p style={{ fontSize: "0.8rem", fontStyle: "italic" }}>
                        {model.citation}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Supported Features */}
          <div className="results-section">
            <h3>Supported Features</h3>
            <div className="metadata-grid">
              <div className="metadata-item">
                <h4>Spectroscopy Modalities</h4>
                <p>{systemInfo.supported_modalities?.join(", ") ?? "N/A"}</p>
              </div>
              <div className="metadata-item">
                <h4>Target Spectrum Length</h4>
                <p>{systemInfo.target_spectrum_length} points</p>
              </div>
              <div className="metadata-item">
                <h4>File Formats</h4>
                <p>.txt, .csv, .json</p>
              </div>
              <div className="metadata-item">
                <h4>Analysis Types</h4>
                <p>Single, Batch, Comparison</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default PerformanceTracking;
