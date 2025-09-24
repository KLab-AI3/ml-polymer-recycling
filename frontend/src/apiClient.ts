import { components, operations } from "./types/api";

// Type aliases for better readability
export type SpectrumData = components["schemas"]["SpectrumData"];
export type PredictionResult = components["schemas"]["PredictionResult"];
export type BatchPredictionResult =
  components["schemas"]["BatchPredictionResult"];
export type ComparisonResult = components["schemas"]["ComparisonResult"];
export type ModelInfo = components["schemas"]["ModelInfo"];
export type SystemInfo = components["schemas"]["SystemInfo"];
export type AnalysisRequest = components["schemas"]["AnalysisRequest"];
export type BatchAnalysisRequest =
  components["schemas"]["BatchAnalysisRequest"];
export type ComparisonRequest = components["schemas"]["ComparisonRequest"];

// API Response types
type HealthResponse = { status: string; timestamp: string };

export class ApiClient {
  private getBaseUrl(): string {
    // In local dev, frontend served at localhost:3000, backend at localhost:8000.
    // In deployed HF Spaces, use relative paths (same origin).
    if (
      typeof window !== "undefined" &&
      window.location.hostname === "localhost"
    ) {
      return "http://localhost:8000";
    }
    // Default: relative path so requests go to the same origin (HF Space)
    return "";
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const base = this.getBaseUrl();
    const url = endpoint.startsWith("/")
      ? `${base}${endpoint}`
      : `${base}/${endpoint}`;
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    return (await res.json()) as T;
  }
  // Health endpoints
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/api/v1/health");
  }

  // System information
  async getSystemInfo(): Promise<SystemInfo> {
    return this.request<SystemInfo>("/api/v1/system");
  }

  // Model management
  async getModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>("/api/v1/models");
  }

  // Spectrum analysis
  async analyzeSpectrum(request: AnalysisRequest): Promise<PredictionResult> {
    return this.request<PredictionResult>("/api/v1/analyze", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async analyzeBatch(
    request: BatchAnalysisRequest
  ): Promise<BatchPredictionResult> {
    return this.request<BatchPredictionResult>("/api/v1/analyze/batch", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async compareModels(request: ComparisonRequest): Promise<ComparisonResult> {
    return this.request<ComparisonResult>("/api/v1/compare", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // Explainability
  async explainSpectrum(request: AnalysisRequest): Promise<any> {
    return this.request<any>("/api/v1/explain", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async explainBatch(request: BatchAnalysisRequest): Promise<any> {
    return this.request<any>("/api/v1/explain/batch", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // File upload
  async uploadSpectrum(file: File): Promise<SpectrumData> {
    const formData = new FormData();
    formData.append("file", file);

    return this.request<SpectrumData>("/api/v1/upload", {
      method: "POST",
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it for FormData
    });
  }
}

// Create and export a default instance
export const apiClient = new ApiClient();

export interface FeatureImportance {
  method: string;
  top_features: {
    indices: number[];
    values: number[];
  };
  summary: {
    max_importance: number;
    mean_importance: number;
    important_region_start: number;
    important_region_end: number;
  };
  importance_scores: number[]; // <-- Add this line
}

export interface ExplanationResult {
  prediction: number; // The predicted class index (e.g., 0 or 1)
  confidence: number;
  probabilities: number[]; // Array of probabilities for each class
  class_labels: { [key: number]: string }; // Maps class index to a string name (e.g., {0: 'stable', 1: 'weathered'})
  feature_importance: FeatureImportance;
  model_used: string;
  spectrum_filename: string;
}

// Export individual client methods as hooks-ready functions
export const useApiClient = () => {
  return {
    health: () => apiClient.health(),
    getSystemInfo: () => apiClient.getSystemInfo(),
    getModels: () => apiClient.getModels(),
    analyzeSpectrum: (request: AnalysisRequest) =>
      apiClient.analyzeSpectrum(request),
    analyzeBatch: (request: BatchAnalysisRequest) =>
      apiClient.analyzeBatch(request),
    compareModels: (request: ComparisonRequest) =>
      apiClient.compareModels(request),
    explainSpectrum: (request: AnalysisRequest) =>
      apiClient.explainSpectrum(request),
    explainBatch: (request: BatchAnalysisRequest) =>
      apiClient.explainBatch(request),
    uploadSpectrum: (file: File) => apiClient.uploadSpectrum(file),
  };
};
