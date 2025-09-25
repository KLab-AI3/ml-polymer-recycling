import React, { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import StandardAnalysis from "./components/StandardAnalysis";
import ModelComparison from "./components/ModelComparison";
import PerformanceTracking from "./components/PerformanceTracking";
import ExplainabilityPanel from "./components/ExplainabilityPanel";
import { SpectrumData } from "./apiClient";

function App() {
  const [activeTab, setActiveTab] = useState("standard");
  const [selectedModel, setSelectedModel] = useState("figure2");
  const [modality, setModality] = useState<"raman" | "ftir">("raman");
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);

  const renderActiveTab = () => {
    switch (activeTab) {
      case "standard":
        return (
          <StandardAnalysis
            selectedModel={selectedModel}
            modality={modality}
            onSpectrumChange={setSpectrumData}
          />
        );
      case "comparison":
        return <ModelComparison modality={modality} />;
      case "performance":
        return <PerformanceTracking />;
      case "explainability":
        return (
          <ExplainabilityPanel
            spectrumData={spectrumData}
            selectedModel={selectedModel}
            modality={modality}
            onExplainabilityResult={(result: any) =>
              console.log("Explainability result:", result)
            }
          />
        );
      default:
        return (
          <StandardAnalysis
            selectedModel={selectedModel}
            modality={modality}
            onSpectrumChange={setSpectrumData}
          />
        );
    }
  };

  return (
    <div className="app-container">
      <Header />
      <div className="app-layout">
        <Sidebar
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          modality={modality}
          setModality={setModality}
        />
        <main className="main-content">
          <div className="tabs">
            <button
              className={`tab ${activeTab === "standard" ? "active" : ""}`}
              onClick={() => setActiveTab("standard")}
            >
              Standard Analysis
            </button>
            <button
              className={`tab ${activeTab === "comparison" ? "active" : ""}`}
              onClick={() => setActiveTab("comparison")}
            >
              Model Comparison
            </button>
            <button
              className={`tab ${activeTab === "explainability" ? "active" : ""}`}
              onClick={() => setActiveTab("explainability")}
            >
              AI Explainability
            </button>
            <button
              className={`tab ${activeTab === "performance" ? "active" : ""}`}
              onClick={() => setActiveTab("performance")}
            >
              Performance Tracking
            </button>
          </div>
          <div className="tab-content">{renderActiveTab()}</div>
        </main>
      </div>
    </div>
  );
}

export default App;
