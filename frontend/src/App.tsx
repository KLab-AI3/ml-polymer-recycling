/// <reference types="react" />
// Provide a minimal JSX global declaration to satisfy TypeScript in environments
// where the React JSX types or the new jsx-runtime types are not available.
declare global {
  namespace JSX {
    interface IntrinsicElements {
      [elemName: string]: any;
    }
  }
}

import React, { useState } from "react";
import "./static/style.css";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import StandardAnalysis from "./components/StandardAnalysis";
import ModelComparison from "./components/ModelComparison";
import PerformanceTracking from "./components/PerformanceTracking";
import ExplainabilityPanel from "./components/ExplainabilityPanel";
import { SpectrumData } from "./apiClient";

const h = React.createElement;

function App() {
  const [activeTab, setActiveTab] = useState("standard");
  const [selectedModel, setSelectedModel] = useState("figure2");
  const [modality, setModality] = useState<"raman" | "ftir">("raman");
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);

  const renderActiveTab = () => {
    switch (activeTab) {
      case "standard":
        return h(StandardAnalysis, {
          selectedModel,
          modality,
          onSpectrumChange: setSpectrumData,
        });
      case "comparison":
        return h(ModelComparison, { modality });
      case "performance":
        return h(PerformanceTracking, null);
      case "explainability":
        return h(ExplainabilityPanel, {
          spectrumData,
          selectedModel,
          modality,
          onExplainabilityResult: (result: any) =>
            console.log("Explainability result:", result),
        });
      default:
        return h(StandardAnalysis, {
          selectedModel,
          modality,
          onSpectrumChange: setSpectrumData,
        });
    }
  };

  return h(
    "div",
    { className: "app-container" },
    h(Header, null),
    h(
      "div",
      { className: "app-layout" },
      h(Sidebar, {
        selectedModel,
        setSelectedModel,
        modality,
        setModality,
      }),
      h(
        "main",
        { className: "main-content" },
        h(
          "div",
          { className: "tabs" },
          h(
            "button",
            {
              className: `tab ${activeTab === "standard" ? "active" : ""}`,
              onClick: () => setActiveTab("standard"),
            },
            "Standard Analysis"
          ),
          h(
            "button",
            {
              className: `tab ${activeTab === "comparison" ? "active" : ""}`,
              onClick: () => setActiveTab("comparison"),
            },
            "Model Comparison"
          ),
          h(
            "button",
            {
              className: `tab ${activeTab === "explainability" ? "active" : ""}`,
              onClick: () => setActiveTab("explainability"),
            },
            "AI Explainability"
          ),
          h(
            "button",
            {
              className: `tab ${activeTab === "performance" ? "active" : ""}`,
              onClick: () => setActiveTab("performance"),
            },
            "Performance Tracking"
          )
        ),
        h("div", { className: "tab-content" }, renderActiveTab())
      )
    )
  );
}

export default App;
