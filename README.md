---
title: "PolymerOS: Predictive Framework for Polymer Aging"
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# PolymerOS: A Computational Framework for Degradation-Aware Plastic Classification

![React](https://img.shields.io/badge/React-18.2-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)
![OCI](https://img.shields.io/badge/container-ready-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

[**Live Interactive Dashboard**](https://huggingface.co/spaces/dev-jas/polymer-aging-with-ml) | [**Official Repository**](https://github.com/KLab-AI3/ml-polymer-recycling)

---

## Overview

**PolymerOS** is the official computational framework for the predictive aging of plastics as described in the manuscript *"Predictive Framework to Indicate the Age of Plastics for Proper Recycling."*

Conventional mechanical recycling often overlooks the degradation history of materials, leading to inconsistent product quality. This framework utilizes **deep learning applied to Raman and FTIR spectroscopy** to identify early-stage chemical and physical aging signatures. It provides a standardized, secure, and reproducible environment for the scientific classification of aged versus unaged polymers.

---

## Core Research Artifacts

### 1. Model Zoo
Verified architectures and weights for the following models are provided:
- **Figure2CNN**: High-performance binary classifier (Aged vs. Unaged) optimized for spectral data.
- **ResNet1D**: Benchmarked 1D-convolutional architecture for spectral feature extraction.
- **Preprocessing Pipeline**: A standardized 4-step sequence including asymmetric least-squares baseline correction, Savitzky–Golay smoothing, min-max normalization, and resampling to 4000 spectral points.

### 2. Standalone Scientific Appliance
To facilitate reproducibility and practical use by researchers, the entire pipeline—including the interactive dashboard, inference engine, and preprocessing logic—is delivered as a **portable OCI container**.

---

## Technical Architecture

```text
ml-polymer-recycling/
├── backend/
│   ├── main.py               # API entrypoint and static asset server
│   ├── models/
│   │   └── weights/          # Model binaries (.pth managed via Git LFS)
│   ├── utils/
│   │   ├── model_manager.py  # Hardened PyTorch 2.6 safe-loading logic
│   │   └── preprocessing.py  # Standardized 4-step spectral preprocessing
│   └── service.py            # Core inference orchestration
├── frontend/
│   ├── src/                  # React/TypeScript source code
│   │   └── apiClient.ts      # Location-agnostic API Client
│   └── dist/                 # Compiled production assets
├── Dockerfile                # Multi-stage hardened OCI build configuration
├── requirements.txt          # Python environment specifications
└── .gitattributes            # Git LFS tracking for model weights
```

---

## Reproducibility & Local Operation

### 1. Prerequisites
- **Git LFS** (Required to download model weight binaries).
- **Docker** or **Podman**.

### 2. Setup
```bash
# Clone the official repository
git clone https://github.com/KLab-AI3/ml-polymer-recycling.git
cd ml-polymer-recycling

# Initialize LFS and pull model weights
git lfs install
git lfs pull
```

### 3. Running the Dashboard
To ensure bit-perfect scientific parity with the benchmarks reported in the manuscript, we recommend running the framework as a standalone appliance. The container is internally hardened to run in a restricted, read-only state.

```bash
# Build the appliance
docker build -t polymer-os .

# Launch the dashboard
docker run -p 7860:7860 polymer-os
```
The interactive interface will be available at: **http://localhost:7860**

---

## Security & Agnosticism

- **PyTorch 2.6 Enforcement**: Models are loaded using the hardened `weights_only=True` standard to ensure safe execution in public environments.
- **Environment Aware**: The "Agnostic" API client automatically detects host and port settings, ensuring seamless transitions between local workstations and cloud providers like Hugging Face Spaces.

---

## Contributors

- **Dhoopshikha Lakshmi Devi Basgeet** — Lead Author
- **Jaser Hasan** — Lead Developer / Technical Audit
- **Konpal Raheja**
- **Divita Mathur**
- **Dr. Sanmukh Kuppannagari** — Corresponding Author
- **Dr. Metin Karayilan** — Corresponding Author

---

## License

Licensed under the Apache License, Version 2.0 (the "License"); see [LICENSE](LICENSE) for details.
