
# 🔬 AI-Driven Polymer Aging Prediction and Classification System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research project developed as part of AIRE 2025. This system applies deep learning to Raman spectral data to classify polymer aging — a critical proxy for recyclability — using a fully reproducible and modular ML pipeline.

---

## 🎯 Project Objective

- Build a validated machine learning system for classifying polymer spectra (predict degradation levels as a proxy for recyclability)
- Compare literature-based and modern CNN architectures (Figure2CNN vs. ResNet1D) on Raman spectral data
- Ensure scientific reproducibility through structured diaignostics and artifact control
- Support sustainability and circular materials research through spectrum-based classification.

---

## 🧠 Model Architectures

| Model| Description |
|------|-------------|
| `Figure2CNN`  | Baseline model from literature |
| `ResNet1D`    | Deeper candidate model with skip connections |

> Both models support flexible input lengths; Figure2CNN relies on reshape logic, while ResNet1D uses native global pooling.

---

## 📁 Project Structure (Cleaned and Current)

```text
polymer_project/
├── datasets/rdwp     # Raman spectra  
├── models/           # Model architectures
├── scripts/          # Training, inference, utilities
├── outputs/          # Artifacts: models, logs, plots
├── docs/             # Documentation & reports
└── environment.yml   # (local) Conda execution environment
```

---

## ✅ Current Status

| Track     | Status               | Test Accuracy |
|-----------|----------------------|----------------|
| **Raman** | ✅ Active & validated  | **87.81% ± 7.59%** |
| **FTIR**  | ⏸️ Deferred (modeling only) | N/A |

**Note:** FTIR preprocessing scripts are preserved but inactive. Modeling work is deferred until a suitable architecture is identified.

**Artifacts:**

- `outputs/figure2_model.pth`
- `outputs/resnet_model.pth`
- `outputs/logs/raman_{model}_diagnostics.json`

---

## 🔬 Key Features

- ✅ 10-Fold Stratified Cross-Validation
- ✅ CLI Training: `train_model.py`
- ✅ CLI Inference `run_inference.py`
- ✅ Output artifact naming per model
- ✅ Raman-only preprocessing with baseline correction, smoothing, normalization
- ✅ Structured diagnostics JSON (accuracies, confusion matrices)
- ✅ Canonical validation script (`validate_pipeline.sh`) confirms reproducibility of all core components

---

## 🔀 Branching Strategy

| Branch | Purpose|
|--------|--------|
| `main` | Local development (CPU) |
| `hpc_main` | Cluster-ready (HPC; GPU) |

**Environments:**

```bash

# Local
git checkout main
conda env create -f environment.yml
conda activate polymer_env

# HPC
git checkout hpc-main
conda env create -f environment_hpc.yml
conda activate polymer_env
```

## 📊 Sample Training & Inference

### Training (10-Fold CV)

```bash

python scripts/train_model.py --model resnet --target-len 4000 --baseline --smooth --normalize
```

### Inference (Raman)

```bash

python scripts/run_inference.py --target-len 4000 
--input datasets/rdwp/sample123.txt --model outputs/resnet_model.pth 
--output outputs/inference/prediction.txt
```

### Inference Output Example:

```bash
Predicted Label: 1 True Label: 1
Raw Logits: [[-569.544, 427.996]]
```

### Validation Script (Raman Pipeline)

```bash
./validate_pipeline.sh
# Runs preprocessing, training, inference, and plotting checks
# Confirms artifact integrity and logs test results
```

---

## 📚 Dataset Resources

| Type  | Dataset | Source |
|-------|---------|--------|
| Raman | RDWP    | [A Raman database of microplastics weathered under natural environments](https://data.mendeley.com/datasets/kpygrf9fg6/1) |

| Datasets should be downloaded separately and placed here:

```bash
datasets/
└── rdwp/
  ├── sample1.txt
  ├── sample2.txt
  └── ...
```

These files are intentionally excluded from version control via `.gitignore`

---

## 🛠 Dependencies

- `Python 3.10+`
- `Conda, Git`
- `PyTorch (CPU & CUDA)`
- `Numpy, SciPy, Pandas`
- `Scikit-learn`
- `Matplotlib, Seaborn`
- `ArgParse, JSON`

---

## 🧑‍🤝‍🧑 Contributors

- **Dr. Sanmukh Kuppannagari** — Research Mentor
- **Dr. Metin Karailyan** — Research Mentor
- **Jaser H.** — AIRE 2025 Intern, Developer  

---

## 🎯 Strategic Expansion Objectives

> Following Dr. Kuppannagari’s updated guidance, the project scope now extends beyond the Raman-only validated baseline. The roadmap defines three major expansion paths designed to broaden the system’s capabilities and impact:

1. **Model Expansion: Multi-Model Dashboard**

    > The dashboard will evolve into a hub for multiple model architectures rather than being tied to a single baseline. Planned work includes:

   - **Retraining & Fine-Tuning**: Incorporating publicly available vision models and retraining them with the polymer dataset.
   - **Model Registry**: Automatically detecting available .pth weights and exposing them in the dashboard for easy selection.
   - **Side-by-Side Reporting**: Running comparative experiments and reporting each model’s accuracy and diagnostics in a standardized format.
   - **Reproducible Integration**: Maintaining modular scripts and pipelines so each model’s results can be replicated without conflict.

   This ensures flexibility for future research and transparency in performance comparisons.

2. **Image Input Modality**

    > The system will support classification on images as an additional modality, extending beyond spectra. Key features will include:

   - **Upload Support**: Users can upload single images or batches directly through the dashboard.
   - **Multi-Model Execution**: Selected models from the registry can be applied to all uploaded images simultaneously.
   - **Batch Results**: Output will be returned in a structured, accessible way, showing both individual predictions and aggregate statistics.
   - **Enhanced Feedback**: Outputs will include predicted class, model confidence, and potentially annotated image previews.

   This expands the system toward a multi-modal framework, supporting broader research workflows.

3. **FTIR Dataset Integration**

    > Although previously deferred, FTIR support will be added back in a modular, distinct fashion. Planned steps are:

    - **Dedicated Preprocessing**: Tailored scripts to handle FTIR-specific signal characteristics (multi-layer handling, baseline correction, normalization).
    - **Architecture Compatibility**: Ensuring existing and retrained models can process FTIR data without mixing it with Raman workflows.
    - **UI Integration**: Introducing FTIR as a separate option in the modality selector, keeping Raman, Image, and FTIR workflows clearly delineated.
    - **Phased Development**: Implementation details to be refined during meetings to ensure scientific rigor.

    This guarantees FTIR becomes a supported modality without undermining the validated Raman foundation.

## 🔑 Guiding Principles

- **Preserve the Raman baseline** as the reproducible ground truth
- **Additive modularity**: Models, images, and FTIR added as clean, distinct layers rather than overwriting core functionality
- **Transparency & reproducibility**: All expansions documented, tested, and logged with clear outputs.
- **Future-oriented design**: Workflows structured to support ongoing collaboration and successor-safe research.
