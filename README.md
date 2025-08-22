
# 🔬 AI-Driven Polymer Aging Prediction and Classification System

A research project developed as part of AIRE 2025. This system applies deep learning to spectral data to classify polymer aging a critical proxy for recyclability using a fully reproducible and modular ML pipeline.

The broader research vision is a multi-modal evaluation platform, benchmarking not only Raman spectra but also image-based models and FTIR spectral data, ensuring reproducibility, extensibility, and scientific rigor.

---

## 🎯 Project Objective

- Build a validated machine learning system for classifying polymer spectra (predict degradation levels as a proxy for recyclability)
- Evaluate and compare multiple CNN architectures, beginning with Figure2CNN and ResNet variants, and expand to additional trained models.
- Ensure scientific reproducibility through structured diaignostics and artifact control
- Support sustainability and circular materials research through spectrum-based classification.

  **Reference (for Figure2CNN baseline):**

  > Neo, E.R.K., Low, J.S.C., Goodship, V., Debattista, K. (2023).
  > Deep learning for chemometric analysis of plastic spectral data from infrared and Raman databases.
  > Resources, Conservation & Recycling, 188, 106718.
  > https://doi.org/10.1016/j.resconrec.2022.106718
---

## 🧠 Model Architectures

| Model| Description |
|------|-------------|
| `Figure2CNN`  | Baseline model from literature |
| `ResNet1D`    | Deeper candidate model with skip connections |
| `ResNet18Vision` | Image-focused CNN architecture, retrained on polymer dataset (roadmap) |

  Future expansions will add additional trained CNNs, supporting direct benchmarking and comparative reporting.

---

## 📁 Project Structure (Cleaned and Current)

```text
ml-polymer-recycling/
├── datasets/
├── models/           # Model architectures
├── scripts/          # Training, inference, utilities
├── outputs/          # Artifacts: models, logs, plots
├── docs/             # Documentation & reports
└── environment.yml   # (local) Conda execution environment
```

![ml-polymer-gitdiagram-0](https://github.com/user-attachments/assets/bb5d93dc-7ab9-4259-8513-fb680ae59d64)

---

## ✅ Current Status

| Track     | Status               | Test Accuracy |
|-----------|----------------------|----------------|
| **Raman** | ✅ Active & validated  | **87.81% ± 7.59%** |
| **Image**  | 🚧 Planned Expansion | N/A |
| **FTIR**  | ⏸️ Deferred/Modularized | N/A |

## 🔬 Key Features

- ✅ 10-Fold Stratified Cross-Validation
- ✅ CLI Training: `train_model.py`
- ✅ CLI Inference `run_inference.py`
- ✅ Output artifact naming per model
- ✅ Raman-only preprocessing with baseline correction, smoothing, normalization
- ✅ Structured diagnostics JSON (accuracies, confusion matrices)
- ✅ Canonical validation script (`validate_pipeline.sh`) confirms reproducibility of all core components

---

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

## 🎯 Strategic Expansion Objectives (Roadmap)

> The roadmap defines three major expansion paths designed to broaden the system’s capabilities and impact:

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

