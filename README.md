# AI-Driven Polymer Aging Prediction and Classification System

This system applies deep learning to spectral data to classify polymer aging a critical proxy for recyclability using a fully reproducible and modular ML pipeline.

The broader research vision is a multi-modal evaluation platform, benchmarking not only Raman spectra but also image-based models and FTIR spectral data, ensuring reproducibility, extensibility, and scientific rigor.

## Model Architectures

| Model            | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `Figure2CNN`     | Baseline model from literature                                         |
| `ResNet1D`       | Deeper candidate model with skip connections                           |
| `ResNet18Vision` | Image-focused CNN architecture, retrained on polymer dataset (roadmap) |

Future expansions will add additional trained CNNs, supporting direct benchmarking and comparative reporting.


## Project Structure (Cleaned and Current)

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

## Current Status

| Track     | Status                  | Test Accuracy      |
| --------- | ----------------------- | ------------------ |
| **Raman** | Active & validated   | **87.81% ± 7.59%** |
| **Image** | Planned Expansion    | N/A                |
| **FTIR**  | Reactivating Modules Soon | N/A                |

## Key Features

- 10-Fold Stratified Cross-Validation
- CLI Training: `train_model.py`
- CLI Inference `run_inference.py`
- Output artifact naming per model
- Raman-only preprocessing with baseline correction, smoothing, normalization
- Structured diagnostics JSON (accuracies, confusion matrices)
- Canonical validation script (`validate_pipeline.sh`) confirms reproducibility of all core components


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

## Sample Training & Inference

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

## Dataset Resources

| Type  | Dataset | Source                                                                                                                    |
| ----- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
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


## Dependencies

- `Python 3.10+`
- `Conda, Git`
- `PyTorch (CPU & CUDA)`
- `Numpy, SciPy, Pandas`
- `Scikit-learn`
- `Matplotlib, Seaborn`
- `ArgParse, JSON`

---

## Contributors

- **Dr. Sanmukh Kuppannagari**
- **Dr. Metin Karailyan**
- **Jaser H.**


  **Reference (for Figure2CNN baseline):**

  > Neo, E.R.K., Low, J.S.C., Goodship, V., Debattista, K. (2023).
  > Deep learning for chemometric analysis of plastic spectral data from infrared and Raman databases.
  > Resources, Conservation & Recycling, 188, 106718.
  > https://doi.org/10.1016/j.resconrec.2022.106718
