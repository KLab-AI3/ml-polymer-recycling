# 📅 PROJECT_TIMELINE.md

## AI-Driven Polymer Aging Prediction and Classification System

**Intern:** Jaser Hasan

### ✅ PHASE 1 – Project Kickoff and Faculty Guidance

**Tag:** `@project-init-complete`

Received first set of research tasks from Prof. Kuppannagari

- Reeived research plan
- Objectives defined: download datasets, analyze spectra, implement CNN, run initial inference

---

### ✅ PHASE 2 – Dataset Acquisition (Local System)

**Tag:** `@data-downloaded`

- Downloaded Raman `.txt`  (RDWP) and FTIR `.csv` data (polymer packaging)
- Structured into:
- `datasets/rdwp`
- `datasets/ftir`

---

### ✅ PHASE 3 – Data Exploration & Spectral Validation

**Tag:** `@data-exploration-complete`

- Built plotting tools for Raman and FTIR
- Validated spectrum structure, removed malformed samples
- Observed structural inconsistencies in FTIR multi-layer grouping

---

### ✅ PHASE 4 – Preprocessing Pipeline Implementation

**Tag:** `@data-prep`

- Implemented `preprocess_dataset.py` for Raman
- Applied: Resampling -> Baseline correction -> Smoothing -> Normalization
- Confirmed reproducible input/output behavior and dynamic CLI control

### ✅ PHASE 5 – Figure2CNN Architecture Build

**Tag:** `@figure2cnn-complete`

- Constructed `Figure2CNN` modeled after Figure 2 CNN from research paper
- `Figure2CNN`: 4 conv layers + 3 FC layers
- Verified dynamic input length handling (e.g., 500, 1000, 4000)

---

### ✅ PHASE 6 – Local Training and Inference

**Tag:** `@figure2cnn-training-local`

- Trained Raman models locally (FTIR now deferred)
- Canonical Raman accuracy: **87.29% ± 6.30%**
- FTIR accuracy results archived and excluded from current validation
- CLI tools for training, inference, plotting implemented

---

### ✅ PHASE 7 –  Reproducibility and Documentation Setup

**Tag:** `@project-docs-started`

- Authored `README.md`, `PROJECT_REPORT.md`, and `ENVIRONMENT_GUIDE.md`
- Defined reproducibility guidelines
- Standardized project structure and versioning

---

### ✅ PHASE 8 – HPC Access and Venv Strategy

**Tag:** `@hpc-login-successful`

- Logged into CWRU Pioneer (SSH via PuTTY)
- Setup up FortiClient VPN as it is required to access Pioneer remotely
- Explored module system; selected venv over Conda for compatibility
- Loaded Python 3.12.3 + created `polymer_env`

---

### ✅ PHASE 9 – HPC Environment Sync

**Tag:** `@venv-alignment-complete`

- Created `environment_hpc.yml`
- Installed dependencies into `polymer_env`
- Validated imports, PyTorch installation, and CLI script execution

---

### ✅ PHASE 10 – Full Instruction Validation on HPC

**Tag:** `@prof-k-instruction-validation-complete`

- Ran Raman preprocessing and plotting scripts
- Executed `run_inference.py` with CLI on raw Raman `.txt` file
- Verified consistent predictions and output logging across local and HPC

---

### ✅ PHASE 11 – FTIR Path Paused, Raman Declared Primary

**Tag:** `@raman-pipeline-focus-milestone`

- FTIR modeling formally deferred
- FTIR preprocessing scripts preserved and archived for future use
- All resources directed toward Raman pipeline finalization
- Saliency, FTIR ingestion, and `train_ftir_model.py` archived

---

### ✅ PHASE 12 – ResNet1D Prototyping & Benchmark Setup

**Tag:** `@resnet-prototype-complete`

- Built `ResNet1D` architecture in `models/resnet_cnn.py`
- Integrated `train_model.py` via `--model resnet`
- Ran initial CV training with successful results

---

### ✅ PHASE 13 – Output Artifact Isolation

**Tag:** `@artifact-isolation-complete`  

- Patched `train_model.py` to save:
  - `figure2_model.pth`, `resnet_model.pth`
  - `raman_figure2_diagnostics.json`. `raman_resnet_diagnostics.json`
- Prevented all overwrites by tying output filenames to `args.model`
- Snapshotted as reproducibility milestone. Enabled downstream validation harness.

### ✅ PHASE 14 – Canonical Validation Achieved

**Tag:** `@validation-loop-complete`

- Created `validate_pipeline.sh` to verify preprocessing, training, inferece, plotting
- Ran full validation using `Figure2CNN` with reproducible CLI config
- All ouputs verified: logs, artifacts, predictions, plots
- Declared Raman pipeline scientifically validated and stable

---

### ⏭️ NEXT - Results Analysis & Finalization

- Analyze logged diagnostics for both models
- Conduct optional hyperparameter tuning (batch size, LR)
- Begin deliverable prep: visuals, posters, cards
- Resume FTIR work only after Raman path is fully stablized and documented & open FTIR conceptual error is resolved