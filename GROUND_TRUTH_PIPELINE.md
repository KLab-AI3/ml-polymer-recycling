# 🧾 GROUND_TRUTH_PIPELINE.md

**AI-Driven Polymer Aging Prediction & Classification System**  
*Empirical Baseline Inventory - CLI and UI Workflows*

> **Purpose**  
> This document provides a comprehensive, empirical inventory of the complete ML pipeline as it exists today. It serves as the definitive reference baseline for all future development, ensuring reproducibility and preventing drift between CLI and UI components.

## 📋 Table of Contents

1. [Data Handling](#1-data-handling)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)  
3. [Models and Weights](#3-models-and-weights)
4. [CLI Workflow](#4-cli-workflow)
5. [Streamlit Dashboard Workflow](#5-streamlit-dashboard-workflow)
6. [Empirical Outputs and Known Differences](#6-empirical-outputs-and-known-differences)
7. [Configuration Constants](#7-configuration-constants)
8. [Dependencies and Environment](#8-dependencies-and-environment)
9. [Validation and Testing](#9-validation-and-testing)
10. [Critical Gaps and Inconsistencies](#10-critical-gaps-and-inconsistencies)
11. [Future Development Guidelines](#11-future-development-guidelines)
12. [Artifact Inventory](#12-artifact-inventory)

---

## 📊 Pipeline Overview

The system consists of two primary workflows:
1. **CLI Pipeline** - Canonical end-to-end training and inference system
2. **Streamlit Dashboard** - User interface for inference demonstrations

**Current Status:**
- ✅ CLI Pipeline: Fully validated, reproducible
- ⚠️ Streamlit Dashboard: Functional but missing preprocessing steps (identified gap)

---

## 1. Data Handling

### 1.1 Accepted Input Formats

**File Requirements:**
- Format: `.txt` files containing Raman spectral data
- Columns: 1-2 columns, space or comma separated
- Column 1: Wavenumber values (cm⁻¹) 
- Column 2: Intensity values
- Single column files: Intensity only (wavenumber auto-generated)

**Parsing Behavior:**
- Comments (lines starting with `#`) are ignored
- Minimum requirement: 10 data points
- Validation: Checks for monotonic wavenumber range
- Expected range: ~400–4000 cm⁻¹ (minimum span > 100 cm⁻¹)

**File Naming Convention (for labels):**
- `sta-*.txt` → Label 0 (Stable/Unweathered)
- `wea-*.txt` → Label 1 (Weathered/Degraded)
- Other patterns → No automatic labeling

**Implementation:**
- CLI: `scripts/discover_raman_files.py::label_file()`
- UI: `deploy/hf-space/app.py::label_file()`

### 1.2 Known-Good Sample Files

**Expected Structure:**
```
datasets/rdwp/
├── sta-001.txt    # Stable samples
├── sta-002.txt
├── wea-100.txt    # Weathered samples  
├── wea-101.txt
└── ...
```

**Validation References:**
- CLI smoke test uses: `datasets/rdwp/wea-100.txt`
- CLI plotting test uses: `datasets/rdwp/sta-10.txt`

---

## 2. Preprocessing Pipeline

### 2.1 Core Resampling Function

**Function:** `resample_spectrum(x, y, target_len=500)`

**Behavior:**
- Linear interpolation to uniform grid
- Handles duplicate x-values by averaging y-values
- Sorts data by x-values to ensure monotonic order
- Default target length: 500 points
- Range: Uses full span of input wavenumber range

**Implementation:**
- Primary: `utils/preprocessing.py::resample_spectrum()`
- Fallback: `scripts/preprocess_dataset.py::resample_spectrum()`

**Complete Pipeline Function Available:**
- `utils/preprocessing.py::preprocess_spectrum()` - Full pipeline with all optional steps
- **⚠️ CRITICAL:** This complete function exists but is NOT used in UI

### 2.2 Optional Preprocessing Steps

**Available in CLI (`preprocess_dataset.py`, `train_model.py`):**

1. **Baseline Correction** (`--baseline`)
   - Method: Polynomial fitting (order 2)
   - Function: `remove_baseline(y)`
   - Applied after resampling

2. **Smoothing** (`--smooth`)
   - Method: Savitzky-Golay filter
   - Parameters: window_length=11, polyorder=2
   - Function: `smooth_spectrum(y)`
   - Applied after baseline correction

3. **Normalization** (`--normalize`)
   - Method: Min-max scaling to [0, 1]
   - Function: `normalize_spectrum(y)`
   - Applied last in preprocessing chain

**Processing Order:**
```
Raw Spectrum → Resample → Baseline → Smooth → Normalize → Model Input
```

### 2.3 Validation Rules

**Input Validation:**
- Minimum points: 10 for meaningful interpolation
- Wavenumber range: ~400–4000 cm⁻¹ expected
- Minimum span: > 100 cm⁻¹ for reliable resampling

**Output Validation:**
- Fixed length: 500 points (default)
- All NaN values checked and handled

---

## 3. Models and Weights

### 3.1 Figure2CNN (Baseline Model)

**Architecture:**
- Type: 1D CNN with 4 convolutional layers + 3 fully connected layers
- Input: (1, 1, target_len) tensor
- Output: 2-class logits [stable, weathered]
- File: `models/figure2_cnn.py`

**Training Configuration:**
- Target length: 500 points
- Preprocessing: `--baseline --smooth --normalize`
- Training method: 10-fold stratified cross-validation

**Model Weights:**
- Path: `outputs/figure2_model.pth`
- Training provenance: CLI `train_model.py --model figure2`
- Expected behavior: Balanced probabilities, moderate confidence

**Performance (from CLI training):**
- Accuracy: ~87.81% ± 7.59%
- Output: Raw logits in range [-10, +10] typical

### 3.2 ResNet1D (Advanced Model)

**Architecture:**
- Type: 1D ResNet with residual connections
- Input: (1, 1, target_len) tensor  
- Output: 2-class logits [stable, weathered]
- File: `models/resnet_cnn.py`

**Training Configuration:**
- Target length: 500 points (CLI), 4000 points (optional)
- Preprocessing: `--baseline --smooth --normalize`
- Training method: 10-fold stratified cross-validation

**Model Weights:**
- Path: `outputs/resnet_model.pth`
- Training provenance: CLI `train_model.py --model resnet`
- Expected behavior: High confidence, calibrated logits

**Performance (from CLI training):**
- Accuracy: Higher than Figure2CNN
- Output: Raw logits in range [-500, +500] typical

### 3.3 Training Diagnostics

**Available Models (Registry):**
```python
# models/registry.py
choices = ["figure2", "resnet", "resnet18vision"]
```
- `figure2` → `Figure2CNN`
- `resnet` → `ResNet1D` 
- `resnet18vision` → `ResNet18Vision` (defined but not trained in current pipeline)

**Logged Artifacts:**
- `outputs/logs/raman_figure2_diagnostics.json`
- `outputs/logs/raman_resnet_diagnostics.json`

**Content:**
- Fold-wise accuracies
- Confusion matrices
- Training timestamps
- Model configuration parameters

---

## 4. CLI Workflow

### 4.1 Complete Pipeline Scripts

**Canonical Sequence:**
1. `scripts/preprocess_dataset.py` - Data preprocessing and validation
2. `scripts/train_model.py` - 10-fold CV training
3. `scripts/run_inference.py` - Single-file inference

### 4.2 Preprocessing Command

```bash
python scripts/preprocess_dataset.py datasets/rdwp \
    --target-len 500 --baseline --smooth --normalize
```

**Output:** Validates dataset and prints shape information
**Validation:** Must print "X shape:" for successful completion

### 4.3 Training Command

**Figure2CNN:**
```bash
python scripts/train_model.py \
    --target-len 500 --baseline --smooth --normalize \
    --model figure2
```

**ResNet1D:**
```bash
python scripts/train_model.py \
    --target-len 4000 --baseline --smooth --normalize \
    --model resnet
```

**Outputs:**
- Model weights: `outputs/{model}_model.pth`
- Diagnostics: `outputs/logs/raman_{model}_diagnostics.json`

### 4.4 Inference Command

**Canonical Example:**
```bash
python scripts/run_inference.py \
    --target-len 500 \
    --arch figure2 \
    --input datasets/rdwp/wea-100.txt \
    --model outputs/figure2_model.pth \
    --output outputs/inference/smoke_prediction_figure2.json
```

**Expected Output Format:**
```
Predicted Label: 1 True Label: 1
Raw Logits: [[-569.544, 427.996]]
```

**Output Saved To:** JSON file with prediction results

### 4.5 Validation Harness

**Command:** `./validate_pipeline.sh`

**Steps Executed:**
1. Preprocessing validation
2. Figure2CNN training (if dataset present)
3. Inference test
4. Spectrum plotting validation

**Success Criteria:**
```
[PASS] Preprocessing
[PASS] Training & artifacts
[PASS] Inference
[PASS] Plotting
All validation checks passed!
```

**Artifacts Created:**
- `outputs/figure2_model.pth`
- `outputs/logs/raman_figure2_diagnostics.json`
- `outputs/inference/test_prediction.json`
- `outputs/plots/validation_plot.png`

---

## 5. Streamlit Dashboard Workflow

### 5.1 Application Structure

**File:** `deploy/hf-space/app.py`
**Purpose:** Hugging Face Spaces deployment with inference UI

### 5.2 Model Configuration

**Available Models:**
```python
MODEL_CONFIG = {
    "Figure2CNN (Baseline)": {
        "class": Figure2CNN,
        "path": "outputs/figure2_model.pth",
        "accuracy": "94.80%",
        "f1": "94.30%"
    },
    "ResNet1D (Advanced)": {
        "class": ResNet1D,
        "path": "outputs/resnet_model.pth",  
        "accuracy": "96.20%",
        "f1": "95.90%"
    }
}
```

### 5.3 File Upload and Processing

**Input Methods:**
1. File upload (.txt files)
2. Sample file selection (if available)

**Processing Steps:**
1. Parse spectrum data: `parse_spectrum_data(raw_text)`
2. Resample spectrum: `resample_spectrum(x_raw, y_raw, TARGET_LEN)`
3. Convert to tensor for model input
4. Model inference
5. Result display

### 5.4 Model Loading and Caching

**Function:** `load_model(model_name)` with `@st.cache_resource`

**Behavior:**
- Loads model weights from configured paths
- Falls back to random initialization if weights missing
- Caches loaded models for performance
- Error handling with user feedback

### 5.5 Inference Process

**Input Processing:**
- Target length: 500 points (hardcoded)
- **⚠️ IDENTIFIED GAP:** No baseline correction, smoothing, or normalization applied
- Raw resampling only

**Model Inference:**
- Input tensor shape: (1, 1, 500)
- Output: Raw logits [stable, weathered]
- Confidence calculation: Margin between logits

**Result Display:**
- Predicted class with confidence level
- Raw logits values
- Ground truth comparison (if filename matches convention)
- Processing time
- Spectrum visualization

### 5.6 Output Format

**Confidence Levels:**
- Logit margin > 100: "Very High"
- Logit margin > 50: "High"  
- Logit margin > 10: "Moderate"
- Logit margin ≤ 10: "Low"

**Spectrum Statistics:**
- Original length
- Resampled length (500)
- Wavenumber range

---

## 6. Empirical Outputs and Known Differences

### 6.1 CLI Reference Behavior

**Known-Good CLI Output (Figure2CNN on wea-100.txt):**
```
Predicted Label: 1 True Label: 1
Raw Logits: [[-569.544, 427.996]]
```

**Characteristics:**
- High logit magnitudes (hundreds)
- Clear class separation
- Consistent with weathered sample (label 1)

### 6.2 UI Behavior Differences

**⚠️ CRITICAL GAP IDENTIFIED:**

**Issue:** Streamlit dashboard skips preprocessing steps
- CLI applies: resampling → baseline → smoothing → normalization
- UI applies: resampling only

**Observed Symptoms:**
- Figure2CNN shows very low confidence in UI vs. CLI
- ResNet1D shows very high logits in UI vs. CLI
- Inconsistent predictions between workflows

**Root Cause:** Missing preprocessing in UI inference path

### 6.3 Expected vs. Actual UI Behavior

**Expected (should match CLI):**
- Figure2CNN: Balanced confidence, logits in [-10, +10] range
- ResNet1D: High confidence, logits in [-500, +500] range

**Actual (due to missing preprocessing):**
- Figure2CNN: Very low confidence margins
- ResNet1D: Extreme logit values
- Inconsistent with training data distribution

---

## 7. Configuration Constants

### 7.1 Default Parameters

**Target Lengths:**
- CLI default: 500 points
- CLI optional: 4000 points (ResNet)
- UI fixed: 500 points

**Preprocessing Defaults:**
- Baseline correction: Order 2 polynomial
- Smoothing: Savitzky-Golay (window=11, order=2)  
- Normalization: Min-max to [0, 1]

**Model Paths:**
- Figure2CNN: `outputs/figure2_model.pth`
- ResNet1D: `outputs/resnet_model.pth`
- Diagnostics: `outputs/logs/raman_{model}_diagnostics.json`

### 7.2 Label Mapping

```python
LABEL_MAP = {
    0: "Stable (Unweathered)",
    1: "Weathered (Degraded)"
}
```

---

## 8. Dependencies and Environment

### 8.1 Core Dependencies

**Python Version:** 3.10+

**Key Libraries:**
- `torch` - Model definitions and inference
- `numpy` - Array operations
- `scipy` - Signal processing (interpolation, smoothing)
- `sklearn` - Preprocessing utilities
- `matplotlib` - Plotting (CLI)
- `streamlit` - Web interface (UI)

### 8.2 Environment Setup

**Missing:** No `environment.yml` in root (referenced in docs but not present)

**Expected Setup:**
```bash
conda env create -f environment.yml
conda activate polymer_env
```

---

## 9. Validation and Testing

### 9.1 Smoke Test Coverage

**Pipeline Validation:** `./validate_pipeline.sh`
- ✅ Preprocessing validation
- ✅ Training artifact generation  
- ✅ Inference execution
- ✅ Plotting functionality

**Test Files Used:**
- Training: Full dataset (`datasets/rdwp/`)
- Inference: `datasets/rdwp/wea-100.txt`
- Plotting: `datasets/rdwp/sta-10.txt`

### 9.2 Success Criteria

**All Components Must Pass:**
1. Preprocessing prints "X shape:" 
2. Model files created (.pth)
3. Diagnostics JSON generated
4. Inference JSON output created
5. No runtime errors in any step

---

## 10. Critical Gaps and Inconsistencies

### 10.1 Preprocessing Gap in UI

**Status:** ⚠️ HIGH PRIORITY

**Description:** Streamlit dashboard omits baseline correction, smoothing, and normalization steps that are applied in CLI training.

**Critical Finding:** The complete preprocessing pipeline function `utils/preprocessing.py::preprocess_spectrum()` exists and implements the full workflow but is NOT used in the UI inference path.

**Current UI Implementation:**
```python
# UI only does:
y_resampled = resample_spectrum(x_raw, y_raw, TARGET_LEN)
```

**Should Be (to match CLI):**
```python  
# Should use complete pipeline:
y_processed = preprocess_spectrum(x_raw, y_raw, TARGET_LEN, 
                                 baseline_correction=True,
                                 apply_smoothing=True, 
                                 normalize=True)
```

**Impact:** 
- Inconsistent predictions between CLI and UI
- Model receives different data distributions
- Undermines reproducibility
- Available solution exists but not implemented

**Location:** `deploy/hf-space/app.py` inference pipeline

### 10.2 Environment Configuration

**Status:** ⚠️ MEDIUM PRIORITY

**Description:** Missing `environment.yml` file referenced in documentation.

**Impact:**
- Difficult environment setup for new users
- Inconsistent dependency versions

### 10.3 Model Performance Reporting

**Status:** 📊 INFORMATIONAL

**Description:** UI shows hardcoded performance metrics that may not match actual trained model performance.

**Current UI Values:**
- Figure2CNN: "94.80%" accuracy, "94.30%" F1
- ResNet1D: "96.20%" accuracy, "95.90%" F1

**Recommendation:** Dynamically load from diagnostics JSON files.

---

## 11. Future Development Guidelines

### 11.1 Modification Principles

**Before Any Changes:**
1. Run full CLI validation: `./validate_pipeline.sh`
2. Document current outputs as baseline
3. Test on known-good samples

**After Any Changes:**
1. Re-run CLI validation
2. Compare outputs against this baseline
3. Document any differences
4. Update this inventory if behavior changes

### 11.2 Integration Requirements

**For UI Modifications:**
- Must maintain consistency with CLI preprocessing
- Should load actual model performance from diagnostics
- Must preserve reproducible inference behavior

**For CLI Modifications:**
- Must maintain backward compatibility with existing model weights
- Should preserve artifact naming conventions
- Must update validation harness accordingly

---

## 12. Artifact Inventory

### 12.1 Expected Outputs

**After Full Pipeline Run:**
```
outputs/
├── figure2_model.pth              # Figure2CNN weights
├── resnet_model.pth              # ResNet1D weights  
├── logs/
│   ├── raman_figure2_diagnostics.json
│   └── raman_resnet_diagnostics.json
├── inference/
│   └── test_prediction.json
└── plots/
    └── validation_plot.png
```

### 12.2 Reference Checksums

**Status:** 🔄 TO BE GENERATED

Future versions should include file checksums for exact reproducibility verification.

---

## 📌 Summary

This ground truth inventory establishes the definitive baseline for the ML polymer recycling pipeline as of the current state. The most critical finding is the **preprocessing gap in the Streamlit dashboard** that causes inconsistent behavior between CLI and UI workflows.

**Key Baseline Behaviors:**
- ✅ CLI pipeline fully validated and reproducible
- ✅ Two model architectures with known performance characteristics  
- ✅ Comprehensive preprocessing pipeline with optional steps
- ⚠️ UI missing preprocessing steps (critical gap)
- ✅ Validation harness confirms all components

This document serves as the objective reference point for all future development, ensuring no drift occurs between system components.

---

*Document Version: 1.0*  
*Created: August 2024*  
*Purpose: Empirical baseline for ml-polymer-recycling pipeline*  
*Next Action: Address preprocessing gap in UI to achieve CLI/UI consistency*