# Codebase Inventory: ml-polymer-recycling

## Overview

A comprehensive machine learning system for AI-driven polymer aging prediction and classification using spectral data analysis. The project implements multiple CNN architectures (Figure2CNN, ResNet1D, ResNet18Vision) to classify polymer degradation levels as a proxy for recyclability, built with Python, PyTorch, and featuring both CLI and Streamlit UI workflows.

## Inventory by Category

### 1. Core Application Modules

- **Module Name**: `models/registry.py`
  - **Purpose**: Central registry system for model architectures providing dynamic model selection and instantiation
  - **Key Exports/Functions**: `choices()`, `build(name, input_length)`, `_REGISTRY`
  - **Key Dependencies**: `models.figure2_cnn`, `models.resnet_cnn`, `models.resnet18_vision`
  - **External Dependencies**: `typing`

- **Module Name**: `models/figure2_cnn.py`
  - **Purpose**: CNN architecture implementation based on literature (Neo et al. 2023) for 1D Raman spectral classification
  - **Key Exports/Functions**: `Figure2CNN` class with conv blocks and classifier layers
  - **Key Dependencies**: None (self-contained)
  - **External Dependencies**: `torch`, `torch.nn`

- **Module Name**: `models/resnet_cnn.py`
  - **Purpose**: ResNet1D implementation with residual blocks for deeper spectral feature learning
  - **Key Exports/Functions**: `ResNet1D`, `ResidualBlock1D` classes
  - **Key Dependencies**: None (self-contained)
  - **External Dependencies**: `torch`, `torch.nn`

- **Module Name**: `models/resnet18_vision.py`
  - **Purpose**: ResNet18 architecture adapted for 1D spectral data processing
  - **Key Exports/Functions**: `ResNet18Vision` class
  - **Key Dependencies**: None (self-contained)
  - **External Dependencies**: `torch`, `torch.nn`

- **Module Name**: `utils/preprocessing.py`
  - **Purpose**: Spectral data preprocessing utilities including resampling, baseline correction, smoothing, and normalization
  - **Key Exports/Functions**: `preprocess_spectrum()`, `resample_spectrum()`, `remove_baseline()`, `normalize_spectrum()`, `smooth_spectrum()`
  - **Key Dependencies**: None (self-contained)
  - **External Dependencies**: `numpy`, `scipy.interpolate`, `scipy.signal`, `sklearn.preprocessing`

- **Module Name**: `scripts/preprocess_dataset.py`
  - **Purpose**: Comprehensive dataset preprocessing pipeline with CLI interface for Raman spectral data
  - **Key Exports/Functions**: `preprocess_dataset()`, `resample_spectrum()`, `label_file()`, preprocessing helper functions
  - **Key Dependencies**: `scripts.discover_raman_files`, `scripts.plot_spectrum`
  - **External Dependencies**: `numpy`, `scipy`, `sklearn.preprocessing`

### 2. Scripts & Automation

- **Script Name**: `validate_pipeline.sh`
  - **Trigger**: Manual execution (`./validate_pipeline.sh`)
  - **Apparent Function**: Canonical smoke test validating the complete Raman pipeline from preprocessing through training to inference
  - **Dependencies**: `conda`, `scripts/preprocess_dataset.py`, `scripts/train_model.py`, `scripts/run_inference.py`, `scripts/plot_spectrum.py`

- **Script Name**: `scripts/train_model.py`
  - **Trigger**: CLI execution (`python scripts/train_model.py`)
  - **Apparent Function**: 10-fold stratified cross-validation training with multiple model architectures and preprocessing options
  - **Dependencies**: `scripts/preprocess_dataset`, `models/registry`, reproducibility seeds, PyTorch training loop

- **Script Name**: `scripts/run_inference.py`
  - **Trigger**: CLI execution (`python scripts/run_inference.py`)
  - **Apparent Function**: Single spectrum inference with model loading, preprocessing, and prediction output to JSON
  - **Dependencies**: `models/registry`, `scripts/preprocess_dataset`, trained model weights

- **Script Name**: `scripts/plot_spectrum.py`
  - **Trigger**: CLI execution (`python scripts/plot_spectrum.py`)
  - **Apparent Function**: Visualization tool for Raman spectra with matplotlib plotting and file I/O
  - **Dependencies**: Spectrum loading utilities

- **Script Name**: `scripts/discover_raman_files.py`
  - **Trigger**: Imported by other scripts
  - **Apparent Function**: File discovery and labeling utilities for Raman dataset management
  - **Dependencies**: File system operations, regex pattern matching

- **Script Name**: `scripts/list_spectra.py`
  - **Trigger**: CLI or import
  - **Apparent Function**: Dataset inventory and spectrum listing utilities
  - **Dependencies**: File system scanning

### 3. Configuration & Data

- **File Name**: `deploy/hf-space/requirements.txt`
  - **Purpose**: Python dependencies for Hugging Face Spaces deployment
  - **Key Contents/Structure**: `streamlit`, `torch`, `torchvision`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `fastapi`, `altair`, `huggingface-hub`

- **File Name**: `deploy/hf-space/Dockerfile`
  - **Purpose**: Container configuration for Hugging Face Spaces deployment
  - **Key Contents/Structure**: Python 3.13-slim base, build tools installation, Streamlit server configuration on port 8501

- **File Name**: `deploy/hf-space/sample_data/sta-1.txt`
  - **Purpose**: Sample Raman spectrum for UI demonstration
  - **Key Contents/Structure**: Two-column wavenumber/intensity data format

- **File Name**: `deploy/hf-space/sample_data/sta-2.txt`
  - **Purpose**: Additional sample Raman spectrum for UI testing
  - **Key Contents/Structure**: Two-column wavenumber/intensity data format

- **File Name**: `.gitignore`
  - **Purpose**: Version control exclusions for datasets, build artifacts, and system files
  - **Key Contents/Structure**: `datasets/`, `__pycache__/`, model weights, logs, environment files, deprecated scripts

- **File Name**: `MANIFEST.git`
  - **Purpose**: Git object manifest listing all tracked files with hashes
  - **Key Contents/Structure**: File paths, permissions, and SHA hashes for repository contents

### 4. Assets & Documentation

- **Asset Name**: `README.md`
  - **Purpose**: Primary project documentation with objectives, architecture overview, and usage instructions
  - **Key Contents/Structure**: Project goals, model architectures table, structure diagram, installation guides, sample commands

- **Asset Name**: `GROUND_TRUTH_PIPELINE.md`
  - **Purpose**: Comprehensive empirical baseline inventory documenting every aspect of the current system
  - **Key Contents/Structure**: 635-line detailed documentation of data handling, preprocessing, models, CLI workflow, UI workflow, and gap identification

- **Asset Name**: `docs/ENVIRONMENT_GUIDE.md`
  - **Purpose**: Environment management guide for local and HPC deployment
  - **Key Contents/Structure**: Conda vs venv setup instructions, platform-specific configurations, dependency management

- **Asset Name**: `docs/PROJECT_TIMELINE.md`
  - **Purpose**: Development milestone tracking and project progression documentation
  - **Key Contents/Structure**: Phase-based timeline from project kickoff through model expansion, tagged milestones

- **Asset Name**: `docs/sprint_log.md`
  - **Purpose**: Sprint-based development log with specific technical changes and testing results
  - **Key Contents/Structure**: Chronological entries with goals, changes, tests, and notes for each development sprint

- **Asset Name**: `docs/REPRODUCIBILITY.md`
  - **Purpose**: Scientific reproducibility guidelines and artifact control documentation
  - **Key Contents/Structure**: Validation procedures, artifact integrity, experimental controls

- **Asset Name**: `docs/HPC_REMOTE_SETUP.md`
  - **Purpose**: High-performance computing environment setup for CWRU Pioneer cluster
  - **Key Contents/Structure**: HPC-specific configurations, remote access procedures, computational resource management

- **Asset Name**: `docs/BACKEND_MIGRATION_LOG.md`
  - **Purpose**: Technical migration documentation for backend architecture changes
  - **Key Contents/Structure**: Migration procedures, compatibility notes, system architecture evolution

### 5. Deployment & UI Components

- **Module Name**: `deploy/hf-space/app.py`
  - **Purpose**: Streamlit web application for polymer classification with file upload and model inference
  - **Key Exports/Functions**: Streamlit UI components, model loading, preprocessing pipeline, prediction display
  - **Key Dependencies**: `models.figure2_cnn`, `models.resnet_cnn`, `utils.preprocessing` (fallback), `scripts.preprocess_dataset`
  - **External Dependencies**: `streamlit`, `torch`, `matplotlib`, `PIL`, `numpy`

### 6. Model Artifacts & Outputs

- **File Name**: `outputs/resnet_model.pth`
  - **Purpose**: Trained ResNet1D model weights for Raman spectrum classification
  - **Key Contents/Structure**: PyTorch state dictionary with model parameters

## Workflows & Interactions

- **CLI Training Pipeline**: The main training workflow starts with `scripts/train_model.py` which imports the model registry (`models/registry.py`) to dynamically select architectures (Figure2CNN, ResNet1D, or ResNet18Vision). It uses `scripts/preprocess_dataset.py` to load and preprocess Raman spectra from `datasets/rdwp/`, applying resampling, baseline correction, smoothing, and normalization. The script performs 10-fold stratified cross-validation and saves trained models to `outputs/{model}_model.pth` with diagnostics to `outputs/logs/`.

- **CLI Inference Pipeline**: Running `scripts/run_inference.py` loads a trained model via the registry, processes a single Raman spectrum file through the same preprocessing pipeline, and outputs predictions in JSON format to `outputs/inference/`.

- **UI Workflow**: The Streamlit application (`deploy/hf-space/app.py`) provides a web interface that loads trained models, accepts file uploads or sample data selection, but currently bypasses the full preprocessing pipeline (missing baseline correction, smoothing, and normalization steps) before running inference.

- **Validation Workflow**: The `validate_pipeline.sh` script orchestrates a complete pipeline test by sequentially running preprocessing, training, inference, and plotting scripts to ensure reproducibility and catch regressions.

- **Model Registry System**: All model architectures are centrally managed through `models/registry.py`, which provides dynamic model selection for both CLI training and inference scripts, ensuring consistent model instantiation across the codebase.

## External Dependencies Summary

- **PyTorch Ecosystem**: `torch`, `torchvision` for deep learning model implementation and training
- **Scientific Computing**: `numpy`, `scipy` for numerical operations and signal processing
- **Machine Learning**: `scikit-learn` for preprocessing, metrics, and cross-validation utilities
- **Data Handling**: `pandas` for structured data manipulation
- **Visualization**: `matplotlib`, `seaborn` for plotting and data visualization
- **Web Framework**: `streamlit` for interactive web application deployment
- **Image Processing**: `PIL` (Pillow) for image handling in the UI
- **Development Tools**: `argparse` for CLI interfaces, `json` for data serialization
- **Deployment**: `fastapi`, `uvicorn` for potential API deployment, `huggingface-hub` for model hosting

## Key Findings & Assumptions

- **Critical Preprocessing Gap**: The UI workflow in `deploy/hf-space/app.py` bypasses essential preprocessing steps (baseline correction, smoothing, normalization) that are standard in the CLI pipeline, potentially causing prediction inconsistencies.

- **Model Architecture Assumptions**: Three CNN architectures are registered (`figure2`, `resnet`, `resnet18vision`) but the codebase suggests only two are currently trained and validated in the standard pipeline.

- **Dataset Structure**: The system assumes Raman spectra are stored as two-column text files (wavenumber, intensity) in the `datasets/rdwp/` directory, with filenames indicating weathering conditions for automated labeling.

- **Environment Fragmentation**: The project uses different dependency management systems (Conda for local development, venv for HPC, pip requirements for deployment) which could lead to environment inconsistencies.

- **Reproducibility Controls**: Strong emphasis on scientific reproducibility with fixed random seeds, deterministic algorithms, and comprehensive validation scripts, indicating this is research-oriented code requiring strict experimental controls.

- **Deployment Readiness**: The Hugging Face Spaces deployment setup suggests the project is intended for public demonstration or research sharing, but the preprocessing gap needs resolution for production use.

- **Legacy Code Management**: The `.gitignore` and documentation references suggest active management of deprecated FTIR-related components, indicating focused scope refinement to Raman-only analysis.
