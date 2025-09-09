# POLYMEROS: AI-Driven Polymer Aging Analysis & Classification

POLYMEROS is an advanced, AI-driven platform for analyzing and classifying polymer degradation using spectroscopic data. This project extends a baseline CNN model to incorporate multi-modal analysis (Raman & FTIR), modern machine learning architectures, a comprehensive data pipeline, and an interactive educational framework.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)

---

## 🚀 Key Features & Recent Enhancements

This platform has been significantly enhanced with a suite of research-grade features. Recent architectural improvements have focused on creating a robust, maintainable, and unified training system.

### Unified Training Architecture

Previously, the project contained two separate implementations of the model training logic: one in the command-line script (`scripts/train_model.py`) and another powering the backend of the web UI (`utils/training_manager.py`). This duplication led to inconsistencies and made maintenance difficult.

The system has been refactored to follow the **Don't Repeat Yourself (DRY)** principle:

1. **Central `TrainingEngine`**: A new `utils/training_engine.py` module was created to house the core, canonical training and cross-validation loop. This engine is now the single source of truth for how models are trained.

2. **Decoupled Data Structures**: Shared data classes like `TrainingConfig` and `TrainingStatus` were moved to a dedicated `utils/training_types.py` file. This resolved circular import errors and improved modularity.

3. **Refactored Interfaces**:
   - The **CLI script** (`scripts/train_model.py`) is now a lightweight wrapper that parses command-line arguments and calls the `TrainingEngine`.
   - The **UI backend** (`utils/training_manager.py`) now also uses the `TrainingEngine` to run training jobs submitted from the "Model Training Hub".

This unified architecture ensures that any improvements to the training process are immediately available to both developers using the CLI and users interacting with the web UI.

---

## � Acquiring and Preparing Datasets

To train a model, you need a dataset of polymer spectra organized in a specific way. The training engine expects a directory containing two subdirectories:

- `stable/`: Contains spectra for unweathered, stable polymers.
- `weathered/`: Contains spectra for weathered, degraded polymers.

**Example Directory Structure:**

```
/my_dataset
├── /stable
│   ├── sample_01.txt
│   ├── sample_02.csv
│   └── ...
└── /weathered
    ├── sample_101.txt
    ├── sample_102.json
    └── ...
```

### Data Format

Each file inside the `stable` and `weathered` folders should be a two-column text-based format representing a single spectrum:

- **Column 1**: Wavenumber (in cm⁻¹)
- **Column 2**: Intensity / Absorbance
- **Supported File Types**: `.txt`, `.csv`, `.json`
- **Separators**: Comma, space, or tab.

### Finding Public Datasets

If you don't have your own data, you can find public datasets from various sources. Here are some starting points and keywords for your search:

- **Open Specy**: A fantastic community-driven library for Raman and FTIR spectra. You can search for specific polymers and download data.
- **RRUFF™ Project**: An integrated database of Raman spectra, X-ray diffraction, and chemistry data for minerals. While not polymer-focused, it's a great example of a spectral database.
- **NIST Chemistry WebBook**: Contains FTIR spectra for many chemical compounds.
- **GitHub & Kaggle**: Search for "polymer spectroscopy dataset", "Raman spectra plastic", or "FTIR microplastics".

When using public data, you may need to manually classify and organize the files into the `stable`/`weathered` structure based on the sample descriptions provided with the dataset.

---

## �🛠️ How to Train Models

With the new unified architecture, you can train models using either the command line or the interactive web UI, depending on your needs.

### 1. CLI Training (For Developers & Automation)

The command-line interface is the ideal method for reproducible experiments, automated workflows, or training on a remote server. It provides full control over all training hyperparameters.

**Why use the CLI?**

- For scripting multiple training runs.
- For integration into CI/CD pipelines.
- When working in a non-GUI environment.

**Example Command:**
To run a 10-fold cross-validation for the `figure2` model, run the following from the project's root directory:

```bash
python scripts/train_model.py --model figure2 --epochs 15 --baseline --smooth --normalize
```

This command will:

- Load the default dataset from `datasets/rdwp`.
- Apply the specified preprocessing steps.
- Run the training using the central `TrainingEngine`.
- Save the final model weights to `outputs/weights/figure2_model.pth` and a detailed JSON log to `outputs/logs/`.

### 2. UI Training Hub (For Interactive Use)

The "Model Training Hub" within the web application provides a user-friendly, graphical interface for training models. It's designed for interactive experimentation and for users who may not be comfortable with the command line.

**Why use the UI?**

- To easily train models on your own uploaded datasets.
- To interactively tweak hyperparameters and see their effect.
- To monitor training progress in real-time with visual feedback.

**How to use it:**

1. Navigate to the **Model Training Hub** tab in the application.
2. **Configure Your Job**:
   - Select a model architecture.
   - Upload a new dataset or choose an existing one.
   - Adjust training parameters like epochs, learning rate, and batch size.
3. Click **"🚀 Start Training"**.
4. The job will run in the background, and you can monitor its progress in the "Training Status" and "Training Progress" sections. Completed models and logs can be downloaded directly from the UI.

---

## Project Structure Overview

- `app.py`: Main Streamlit application entry point.
- `modules/`: Contains all major feature modules.
  - `training_ui.py`: Renders the "Model Training Hub" tab.
- `scripts/`: Contains command-line tools.
  - `train_model.py`: The CLI for running training jobs.
  - `inspect_weights.py`: A diagnostic tool to check model weight files.
- `utils/`: Core utilities for the application.
  - `training_engine.py`: **The new central training logic.**
  - `training_manager.py`: The backend manager for UI-based training jobs.
  - `training_types.py`: **New file for shared training data structures.**
- `models/`: Model definitions and the central model registry.
- `outputs/`: Default directory for saved model weights and training logs.
