---
title: AI Polymer Classification (Raman & FTIR)
emoji: 🔬
colorFrom: indigo
colorTo: yellow
sdk: streamlit
app_file: App.py
pinned: false
license: apache-2.0
---

## AI-Driven Polymer Aging Prediction and Classification (v0.1)

This web application classifies the degradation state of polymers using **Raman and FTIR spectroscopy** and deep learning.
It is a prototype pipeline for evaluating multiple convolutional neural networks (CNNs) on spectral data.

---

## 🧪 Current Scope

- 🔬 **Modalities**: Raman & FTIR spectroscopy
- 💾 **Input Formats**: `.txt`, `.csv`, `.json` (with auto-detection)
- 🧠 **Models**: Figure2CNN (baseline), ResNet1D, ResNet18Vision
- 📊 **Task**: Binary classification — Stable vs Weathered polymers
- 🚀 **Features**: Multi-model comparison, performance tracking dashboard
- 🛠️ **Architecture**: PyTorch + Streamlit

---

## 🚧 Roadmap

- [x] Inference from Raman `.txt` files
- [x] Model selection (Figure2CNN, ResNet1D)
- [x] **FTIR support** (modular integration complete)
- [x] **Multi-model comparison dashboard**
- [x] **Performance tracking dashboard**
- [ ] Add more trained CNNs for comparison
- [ ] Image-based inference (future modality)
- [ ] RESTful API for programmatic access

---

## 🧭 How to Use

The application provides three main analysis modes in a tabbed interface:

1.  **Standard Analysis**:

    - Upload a single spectrum file (`.txt`, `.csv`, `.json`) or a batch of files.
    - Choose a model from the sidebar.
    - Run analysis and view the prediction, confidence, and technical details.

2.  **Model Comparison**:

    - Upload a single spectrum file.
    - The app runs inference with all available models.
    - View a side-by-side comparison of the models' predictions and performance.

3.  **Performance Tracking**:
    - Explore a dashboard with visualizations of historical performance data.
    - Compare model performance across different metrics.
    - Export performance data in CSV or JSON format.

### Supported Input

- Plaintext `.txt`, `.csv`, or `.json` files.
- Data can be space-, comma-, or tab-separated.
- Comment lines (`#`, `%`) are ignored.
- The app automatically detects the file format and resamples the data to a standard length.

---

## Contributors

Dr. Sanmukh Kuppannagari (Mentor)
Dr. Metin Karailyan (Mentor)
Jaser Hasan (Author/Developer)

## Model Credit

Baseline model inspired by:

Neo, E.R.K., Low, J.S.C., Goodship, V., Debattista, K. (2023).
_Deep learning for chemometric analysis of plastic spectral data from infrared and Raman databases._
_Resources, Conservation & Recycling_, **188**, 106718.
[https://doi.org/10.1016/j.resconrec.2022.106718](https://doi.org/10.1016/j.resconrec.2022.106718)

---

## 🔗 Links

- **Live App**: [Hugging Face Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)
- **GitHub Repo**: [ml-polymer-recycling](https://github.com/KLab-AI3/ml-polymer-recycling)

## 🚀 Technical Architecture

**The system is built on a modular, production-ready architecture designed for scalability and maintainability.**

- **Frontend**: A Streamlit-based web application (`app.py`) provides an interactive, multi-tab user interface.
- **Backend**: PyTorch handles all deep learning operations, including model loading and inference.
- **Model Management**: A registry pattern (`models/registry.py`) allows for dynamic model loading and easy integration of new architectures.
- **Data Processing**: A robust, modality-aware preprocessing pipeline (`utils/preprocessing.py`) ensures data integrity and standardization for both Raman and FTIR data.
- **Multi-Format Parsing**: The `utils/multifile.py` module handles parsing of `.txt`, `.csv`, and `.json` files.
- **Results Management**: The `utils/results_manager.py` module manages session and persistent results, with support for multi-model comparison and data export.
- **Performance Tracking**: The `utils/performance_tracker.py` module logs performance metrics to a SQLite database and provides a dashboard for visualization.
- **Deployment**: The application is containerized using Docker (`Dockerfile`) for reproducible, cross-platform execution.
