---
title: AI Polymer Classification
emoji: 🔬
colorFrom: indigo
colorTo: teal
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
---
## AI-Driven Polymer Aging Prediction and Classification (v0.1)

This web application classifies the degradation state of polymers using Raman spectroscopy and deep learning.

It was developed as part of the AIRE 2025 internship project at the Imageomics Institute and demonstrates a prototype pipeline for evaluating multiple convolutional neural networks (CNNs) on spectral data.

---

## 🧪 Current Scope

- 🔬 **Modality**: Raman spectroscopy (.txt)
- 🧠 **Model**: Figure2CNN (baseline)
- 📊 **Task**: Binary classification — Stable vs Weathered polymers
- 🛠️ **Architecture**: PyTorch + Streamlit

---

## 🚧 Roadmap

- [x] Inference from Raman `.txt` files
- [x] Model selection (Figure2CNN, ResNet1D)
- [ ] Add more trained CNNs for comparison
- [ ] FTIR support (modular integration planned)
- [ ] Image-based inference (future modality)

---

## 🧭 How to Use

1. Upload a Raman spectrum `.txt` file (or select a sample)
2. Choose a model from the sidebar
3. Run analysis
4. View prediction, logits, and technical information

Supported input:

- Plaintext `.txt` files with 1–2 columns
- Space- or comma-separated
- Comment lines (#) are ignored
- Automatically resampled to 500 points

---

## Contributors

  👨‍🏫 Dr. Sanmukh Kuppannagari (Mentor)  
  👨‍🏫 Dr. Metin Karailyan (Mentor)  
  👨‍💻 Jaser Hasan (Author/Developer)

## 🧠 Model Credit

Baseline model inspired by:

Neo, E.R.K., Low, J.S.C., Goodship, V., Debattista, K. (2023).  
*Deep learning for chemometric analysis of plastic spectral data from infrared and Raman databases.*  
_Resources, Conservation & Recycling_, **188**, 106718.  
[https://doi.org/10.1016/j.resconrec.2022.106718](https://doi.org/10.1016/j.resconrec.2022.106718)

---

## 🔗 Links

- 💻 **Live App**: [Hugging Face Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)
- 📂 **GitHub Repo**: [ml-polymer-recycling](https://github.com/KLab-AI3/ml-polymer-recycling)


## 🎯 Strategic Expansion Objectives (Roadmap)

**The roadmap defines three major expansion paths designed to broaden the system’s capabilities and impact:**

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
