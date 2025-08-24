---
title: AI Polymer Classification
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🔬 AI-Driven Polymer Aging Prediction and Classification

This application uses deep learning to classify polymer degradation states from Raman spectroscopy data, helping researchers assess material longevity and recycling potential.

## 🚀 Features

- 🔬 **Real-time Analysis**: Upload Raman spectra and get instant classification
- 🧠 **Multiple AI Models**: Choose between Figure2CNN (baseline) and ResNet1D (advanced)
- 📊 **Interactive Visualization**: View raw and processed spectral data
- 🎯 **Binary Classification**: Stable (Unweathered) vs Weathered (Degraded) polymers
- 📈 **Confidence Scoring**: Detailed prediction confidence and model insights

## 🔧 How to Use

1. **Select Model**: Choose between Figure2CNN or ResNet1D architectures
2. **Upload Data**: Upload a Raman spectrum file (.txt format)
3. **Run Analysis**: Click "Run Inference" to process your data
4. **View Results**: Examine predictions, confidence scores, and visualizations

## 📊 Model Performance

| Model | Accuracy | F1 Score | Description |
|-------|----------|----------|-------------|
| Figure2CNN | 94.80% | 94.30% | Baseline CNN with standard filters |
| ResNet1D | 96.20% | 95.90% | Advanced residual network with deeper learning |

## 📁 Input Format

Upload Raman spectroscopy data as `.txt` files with:
- Two columns: wavenumber and intensity
- Space or comma-separated values
- Any length (will be resampled to 500 points)

Example format:
```
200.0 1542.3
201.0 1543.1
202.0 1544.8
...
```

## 🧪 Sample Data

The app includes sample spectra for testing:
- Stable polymer samples (prefix: "sta")
- Weathered polymer samples (prefix: "wea")

## 🔬 About the Research

Part of the **AIRE 2025 Internship Project**: AI-Driven Polymer Aging Prediction and Classification

This research applies machine learning to materials science, specifically:
- **Material Longevity Assessment**: Predict how polymers degrade over time
- **Recycling Optimization**: Identify polymers suitable for recycling
- **Sustainability Research**: Support environmental impact studies

### Research Team

- **Author**: Jaser Hasan
- **Mentor**: Dr. Sanmukh Kuppannagari
- **Institution**: Case Western Reserve University
- **Project**: [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)

## 🛠️ Technical Details

### Model Architectures

**Figure2CNN (Baseline)**
- 4-layer CNN with MaxPooling
- 256-dimensional fully connected layers
- Input: 500-point Raman spectrum
- Output: Binary classification logits

**ResNet1D (Advanced)**
- 3-stage residual network
- Skip connections for improved gradient flow
- Global average pooling
- Optimized for 1D spectral data

### Data Processing Pipeline

1. **Upload**: Raw Raman spectrum file
2. **Parsing**: Extract wavenumber and intensity values
3. **Resampling**: Interpolate to 500 points using scipy
4. **Normalization**: Apply preprocessing (optional baseline correction)
5. **Inference**: Feed through selected CNN model
6. **Classification**: Generate prediction with confidence scores

### Performance Metrics

Models evaluated using 10-fold cross-validation on a balanced dataset of polymer samples with known degradation states.

## 📚 Citations and References

If you use this tool in your research, please cite:

```bibtex
@software{hasan2025polymer,
  title={AI-Driven Polymer Aging Prediction and Classification},
  author={Hasan, Jaser and Kuppannagari, Sanmukh R.},
  year={2025},
  institution={Case Western Reserve University},
  url={https://huggingface.co/spaces/YOUR_USERNAME/polymer-classification}
}
```

## 🤝 Contributing

This is an open-source research project. Contributions are welcome:

- **Data**: Share additional Raman spectra datasets
- **Models**: Contribute improved architectures
- **Features**: Suggest new functionality
- **Documentation**: Improve user guides and tutorials

## 📄 License

Licensed under Apache 2.0. See [LICENSE](https://github.com/KLab-AI3/ml-polymer-recycling/blob/main/LICENSE) for details.

## 🔗 Links

- **GitHub Repository**: [ml-polymer-recycling](https://github.com/KLab-AI3/ml-polymer-recycling)
- **Documentation**: [Full Project Docs](https://github.com/KLab-AI3/ml-polymer-recycling/tree/main/docs)
- **Research Paper**: [Coming Soon]
- **Contact**: [Jaser Hasan](mailto:jxh369@case.edu)

---

*Advancing materials science through artificial intelligence* 🧪🤖