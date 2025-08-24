# 🚀 Complete Guide: Migrating Streamlit App to Hugging Face Spaces

## Overview

This comprehensive tutorial guides you through migrating the ML Polymer Recycling Streamlit application from a local development environment to **Hugging Face Spaces**, making your AI-powered polymer classification tool accessible to researchers worldwide.

### What You'll Learn

- 📋 Prerequisites and setup requirements
- 🔧 Preparing your Streamlit app for Hugging Face Spaces
- 📁 Creating the required configuration files
- 🚀 Deploying to Hugging Face Spaces
- 🐛 Troubleshooting common issues
- 🔄 Managing updates and maintenance

### What You'll Build

By the end of this tutorial, you'll have:
- A fully functional Hugging Face Space running your polymer classification app
- Public access to your AI tool for the research community
- Automated deployment pipeline
- Professional documentation and presentation

---

## 📋 Prerequisites

### Required Accounts
- **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
- **Git**: Installed on your local machine
- **Python 3.8+**: For local testing

### Required Knowledge
- Basic understanding of Streamlit applications
- Git version control fundamentals
- Python package management
- Basic command line operations

### Project Setup
Ensure you have the ML Polymer Recycling project cloned locally:
```bash
git clone https://github.com/KLab-AI3/ml-polymer-recycling.git
cd ml-polymer-recycling
```

---

## 🎯 Understanding Hugging Face Spaces

### What are Hugging Face Spaces?

Hugging Face Spaces is a platform that allows you to host machine learning applications with zero-configuration deployment. It supports:

- **Streamlit** applications (what we're using)
- Gradio interfaces
- Static HTML sites
- Docker containers

### Why Choose Hugging Face Spaces?

✅ **Zero DevOps**: No server management required
✅ **Free Hosting**: Community tier available
✅ **Automatic Deployment**: Git-based deployment
✅ **GPU Support**: Available for compute-intensive apps
✅ **Community Reach**: Built-in discovery and sharing
✅ **Version Control**: Integrated with Git

---

## 🔧 Step 1: Analyzing the Current Application

### Current App Structure

Our Streamlit app (`app/app.py`) currently has these key components:

```python
# Key imports
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
from scripts.preprocess_dataset import resample_spectrum
```

### Dependencies Analysis

The app requires:
- **Core ML**: PyTorch, NumPy, SciPy
- **UI**: Streamlit
- **Visualization**: Matplotlib, PIL
- **Data Processing**: Pandas (implicit)
- **Custom Modules**: Local model and script imports

### Current Challenges for Migration

1. **Local Module Dependencies**: App imports from `models/` and `scripts/`
2. **Missing Requirements File**: No explicit dependency list
3. **Model File Paths**: Hardcoded paths to model weights
4. **File Structure**: Not optimized for Hugging Face Spaces

---

## 📁 Step 2: Preparing the File Structure

### Create Hugging Face Spaces Directory

Create a new directory for your Hugging Face Space:

```bash
mkdir huggingface-space
cd huggingface-space
```

### Required File Structure

Your Hugging Face Space needs this structure:

```
huggingface-space/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Space description and documentation
├── models/               # Model architecture files
│   ├── __init__.py
│   ├── figure2_cnn.py
│   └── resnet_cnn.py
├── utils/                # Utility functions
│   ├── __init__.py
│   └── preprocessing.py
├── sample_data/          # Sample spectra for testing
│   └── *.txt
└── model_weights/        # Pre-trained model files
    ├── figure2_model.pth
    └── resnet_model.pth
```

---

## 🔨 Step 3: Creating the Requirements File

### Analyzing Dependencies

Create `requirements.txt` with all necessary packages:

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
Pillow>=9.5.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

### Version Considerations

- **Pin major versions** to ensure compatibility
- **Use >= operators** for flexibility with security updates
- **Test locally** with these exact versions

---

## 📝 Step 4: Adapting the Streamlit App

### Create the Main App File

Copy and adapt your `app/app.py` to the new structure:

```python
# huggingface-space/app.py
import os
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Import local modules
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
from utils.preprocessing import resample_spectrum

# Configuration
st.set_page_config(
    page_title="AI Polymer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the rest of your Streamlit app code here...
```

### Key Modifications Needed

1. **Update Import Paths**: Adjust module imports for new structure
2. **Model Path Configuration**: Use relative paths for model weights
3. **Add Error Handling**: Robust error handling for missing files
4. **Environment Variables**: Support for configuration via environment

### Model Loading Strategy

```python
@st.cache_resource
def load_model(model_type, model_path):
    """Load and cache the specified model"""
    try:
        if model_type == "figure2":
            model = Figure2CNN(input_length=500)
        elif model_type == "resnet":
            model = ResNet1D(input_length=500)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
```

---

## 🏗️ Step 5: Creating Supporting Files

### Create README.md for Your Space

```markdown
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

# AI-Driven Polymer Aging Prediction and Classification

This application uses deep learning to classify polymer degradation states from Raman spectroscopy data.

## Features

- 🔬 Real-time Raman spectrum analysis
- 🧠 Multiple AI model architectures (Figure2CNN, ResNet1D)
- 📊 Interactive visualization of spectral data
- 🎯 Binary classification: Stable vs Weathered polymers

## How to Use

1. Select your preferred AI model
2. Upload a Raman spectrum (.txt file)
3. Click "Run Inference" to get predictions
4. View detailed results and confidence scores

## Model Information

- **Figure2CNN**: Baseline CNN with 94.80% accuracy
- **ResNet1D**: Advanced residual network with 96.20% accuracy

## About

Part of the AIRE 2025 Internship Project on AI-Driven Polymer Aging Prediction and Classification.

**Author**: Jaser Hasan  
**Mentor**: Dr. Sanmukh Kuppannagari
```

### Copy Model Files

Copy your model architecture files:

```bash
# Copy model definitions
cp -r ../models/ ./models/

# Copy preprocessing utilities
mkdir utils
cp ../scripts/preprocess_dataset.py ./utils/preprocessing.py
```

### Create Model Utilities

Create `utils/__init__.py`:

```python
"""Utility functions for the polymer classification app"""
from .preprocessing import resample_spectrum

__all__ = ['resample_spectrum']
```

---

## 🚀 Step 6: Deploying to Hugging Face Spaces

### Method 1: Web Interface (Recommended for Beginners)

1. **Go to Hugging Face Spaces**: Visit [huggingface.co/spaces](https://huggingface.co/spaces)

2. **Create New Space**:
   - Click "Create new Space"
   - Choose a name: `polymer-classification`
   - Select "Streamlit" as SDK
   - Choose visibility (Public recommended)

3. **Upload Files**:
   - Use the web interface to upload all your files
   - Maintain the directory structure

4. **Wait for Build**:
   - Hugging Face will automatically build your app
   - Monitor the build logs for any errors

### Method 2: Git-Based Deployment (Recommended for Advanced Users)

1. **Clone Your Space Repository**:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/polymer-classification
cd polymer-classification
```

2. **Copy Your Files**:
```bash
cp -r ../huggingface-space/* ./
```

3. **Commit and Push**:
```bash
git add .
git commit -m "Initial deployment of polymer classification app"
git push
```

### Method 3: Using Hugging Face Hub

Install the hub library:
```bash
pip install huggingface_hub
```

Use Python to upload:
```python
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create the space
create_repo(
    repo_id="polymer-classification",
    repo_type="space",
    space_sdk="streamlit"
)

# Upload files
api.upload_folder(
    folder_path="./huggingface-space",
    repo_id="YOUR_USERNAME/polymer-classification",
    repo_type="space"
)
```

---

## 🔧 Step 7: Configuration and Optimization

### Environment Variables

Set up configuration in your Space settings:

```bash
# In your Space settings, add these secrets
MODEL_CACHE_DIR=/tmp/models
TORCH_HOME=/tmp/torch
```

### Performance Optimization

#### Memory Management

```python
# Add to your app.py
import gc

@st.cache_resource
def cleanup_memory():
    """Clean up memory after inference"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### Model Caching

```python
@st.cache_resource
def load_all_models():
    """Load and cache all models at startup"""
    models = {}
    model_configs = {
        "figure2": {
            "class": Figure2CNN,
            "path": "model_weights/figure2_model.pth"
        },
        "resnet": {
            "class": ResNet1D, 
            "path": "model_weights/resnet_model.pth"
        }
    }
    
    for name, config in model_configs.items():
        models[name] = load_model(config["class"], config["path"])
    
    return models
```

### Resource Allocation

For CPU-only deployment:
```python
# Force CPU usage
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
```

---

## 🐛 Step 8: Troubleshooting Common Issues

### Build Failures

#### Issue: Dependencies Not Installing
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solution**: Use compatible versions
```txt
# requirements.txt
torch==1.13.1
torchvision==0.14.1
```

#### Issue: Import Errors
```
ModuleNotFoundError: No module named 'models'
```

**Solution**: Fix Python path
```python
import sys
import os
sys.path.append(os.path.dirname(__file__))
```

### Runtime Errors

#### Issue: Model Files Not Found
```
FileNotFoundError: model_weights/figure2_model.pth
```

**Solution**: Implement graceful fallbacks
```python
def load_model_safe(model_path):
    if not os.path.exists(model_path):
        st.warning(f"Model not found: {model_path}")
        return None
    return torch.load(model_path, map_location="cpu")
```

#### Issue: Memory Limitations
```
RuntimeError: CUDA out of memory
```

**Solution**: Optimize memory usage
```python
# Clear cache after each inference
with torch.no_grad():
    output = model(input_tensor)
    torch.cuda.empty_cache()  # If using GPU
```

### Performance Issues

#### Issue: Slow Loading Times

**Solutions**:
1. Use `@st.cache_resource` for model loading
2. Optimize model size
3. Use CPU-optimized models

#### Issue: Large File Sizes

**Solutions**:
1. Use Git LFS for model files
2. Compress model weights
3. Use quantized models

---

## 📊 Step 9: Testing and Validation

### Local Testing

Before deploying, test locally:

```bash
# Navigate to your space directory
cd huggingface-space

# Install dependencies
pip install -r requirements.txt

# Run Streamlit locally
streamlit run app.py
```

### Validation Checklist

- [ ] ✅ App loads without errors
- [ ] ✅ File upload works correctly
- [ ] ✅ Model inference produces results
- [ ] ✅ Visualizations render properly
- [ ] ✅ All buttons and interactions work
- [ ] ✅ Error handling works gracefully
- [ ] ✅ Sample data loads correctly

### Performance Testing

```python
import time

def test_inference_speed():
    """Test model inference speed"""
    start_time = time.time()
    
    # Your inference code here
    result = model(test_input)
    
    end_time = time.time()
    st.write(f"Inference time: {end_time - start_time:.2f} seconds")
```

---

## 🔄 Step 10: Maintenance and Updates

### Version Control Strategy

#### Development Workflow
1. Make changes locally
2. Test thoroughly
3. Push to Hugging Face Space repository
4. Monitor deployment logs

#### Rollback Strategy
```bash
# If something breaks, rollback
git revert HEAD
git push
```

### Monitoring Your Space

#### Usage Analytics
- Monitor Space usage in Hugging Face dashboard
- Track user engagement metrics
- Monitor error rates

#### Performance Monitoring
```python
# Add to your app
import streamlit as st

# Track usage
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0

st.session_state.usage_count += 1
st.sidebar.write(f"Predictions made: {st.session_state.usage_count}")
```

### Updating Models

#### Adding New Models
1. Train your new model
2. Add to `model_weights/` directory
3. Update model configuration
4. Test thoroughly
5. Deploy update

#### Model Versioning
```python
MODEL_VERSIONS = {
    "figure2_v1": "model_weights/figure2_v1.pth",
    "figure2_v2": "model_weights/figure2_v2.pth",
    "resnet_v1": "model_weights/resnet_v1.pth"
}
```

---

## 🌟 Step 11: Advanced Features

### Adding Authentication

For private spaces, add authentication:

```python
import streamlit_authenticator as stauth

# Configuration
config = {
    'credentials': {
        'usernames': {
            'jdoe': {
                'name': 'John Doe',
                'password': 'hashed_password_here'
            }
        }
    }
}

authenticator = stauth.Authenticate(
    config['credentials'],
    'polymer_classifier',
    'auth_key'
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Your app code here
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
else:
    st.warning('Please enter your username and password')
```

### Analytics Integration

Add usage analytics:

```python
import streamlit.components.v1 as components

# Google Analytics
ga_code = """
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
"""

components.html(ga_code, height=0)
```

### API Integration

Add API endpoints:

```python
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# Create FastAPI app
api = FastAPI()

@api.post("/predict")
async def predict_api(spectrum_data: dict):
    # Your prediction logic
    return {"prediction": "stable", "confidence": 0.95}

# Mount Streamlit app
app = WSGIMiddleware(streamlit_app)
```

---

## 📚 Step 12: Documentation and Community

### Creating Comprehensive Documentation

#### Model Card
Create a detailed model card:

```markdown
# Model Card: Polymer Classification Models

## Model Details
- **Model Type**: Convolutional Neural Network
- **Architecture**: Figure2CNN and ResNet1D
- **Input**: Raman spectroscopy data (500 points)
- **Output**: Binary classification (Stable/Weathered)

## Training Data
- **Dataset Size**: [Specify number of samples]
- **Data Source**: Raman spectroscopy measurements
- **Preprocessing**: Resampling, normalization, baseline correction

## Performance Metrics
- **Figure2CNN Accuracy**: 94.80% ± 6.30%
- **ResNet1D Accuracy**: 96.20% ± 4.10%
- **Validation Method**: 10-fold cross-validation

## Limitations
- Limited to Raman spectroscopy data
- Binary classification only
- Requires 500-point input format

## Ethical Considerations
- No known biases in spectroscopic data
- Open source for research transparency
- Educational and research use recommended
```

#### API Documentation

If you add API endpoints:

```markdown
# API Documentation

## Endpoints

### POST /predict
Predict polymer degradation state from spectrum data.

**Request Body:**
```json
{
    "spectrum": [float array of 500 points],
    "model": "figure2" | "resnet"
}
```

**Response:**
```json
{
    "prediction": "stable" | "weathered",
    "confidence": float,
    "processing_time": float
}
```
```

### Community Engagement

#### Promote Your Space
1. **Share on Social Media**: Twitter, LinkedIn, research communities
2. **Academic Conferences**: Present at relevant conferences
3. **Research Papers**: Include Space URL in publications
4. **GitHub README**: Add Space badge to your repository

#### Collect Feedback
```python
# Add feedback collection
feedback = st.text_area("Share your feedback:")
if st.button("Submit Feedback"):
    # Log feedback (consider using a simple form service)
    st.success("Thank you for your feedback!")
```

---

## ✅ Step 13: Final Checklist

### Pre-Deployment Checklist

- [ ] All dependencies in requirements.txt
- [ ] README.md properly formatted with metadata
- [ ] Model files uploaded and accessible
- [ ] Error handling implemented
- [ ] Local testing completed successfully
- [ ] Performance optimizations applied
- [ ] Documentation complete

### Post-Deployment Checklist

- [ ] Space loads correctly
- [ ] All features work as expected
- [ ] Performance is acceptable
- [ ] Error handling works in production
- [ ] Analytics/monitoring set up
- [ ] Community documentation published
- [ ] Feedback mechanism active

### Long-term Maintenance

- [ ] Regular model updates planned
- [ ] Performance monitoring active
- [ ] User feedback collection system
- [ ] Version control strategy implemented
- [ ] Backup and recovery plan
- [ ] Security updates scheduled

---

## 🎓 Conclusion

Congratulations! You've successfully migrated your Streamlit polymer classification app to Hugging Face Spaces. Your AI tool is now accessible to researchers worldwide, contributing to the advancement of materials science and sustainability research.

### What You've Accomplished

✅ **Professional Deployment**: Your app runs on enterprise-grade infrastructure
✅ **Global Accessibility**: Researchers worldwide can access your tool
✅ **Zero Maintenance**: Automatic updates and scaling
✅ **Community Impact**: Contributing to open science
✅ **Portfolio Enhancement**: Professional demonstration of your ML skills

### Next Steps

1. **Share Your Work**: Promote your Space in research communities
2. **Collect Data**: Analyze usage patterns and user feedback
3. **Iterate**: Improve based on user needs and new research
4. **Expand**: Add new models and features
5. **Collaborate**: Invite other researchers to contribute

### Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Model Deployment Guide](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [ML Model Documentation Best Practices](https://huggingface.co/docs/hub/model-cards)

### Support and Community

- **Hugging Face Community**: [huggingface.co/join/discord](https://huggingface.co/join/discord)
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io/)
- **GitHub Issues**: Use for technical problems
- **Research Collaboration**: Reach out to the authors for academic partnerships

---

**Happy Deploying! 🚀**

*This tutorial was created as part of the AIRE 2025 Internship Project on AI-Driven Polymer Aging Prediction and Classification.*