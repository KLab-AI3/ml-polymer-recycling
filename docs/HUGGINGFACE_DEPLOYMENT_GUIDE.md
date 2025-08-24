# 🚀 Quick Deployment Guide for Hugging Face Spaces

## Overview
This guide provides step-by-step instructions for deploying the polymer classification app to Hugging Face Spaces using the prepared files in the `huggingface-space/` directory.

## 📋 Prerequisites
- Hugging Face account ([sign up here](https://huggingface.co/join))
- Git installed on your local machine
- Basic familiarity with command line operations

## 🗂️ Prepared Files Structure

The `huggingface-space/` directory contains everything needed for deployment:

```
huggingface-space/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Space description with metadata
├── .gitignore               # Git ignore patterns
├── models/                  # Model architecture files
│   ├── __init__.py
│   ├── figure2_cnn.py
│   └── resnet_cnn.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── preprocessing.py
├── sample_data/             # Sample spectra for testing
│   ├── sta_sample_01.txt    # Stable polymer sample
│   └── wea_sample_01.txt    # Weathered polymer sample
└── model_weights/           # Directory for trained model files
    └── (model .pth files go here)
```

## 🔧 Method 1: Web Interface Deployment (Recommended for Beginners)

### Step 1: Create New Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `polymer-classification` (or your preferred name)
   - **License**: Apache 2.0
   - **SDK**: Streamlit
   - **Visibility**: Public (recommended)
4. Click **"Create Space"**

### Step 2: Upload Files
1. In your new Space, click **"Upload files"**
2. Drag and drop all files from `huggingface-space/` directory
3. Maintain the directory structure when uploading
4. For large model files (.pth), use Git LFS (see below)

### Step 3: Wait for Build
1. Hugging Face automatically builds your app
2. Monitor the **"Settings" > "Logs"** for build progress
3. The app will be live once the build completes

## 🐙 Method 2: Git-Based Deployment (Recommended for Advanced Users)

### Step 1: Clone Your Space Repository
```bash
# Replace YOUR_USERNAME with your Hugging Face username
git clone https://huggingface.co/spaces/YOUR_USERNAME/polymer-classification
cd polymer-classification
```

### Step 2: Copy Prepared Files
```bash
# Copy all files from the prepared directory
cp -r ../path/to/huggingface-space/* ./

# If you're in the ml-polymer-recycling repository:
cp -r huggingface-space/* ./
```

### Step 3: Add Model Weights (Optional)
```bash
# If you have trained model files, add them to model_weights/
# For large files, use Git LFS:
git lfs track "*.pth"
git add .gitattributes

# Add your model files
cp path/to/your/figure2_model.pth model_weights/
cp path/to/your/resnet_model.pth model_weights/
```

### Step 4: Commit and Push
```bash
git add .
git commit -m "Deploy polymer classification app to Hugging Face Spaces"
git push
```

## 📦 Method 3: Using Hugging Face Hub Python Library

### Step 1: Install Hub Library
```bash
pip install huggingface_hub
```

### Step 2: Login and Upload
```python
from huggingface_hub import HfApi, create_repo, login

# Login (will prompt for token)
login()

# Create the space
create_repo(
    repo_id="polymer-classification",
    repo_type="space",
    space_sdk="streamlit",
    exist_ok=True
)

# Upload files
api = HfApi()
api.upload_folder(
    folder_path="./huggingface-space",
    repo_id="YOUR_USERNAME/polymer-classification",
    repo_type="space"
)
```

## 🏋️ Handling Large Model Files

### Using Git LFS (Large File Storage)

If you have trained model weights (`.pth` files):

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git add .gitattributes

# Add your model files
git add model_weights/figure2_model.pth
git add model_weights/resnet_model.pth

# Commit and push
git commit -m "Add trained model weights"
git push
```

### Alternative: Hugging Face Model Hub

For very large models, consider hosting on Hugging Face Model Hub:

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model to Model Hub
api.upload_file(
    path_or_fileobj="path/to/figure2_model.pth",
    path_in_repo="figure2_model.pth",
    repo_id="YOUR_USERNAME/polymer-models",
    repo_type="model"
)

# Then modify app.py to download from Model Hub
```

## 🧪 Testing Your Deployment

### Local Testing Before Deployment
```bash
# Navigate to huggingface-space directory
cd huggingface-space

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Post-Deployment Testing
1. Wait for the Space to build (check logs)
2. Visit your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/polymer-classification`
3. Test all features:
   - Model selection
   - File upload
   - Sample data
   - Inference functionality

## 🐛 Common Issues and Solutions

### Build Failures

**Issue**: `ERROR: Could not find a version that satisfies the requirement torch>=1.13.0`

**Solution**: Update `requirements.txt` with compatible versions:
```txt
torch==1.13.1+cpu
torchvision==0.14.1+cpu
```

**Issue**: `ModuleNotFoundError: No module named 'models'`

**Solution**: Ensure all `__init__.py` files are present and the directory structure is correct.

### Runtime Issues

**Issue**: Model files not found

**Solution**: The app is designed to work without model weights (demo mode). To add weights:
1. Upload `.pth` files to `model_weights/` directory
2. Use Git LFS for large files
3. Or modify app to download from external source

**Issue**: Memory errors

**Solution**: Add to `requirements.txt`:
```txt
torch==1.13.1+cpu
```

## 🔄 Updating Your Space

### Method 1: Git Push
```bash
# Make changes locally
# Test changes
git add .
git commit -m "Update: description of changes"
git push
```

### Method 2: Web Interface
1. Go to your Space
2. Edit files directly in the web interface
3. Commit changes

## 📊 Monitoring and Analytics

### Built-in Metrics
- View usage stats in Space settings
- Monitor performance in logs
- Track user engagement

### Custom Analytics
Add to your `app.py`:
```python
# Track usage
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0

st.session_state.usage_count += 1
```

## 🔒 Privacy and Security

### Public vs Private Spaces
- **Public**: Accessible to everyone, discoverable
- **Private**: Requires authentication, not discoverable

### Environment Variables
Set in Space settings for sensitive data:
```bash
API_KEY=your_secret_key
MODEL_PATH=/custom/path
```

## 🎯 Optimization Tips

### Performance
- Use `@st.cache_resource` for model loading
- Implement memory cleanup after inference
- Optimize image sizes and plots

### User Experience
- Add loading spinners for long operations
- Provide clear error messages
- Include help text and tooltips

### SEO and Discovery
- Use descriptive Space name and description
- Add relevant tags in README.md metadata
- Share on social media and research communities

## ✅ Final Checklist

Before going live:
- [ ] All files uploaded correctly
- [ ] App builds without errors
- [ ] All features work as expected
- [ ] Documentation is clear and complete
- [ ] Model files uploaded (if available)
- [ ] Space description is professional
- [ ] Privacy settings configured correctly

## 🆘 Getting Help

### Resources
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Community Discord](https://huggingface.co/join/discord)

### Support Channels
- Hugging Face Community Forums
- GitHub Issues on the original repository
- Streamlit Community Forum

---

**You're ready to deploy! 🚀**

Your polymer classification app will be accessible worldwide, contributing to materials science research and sustainability efforts.