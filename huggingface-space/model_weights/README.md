# Model Weights Directory

This directory is intended to contain the trained PyTorch model weights (.pth files) for the polymer classification models.

## Expected Files

- `figure2_model.pth` - Weights for the Figure2CNN baseline model
- `resnet_model.pth` - Weights for the ResNet1D advanced model

## How to Add Model Weights

### Option 1: Copy from Existing Training Output
If you have trained models in the main repository:
```bash
cp ../outputs/figure2_model.pth ./figure2_model.pth
cp ../outputs/resnet_model.pth ./resnet_model.pth
```

### Option 2: Download from External Source
If models are hosted elsewhere, download them here:
```bash
wget https://your-model-host.com/figure2_model.pth
wget https://your-model-host.com/resnet_model.pth
```

### Option 3: Use Git LFS (Large File Storage)
For version control of large model files:
```bash
git lfs track "*.pth"
git add .gitattributes
git add figure2_model.pth resnet_model.pth
git commit -m "Add model weights with LFS"
```

## Demo Mode

If no model weights are present, the app will run in demo mode with randomly initialized weights. This is useful for:
- Testing the interface
- Demonstrating the app structure
- Development without requiring large model files

## Model Information

- **Figure2CNN**: ~2-5 MB baseline CNN model
- **ResNet1D**: ~1-3 MB residual network model
- **Input Format**: PyTorch state dictionaries saved with `torch.save()`
- **Target Length**: 500 points (Raman spectra)

## Important Notes

- Model files are ignored by Git due to size constraints
- Hugging Face Spaces supports files up to 50MB
- For larger models, consider using Hugging Face Model Hub
- Always test models locally before deployment