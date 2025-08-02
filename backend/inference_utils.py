def load_model(name):
    return "mock_model"

def run_inference(model, spectrum):
    return {
        "prediction": "Stubbed Output",
        "class_index": 0,
        "logits": [0.0, 1.0],
        "class_labels": ["Stub", "Output"]
    }


# ---------- ACTUAL MODEL LOADING/INFERENCE CODE ---------------------|
# import torch
# import numpy as np
# from pathlib import Path 
# from scripts.preprocess_dataset import resample_spectrum
# from models.figure2_cnn import Figure2CNN
# from models.resnet_cnn import ResNet1D

# # -- Label Map --
# LABELS = ["Stable (Unweathered)", "Weathered (Degraded)"]

# # -- Model Paths --
# MODEL_CONFIG = {
#     "figure2": {
#         "class": Figure2CNN,
#         "path": "outputs/figure2_model.pth"
#     },
#     "resnet": {
#         "class": ResNet1D,
#         "path": "outputs/resnet_model.pth"
#     }
# }

# def load_model(model_name: str):
#     if model_name not in MODEL_CONFIG:
#         raise ValueError(f"Unknown model '{model_name}'. Valid options: {list(MODEL_CONFIG.keys())}")

#     config = MODEL_CONFIG[model_name]
#     model = config["class"]()
#     state_dict = torch.load(config["path"], map_location=torch.device("cpu"), weights_only=True)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# def run_inference(model, spectrum: list):
#     # -- Validate Input --
#     if not isinstance(spectrum, list) or len(spectrum) < 10:
#         raise ValueError("Spectrum must be a list of floats with reasonable length")

#     # -- Convert to Numpy --
#     spectrum = np.array(spectrum, dtype=np.float32)

#     # -- Resample --
#     x_vals = np.arange(len(spectrum))
#     spectrum = resample_spectrum(x_vals, spectrum, target_len=500)

#     # -- Normalize --
#     mean = np.mean(spectrum)
#     std = np.std(spectrum)
#     if std == 0:
#         raise ValueError("Standard deviation of spectrum is zero; normalization will fail.")
#     spectrum = (spectrum - mean) / std

#     # -- To Tensor --
#     x = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0)   # Shape (1, 1, 500)

#     with torch.no_grad():
#         logits = model(x)
#         pred_index = torch.argmax(logits, dim=1).item()

#     return {
#         "prediction": LABELS[pred_index],
#         "class_index": pred_index,
#         "logits": logits.squeeze().tolist(),
#         "class_labels": LABELS
#     }
# ---------- ACTUAL MODEL LOADING/INFERENCE CODE ---------------------|