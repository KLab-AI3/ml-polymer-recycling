from pathlib import Path
import os
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D

KEEP_KEYS = {
    # ==global UI context we want to keep after "Reset"==
    "model_select",     # sidebar model key
    "input_mode",       # radio for Upload|Sample
    "uploader_version",  # version counter for file uploader
    "input_registry",   # radio controlling Upload vs Sample
}

TARGET_LEN = 500
SAMPLE_DATA_DIR = Path("sample_data")

MODEL_WEIGHTS_DIR = (
    os.getenv("WEIGHTS_DIR")
    or ("model_weights" if os.path.isdir("model_weights") else "outputs")
)

# Model configuration
MODEL_CONFIG = {
    "Figure2CNN (Baseline)": {
        "class": Figure2CNN,
        "path": f"{MODEL_WEIGHTS_DIR}/figure2_model.pth",
        "emoji": "",
        "description": "Baseline CNN with standard filters",
        "accuracy": "94.80%",
        "f1": "94.30%"
    },
    "ResNet1D (Advanced)": {
        "class": ResNet1D,
        "path": f"{MODEL_WEIGHTS_DIR}/resnet_model.pth",
        "emoji": "",
        "description": "Residual CNN with deeper feature learning",
        "accuracy": "96.20%",
        "f1": "95.90%"
    }
}

# ==Label mapping==
LABEL_MAP = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}
