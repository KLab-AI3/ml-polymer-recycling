from pathlib import Path
import os

KEEP_KEYS = {
    # ==global UI context we want to keep after "Reset"==
    "model_select",  # sidebar model key
    "input_mode",  # radio for Upload|Sample
    "uploader_version",  # version counter for file uploader
    "input_registry",  # radio controlling Upload vs Sample
}

TARGET_LEN = 500
SAMPLE_DATA_DIR = Path("sample_data")

MODEL_WEIGHTS_DIR = os.getenv("WEIGHTS_DIR") or (
    "model_weights" if os.path.isdir("model_weights") else "outputs"
)

# ==Label mapping==
LABEL_MAP = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}
