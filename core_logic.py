import os

# --- New Imports ---
from config import MODEL_CONFIG, TARGET_LEN
import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from pathlib import Path
from config import SAMPLE_DATA_DIR


def label_file(filename: str) -> int:
    """Extract label from filename based on naming convention"""
    name = Path(filename).name.lower()
    if name.startswith("sta"):
        return 0
    elif name.startswith("wea"):
        return 1
    else:
        # Return None for unknown patterns instead of raising error
        return -1  # Default value for unknown patterns


@st.cache_data
def load_state_dict(_mtime, model_path):
    """Load state dict with mtime in cache key to detect file changes"""
    try:
        return torch.load(model_path, map_location="cpu")
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"Error loading state dict: {e}")
        return None


@st.cache_resource
def load_model(model_name):
    """Load and cache the specified model with error handling"""
    try:
        config = MODEL_CONFIG[model_name]
        model_class = config["class"]
        model_path = config["path"]

        # Initialize model
        model = model_class(input_length=TARGET_LEN)

        # Check if model file exists
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model weights not found: {model_path}")
            st.info("Using randomly initialized model for demonstration purposes.")
            return model, False

        # Get mtime for cache invalidation
        mtime = os.path.getmtime(model_path)

        # Load weights
        state_dict = load_state_dict(mtime, model_path)
        if state_dict:
            model.load_state_dict(state_dict, strict=True)
            if model is None:
                raise ValueError(
                    "Model is not loaded. Please check the model configuration or weights."
                )
            if model is None:
                raise ValueError(
                    "Model is not loaded. Please check the model configuration or weights."
                )
            if model is None:
                raise ValueError(
                    "Model is not loaded. Please check the model configuration or weights."
                )
            model.eval()
            return model, True
        else:
            return model, False

    except (FileNotFoundError, KeyError, RuntimeError) as e:
        st.error(f"❌ Error loading model {model_name}: {str(e)}")
        return None, False


def cleanup_memory():
    """Clean up memory after inference"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_data
def run_inference(y_resampled, model_choice, _cache_key=None):
    """Run model inference and cache results"""
    model, model_loaded = load_model(model_choice)
    if not model_loaded:
        return None, None, None, None, None

    input_tensor = (
        torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        if model is None:
            raise ValueError(
                "Model is not loaded. Please check the model configuration or weights."
            )
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
        logits_list = logits.detach().numpy().tolist()[0]
        probs = F.softmax(logits.detach(), dim=1).cpu().numpy().flatten()
    inference_time = time.time() - start_time
    cleanup_memory()
    return prediction, logits_list, probs, inference_time, logits


@st.cache_data
def get_sample_files():
    """Get list of sample files if available"""
    sample_dir = Path(SAMPLE_DATA_DIR)
    if sample_dir.exists():
        return sorted(list(sample_dir.glob("*.txt")))
    return []


def parse_spectrum_data(raw_text):
    """Parse spectrum data from text with robust error handling and validation"""
    x_vals, y_vals = [], []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        try:
            # Handle different separators
            parts = line.replace(",", " ").split()
            numbers = [
                p
                for p in parts
                if p.replace(".", "", 1)
                .replace("-", "", 1)
                .replace("+", "", 1)
                .isdigit()
            ]

            if len(numbers) >= 2:
                x, y = float(numbers[0]), float(numbers[1])
                x_vals.append(x)
                y_vals.append(y)

        except ValueError:
            # Skip problematic lines but don't fail completely
            continue

    if len(x_vals) < 10:  # Minimum reasonable spectrum length
        raise ValueError(
            f"Insufficient data points: {len(x_vals)}. Need at least 10 points."
        )

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Check for NaNs
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")

    # Check monotonic increasing x
    if not np.all(np.diff(x) > 0):
        raise ValueError("Wavenumbers must be strictly increasing")

    # Check reasonable range for Raman spectroscopy
    if min(x) < 0 or max(x) > 10000 or (max(x) - min(x)) < 100:
        raise ValueError(
            f"Invalid wavenumber range: {min(x)} - {max(x)}. Expected ~400-4000 cm⁻¹ with span >100"
        )

    return x, y
