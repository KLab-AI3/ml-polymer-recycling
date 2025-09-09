import os

# --- New Imports ---
from config import TARGET_LEN
import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from pathlib import Path
from config import SAMPLE_DATA_DIR
from models.registry import build, choices


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
def load_state_dict(mtime, model_path):
    """Load state dict with mtime in cache key to detect file changes"""
    try:
        return torch.load(model_path, map_location="cpu")
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"Error loading state dict: {e}")
        return None


@st.cache_resource
def load_model(model_name):
    # First try registry system (new approach)
    if model_name in choices():
        # Use registry system
        model = build(model_name, TARGET_LEN)

        # Try to load weights from standard locations
        weight_paths = [
            f"model_weights/{model_name}_model.pth",
            f"outputs/{model_name}_model.pth",
            f"model_weights/{model_name}.pth",
            f"outputs/{model_name}.pth",
        ]

        weights_loaded = False
        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                try:
                    mtime = os.path.getmtime(weight_path)
                    state_dict = load_state_dict(mtime, weight_path)
                    if state_dict:
                        model.load_state_dict(state_dict, strict=True)
                        model.eval()
                        weights_loaded = True
                        break  # Exit loop after successful load

                except (OSError, RuntimeError):
                    continue

        if not weights_loaded:
            st.warning(
                f"⚠️ Model weights not found for '{model_name}'. Using randomly initialized model."
            )
            st.info(
                "This model will provide random predictions for demonstration purposes."
            )

        return model, weights_loaded

    # If model not in registry, raise error
    st.error(f"Unknown model '{model_name}'. Available models: {choices()}")
    return None, False


def cleanup_memory():
    """Clean up memory after inference"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_data
def run_inference(y_resampled, model_choice, modality: str, cache_key=None):
    """Run model inference and cache results with performance tracking"""
    from utils.performance_tracker import get_performance_tracker, PerformanceMetrics
    from datetime import datetime

    model, model_loaded = load_model(model_choice)
    if not model_loaded:
        return None, None, None, None, None

    # Performance tracking setup
    tracker = get_performance_tracker()

    input_tensor = (
        torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

    # Track inference performance
    start_time = time.time()
    start_memory = _get_memory_usage()

    model.eval()  # type: ignore
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
    end_memory = _get_memory_usage()
    memory_usage = max(end_memory - start_memory, 0)

    # Log performance metrics
    try:
        confidence = float(max(probs)) if probs is not None and len(probs) > 0 else 0.0

        metrics = PerformanceMetrics(
            model_name=model_choice,
            prediction_time=inference_time,
            preprocessing_time=0.0,  # Will be updated by calling function if available
            total_time=inference_time,
            memory_usage_mb=memory_usage,
            accuracy=None,  # Will be updated if ground truth is available
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            input_size=(
                len(y_resampled) if hasattr(y_resampled, "__len__") else TARGET_LEN
            ),
            modality=modality,
        )

        tracker.log_performance(metrics)
    except (AttributeError, ValueError, KeyError) as e:
        # Don't fail inference if performance tracking fails
        print(f"Performance tracking failed: {e}")

    cleanup_memory()
    return prediction, logits_list, probs, inference_time, logits


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0  # psutil not available


@st.cache_data
def get_sample_files():
    """Get list of sample files if available"""
    sample_dir = Path(SAMPLE_DATA_DIR)
    if sample_dir.exists():
        return sorted(list(sample_dir.glob("*.txt")))
    return []
