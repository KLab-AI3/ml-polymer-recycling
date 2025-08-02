"""
This script preprocesses a dataset of spectra by resampling and labeling the data.

Functions:
- resample_spectrum(x, y, target_len): Resamples a spectrum to a fixed number of points.
- preprocess_dataset(...): Loads, resamples, and applies optional preprocessing steps:
  - baseline correction
  - Savitzky-Golay smoothing
  - min-max normalization

The script expects the dataset directory to contain text files representing spectra.
Each file is:
1. Listed using `list_txt_files()`
2. Labeled using `label_file()`
3. Loaded using `load_spectrum()`
4. Resampled and optionally cleaned
5. Returned as arrays suitable for ML training

Dependencies:
- numpy
- scipy.interpolate, scipy.signal
- sklearn.preprocessing
- list_spectra (custom)
- plot_spectrum (custom)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import minmax_scale
from scripts.discover_raman_files import list_txt_files, label_file
from scripts.plot_spectrum import load_spectrum

# Default resample target
TARGET_LENGTH = 500

# Optional preprocessing steps
def remove_baseline(y):
    """Simple baseline correction using polynomial fitting (order 2)"""
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, deg=2)
    baseline = np.polyval(coeffs, x)
    return y - baseline

def normalize_spectrum(y):
    """Min-max normalization to [0, 1]"""
    return minmax_scale(y)

def smooth_spectrum(y, window_length=11, polyorder=2):
    """Apply Savitzky-Golay smoothing."""
    return savgol_filter(y, window_length, polyorder)

def resample_spectrum(x, y, target_len=TARGET_LENGTH):
    """Resample a spectrum to a fixed number of points."""
    f_interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
    x_uniform = np.linspace(min(x), max(x), target_len)
    y_uniform = f_interp(x_uniform)
    return y_uniform

def preprocess_dataset(
    dataset_dir,
    target_len=500,
    baseline_correction=False,
    apply_smoothing=False,
    normalize=False
):
    """
    Load, resample, and preprocess all valid spectra in the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset
        target_len (int): Number of points to resample to
        baseline_correction (bool): Whether to apply baseline removal
        apply_smoothing (bool): Whether to apply Savitzky-Golay smoothing
        normalize (bool): Whether to apply min-max normalization

    Returns:
        X (np.ndarray): Preprocessed spectra
        y (np.ndarray): Corresponding labels
    """
    txt_paths = list_txt_files(dataset_dir)
    X, y_labels = [], []

    for path in txt_paths:
        label = label_file(path)
        if label is None:
            continue

        x_raw, y_raw = load_spectrum(path)
        if len(x_raw) < 10:
            continue  # Skip files with too few points

        # Resample
        y_processed = resample_spectrum(x_raw, y_raw, target_len=target_len)

        # Optional preprocessing
        if baseline_correction:
            y_processed = remove_baseline(y_processed)
        if apply_smoothing:
            y_processed = smooth_spectrum(y_processed)
        if normalize:
            y_processed = normalize_spectrum(y_processed)

        X.append(y_processed)
        y_labels.append(label)

    return np.array(X), np.array(y_labels)

# Optional: Run directly for testing
if __name__ == "__main__":
    dataset_dir = os.path.join(
        "datasets", "rdwp"
    )
    X, y = preprocess_dataset(dataset_dir)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
