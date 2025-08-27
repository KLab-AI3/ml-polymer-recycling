"""preprocess_dataset.py

Canonical Raman preprocessing for dataset splits.
Uses the single source of truth in utils.preprocessing:
resample → baseline (deg=2) → smooth (w=11,o=2) → normalize.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from utils.preprocessing import (
    TARGET_LENGTH,
    preprocess_spectrum
)

from scripts.discover_raman_files import list_txt_files, label_file
from scripts.plot_spectrum import load_spectrum

def preprocess_dataset(
    dataset_dir: str,
    target_len: int = TARGET_LENGTH,
    baseline_correction: bool = True,
    apply_smoothing: bool = True,
    normalize: bool = True,
    out_dtype: str = "float32",
):
    """
    Load, preprocess, and label Raman spectra in dataset_dir.

    Returns
    -------
    X : np.ndarray, shape (N, target_len), dtype=out_dtype
        Preprocessed spectra (resampled and transformed).
    y : np.ndarray, shape (N,), dtype=int64
        Integer labels (e.g., 0 = stable, 1 = weathered).
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


        # === Single-source-of-truth path ===
        _, y_processed = preprocess_spectrum(
            np.asarray(x_raw), 
            np.asarray(y_raw), 
            target_len=target_len,
            do_baseline=baseline_correction,
            do_smooth=apply_smoothing,
            do_normalize=normalize,
            out_dtype=out_dtype # str is OK (DTypeLike),
        )
        
        # === Collect ===
        X.append(y_processed)
        y_labels.append(int(label))

    if not X:
        # === No valid samples ===
        return np.empty((0, target_len), dtype=out_dtype), np.empty((0,), dtype=np.int64)

    X_arr = np.asarray(X, dtype=np.dtype(out_dtype))
    Y_arr = np.asarray(y_labels, dtype=np.int64)

    return X_arr, Y_arr

# === Optional: Run directly for quick smoke test ===
if __name__ == "__main__":
    test_dataset_dir = os.path.join("datasets", "rdwp")
    X, y = preprocess_dataset(test_dataset_dir)

    print(f"X shape: {X.shape} dtype={X.dtype}")
    print(f"y shape: {y.shape} dtype={y.dtype}")
    if y.size:
        try:
            counts = np.bincount(y, minlength=2)
            print(f"Label distribution: {counts} (stable, weathered)")
        except Exception as e:
            print(f"Could not compute label distribution {e}")
