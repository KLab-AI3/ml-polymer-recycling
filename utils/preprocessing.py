"""
Preprocessing utilities for polymer classification app.
Adapted from the original scripts/preprocess_dataset.py for Hugging Face Spaces deployment.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import DTypeLike
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

TARGET_LENGTH = 500     # Frozen default per PREPROCESSING_BASELINE

def _ensure_1d_equal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y must be 1D arrays of equal length >= 2")
    return x, y

def resample_spectrum(x: np.ndarray, y: np.ndarray, target_len: int = TARGET_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Linear re-sampling onto a uniform grid of length target_len."""
    x, y = _ensure_1d_equal(x, y)
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    x_new = np.linspace(x_sorted[0], x_sorted[-1], int(target_len))
    f = interp1d(x_sorted, y_sorted, kind="linear", assume_sorted=True)
    y_new = f(x_new)
    return x_new, y_new

def remove_baseline(y: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial baseline subtraction (degree=2 default)"""
    y = np.asarray(y, dtype=float)
    x_idx = np.arange(y.size, dtype=float)
    coeffs = np.polyfit(x_idx, y, deg=int(degree))
    baseline = np.polyval(coeffs, x_idx)
    return y - baseline

def smooth_spectrum(y: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing with safe/odd window enforcement"""
    y = np.asarray(y, dtype=float)
    window_length = int(window_length)
    polyorder = int(polyorder)
    # === window must be odd and >= polyorder+1 ===
    if window_length % 2 == 0:
        window_length += 1 
    min_win = polyorder + 1
    if min_win % 2 == 0:
        min_win += 1
    window_length = max(window_length, min_win)
    return savgol_filter(y, window_length=window_length, polyorder=polyorder, mode="interp")

def normalize_spectrum(y: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] with constant-signal guard."""
    y = np.asarray(y, dtype=float)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if np.isclose(y_max - y_min, 0.0):
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)

def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_len: int = TARGET_LENGTH,
    do_baseline: bool = True,
    degree: int = 2,
    do_smooth: bool = True,
    window_length: int = 11,
    polyorder: int = 2,
    do_normalize: bool = True,
    out_dtype: DTypeLike = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact CLI baseline: resample -> baseline -> smooth -> normalize"""
    x_rs, y_rs = resample_spectrum(x, y, target_len=target_len)
    if do_baseline:
        y_rs = remove_baseline(y_rs, degree=degree)
    if do_smooth:
        y_rs = smooth_spectrum(y_rs, window_length=window_length, polyorder=polyorder)
    if do_normalize:
        y_rs = normalize_spectrum(y_rs)
    # === Coerce to a real dtype to satisfy static checkers & runtime ===
    out_dt = np.dtype(out_dtype)
    return x_rs.astype(out_dt, copy=False), y_rs.astype(out_dt, copy=False)