"""
Preprocessing utilities for polymer classification app.
Adapted from the original scripts/preprocess_dataset.py for Hugging Face Spaces deployment.
Supports both Raman and FTIR spectroscopy modalities.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import DTypeLike
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from typing import Tuple, Literal, Optional

TARGET_LENGTH = 500  # Frozen default per PREPROCESSING_BASELINE

# Modality-specific validation ranges (cm⁻¹)
MODALITY_RANGES = {
    "raman": (200, 4000),  # Typical Raman range
    "ftir": (400, 4000),  # FTIR wavenumber range
}

# Modality-specific preprocessing parameters
MODALITY_PARAMS = {
    "raman": {
        "baseline_degree": 2,
        "smooth_window": 11,
        "smooth_polyorder": 2,
        "cosmic_ray_removal": False,
    },
    "ftir": {
        "baseline_degree": 2,
        "smooth_window": 13,  # Slightly larger window for FTIR
        "smooth_polyorder": 2,
        "cosmic_ray_removal": False,
        "atmospheric_correction": False,  # Placeholder for future implementation
    },
}


def _ensure_1d_equal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y must be 1D arrays of equal length >= 2")
    return x, y


def resample_spectrum(
    x: np.ndarray, y: np.ndarray, target_len: int = TARGET_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
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


def smooth_spectrum(
    y: np.ndarray, window_length: int = 11, polyorder: int = 2
) -> np.ndarray:
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
    return savgol_filter(
        y, window_length=window_length, polyorder=polyorder, mode="interp"
    )


def normalize_spectrum(y: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] with constant-signal guard."""
    y = np.asarray(y, dtype=float)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if np.isclose(y_max - y_min, 0.0):
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)


def validate_spectrum_range(x: np.ndarray, modality: str = "raman") -> bool:
    """Validate that spectrum wavenumbers are within expected range for modality."""
    if modality not in MODALITY_RANGES:
        raise ValueError(
            f"Unknown modality '{modality}'. Supported: {list(MODALITY_RANGES.keys())}"
        )

    min_range, max_range = MODALITY_RANGES[modality]
    x_min, x_max = np.min(x), np.max(x)

    # Check if majority of data points are within range
    in_range = np.sum((x >= min_range) & (x <= max_range))
    total_points = len(x)

    return bool((in_range / total_points) >= 0.7)  # At least 70% should be in range


def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_len: int = TARGET_LENGTH,
    modality: str = "raman",  # New parameter for modality-specific processing
    do_baseline: bool = True,
    degree: int | None = None,  # Will use modality default if None
    do_smooth: bool = True,
    window_length: int | None = None,  # Will use modality default if None
    polyorder: int | None = None,  # Will use modality default if None
    do_normalize: bool = True,
    out_dtype: DTypeLike = np.float32,
    validate_range: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modality-aware preprocessing: resample -> baseline -> smooth -> normalize

    Args:
        x, y: Input spectrum data
        target_len: Target length for resampling
        modality: 'raman' or 'ftir' for modality-specific processing
        do_baseline: Enable baseline correction
        degree: Polynomial degree for baseline (uses modality default if None)
        do_smooth: Enable smoothing
        window_length: Smoothing window length (uses modality default if None)
        polyorder: Polynomial order for smoothing (uses modality default if None)
        do_normalize: Enable normalization
        out_dtype: Output data type
        validate_range: Check if wavenumbers are in expected range for modality

    Returns:
        Tuple of (resampled_x, processed_y)
    """
    # Validate modality
    if modality not in MODALITY_PARAMS:
        raise ValueError(
            f"Unsupported modality '{modality}'. Supported: {list(MODALITY_PARAMS.keys())}"
        )

    # Get modality-specific parameters
    modality_config = MODALITY_PARAMS[modality]

    # Use modality defaults if parameters not specified
    if degree is None:
        degree = modality_config["baseline_degree"]
    if window_length is None:
        window_length = modality_config["smooth_window"]
    if polyorder is None:
        polyorder = modality_config["smooth_polyorder"]

    # Validate spectrum range if requested
    if validate_range:
        if not validate_spectrum_range(x, modality):
            print(
                f"Warning: Spectrum wavenumbers may not be optimal for {modality.upper()} analysis"
            )

    # Standard preprocessing pipeline
    x_rs, y_rs = resample_spectrum(x, y, target_len=target_len)

    if do_baseline:
        y_rs = remove_baseline(y_rs, degree=degree)

    if do_smooth:
        y_rs = smooth_spectrum(y_rs, window_length=window_length, polyorder=polyorder)

    # FTIR-specific processing
    if modality == "ftir":
        if modality_config.get("atmospheric_correction", False):
            y_rs = remove_atmospheric_interference(y_rs)
        if modality_config.get("water_correction", False):
            y_rs = remove_water_vapor_bands(y_rs, x_rs)

    if do_normalize:
        y_rs = normalize_spectrum(y_rs)

    # === Coerce to a real dtype to satisfy static checkers & runtime ===
    out_dt = np.dtype(out_dtype)
    return x_rs.astype(out_dt, copy=False), y_rs.astype(out_dt, copy=False)


def remove_atmospheric_interference(y: np.ndarray) -> np.ndarray:
    """Remove atmospheric CO2 and H2O interference common in FTIR."""
    y = np.asarray(y, dtype=float)

    # Simple atmospheric correction using median filtering
    # This is a basic implementation - in practice would use reference spectra
    from scipy.signal import medfilt

    # Apply median filter to reduce sharp atmospheric lines
    corrected = medfilt(y, kernel_size=5)

    # Blend with original to preserve peak structure
    alpha = 0.7  # Weight for original spectrum
    return alpha * y + (1 - alpha) * corrected


def remove_water_vapor_bands(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove water vapor interference bands in FTIR spectra."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # Common water vapor regions in FTIR (cm⁻¹)
    water_regions = [(3500, 3800), (1300, 1800)]

    corrected_y = y.copy()

    for low, high in water_regions:
        # Find indices in water vapor region
        mask = (x >= low) & (x <= high)
        if np.any(mask):
            # Simple linear interpolation across water regions
            indices = np.where(mask)[0]
            if len(indices) > 2:
                start_idx, end_idx = indices[0], indices[-1]
                if start_idx > 0 and end_idx < len(y) - 1:
                    # Linear interpolation between boundary points
                    start_val = y[start_idx - 1]
                    end_val = y[end_idx + 1]
                    interp_vals = np.linspace(start_val, end_val, len(indices))
                    corrected_y[mask] = interp_vals

    return corrected_y


def apply_ftir_specific_processing(
    x: np.ndarray,
    y: np.ndarray,
    atmospheric_correction: bool = False,
    water_correction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FTIR-specific preprocessing steps."""
    processed_y = y.copy()

    if atmospheric_correction:
        processed_y = remove_atmospheric_interference(processed_y)

    if water_correction:
        processed_y = remove_water_vapor_bands(processed_y, x)

    return x, processed_y


def get_modality_info(modality: str) -> dict:
    """Get processing parameters and validation ranges for a modality."""
    if modality not in MODALITY_PARAMS:
        raise ValueError(f"Unknown modality '{modality}'")

    return {
        "range": MODALITY_RANGES[modality],
        "params": MODALITY_PARAMS[modality].copy(),
    }
