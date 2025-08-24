"""
Preprocessing utilities for polymer classification app.
Adapted from the original scripts/preprocess_dataset.py for Hugging Face Spaces deployment.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import minmax_scale

# Default resample target
TARGET_LENGTH = 500

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
    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window_length < 3:
            return y
    return savgol_filter(y, window_length, polyorder)

def resample_spectrum(x, y, target_len=TARGET_LENGTH):
    """
    Resample a spectrum to a fixed number of points using linear interpolation.
    
    Args:
        x (array-like): Wavenumber values
        y (array-like): Intensity values  
        target_len (int): Target number of points
        
    Returns:
        np.ndarray: Resampled intensity values
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Check for valid input
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} vs {len(y)}")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 points for interpolation")
    
    # Sort by x values to ensure monotonic order
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate x values
    if len(np.unique(x_sorted)) != len(x_sorted):
        # Remove duplicates by averaging y values for same x
        x_unique, inverse_indices = np.unique(x_sorted, return_inverse=True)
        y_unique = np.zeros_like(x_unique, dtype=float)
        for i in range(len(x_unique)):
            mask = inverse_indices == i
            y_unique[i] = np.mean(y_sorted[mask])
        x_sorted, y_sorted = x_unique, y_unique
    
    # Create interpolation function
    f_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    
    # Generate uniform grid
    x_uniform = np.linspace(min(x_sorted), max(x_sorted), target_len)
    y_uniform = f_interp(x_uniform)
    
    return y_uniform

def preprocess_spectrum(x, y, target_len=500, baseline_correction=False, 
                       apply_smoothing=False, normalize=False):
    """
    Complete preprocessing pipeline for a single spectrum.
    
    Args:
        x (array-like): Wavenumber values
        y (array-like): Intensity values
        target_len (int): Number of points to resample to
        baseline_correction (bool): Whether to apply baseline removal
        apply_smoothing (bool): Whether to apply Savitzky-Golay smoothing
        normalize (bool): Whether to apply min-max normalization
        
    Returns:
        np.ndarray: Preprocessed spectrum
    """
    # Resample first
    y_processed = resample_spectrum(x, y, target_len=target_len)
    
    # Optional preprocessing steps
    if baseline_correction:
        y_processed = remove_baseline(y_processed)
    
    if apply_smoothing:
        y_processed = smooth_spectrum(y_processed)
    
    if normalize:
        y_processed = normalize_spectrum(y_processed)
    
    return y_processed