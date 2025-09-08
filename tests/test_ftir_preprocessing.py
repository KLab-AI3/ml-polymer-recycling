"""Tests for FTIR preprocessing functionality."""

import pytest
import numpy as np
from utils.preprocessing import (
    preprocess_spectrum,
    validate_spectrum_range,
    get_modality_info,
    MODALITY_RANGES,
    MODALITY_PARAMS,
)


def test_modality_ranges():
    """Test that modality ranges are correctly defined."""
    assert "raman" in MODALITY_RANGES
    assert "ftir" in MODALITY_RANGES

    raman_range = MODALITY_RANGES["raman"]
    ftir_range = MODALITY_RANGES["ftir"]

    assert raman_range[0] < raman_range[1]  # Valid range
    assert ftir_range[0] < ftir_range[1]  # Valid range
    assert ftir_range[0] >= 400  # FTIR starts at 400 cm⁻¹
    assert ftir_range[1] <= 4000  # FTIR ends at 4000 cm⁻¹


def test_validate_spectrum_range():
    """Test spectrum range validation for different modalities."""
    # Test Raman range validation
    raman_x = np.linspace(300, 3500, 100)  # Typical Raman range
    assert validate_spectrum_range(raman_x, "raman") == True

    # Test FTIR range validation
    ftir_x = np.linspace(500, 3800, 100)  # Typical FTIR range
    assert validate_spectrum_range(ftir_x, "ftir") == True

    # Test out-of-range data
    out_of_range_x = np.linspace(50, 150, 100)  # Too low for either
    assert validate_spectrum_range(out_of_range_x, "raman") == False
    assert validate_spectrum_range(out_of_range_x, "ftir") == False


def test_ftir_preprocessing():
    """Test FTIR-specific preprocessing parameters."""
    # Generate synthetic FTIR spectrum
    x = np.linspace(400, 4000, 200)  # FTIR range
    y = np.sin(x / 500) + 0.1 * np.random.randn(len(x)) + 2.0  # Synthetic absorbance

    # Test FTIR preprocessing
    x_proc, y_proc = preprocess_spectrum(x, y, modality="ftir", target_len=500)

    assert x_proc.shape == (500,)
    assert y_proc.shape == (500,)
    assert np.all(np.diff(x_proc) > 0)  # Monotonic increasing
    assert np.min(y_proc) >= 0.0  # Normalized to [0, 1]
    assert np.max(y_proc) <= 1.0


def test_raman_preprocessing():
    """Test Raman-specific preprocessing parameters."""
    # Generate synthetic Raman spectrum
    x = np.linspace(200, 3500, 200)  # Raman range
    y = np.exp(-(((x - 1500) / 200) ** 2)) + 0.05 * np.random.randn(
        len(x)
    )  # Gaussian peak

    # Test Raman preprocessing
    x_proc, y_proc = preprocess_spectrum(x, y, modality="raman", target_len=500)

    assert x_proc.shape == (500,)
    assert y_proc.shape == (500,)
    assert np.all(np.diff(x_proc) > 0)  # Monotonic increasing
    assert np.min(y_proc) >= 0.0  # Normalized to [0, 1]
    assert np.max(y_proc) <= 1.0


def test_modality_specific_parameters():
    """Test that different modalities use different default parameters."""
    x = np.linspace(400, 4000, 200)
    y = np.sin(x / 500) + 1.0

    # Test that FTIR uses different window length than Raman
    ftir_params = MODALITY_PARAMS["ftir"]
    raman_params = MODALITY_PARAMS["raman"]

    assert ftir_params["smooth_window"] != raman_params["smooth_window"]

    # Preprocess with both modalities (should use different parameters)
    x_raman, y_raman = preprocess_spectrum(x, y, modality="raman")
    x_ftir, y_ftir = preprocess_spectrum(x, y, modality="ftir")

    # Results should be slightly different due to different parameters
    assert not np.allclose(y_raman, y_ftir, rtol=1e-10)


def test_get_modality_info():
    """Test modality information retrieval."""
    raman_info = get_modality_info("raman")
    ftir_info = get_modality_info("ftir")

    assert "range" in raman_info
    assert "params" in raman_info
    assert "range" in ftir_info
    assert "params" in ftir_info

    # Check that ranges match expected values
    assert raman_info["range"] == MODALITY_RANGES["raman"]
    assert ftir_info["range"] == MODALITY_RANGES["ftir"]

    # Check that parameters are present
    assert "baseline_degree" in raman_info["params"]
    assert "smooth_window" in ftir_info["params"]


def test_invalid_modality():
    """Test handling of invalid modality."""
    x = np.linspace(1000, 2000, 100)
    y = np.sin(x / 100)

    with pytest.raises(ValueError, match="Unsupported modality"):
        preprocess_spectrum(x, y, modality="invalid")

    with pytest.raises(ValueError, match="Unknown modality"):
        validate_spectrum_range(x, "invalid")

    with pytest.raises(ValueError, match="Unknown modality"):
        get_modality_info("invalid")


def test_modality_parameter_override():
    """Test that modality defaults can be overridden."""
    x = np.linspace(400, 4000, 100)
    y = np.sin(x / 500) + 1.0

    # Override FTIR default window length
    custom_window = 21  # Different from FTIR default (13)

    x_proc, y_proc = preprocess_spectrum(
        x, y, modality="ftir", window_length=custom_window
    )

    assert x_proc.shape[0] > 0
    assert y_proc.shape[0] > 0


def test_range_validation_warning():
    """Test that range validation warnings work correctly."""
    # Create spectrum outside typical FTIR range
    x_bad = np.linspace(100, 300, 50)  # Too low for FTIR
    y_bad = np.ones_like(x_bad)

    # Should still process but with validation disabled
    x_proc, y_proc = preprocess_spectrum(
        x_bad, y_bad, modality="ftir", validate_range=False  # Disable validation
    )

    assert len(x_proc) > 0
    assert len(y_proc) > 0


def test_backwards_compatibility():
    """Test that old preprocessing calls still work (defaults to Raman)."""
    x = np.linspace(1000, 2000, 100)
    y = np.sin(x / 100)

    # Old style call (should default to Raman)
    x_old, y_old = preprocess_spectrum(x, y)

    # New style call with explicit Raman
    x_new, y_new = preprocess_spectrum(x, y, modality="raman")

    # Should be identical
    np.testing.assert_array_equal(x_old, x_new)
    np.testing.assert_array_equal(y_old, y_new)


if __name__ == "__main__":
    pytest.main([__file__])
