"""
Generate demo datasets for testing the training functionality.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_synthetic_spectrum(
    wavenumbers, base_intensity=0.5, noise_level=0.05, peaks=None
):
    """Generate a synthetic spectrum with specified characteristics"""
    spectrum = np.full_like(wavenumbers, base_intensity)

    # Add some peaks
    if peaks is None:
        peaks = [
            (1000, 0.3, 50),
            (1500, 0.5, 80),
            (2000, 0.2, 40),
        ]  # (center, height, width)

    for center, height, width in peaks:
        peak = height * np.exp(-(((wavenumbers - center) / width) ** 2))
        spectrum += peak

    # Add noise
    spectrum += np.random.normal(0, noise_level, len(wavenumbers))

    # Ensure positive values
    spectrum = np.maximum(spectrum, 0.01)

    return spectrum


def create_demo_datasets():
    """Create demo datasets for training"""

    # Define wavenumber range (typical for Raman)
    wavenumbers = np.linspace(400, 3500, 200)

    # Create stable polymer samples
    stable_dir = Path("datasets/demo_dataset/stable")
    stable_dir.mkdir(parents=True, exist_ok=True)

    print("Generating stable polymer samples...")
    for i in range(20):
        # Stable polymers - higher intensity, sharper peaks
        stable_peaks = [
            (
                800 + np.random.normal(0, 20),
                0.4 + np.random.normal(0, 0.05),
                30 + np.random.normal(0, 5),
            ),
            (
                1200 + np.random.normal(0, 30),
                0.6 + np.random.normal(0, 0.08),
                40 + np.random.normal(0, 8),
            ),
            (
                1600 + np.random.normal(0, 25),
                0.3 + np.random.normal(0, 0.04),
                35 + np.random.normal(0, 6),
            ),
            (
                2900 + np.random.normal(0, 40),
                0.8 + np.random.normal(0, 0.1),
                60 + np.random.normal(0, 10),
            ),
        ]

        spectrum = generate_synthetic_spectrum(
            wavenumbers,
            base_intensity=0.4 + np.random.normal(0, 0.05),
            noise_level=0.02,
            peaks=stable_peaks,
        )

        # Save as two-column format
        data = np.column_stack([wavenumbers, spectrum])
        np.savetxt(stable_dir / f"stable_sample_{i:02d}.txt", data, fmt="%.6f")

    # Create weathered polymer samples
    weathered_dir = Path("datasets/demo_dataset/weathered")
    weathered_dir.mkdir(parents=True, exist_ok=True)

    print("Generating weathered polymer samples...")
    for i in range(20):
        # Weathered polymers - lower intensity, broader peaks, additional oxidation peaks
        weathered_peaks = [
            (
                800 + np.random.normal(0, 30),
                0.2 + np.random.normal(0, 0.04),
                45 + np.random.normal(0, 10),
            ),
            (
                1200 + np.random.normal(0, 40),
                0.3 + np.random.normal(0, 0.06),
                55 + np.random.normal(0, 12),
            ),
            (
                1600 + np.random.normal(0, 35),
                0.15 + np.random.normal(0, 0.03),
                50 + np.random.normal(0, 8),
            ),
            (
                1720 + np.random.normal(0, 20),
                0.25 + np.random.normal(0, 0.04),
                40 + np.random.normal(0, 7),
            ),  # Oxidation peak
            (
                2900 + np.random.normal(0, 50),
                0.4 + np.random.normal(0, 0.08),
                80 + np.random.normal(0, 15),
            ),
        ]

        spectrum = generate_synthetic_spectrum(
            wavenumbers,
            base_intensity=0.25 + np.random.normal(0, 0.04),
            noise_level=0.03,
            peaks=weathered_peaks,
        )

        # Save as two-column format
        data = np.column_stack([wavenumbers, spectrum])
        np.savetxt(weathered_dir / f"weathered_sample_{i:02d}.txt", data, fmt="%.6f")

    print(f"✅ Demo dataset created:")
    print(f"   Stable samples: {len(list(stable_dir.glob('*.txt')))}")
    print(f"   Weathered samples: {len(list(weathered_dir.glob('*.txt')))}")
    print(f"   Location: datasets/demo_dataset/")


if __name__ == "__main__":
    create_demo_datasets()
