"""
plot_spectrum.py

This script provides functionality to load and plot Raman spectra from two-column `.txt` files.

Functions:
    - load_spectrum(filepath): Reads a spectrum file and extracts Raman shift and intensity values.
    - plot_spectrum(x, y, title): Plots the Raman spectrum with basic styling.

Command-line Usage:
    The script can be run directly to load and plot a predefined spectrum file. Modify the `spectrum_file` variable to specify the file path.

Dependencies:
    - os: For file path operations.
    - matplotlib.pyplot: For plotting the spectrum.

Example:
    python plot_spectrum.py

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt

def load_spectrum(filepath):
    """Loads a Raman spectrum from a two-column .txt file."""
    x_vals, y_vals = [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue  # Skip lines that can't be converted
    return x_vals, y_vals


def plot_spectrum(x, y, title="Raman Spectrum"):
    """Plots the spectrum data with basic styling."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a Raman spectrum from a .txt file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .txt file")
    parser.add_argument("--output", type=str, required=False, help="Path to save .png image")

    args = parser.parse_args()
    spectrum_file = args.input
    output_file = args.output

    x, y = load_spectrum(spectrum_file)
    plot_spectrum(x, y, title=os.path.basename(spectrum_file))

    if output_file:
        plt.savefig(output_file)
        print(f"✅ Plot saved to {output_file}")
    else:
        plt.show()