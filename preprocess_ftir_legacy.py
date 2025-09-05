import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import minmax_scale

def parse_ftir_file(filepath, num_points=500):
    df = pd.read_csv(filepath, skiprows=5, header=None, names=["Wavenumber", "Transmittance"])
    df.dropna(inplace=True)
    x = df["Wavenumber"].values
    y = df["Transmittance"].values

    if x[0] > x[-1]:  # Ensure increasing order
        x = x[::-1]
        y = y[::-1]

    f_interp = interp1d(x, y, kind="linear", fill_value="extrapolate")
    x_uniform = np.linspace(x.min(), x.max(), num_points)
    y_uniform = f_interp(x_uniform)

    return y_uniform

def label_from_filename(filename):
    """ Assign label based on sample prefix:
        - '2015a', '2015b', '2015c', '2016a', '2016b' => 1 (Aged)
        - All other number prefixes (1 . . 19)        => 0 (Unaged)
    """
    lower_filename = filename.lower()
    if lower_filename.startswith(("2015a", "2015b", "2015c", "2016a", "2016b")):
        return 1    # Aged
    else:
        return 0    # Unaged

def remove_baseline(y):
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, deg=2)
    baseline = np.polyval(coeffs, x)
    return y - baseline

def normalize_spectrum(y):
    return minmax_scale(y)

def smooth_spectrum(y, window_length=11, polyorder=2):
    return savgol_filter(y, window_length, polyorder)

def preprocess_ftir(directory, target_len=500, baseline_correction=False, apply_smoothing=False, normalize=False):
    X, y = [], []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and "fcl" in file.lower():
                filepath = os.path.join(root, file)
                try:
                    spectrum = parse_ftir_file(filepath, num_points=target_len)

                    if baseline_correction:
                        spectrum = remove_baseline(spectrum)
                    if apply_smoothing:
                        spectrum = smooth_spectrum(spectrum)
                    if normalize:
                        spectrum = normalize_spectrum(spectrum)

                    label = label_from_filename(file)
                    X.append(spectrum)
                    y.append(label)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
                    with open("scripts/ftir_failures.log", "a", encoding="utf-8") as log_file:
                        log_file.write(f"{file} - {e}\n")

    X = np.array(X)
    y = np.array(y)

    print(f"Processed {len(X)} FTIR samples.")
    return X, y

if __name__ == "__main__":
    data_dir = os.path.join("datasets", "ftir")
    X, y = preprocess_ftir(data_dir)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
