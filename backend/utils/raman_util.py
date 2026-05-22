from typing import List, Dict
import os
from pathlib import Path
import matplotlib.pyplot as plt

import json

def list_txt_files(root_dir):
    """Recursively lists all .txt files in a directory."""
    txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".txt"):
                full_path = os.path.join(dirpath, file)
                txt_files.append(full_path)
    return txt_files

def label_file(filepath):
    """
    Assigns label based on filename prefix:
    - 'sta-' => 0 (pristine)
    - 'wea-' => 1 (weathered)
    Returns None if prefix is unknown.
    """
    filename = os.path.basename(filepath).lower()
    if filename.startswith("sta-"):
        return 0
    elif filename.startswith("wea-"):
        return 1
    else:
        return None  # Unknown or irrelevant

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
