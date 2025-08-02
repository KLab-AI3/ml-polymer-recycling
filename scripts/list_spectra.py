"""
list_spectra.py

This script provides functionality to recursively list all `.txt` files 
within a specified directory. It is designed to assist in managing and 
exploring datasets, particularly for Raman spectrum data stored in text files.

Functions:
-   list_txt_files(root_dir): Recursively finds and returns a list of all `.txt` 
    files in the given directory.

Usage:
-   The script can be executed directly to list `.txt` files in a predefined 
    dataset directory and print a summary, including the total count and a 
    sample of file paths.

Example:
    $ python list_spectra.py
    Found 100 .txt files.
    Sample Files:
     - datasets/rdwp/.../file1.txt
     - datasets/rdwp/.../file2.txt
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


if __name__ == "__main__":
    dataset_dir = os.path.join(
        "datasets", "rdwp",
        "A Raman database of microplastics weathered under natural environments"
    )

    txt_paths = list_txt_files(dataset_dir)

    print(f"Found {len(txt_paths)} .txt files.")
    print("Sample Files: ")
    for path in txt_paths[:5]:
        print(" -", path)

    labeled_files = []
    for path in txt_paths:
        label = label_file(path)
        if label is not None:
            labeled_files.append((path, label))

    print(f"\nLabeled {len(labeled_files)} files:")
    for path, label in labeled_files[:5]:
        print(f" - {os.path.basename(path)} => Label: {label}")
