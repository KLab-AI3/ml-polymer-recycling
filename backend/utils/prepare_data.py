"""
Data Preparation Script

This script takes a raw dataset, performs a stratified split into
training, validation, and test sets, and saves them to a processed
data directory. This ensures a consistent and reproducible data
splitting strategy.

Usage:
    python scripts/prepare_data.py --data-path /path/to/raw/data.csv --output-path data/processed
"""

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(data_path: Path, output_path: Path, test_size: float = 0.2, val_size: float = 0.15):
    """
    Loads data, performs stratified train-val-test split, and saves the splits.

    Args:
        data_path (Path): Path to the raw data file (CSV expected).
        output_path (Path): Directory to save the processed data splits.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to use for validation.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {data_path}")

    print(f"Loading data from {data_path}...")
    # This assumes a CSV with a 'spectra' column and a 'label' column.
    # You will need to adapt this to your actual raw data format.
    df = pd.read_csv(data_path)

    # Ensure the 'label' column exists in the dataset
    if 'label' not in df.columns:
        raise ValueError(
            "The input data must contain a 'label' column for stratified splitting.")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    print("Performing stratified train-test split...")
    # Split off the test set first
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )

    # Split the remaining data into training and validation sets
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42
    )

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Save the splits
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "validation.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print(f"✅ Data splits saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and split spectral data.")
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to the raw data CSV file.")
    parser.add_argument("--output-path", type=Path, default=Path(
        "data/processed"), help="Directory to save data splits.")
    args = parser.parse_args()
    prepare_data(args.data_path, args.output_path)
