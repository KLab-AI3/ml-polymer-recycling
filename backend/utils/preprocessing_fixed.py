# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring, redefined-outer-name, unused-argument, unused-import, singleton-comparison, invalid-name, wrong-import-position, too-many-arguments, too-many-locals, too-many-statements, wrong-import-order
"""
preprocessing_fixed.py
Data leakage-free preprocessing pipeline for polymer aging classification.
This module ensures that preprocessing transformations (normalization, scaling, etc.)
are fitted only on training data within each cross-validation fold.
CRITICAL: This fixes the data leakage issue where preprocessing was applied
to the entire dataset before cross-validation splits.
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.discover_raman_files import list_txt_files, label_file
from scripts.plot_spectrum import load_spectrum
from backend.utils.preprocessing import preprocess_spectrum, TARGET_LENGTH

class SpectrumPreprocessor:
    """
    Data leakage-free preprocessing pipeline for spectral data.

    This class ensures that normalization and other transformations
    are fitted only on training data within each CV fold.
    """

    def __init__(
        self,
        target_len: int = TARGET_LENGTH,
        do_baseline: bool = True,
        do_smooth: bool = True,
        do_normalize: bool = True,
        modality: str = "raman"
    ):
        """
        Initialize the preprocessor with configuration.

        Args:
            target_len (int): Target length for resampling
            do_baseline (bool): Whether to apply baseline correction
            do_smooth (bool): Whether to apply smoothing
            do_normalize (bool): Whether to apply normalization
            modality (str): Spectroscopy modality ('raman' or 'ftir')
        """
        self.target_len = target_len
        self.do_baseline = do_baseline
        self.do_smooth = do_smooth
        self.do_normalize = do_normalize
        self.modality = modality

        # Stats fitted on training data only
        self.normalization_stats = None
        self.is_fitted = False

    def load_raw_data(self, dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load raw spectrum data without preprocessing.

        Args:
            dataset_dir (str): Path to dataset directory

        Returns:
            tuple: (raw_spectra, labels, file_paths)
        """
        txt_paths = list_txt_files(dataset_dir)
        raw_spectra = []
        labels = []
        valid_files = []

        for path in txt_paths:
            label = label_file(path)
            if label is None:
                continue

            try:
                x_raw, y_raw = load_spectrum(path)
                if len(x_raw) < 10:
                    continue  # Skip files with too few points

                raw_spectra.append((x_raw, y_raw))
                labels.append(int(label))
                valid_files.append(path)

            except (IOError, ValueError) as e:
                print(f"⚠️ Warning: Failed to load {path}: {e}")
                continue

        return np.array(raw_spectra, dtype=object), np.array(labels), valid_files

    def preprocess_single_spectrum(
        self,
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        use_fitted_stats: bool = False
    ) -> np.ndarray:
        """
        Preprocess a single spectrum.

        Args:
            x_raw (np.ndarray): Raw wavenumber values
            y_raw (np.ndarray): Raw intensity values
            use_fitted_stats (bool): Whether to use fitted normalization stats

        Returns:
            np.ndarray: Preprocessed spectrum
        """
        # Apply resampling, baseline correction, and smoothing
        # These don't cause data leakage as they're applied per-sample
        _, y_processed = preprocess_spectrum(
            np.asarray(x_raw),
            np.asarray(y_raw),
            target_len=self.target_len,
            modality=self.modality,
            do_baseline=self.do_baseline,
            do_smooth=self.do_smooth,
            do_normalize=False,  # We handle normalization separately
            out_dtype=np.float32
        )

        # Apply normalization using fitted stats if available
        if self.do_normalize and use_fitted_stats and self.is_fitted:
            y_processed = self._apply_fitted_normalization(y_processed)
        elif self.do_normalize and not use_fitted_stats:
            # Apply per-sample normalization (min-max)
            y_min, y_max = y_processed.min(), y_processed.max()
            if y_max > y_min:
                y_processed = (y_processed - y_min) / (y_max - y_min)

        return y_processed

    def fit_normalization_stats(self, train_spectra: list) -> None:
        """
        Fit normalization statistics on training data only.

        Args:
            train_spectra (list): List of (x_raw, y_raw) tuples for training
        """
        if not self.do_normalize:
            return

        # Preprocess training spectra without normalization
        processed_spectra = []
        for x_raw, y_raw in train_spectra:
            y_processed = self.preprocess_single_spectrum(
                x_raw, y_raw, use_fitted_stats=False
            )
            processed_spectra.append(y_processed)

        # Calculate global statistics from training data
        all_values = np.concatenate(processed_spectra)
        self.normalization_stats = {
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values)
        }

        self.is_fitted = True
        print("✅ Fitted normalization statistics on training data")

    def _apply_fitted_normalization(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply fitted normalization to a spectrum.

        Args:
            spectrum (np.ndarray): Preprocessed spectrum

        Returns:
            np.ndarray: Normalized spectrum
        """
        if not self.is_fitted:
            raise ValueError("Normalization stats not fitted. Call fit_normalization_stats first.")

        # Use min-max normalization based on training data
        stats = self.normalization_stats
        if stats is not None and stats['max'] > stats['min']:
            spectrum = (spectrum - stats['min']) / (stats['max'] - stats['min'])

        return spectrum

    def transform_fold(
        self,
        raw_spectra: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data for a single CV fold without data leakage.

        Args:
            raw_spectra (np.ndarray): Array of (x_raw, y_raw) tuples
            train_indices (np.ndarray): Training indices for this fold
            val_indices (np.ndarray): Validation indices for this fold

        Returns:
            tuple: (X_train, X_val) preprocessed data
        """
        # Get training and validation raw data
        train_raw = raw_spectra[train_indices]
        val_raw = raw_spectra[val_indices]

        # Fit normalization stats on training data only
        self.fit_normalization_stats(train_raw.tolist())

        # Preprocess training data
        X_train = []
        for x_raw, y_raw in train_raw:
            processed = self.preprocess_single_spectrum(
                x_raw, y_raw, use_fitted_stats=True
            )
            X_train.append(processed)

        # Preprocess validation data using fitted stats
        X_val = []
        for x_raw, y_raw in val_raw:
            processed = self.preprocess_single_spectrum(
                x_raw, y_raw, use_fitted_stats=True
            )
            X_val.append(processed)

        return np.array(X_train), np.array(X_val)

def load_data_for_cv(
    dataset_dir: str,
    preprocessor_config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, SpectrumPreprocessor]:
    """
    Load raw data for cross-validation without data leakage.

    Args:
        dataset_dir (str): Path to dataset directory
        preprocessor_config (dict): Configuration for preprocessor

    Returns:
        tuple: (raw_spectra, labels, preprocessor)
    """
    config = preprocessor_config or {}
    preprocessor = SpectrumPreprocessor(**config)

    raw_spectra, labels, _ = preprocessor.load_raw_data(dataset_dir)

    print(f"✅ Loaded {len(raw_spectra)} raw spectra for CV")
    print(f"Class distribution: {np.bincount(labels)}")

    return raw_spectra, labels, preprocessor

def preprocess_holdout_test_set(
    test_spectra: np.ndarray,
    fitted_preprocessor: SpectrumPreprocessor
) -> np.ndarray:
    """
    Preprocess hold-out test set using fitted preprocessor.

    Args:
        test_spectra (np.ndarray): Raw test spectra
        fitted_preprocessor (SpectrumPreprocessor): Preprocessor fitted on training data

    Returns:
        np.ndarray: Preprocessed test data
    """
    if not fitted_preprocessor.is_fitted:
        raise ValueError("Preprocessor must be fitted on training data first")

    X_test = []
    for x_raw, y_raw in test_spectra:
        processed = fitted_preprocessor.preprocess_single_spectrum(
            x_raw, y_raw, use_fitted_stats=True
        )
        X_test.append(processed)

    return np.array(X_test)

if __name__ == "__main__":
    # Test the data leakage-free preprocessing pipeline
    print("Testing data leakage-free preprocessing pipeline...")

    # Test with sample data
    dataset_dir = "sample_data"

    # Load raw data
    raw_spectra, labels, preprocessor = load_data_for_cv(dataset_dir)

    # Simulate a single CV fold
    from sklearn.model_selection import StratifiedKFold

    if len(raw_spectra) >= 2:
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        train_idx, val_idx = next(cv.split(raw_spectra, labels))

        # Transform without data leakage
        X_train, X_val = preprocessor.transform_fold(raw_spectra, train_idx, val_idx)

        print("✅ Fold transformation completed")
        print(f"   Train: {X_train.shape}")
        print(f"   Val: {X_val.shape}")
        print(f"   Normalization fitted: {preprocessor.is_fitted}")

    print("✅ Data leakage-free preprocessing test completed!")
