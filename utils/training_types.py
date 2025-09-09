"""
Defines core data structures and types for the training system.

This module centralizes data classes like TrainingConfig and helper
functions to avoid circular dependencies between the TrainingManager
and TrainingEngine.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit


class TrainingStatus(Enum):
    """Training job status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CVStrategy(Enum):
    """Cross-validation strategy enumeration"""

    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    TIME_SERIES_SPLIT = "time_series_split"


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    model_name: str
    dataset_path: str
    target_len: int = 500
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    num_folds: int = 10
    baseline_correction: bool = True
    smoothing: bool = True
    normalization: bool = True
    modality: str = "raman"
    device: str = "auto"  # auto, cpu, cuda
    cv_strategy: str = "stratified_kfold"  # New field for CV strategy
    spectral_weight: float = 0.1  # Weight for spectroscopy-specific metrics
    enable_augmentation: bool = False  # Enable data augmentation
    noise_level: float = 0.01  # Noise level for augmentation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class TrainingProgress:
    """Training progress tracking with enhanced metrics"""

    current_fold: int = 0
    total_folds: int = 10
    current_epoch: int = 0
    total_epochs: int = 10
    current_loss: float = 0.0
    current_accuracy: float = 0.0
    fold_accuracies: List[float] = field(default_factory=list)
    confusion_matrices: List[List[List[int]]] = field(default_factory=list)
    spectroscopy_metrics: List[Dict[str, float]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


def get_cv_splitter(strategy: str, n_splits: int = 10, random_state: int = 42):
    """Get cross-validation splitter based on strategy"""
    if strategy == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    elif strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif strategy == "time_series_split":
        return TimeSeriesSplit(n_splits=n_splits)
    else:
        # Default to stratified k-fold
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )


def augment_spectral_data(
    X: np.ndarray,
    y: np.ndarray,
    noise_level: float = 0.01,
    augmentation_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment spectral data with realistic noise and variations"""
    if augmentation_factor <= 1:
        return X, y

    augmented_X = [X]
    augmented_y = [y]

    for i in range(augmentation_factor - 1):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise

        # Add baseline drift (common in spectroscopy)
        baseline_drift = np.random.normal(0, noise_level * 0.5, (X.shape[0], 1))
        X_drift = X_noisy + baseline_drift

        # Add intensity scaling variation
        intensity_scale = np.random.normal(1.0, 0.05, (X.shape[0], 1))
        X_scaled = X_drift * intensity_scale

        # Ensure no negative values
        X_scaled = np.maximum(X_scaled, 0)

        augmented_X.append(X_scaled)
        augmented_y.append(y)

    return np.vstack(augmented_X), np.hstack(augmented_y)
