"""
Training job management system for ML Hub functionality.
Handles asynchronous training jobs, progress tracking, and result management.
"""

import os
import sys
import json
import time
import uuid
import threading
import concurrent.futures
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean

# Add project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.registry import choices as model_choices, build as build_model
from utils.training_engine import TrainingEngine
from utils.training_types import (
    TrainingConfig,
    TrainingProgress,
    TrainingStatus,
    CVStrategy,
    get_cv_splitter,
)
from utils.preprocessing import preprocess_spectrum


def spectral_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate cosine similarity between spectral predictions and true values"""
    # Reshape if needed for cosine similarity calculation
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    return float(cosine_similarity(y_true, y_pred)[0, 0])


def peak_matching_score(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    height_threshold: float = 0.1,
    distance: int = 5,
) -> float:
    """Calculate peak matching score between two spectra"""
    try:
        # Find peaks in both spectra
        peaks1, _ = find_peaks(spectrum1, height=height_threshold, distance=distance)
        peaks2, _ = find_peaks(spectrum2, height=height_threshold, distance=distance)

        if len(peaks1) == 0 or len(peaks2) == 0:
            return 0.0

        # Calculate matching peaks (within tolerance)
        tolerance = 3  # wavenumber tolerance
        matches = 0

        for peak1 in peaks1:
            for peak2 in peaks2:
                if abs(peak1 - peak2) <= tolerance:
                    matches += 1
                    break

        # Return normalized matching score
        return matches / max(len(peaks1), len(peaks2))
    except:
        return 0.0


def spectral_euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate normalized Euclidean distance between spectra"""
    try:
        distance = euclidean(y_true.flatten(), y_pred.flatten())
        # Normalize by the length of the spectrum
        return distance / len(y_true.flatten())
    except:
        return float("inf")


def calculate_spectroscopy_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate comprehensive spectroscopy-specific metrics"""
    metrics = {}

    try:
        # Standard classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")

        # Spectroscopy-specific metrics
        if probabilities is not None and len(probabilities.shape) > 1:
            # For classification with probabilities, use cosine similarity on prob distributions
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                # Convert true labels to one-hot for similarity calculation
                y_true_onehot = np.eye(len(unique_classes))[y_true]
                metrics["cosine_similarity"] = float(
                    cosine_similarity(
                        y_true_onehot.mean(axis=0).reshape(1, -1),
                        probabilities.mean(axis=0).reshape(1, -1),
                    )[0, 0]
                )

        # Add bias audit metric (class distribution comparison)
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

        # Calculate distribution difference (Jensen-Shannon divergence approximation)
        true_dist = counts_true / len(y_true)
        pred_dist = np.zeros_like(true_dist)

        for i, class_label in enumerate(unique_true):
            if class_label in unique_pred:
                pred_idx = np.where(unique_pred == class_label)[0][0]
                pred_dist[i] = counts_pred[pred_idx] / len(y_pred)

        # Simple distribution similarity (1 - average absolute difference)
        metrics["distribution_similarity"] = 1.0 - np.mean(
            np.abs(true_dist - pred_dist)
        )

    except Exception as e:
        print(f"Error calculating spectroscopy metrics: {e}")
        # Return basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0,
            "f1_score": (
                f1_score(y_true, y_pred, average="weighted") if len(y_true) > 0 else 0.0
            ),
            "cosine_similarity": 0.0,
            "distribution_similarity": 0.0,
        }

    return metrics


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""

    enable_augmentation: bool = False
    noise_level: float = 0.01  # Noise level for augmentation


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""

    baseline_correction: bool = True
    smoothing: bool = True
    normalization: bool = True


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
    modality: str = "raman"
    device: str = "auto"  # auto, cpu, cuda
    cv_strategy: str = "stratified_kfold"  # New field for CV strategy
    spectral_weight: float = 0.1  # Weight for spectroscopy-specific metrics
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

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


@dataclass
class TrainingJob:
    """Training job container"""

    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    progress: TrainingProgress = None
    error_message: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    weights_path: Optional[str] = None
    logs_path: Optional[str] = None

    def __post_init__(self):
        if self.progress is None:
            self.progress = TrainingProgress(
                total_folds=self.config.num_folds, total_epochs=self.config.epochs
            )
        if self.created_at is None:
            self.created_at = datetime.now()


class TrainingManager:
    """Manager for training jobs with async execution and progress tracking"""

    def __init__(
        self,
        max_workers: int = 2,
        output_dir: str = "outputs",
        use_multiprocessing: bool = True,
    ):
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing

        # Use ProcessPoolExecutor for CPU/GPU-bound tasks, ThreadPoolExecutor for I/O-bound
        if use_multiprocessing:
            # Limit workers to available CPU cores to prevent oversubscription
            actual_workers = min(max_workers, multiprocessing.cpu_count())
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=actual_workers
            )
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            )

        self.jobs: Dict[str, TrainingJob] = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "weights").mkdir(exist_ok=True)

    def generate_job_id(self) -> str:
        """Generate unique job ID"""
        return f"train_{uuid.uuid4().hex[:8]}_{int(time.time())}"

    def submit_training_job(
        self, config: TrainingConfig, progress_callback: Optional[Callable] = None
    ) -> str:
        """Submit a new training job"""
        job_id = self.generate_job_id()
        job = TrainingJob(job_id=job_id, config=config)

        self.jobs[job_id] = job

        # Submit to thread pool
        self.executor.submit(
            self._run_training_job, job, progress_callback=progress_callback
        )

        return job_id

    def _run_training_job(self, job: TrainingJob) -> None:
        """Execute training job (runs in separate thread)"""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            if job.progress:
                job.progress.start_time = job.started_at

            if progress_callback:
                progress_callback(job)

            # Load and preprocess data
            X, y = self._load_and_preprocess_data(job)
            if X is None or y is None:
                raise ValueError("Failed to load dataset")

            # Define a callback to update the job's progress object
            def engine_progress_callback(progress_data: dict):
                if job.progress:
                    if progress_data["type"] == "fold_start":
                        job.progress.current_fold = progress_data["fold"]
                    elif progress_data["type"] == "epoch_end":
                        job.progress.current_epoch = progress_data["epoch"]
                        job.progress.current_loss = progress_data["loss"]
                if progress_callback:
                    progress_callback(job)

            # Instantiate and run the training engine
            engine = TrainingEngine(job.config)
            results = engine.run(X, y, progress_callback=engine_progress_callback)

            # Update job with results
            if job.progress:
                job.progress.fold_accuracies = results["fold_accuracies"]
                job.progress.confusion_matrices = results["confusion_matrices"]

            # Save model weights and logs
            self._save_model_weights(job, results["model_state_dict"])
            self._save_training_results(job)

            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress.end_time = job.completed_at

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()

        finally:
            if progress_callback:
                progress_callback(job)

    def _load_and_preprocess_data(
        self, job: TrainingJob
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and preprocess dataset with enhanced validation and security"""
        try:
            config = job.config
            dataset_path = Path(config.dataset_path)

            # Enhanced path validation and security
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

            # Validate dataset path is within allowed directories (security)
            try:
                dataset_path = dataset_path.resolve()
                allowed_bases = [
                    Path("datasets").resolve(),
                    Path("data").resolve(),
                    Path("/tmp").resolve(),
                ]
                if not any(
                    str(dataset_path).startswith(str(base)) for base in allowed_bases
                ):
                    raise ValueError(
                        f"Dataset path outside allowed directories: {dataset_path}"
                    )
            except Exception as e:
                print(f"Path validation error: {e}")
                raise ValueError("Invalid dataset path")

            # Load data from dataset directory
            X, y = [], []
            total_files = 0
            processed_files = 0
            max_files_per_class = 1000  # Limit to prevent memory issues
            max_file_size = 10 * 1024 * 1024  # 10MB per file

            # Look for data files in the dataset directory
            for label_dir in dataset_path.iterdir():
                if not label_dir.is_dir():
                    continue

                label = 0 if "stable" in label_dir.name.lower() else 1
                files_in_class = 0

                # Support multiple file formats
                file_patterns = ["*.txt", "*.csv", "*.json"]

                for pattern in file_patterns:
                    for file_path in label_dir.glob(pattern):
                        total_files += 1

                        # Security: Check file size
                        if file_path.stat().st_size > max_file_size:
                            print(
                                f"Skipping large file: {file_path} ({file_path.stat().st_size} bytes)"
                            )
                            continue

                        # Limit files per class
                        if files_in_class >= max_files_per_class:
                            print(
                                f"Reached maximum files per class ({max_files_per_class}) for {label_dir.name}"
                            )
                            break

                        try:
                            # Load spectrum data based on file type
                            if file_path.suffix.lower() == ".txt":
                                data = np.loadtxt(file_path)
                                if data.ndim == 2 and data.shape[1] >= 2:
                                    x_raw, y_raw = data[:, 0], data[:, 1]
                                elif data.ndim == 1:
                                    # Single column data
                                    x_raw = np.arange(len(data))
                                    y_raw = data
                                else:
                                    continue

                            elif file_path.suffix.lower() == ".csv":
                                import pandas as pd

                                df = pd.read_csv(file_path)
                                if df.shape[1] >= 2:
                                    x_raw, y_raw = (
                                        df.iloc[:, 0].values,
                                        df.iloc[:, 1].values,
                                    )
                                else:
                                    x_raw = np.arange(len(df))
                                    y_raw = df.iloc[:, 0].values

                            elif file_path.suffix.lower() == ".json":
                                with open(file_path, "r") as f:
                                    data_dict = json.load(f)
                                if isinstance(data_dict, dict):
                                    if "x" in data_dict and "y" in data_dict:
                                        x_raw, y_raw = np.array(
                                            data_dict["x"]
                                        ), np.array(data_dict["y"])
                                    elif "spectrum" in data_dict:
                                        y_raw = np.array(data_dict["spectrum"])
                                        x_raw = np.arange(len(y_raw))
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                continue

                            # Validate data integrity
                            if len(x_raw) != len(y_raw) or len(x_raw) < 10:
                                print(
                                    f"Invalid data in file {file_path}: insufficient data points"
                                )
                                continue

                            # Check for NaN or infinite values
                            if np.any(np.isnan(y_raw)) or np.any(np.isinf(y_raw)):
                                print(
                                    f"Invalid data in file {file_path}: NaN or infinite values"
                                )
                                continue

                            # Validate reasonable value ranges for spectroscopy
                            if np.min(y_raw) < -1000 or np.max(y_raw) > 1e6:
                                print(
                                    f"Suspicious data values in file {file_path}: outside expected range"
                                )
                                continue

                            # Preprocess spectrum
                            _, y_processed = preprocess_spectrum(
                                x_raw,
                                y_raw,
                                modality=config.modality,
                                target_len=config.target_len,
                                do_baseline=config.baseline_correction,
                                do_smooth=config.smoothing,
                                do_normalize=config.normalization,
                            )

                            # Final validation of processed data
                            if (
                                y_processed is None
                                or len(y_processed) != config.target_len
                            ):
                                print(f"Preprocessing failed for file {file_path}")
                                continue

                            X.append(y_processed)
                            y.append(label)
                            files_in_class += 1
                            processed_files += 1

                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
                            continue

            # Validate final dataset
            if len(X) == 0:
                raise ValueError("No valid data files found in dataset")

            if len(X) < 10:
                raise ValueError(
                    f"Insufficient data: only {len(X)} samples found (minimum 10 required)"
                )

            # Check class balance
            unique_labels, counts = np.unique(y, return_counts=True)
            if len(unique_labels) < 2:
                raise ValueError("Dataset must contain at least 2 classes")

            min_class_size = min(counts)
            if min_class_size < 3:
                raise ValueError(
                    f"Insufficient samples in one class: minimum {min_class_size} (need at least 3)"
                )

            print(f"Dataset loaded: {processed_files}/{total_files} files processed")
            print(f"Class distribution: {dict(zip(unique_labels, counts))}")

            return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None

    def _save_model_weights(self, job: TrainingJob, model_state_dict: dict):
        """Saves the model's state dictionary to a file."""
        weights_dir = self.output_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        job.weights_path = str(weights_dir / f"{job.config.model_name}_model.pth")
        torch.save(model_state_dict, job.weights_path)

    def _save_training_results(self, job: TrainingJob):
        """Save training results and logs with enhanced metrics"""
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        job.logs_path = str(logs_dir / f"{job.job_id}_log.json")

        # Calculate comprehensive summary metrics
        spectro_summary = {}
        if job.progress.spectroscopy_metrics:
            # Average across all folds for each metric
            metric_keys = job.progress.spectroscopy_metrics[0].keys()
            for key in metric_keys:
                values = [
                    fold_metrics.get(key, 0.0)
                    for fold_metrics in job.progress.spectroscopy_metrics
                ]
                spectro_summary[f"mean_{key}"] = float(np.mean(values))
                spectro_summary[f"std_{key}"] = float(np.std(values))

        results = {
            "job_id": job.job_id,
            "config": job.config.to_dict(),
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "progress": {
                "fold_accuracies": job.progress.fold_accuracies,
                "confusion_matrices": job.progress.confusion_matrices,
                "spectroscopy_metrics": job.progress.spectroscopy_metrics,
                "mean_accuracy": (
                    np.mean(job.progress.fold_accuracies)
                    if job.progress.fold_accuracies
                    else 0.0
                ),
                "std_accuracy": (
                    np.std(job.progress.fold_accuracies)
                    if job.progress.fold_accuracies
                    else 0.0
                ),
                "spectroscopy_summary": spectro_summary,
            },
            "weights_path": job.weights_path,
            "error_message": job.error_message,
        }

        if job.logs_path:
            with open(job.logs_path, "w") as f:
                json.dump(results, f, indent=2)

    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get current status of a training job"""
        return self.jobs.get(job_id)

    def list_jobs(
        self, status_filter: Optional[TrainingStatus] = None
    ) -> List[TrainingJob]:
        """List all jobs, optionally filtered by status"""
        jobs = list(self.jobs.values())
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job = self.jobs.get(job_id)
        if job and job.status == TrainingStatus.RUNNING:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            # Note: This is a simple cancellation - actual thread termination is more complex
            return True
        return False

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for job_id, job in self.jobs.items():
            if (
                job.status
                in [
                    TrainingStatus.COMPLETED,
                    TrainingStatus.FAILED,
                    TrainingStatus.CANCELLED,
                ]
                and job.completed_at
                and job.completed_at < cutoff_time
            ):
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]

    def shutdown(self):
        """Shutdown the training manager"""
        self.executor.shutdown(wait=True)


# Global training manager instance
_training_manager = None


def get_training_manager() -> TrainingManager:
    """Get global training manager instance"""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager
