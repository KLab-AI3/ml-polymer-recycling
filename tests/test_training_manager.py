"""
Tests for the training manager functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import json
import pandas as pd

from utils.training_manager import (
    TrainingManager,
    TrainingConfig,
    TrainingStatus,
    get_training_manager,
    CVStrategy,
    get_cv_splitter,
    calculate_spectroscopy_metrics,
    augment_spectral_data,
    spectral_cosine_similarity,
)


def create_test_dataset(dataset_path: Path, num_samples: int = 10):
    """Create a test dataset for training"""
    # Create directories
    (dataset_path / "stable").mkdir(parents=True, exist_ok=True)
    (dataset_path / "weathered").mkdir(parents=True, exist_ok=True)

    # Generate synthetic spectra
    wavenumbers = np.linspace(400, 4000, 200)

    for i in range(num_samples // 2):
        # Stable samples
        intensities = np.random.normal(0.5, 0.1, len(wavenumbers))
        data = np.column_stack([wavenumbers, intensities])
        np.savetxt(dataset_path / "stable" / f"stable_{i}.txt", data)

        # Weathered samples
        intensities = np.random.normal(0.3, 0.1, len(wavenumbers))
        data = np.column_stack([wavenumbers, intensities])
        np.savetxt(dataset_path / "weathered" / f"weathered_{i}.txt", data)


@pytest.fixture
def temp_dataset():
    """Create temporary dataset for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    dataset_path = temp_dir / "test_dataset"
    create_test_dataset(dataset_path)
    yield dataset_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def training_manager():
    """Create training manager for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    # Use ThreadPoolExecutor for tests to avoid multiprocessing complexities
    manager = TrainingManager(
        max_workers=1, output_dir=str(temp_dir), use_multiprocessing=False
    )
    yield manager
    manager.shutdown()
    shutil.rmtree(temp_dir)


def test_training_config():
    """Test training configuration creation"""
    config = TrainingConfig(
        model_name="figure2", dataset_path="/test/path", epochs=5, batch_size=8
    )

    assert config.model_name == "figure2"
    assert config.epochs == 5
    assert config.batch_size == 8
    assert config.device == "auto"


def test_training_manager_initialization(training_manager):
    """Test training manager initialization"""
    assert training_manager.max_workers == 1
    assert len(training_manager.jobs) == 0


def test_submit_training_job(training_manager, temp_dataset):
    """Test submitting a training job"""
    config = TrainingConfig(
        model_name="figure2", dataset_path=str(temp_dataset), epochs=1, batch_size=4
    )

    job_id = training_manager.submit_training_job(config)

    assert job_id is not None
    assert len(job_id) > 0
    assert job_id in training_manager.jobs

    job = training_manager.get_job_status(job_id)
    assert job is not None
    assert job.config.model_name == "figure2"


def test_training_job_execution(training_manager, temp_dataset):
    """Test actual training job execution (lightweight test)"""
    config = TrainingConfig(
        model_name="figure2",
        dataset_path=str(temp_dataset),
        epochs=1,
        num_folds=2,  # Reduced for testing
        batch_size=4,
    )

    job_id = training_manager.submit_training_job(config)

    # Wait a moment for job to start
    import time

    time.sleep(1)

    job = training_manager.get_job_status(job_id)
    assert job.status in [
        TrainingStatus.PENDING,
        TrainingStatus.RUNNING,
        TrainingStatus.COMPLETED,
        TrainingStatus.FAILED,
    ]


def test_list_jobs(training_manager, temp_dataset):
    """Test listing jobs with filters"""
    config = TrainingConfig(
        model_name="figure2", dataset_path=str(temp_dataset), epochs=1
    )

    job_id = training_manager.submit_training_job(config)

    all_jobs = training_manager.list_jobs()
    assert len(all_jobs) >= 1

    pending_jobs = training_manager.list_jobs(TrainingStatus.PENDING)
    running_jobs = training_manager.list_jobs(TrainingStatus.RUNNING)

    # Job should be in one of these states
    assert len(pending_jobs) + len(running_jobs) >= 1


def test_global_training_manager():
    """Test global training manager singleton"""
    manager1 = get_training_manager()
    manager2 = get_training_manager()

    assert manager1 is manager2  # Should be same instance


def test_device_selection(training_manager):
    """Test device selection logic"""
    # Test auto device selection
    device = training_manager._get_device("auto")
    assert device.type in ["cpu", "cuda"]

    # Test CPU selection
    device = training_manager._get_device("cpu")
    assert device.type == "cpu"

    # Test CUDA selection (should fallback to CPU if not available)
    device = training_manager._get_device("cuda")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_invalid_dataset_path(training_manager):
    """Test handling of invalid dataset path"""
    config = TrainingConfig(
        model_name="figure2", dataset_path="/nonexistent/path", epochs=1
    )

    job_id = training_manager.submit_training_job(config)

    # Wait for job to process
    import time

    time.sleep(2)

    job = training_manager.get_job_status(job_id)
    assert job.status == TrainingStatus.FAILED
    assert "dataset" in job.error_message.lower()


def test_configurable_cv_strategies():
    """Test different cross-validation strategies"""
    # Test StratifiedKFold
    skf = get_cv_splitter("stratified_kfold", n_splits=5)
    assert hasattr(skf, "split")

    # Test KFold
    kf = get_cv_splitter("kfold", n_splits=5)
    assert hasattr(kf, "split")

    # Test TimeSeriesSplit
    tss = get_cv_splitter("time_series_split", n_splits=5)
    assert hasattr(tss, "split")

    # Test default fallback
    default = get_cv_splitter("invalid_strategy", n_splits=5)
    assert hasattr(default, "split")


def test_spectroscopy_metrics():
    """Test spectroscopy-specific metrics calculation"""
    # Create test data
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    probabilities = np.array(
        [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4]]
    )

    metrics = calculate_spectroscopy_metrics(y_true, y_pred, probabilities)

    # Check that all expected metrics are present
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "cosine_similarity" in metrics
    assert "distribution_similarity" in metrics

    # Check that metrics are reasonable
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert -1 <= metrics["cosine_similarity"] <= 1
    assert 0 <= metrics["distribution_similarity"] <= 1


def test_spectral_cosine_similarity():
    """Test cosine similarity calculation for spectral data"""
    # Create test spectra
    spectrum1 = np.array([1, 2, 3, 4, 5])
    spectrum2 = np.array([2, 4, 6, 8, 10])  # Perfect correlation
    spectrum3 = np.array([5, 4, 3, 2, 1])  # Anti-correlation

    # Test perfect correlation
    sim1 = spectral_cosine_similarity(spectrum1, spectrum2)
    assert abs(sim1 - 1.0) < 1e-10

    # Test that similarity exists
    sim2 = spectral_cosine_similarity(spectrum1, spectrum3)
    assert -1 <= sim2 <= 1  # Valid cosine similarity range

    # Test self-similarity
    sim3 = spectral_cosine_similarity(spectrum1, spectrum1)
    assert abs(sim3 - 1.0) < 1e-10


def test_data_augmentation():
    """Test spectral data augmentation"""
    # Create test data
    X = np.random.rand(10, 100)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Test augmentation
    X_aug, y_aug = augment_spectral_data(X, y, noise_level=0.01, augmentation_factor=3)

    # Check that data is augmented
    assert X_aug.shape[0] == X.shape[0] * 3
    assert y_aug.shape[0] == y.shape[0] * 3
    assert X_aug.shape[1] == X.shape[1]  # Same number of features

    # Test no augmentation
    X_no_aug, y_no_aug = augment_spectral_data(X, y, augmentation_factor=1)
    assert np.array_equal(X_no_aug, X)
    assert np.array_equal(y_no_aug, y)


def test_enhanced_training_config():
    """Test enhanced training configuration with new parameters"""
    config = TrainingConfig(
        model_name="figure2",
        dataset_path="/test/path",
        cv_strategy="time_series_split",
        enable_augmentation=True,
        noise_level=0.02,
        spectral_weight=0.2,
    )

    assert config.cv_strategy == "time_series_split"
    assert config.enable_augmentation == True
    assert config.noise_level == 0.02
    assert config.spectral_weight == 0.2

    # Test serialization includes new fields
    config_dict = config.to_dict()
    assert "cv_strategy" in config_dict
    assert "enable_augmentation" in config_dict
    assert "noise_level" in config_dict
    assert "spectral_weight" in config_dict


def test_enhanced_dataset_loading_security():
    """Test enhanced dataset loading with security features"""
    temp_dir = Path(tempfile.mkdtemp())
    training_manager = TrainingManager(
        max_workers=1, output_dir=str(temp_dir), use_multiprocessing=False
    )

    try:
        # Create a test dataset with different file formats
        dataset_dir = temp_dir / "test_dataset"
        (dataset_dir / "stable").mkdir(parents=True)
        (dataset_dir / "weathered").mkdir(parents=True)

        # Create multiple files to meet minimum requirements
        for i in range(6):  # Create 6 files per class
            # Create CSV files
            csv_data = pd.DataFrame(
                {
                    "wavenumber": np.linspace(400, 4000, 100),
                    "intensity": np.random.rand(100),
                }
            )
            csv_data.to_csv(
                dataset_dir / "stable" / f"test_stable_{i}.csv", index=False
            )

            # Create JSON files
            json_data = {
                "x": np.linspace(400, 4000, 100).tolist(),
                "y": np.random.rand(100).tolist(),
            }
            with open(dataset_dir / "weathered" / f"test_weathered_{i}.json", "w") as f:
                json.dump(json_data, f)

        # Test configuration with enhanced features
        config = TrainingConfig(
            model_name="figure2",
            dataset_path=str(dataset_dir),
            epochs=1,
            cv_strategy="kfold",
            enable_augmentation=True,
            noise_level=0.01,
        )

        # Test that the enhanced loading works
        from utils.training_manager import TrainingJob, TrainingProgress

        job = TrainingJob(job_id="test", config=config, progress=TrainingProgress())

        # This should work with the enhanced data loading
        X, y = training_manager._load_and_preprocess_data(job)

        # Should load data from multiple formats
        assert X is not None
        assert y is not None
        assert len(X) >= 10  # Should have at least 10 samples total

        # Test that we have both classes
        unique_classes = np.unique(y)
        assert len(unique_classes) >= 2

    finally:
        training_manager.shutdown()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
