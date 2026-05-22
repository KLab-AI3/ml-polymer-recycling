"""
Centralized Spectrum Preprocessing Module
Single source of truth for all preprocessing operations across training, validation, testing, and live inference.
Ensures no drift between different processing stages.
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from backend.utils.preprocessing import (
    preprocess_spectrum,
    validate_spectrum_modality,
    MODALITY_PARAMS,
    MODALITY_RANGES,
    TARGET_LENGTH
)


@dataclass
class PreprocessingConfig:
    """Immutable preprocessing configuration."""
    target_length: int = TARGET_LENGTH
    modality: str = "raman"
    do_baseline: bool = True
    baseline_degree: Optional[int] = None  # Uses modality default if None
    do_smooth: bool = True
    smooth_window: Optional[int] = None  # Uses modality default if None
    smooth_polyorder: Optional[int] = None  # Uses modality default if None
    do_normalize: bool = True
    validate_range: bool = True
    version: str = "1.0.0"  # Version for config compatibility

    def __post_init__(self):
        """Validate configuration and set modality defaults."""
        if self.modality not in MODALITY_PARAMS:
            raise ValueError(f"Invalid modality: {self.modality}")

        # Set modality defaults if None
        modality_config = MODALITY_PARAMS[self.modality]
        if self.baseline_degree is None:
            object.__setattr__(self, 'baseline_degree', modality_config['baseline_degree'])
        if self.smooth_window is None:
            object.__setattr__(self, 'smooth_window', modality_config['smooth_window'])
        if self.smooth_polyorder is None:
            object.__setattr__(self, 'smooth_polyorder', modality_config['smooth_polyorder'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def get_hash(self) -> str:
        """Get deterministic hash of configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class PreprocessingResult:
    """Result of preprocessing with full provenance."""
    x_processed: np.ndarray
    y_processed: np.ndarray
    x_original: np.ndarray
    y_original: np.ndarray
    config: PreprocessingConfig
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: str

    def get_content_hash(self) -> str:
        """Get hash of processed content for drift detection."""
        # Combine config hash with data hash
        config_hash = self.config.get_hash()
        data_hash = hashlib.sha256(
            np.concatenate([self.x_processed, self.y_processed]).tobytes()
        ).hexdigest()[:16]
        return f"{config_hash}_{data_hash}"


class SpectrumPreprocessor:
    """
    Centralized spectrum preprocessor ensuring consistent processing across all stages.
    Single source of truth for preprocessing logic.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize with preprocessing configuration."""
        self.config = config or PreprocessingConfig()
        self._processing_history: List[Dict[str, Any]] = []

    def process(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PreprocessingResult:
        """
        Process spectrum with full provenance tracking.

        Args:
            x: Input wavenumber array
            y: Input intensity array
            metadata: Optional metadata to include

        Returns:
            PreprocessingResult with full provenance
        """
        import time
        start_time = time.time()

        # Store original data
        x_original = np.array(x, copy=True)
        y_original = np.array(y, copy=True)

        # Process using centralized function
        x_processed, y_processed = preprocess_spectrum(
            x, y,
            target_len=self.config.target_length,
            modality=self.config.modality,
            do_baseline=self.config.do_baseline,
            degree=self.config.baseline_degree,
            do_smooth=self.config.do_smooth,
            window_length=self.config.smooth_window,
            polyorder=self.config.smooth_polyorder,
            do_normalize=self.config.do_normalize,
            validate_range=self.config.validate_range
        )

        processing_time = time.time() - start_time

        # Create metadata
        result_metadata = {
            "original_length": len(x_original),
            "processed_length": len(x_processed),
            "wavenumber_range": [float(x_processed.min()), float(x_processed.max())],
            "intensity_range": [float(y_processed.min()), float(y_processed.max())],
            "modality_validated": validate_spectrum_modality(x_original, y_original, self.config.modality)[0],
            **(metadata or {})
        }

        # Create result
        result = PreprocessingResult(
            x_processed=x_processed,
            y_processed=y_processed,
            x_original=x_original,
            y_original=y_original,
            config=self.config,
            metadata=result_metadata,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

        # Track processing
        self._processing_history.append({
            "timestamp": result.timestamp,
            "config_hash": self.config.get_hash(),
            "content_hash": result.get_content_hash(),
            "processing_time": processing_time
        })

        return result

    def process_batch(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray]],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[PreprocessingResult]:
        """Process multiple spectra with consistent configuration."""
        if metadata_list is None:
            metadata_list = [None] * len(spectra)

        results = []
        for i, (x, y) in enumerate(spectra):
            metadata = metadata_list[i] if i < len(metadata_list) else None
            result = self.process(x, y, metadata)
            results.append(result)

        return results

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing operations."""
        if not self._processing_history:
            return {"total_processed": 0}

        return {
            "total_processed": len(self._processing_history),
            "config_hash": self.config.get_hash(),
            "config": self.config.to_dict(),
            "processing_times": {
                "min": min(h["processing_time"] for h in self._processing_history),
                "max": max(h["processing_time"] for h in self._processing_history),
                "mean": np.mean([h["processing_time"] for h in self._processing_history])
            },
            "first_processed": self._processing_history[0]["timestamp"],
            "last_processed": self._processing_history[-1]["timestamp"]
        }


# Global preprocessor instances for different stages
TRAINING_PREPROCESSOR = SpectrumPreprocessor(PreprocessingConfig(modality="raman"))
VALIDATION_PREPROCESSOR = SpectrumPreprocessor(PreprocessingConfig(modality="raman"))
INFERENCE_PREPROCESSOR = SpectrumPreprocessor(PreprocessingConfig(modality="raman"))

# Factory function for creating stage-specific preprocessors
def create_preprocessor(stage: str, modality: str = "raman") -> SpectrumPreprocessor:
    """
    Create preprocessor for specific stage with identical configuration.

    Args:
        stage: 'training', 'validation', 'testing', or 'inference'
        modality: 'raman' or 'ftir'

    Returns:
        SpectrumPreprocessor configured for the stage
    """
    config = PreprocessingConfig(modality=modality)
    return SpectrumPreprocessor(config)


# Utility functions for external compatibility
def preprocess_for_inference(x: np.ndarray, y: np.ndarray, modality: str = "raman") -> Tuple[np.ndarray, np.ndarray]:
    """
    Process spectrum for inference using centralized preprocessor.
    Maintains compatibility with existing code.
    """
    preprocessor = create_preprocessor("inference", modality)
    result = preprocessor.process(x, y)
    return result.x_processed, result.y_processed


def preprocess_for_training(x: np.ndarray, y: np.ndarray, modality: str = "raman") -> Tuple[np.ndarray, np.ndarray]:
    """
    Process spectrum for training using centralized preprocessor.
    Maintains compatibility with existing code.
    """
    preprocessor = create_preprocessor("training", modality)
    result = preprocessor.process(x, y)
    return result.x_processed, result.y_processed
