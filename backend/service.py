# pylint:  wrong-import-order, unused-import, import-outside-toplevel
"""
Backend service layer for ML inference.
Extracts and preserves the current Streamlit application logic for FastAPI.
Maintains scientific fidelity and deterministic outputs.
"""
import time
import gc
from typing import Tuple, List
from pathlib import Path
from datetime import datetime
import psutil
import torch
import torch.nn.functional as F
import numpy as np

from backend.utils.preprocessing import (
    # We will replace these with SpectrumPreprocessor
    # remove_baseline, smooth_spectrum, normalize_spectrum,
    validate_spectrum_modality,
    MODALITY_PARAMS
)
from .config import TARGET_LEN, LABEL_MAP
from backend.models.registry import get_model_info as get_registry_model_info, choices
from backend.utils.performance  import log_model_performance
from .pydantic_models import (
    SpectrumData,
    PredictionResult,
    PreprocessingMetadata,
    QualityControlMetadata,
    ModelMetadata,
    ModelInfo,
    SystemInfo,
    SystemHealth
)
from backend.utils.model_manager import model_manager
from backend.utils.preprocessing_fixed import SpectrumPreprocessor


class MLServiceError(Exception):
    """Custom exception for ML service errors."""

    pass


class MLInferenceService:
    """
    Core ML inference service that preserves the exact behavior of the Streamlit app.
    Maintains scientific fidelity and deterministic outputs.
    """

    def __init__(self, model_manager_instance=model_manager):
        self.model_manager = model_manager_instance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def cleanup_memory(self):
        """Clean up memory after inference"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_preprocessing_metadata(
        self,
        modality: str,
        original_length: int,
        x_data: np.ndarray,
        validation_result: Tuple[bool, List[str]]
    ) -> PreprocessingMetadata:
        """Create preprocessing provenance metadata"""
        params = MODALITY_PARAMS.get(modality, MODALITY_PARAMS["raman"])
        is_valid, issues = validation_result

        return PreprocessingMetadata(
            target_length=TARGET_LEN,
            baseline_degree=params["baseline_degree"],
            smooth_window=params["smooth_window"],
            smooth_polyorder=params["smooth_polyorder"],
            normalization_method="min_max",
            modality_validated=is_valid,
            validation_issues=issues,
            original_length=original_length,
            wavenumber_range=[float(np.min(x_data)), float(np.max(x_data))]
        )

    def create_quality_control_metadata(
        self,
        y_data: np.ndarray,
        y_processed: np.ndarray
    ) -> QualityControlMetadata:
        """Create quality control metadata with basic checks"""
        issues = []

        # Basic signal quality checks
        signal_range = np.max(y_data) - np.min(y_data)
        noise_estimate = np.std(np.diff(y_data))
        snr = signal_range / noise_estimate if noise_estimate > 0 else None

        # Check for saturation (values at extremes)
        if np.any(y_data >= 0.99 * np.max(y_data)):
            issues.append("Potential signal saturation detected")

        # Check for cosmic rays (sudden spikes)
        diff = np.abs(np.diff(y_data))
        if len(diff) > 0:
            threshold = np.mean(diff) + 5 * np.std(diff)
            cosmic_ray_detected = np.any(diff > threshold)
            if cosmic_ray_detected:
                issues.append("Potential cosmic ray spikes detected")
        else:
            cosmic_ray_detected = False

        # Baseline stability
        baseline_stability = 0.0
        if len(y_processed) >= 100:
            baseline_stability = 1.0 - \
                (np.std(y_processed[:50]) + np.std(y_processed[-50:])) / 2
            baseline_stability = max(0.0, min(1.0, float(baseline_stability)))

        return QualityControlMetadata(
            signal_to_noise_ratio=snr,
            baseline_stability=baseline_stability if baseline_stability > 0 else None,
            spectral_resolution=None,
            cosmic_ray_detected=bool(cosmic_ray_detected),
            saturation_detected=any("saturation" in issue.lower()
                                    for issue in issues),
            issues=issues
        )

    def create_model_metadata(
        self,
        model_name: str,
        weights_loaded: bool,
        weights_path: Path
    ) -> ModelMetadata:
        """Create model metadata with calibration details"""
        info = get_registry_model_info(model_name)

        return ModelMetadata(
            model_name=model_name,
            model_description=info.get("description", ""),
            model_version=None,
            training_date=None,
            input_length=info.get("input_length", TARGET_LEN),
            num_classes=info.get("num_classes", 2),
            parameters_count=info.get("parameters", "Unknown"),
            performance_metrics=info.get("performance", {}),
            supported_modalities=info.get("modalities", ["raman", "ftir"]),
            citation=info.get("citation", ""),
            weights_loaded=weights_loaded, # This comes from model_manager
            weights_path=weights_path if weights_loaded else None
        )

    def run_inference(
        self,
        spectrum: SpectrumData,
        model_name: str,
        modality: str,
        include_provenance: bool = True
    ) -> PredictionResult:
        """
        Run model inference preserving exact Streamlit behavior.
        Returns complete result with full provenance metadata.
        """
        start_total = time.time()
        start_memory = self.get_memory_usage()

        # Convert input data
        x_data = np.array(spectrum.x_values)
        y_data = np.array(spectrum.y_values)
        original_length = len(y_data)

        if original_length < 2:
            raise MLServiceError("Spectrum must have at least 2 data points")

        # Validate modality
        validation_result = validate_spectrum_modality(x_data, y_data, modality)

        # Preprocessing
        start_preprocess = time.time()
        # Use SpectrumPreprocessor for consistent preprocessing
        preprocessor = SpectrumPreprocessor(
            target_len=TARGET_LEN,
            do_baseline=True, # Assuming these are desired for standard analysis
            do_smooth=True,
            do_normalize=True,
            modality=modality
        )
        y_processed = preprocessor.preprocess_single_spectrum(x_data, y_data, use_fitted_stats=False)
        # For x_resampled, we can just generate it based on target_len and original range
        x_resampled = np.linspace(np.min(x_data), np.max(x_data), TARGET_LEN)

        preprocessing_time = time.time() - start_preprocess

        # Load model
        model, weights_loaded, weights_path = self.model_manager.load_model(model_name)
        if model is None:
            raise MLServiceError(f"Model '{model_name}' not available")

        # Create input tensor
        input_tensor = torch.tensor(y_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device) # Move to device
        # Inference
        start_inference = time.time()
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=1).item()
            logits_list = logits.detach().numpy().tolist()[0]
            probs = F.softmax(logits.detach(), dim=1).cpu().numpy().flatten()

        inference_time = time.time() - start_inference
        total_time = time.time() - start_total
        end_memory = self.get_memory_usage()
        memory_usage = max(end_memory - start_memory, 0)

        # Log performance metrics for benchmarking
        log_model_performance(
            model_name=model_name,
            inference_time=inference_time,
            preprocessing_time=preprocessing_time,
            total_time=total_time,
            memory_usage=memory_usage,
            spectrum_length=original_length
        )

        # Calculate confidence
        confidence = float(max(probs)) if probs is not None and len(
            probs) > 0 else 0.0

        # Create metadata
        if include_provenance:
            preprocessing_metadata = self.create_preprocessing_metadata(
                modality, original_length, x_data, validation_result
            )
            qc_metadata = self.create_quality_control_metadata(
                y_data, y_processed)
            model_metadata = self.create_model_metadata(
                model_name, weights_loaded, weights_path)
        else:
            preprocessing_metadata = PreprocessingMetadata(
                target_length=TARGET_LEN,
                baseline_degree=2,
                smooth_window=11,
                smooth_polyorder=2,
                normalization_method="min_max",
                modality_validated=validation_result[0],
                validation_issues=validation_result[1],
                original_length=original_length,
                wavenumber_range=[float(np.min(x_data)), float(np.max(x_data))]
            )
            qc_metadata = QualityControlMetadata(
                signal_to_noise_ratio=None,
                baseline_stability=None,
                spectral_resolution=None,
                cosmic_ray_detected=False,
                saturation_detected=False,
                issues=[]
            )
            model_metadata = self.create_model_metadata(
                model_name, weights_loaded, weights_path) # Still need model metadata

        # Create processed spectrum data
        processed_spectrum = SpectrumData(
            x_values=x_resampled.tolist(),
            y_values=y_resampled.tolist(),
            filename=f"processed_{spectrum.filename}" if spectrum.filename else None
        )

        # Clean up memory
        self.cleanup_memory()

        return PredictionResult(
            prediction=prediction,
            prediction_label=LABEL_MAP[prediction] if prediction in LABEL_MAP else "Unknown",
            confidence=confidence,
            probabilities=probs.tolist(),
            logits=logits_list,
            preprocessing=preprocessing_metadata,
            quality_control=qc_metadata,
            model_metadata=model_metadata,
            inference_time=inference_time,
            preprocessing_time=preprocessing_time,
            total_time=total_time,
            memory_usage_mb=memory_usage,
            original_spectrum=spectrum,
            processed_spectrum=processed_spectrum,
            timestamp=datetime.now().isoformat()
        )

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models with their information"""
        return self.model_manager.get_available_models()

    def get_system_info(self) -> SystemInfo:
        """Get system information and health status"""
        models = self.model_manager.get_available_models()

        system_health_data = SystemHealth(
            status="ok",
            timestamp=time.time(),
            models_loaded=sum(1 for m in models if m.available),
            total_models=len(models),
            memory_usage_mb=self.get_memory_usage(),
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available()
        )

        return SystemInfo(
            version="1.0.0",
            available_models=models,
            supported_modalities=["raman", "ftir"],
            max_batch_size=100,
            target_spectrum_length=TARGET_LEN,
            system_health=system_health_data
        )


# Global service instance
ml_service = MLInferenceService()
