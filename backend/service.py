# pylint:  wrong-import-order, unused-import, import-outside-toplevel
"""
Backend service layer for ML inference.
Extracts and preserves the current Streamlit application logic for FastAPI.
Maintains scientific fidelity and deterministic outputs.
"""

import os
import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
import uuid

from .config import TARGET_LEN, LABEL_MAP
from backend.models.registry import build, choices, get_model_info
from backend.utils.preprocessing import (
    preprocess_spectrum,
    validate_spectrum_modality,
    MODALITY_PARAMS
)
from .utils.confidence import calculate_softmax_confidence
from .utils.performance import log_model_performance, PerformanceBenchmark
from backend.models import (
    SpectrumData,
    PredictionResult,
    PreprocessingMetadata,
    QualityControlMetadata,
    ModelMetadata,
    ModelInfo,
    SystemInfo,
    SystemHealth
)


class MLServiceError(Exception):
    """Custom exception for ML service errors."""

    pass


class MLInferenceService:
    """
    Core ML inference service that preserves the exact behavior of the Streamlit app.
    Maintains scientific fidelity and deterministic outputs.
    """

    def __init__(self):
        self._model_cache = {}
        self._weights_cache = {}

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
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

    def load_state_dict(self, model_path: str) -> Optional[Dict]:
        """Load state dict with caching"""
        try:
            if not os.path.exists(model_path):
                return None

            mtime = os.path.getmtime(model_path)
            cache_key = f"{model_path}:{mtime}"

            if cache_key not in self._weights_cache:
                self._weights_cache[cache_key] = torch.load(
                    model_path, map_location="cpu", weights_only=True)

            return self._weights_cache[cache_key]
        except (FileNotFoundError, RuntimeError, OSError) as e:
            print(f"Error loading state dict: {e}")
            return None

    def load_model(self, model_name: str) -> Tuple[Optional[torch.nn.Module], bool, str]:
        """
        Load model with weights. Returns (model, weights_loaded, weights_path).
        Preserves exact Streamlit caching behavior.
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        if model_name not in choices():
            return None, False, ""

        # Build model from registry
        model = build(model_name, TARGET_LEN)

        # Try to load weights from standard locations
        weight_paths = [
            f"model_weights/{model_name}_model.pth",
            f"outputs/{model_name}_model.pth",
            f"model_weights/{model_name}.pth",
            f"outputs/{model_name}.pth",
        ]

        weights_loaded = False
        loaded_path = ""

        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                try:
                    state_dict = self.load_state_dict(weight_path)
                    if state_dict:
                        model.load_state_dict(state_dict, strict=True)
                        model.eval()
                        weights_loaded = True
                        loaded_path = weight_path
                        break
                except (OSError, RuntimeError, KeyError):
                    continue

        result = (model, weights_loaded, loaded_path)
        self._model_cache[model_name] = result
        return result

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
        weights_path: str
    ) -> ModelMetadata:
        """Create model metadata with calibration details"""
        info = get_model_info(model_name)

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
            weights_loaded=weights_loaded,
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
        validation_result = validate_spectrum_modality(
            x_data, y_data, modality)

        # Preprocessing
        start_preprocess = time.time()
        x_resampled, y_resampled = preprocess_spectrum(
            x_data, y_data, modality=modality)
        preprocessing_time = time.time() - start_preprocess

        # Load model
        model, weights_loaded, weights_path = self.load_model(model_name)
        if model is None:
            raise MLServiceError(f"Model '{model_name}' not available")

        # Create input tensor
        input_tensor = torch.tensor(
            y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

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
                y_data, y_resampled)
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
                model_name, weights_loaded, weights_path)

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
        models = []
        for model_name in choices():
            info = get_model_info(model_name)
            _, weights_loaded, _ = self.load_model(model_name)

            models.append(ModelInfo(
                name=model_name,
                description=info.get("description", ""),
                input_length=info.get("input_length", TARGET_LEN),
                num_classes=info.get("num_classes", 2),
                supported_modalities=info.get("modalities", ["raman", "ftir"]),
                performance=info.get("performance", {}),
                parameters=info.get("parameters"),
                speed=info.get("speed"),
                citation=info.get("citation"),
                available=weights_loaded
            ))

        return models

    def get_system_info(self) -> SystemInfo:
        """Get system information and health status"""
        models = self.get_available_models()

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
