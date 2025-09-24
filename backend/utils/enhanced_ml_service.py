# pylint:  disable=wrong-import-order, unused-import
"""
Enhanced API endpoints with explainability features.
Extends the existing FastAPI backend with SHAP-based model explanations
and improved prediction capabilities.
"""

from backend.models.registry import choices  # Ensure choices is imported
from backend.config import TARGET_LEN  # Import TARGET_LEN for model loading
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from fastapi import HTTPException  # Keep HTTPException for API errors
# PredictionResult is not directly returned by this service
from backend.pydantic_models import SpectrumData
from backend.models.registry import build as build_model, choices
from backend.utils.preprocessing_fixed import SpectrumPreprocessor

import os
# Import moved here to the toplevel
from backend.utils.model_manager import model_manager


class EnhancedMLService:
    """
    Enhanced ML service with explainability features.
    Provides predictions with feature importance and model confidence.
    """

    def __init__(self):
        self.model_manager = model_manager
        # Local cache for loaded models (model, preprocessor)
        self._model_cache = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Enhanced ML Service initialized on {self.device}")

    def predict_with_explanation(
        self,
        spectrum_data: SpectrumData,
        model_name: str,
        modality: str = "raman",
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with explainability features.

        Args:
            spectrum_data (SpectrumData): Input spectrum data
            model_name (str): Name of model to use
            modality (str): The spectroscopy modality ('raman' or 'ftir')
            include_feature_importance (bool): Whether to compute feature importance

        Returns:
            dict: Prediction results with explanations
        """
        if model_name not in self._model_cache:
            # Attempt to load model via centralized manager if not in local cache
            model_instance, weights_loaded, _ = self.model_manager.load_model(
                model_name)
            if model_instance is None or not weights_loaded:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {model_name} not loaded or weights not found"
                )

            # Determine model input length robustly: prefer model attribute,
            # fallback to registry/spec, then TARGET_LEN
            input_len = getattr(model_instance, 'input_length', None)
            if input_len is None:
                try:
                    spec_info = registry_spec(model_name)
                    input_len = int(spec_info.get("input_length", TARGET_LEN))
                except Exception:
                    input_len = TARGET_LEN

            # Create preprocessor for this model (use resolved input_len)
            preprocessor = SpectrumPreprocessor(
                target_len=input_len,
                do_baseline=True,
                do_smooth=True,
                do_normalize=True,
                modality=modality  # Use the provided modality
            )
            self._model_cache[model_name] = {
                'model': model_instance, 'preprocessor': preprocessor}

        model_entry = self._model_cache.get(model_name)
        if not model_entry:  # Should not happen if previous block executed
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not loaded"
            )
        model = model_entry['model']

        # --- FIX: Ensure preprocessor has the correct modality ---
        # The preprocessor might have been cached with a default or different modality.
        # We must ensure it matches the one from the current request.
        if model_entry['preprocessor'].modality != modality:
            print(
                f"🔄 Updating preprocessor modality for '{model_name}' from '{model_entry['preprocessor'].modality}' to '{modality}'")
            model_entry['preprocessor'] = SpectrumPreprocessor(
                target_len=model.input_length,
                do_baseline=True, do_smooth=True, do_normalize=True,
                modality=modality
            )

        preprocessor = model_entry['preprocessor']

        try:
            # Preprocess input data
            x_data = np.array(spectrum_data.x_values)
            y_data = np.array(spectrum_data.y_values)

            # Preprocess spectrum
            processed_spectrum = preprocessor.preprocess_single_spectrum(
                x_data, y_data, use_fitted_stats=False
            )

            # Convert to tensor
            input_tensor = torch.tensor(
                processed_spectrum, dtype=torch.float32)
            # Add batch and channel dimensions
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()

            # Basic prediction result
            result = {
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().tolist()[0],
                'class_labels': ['stable', 'weathered'],
                'model_used': model_name,
                'spectrum_filename': spectrum_data.filename
            }

            # Add feature importance if requested
            if include_feature_importance:
                feature_importance = self._compute_feature_importance(
                    model, input_tensor, processed_spectrum
                )
                result['feature_importance'] = feature_importance

            return result

        except (RuntimeError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            ) from e

    def _compute_feature_importance(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        processed_spectrum: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute feature importance using gradient-based methods.

        Args:
            model: PyTorch model
            input_tensor: Preprocessed input tensor
            processed_spectrum: Original processed spectrum

        Returns:
            dict: Feature importance information
        """
        try:
            # Enable gradient computation
            input_tensor.requires_grad_(True)
            torch.set_grad_enabled(True)

            # Forward pass
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

            # Compute gradients with respect to input
            class_score = output[0, predicted_class]
            class_score.backward()

            if input_tensor.grad is not None:
                gradients = input_tensor.grad.data.cpu().numpy().squeeze()
            else:
                raise RuntimeError(
                    "Gradients were not computed. Ensure requires_grad is set "
                    "and gradient computation is enabled."
                )
            gradients = input_tensor.grad.data.cpu().numpy().squeeze()

            # Compute importance metrics
            importance_abs = np.abs(gradients)

            # Find most important regions
            top_indices = np.argsort(importance_abs)[-20:]  # Top 20 features

            # Create interpretable output
            feature_importance = {
                'method': 'gradient_saliency',
                'importance_scores': importance_abs.tolist(),
                'top_features': {
                    'indices': top_indices.tolist(),
                    'values': importance_abs[top_indices].tolist()
                },
                'summary': {
                    'max_importance': float(np.max(importance_abs)),
                    'mean_importance': float(np.mean(importance_abs)),
                    'important_region_start': int(top_indices[0]),
                    'important_region_end': int(top_indices[-1])
                }
            }

            return feature_importance

        except (RuntimeError, ValueError, TypeError) as e:
            print(f"⚠️ Feature importance computation failed: {e}")
            return {
                'method': 'gradient_saliency',
                'error': str(e),
                'importance_scores': [0.0] * len(processed_spectrum)
            }

    def get_model_info(self) -> List[Dict[str, Any]]:
        """
        Get information about loaded models.

        Returns:
            list: List of ModelInfo objects from the centralized manager.
        """
        return self.model_manager.get_available_models()

    def batch_predict_with_explanation(
        self,
        spectra: List[SpectrumData],
        model_name: str,
        modality: str,  # Add modality for preprocessor
        include_feature_importance: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction with explanations.

        Args:
            spectra (list): List of spectrum data
            model_name (str): Model to use
            modality (str): Spectroscopy modality
            include_feature_importance (bool): Whether to include explanations

        Returns:
            list: List of prediction results
        """
        results = []

        for spectrum in spectra:
            try:
                result = self.predict_with_explanation(
                    spectrum,
                    model_name,
                    modality=modality,  # Pass modality down
                    include_feature_importance=include_feature_importance
                )
                results.append(result)
            except (HTTPException, ValueError, RuntimeError) as e:
                results.append({
                    'error': str(e),
                    'spectrum_filename': spectrum.filename
                })

        return results


# Global enhanced service instance
enhanced_ml_service = EnhancedMLService()


def initialize_enhanced_service():
    """Initialize the enhanced ML service with available models."""
    print("Initializing Enhanced ML Service models...")
    # Iterate through all known models in the registry by calling choices() directly
    for model_name in choices():
        try:
            # Attempt to load each model via the centralized manager
            model_instance, weights_loaded, _ = enhanced_ml_service.model_manager.load_model(
                model_name, TARGET_LEN)
            if model_instance and weights_loaded:
                # If successful, create and cache its preprocessor
                preprocessor = SpectrumPreprocessor(
                    target_len=TARGET_LEN,
                    do_baseline=True,
                    do_smooth=True,  # Modality will be set at prediction time
                    do_normalize=True,  # Default to raman, will be updated on-demand
                    modality="raman"
                )
                enhanced_ml_service._model_cache[model_name] = {
                    'model': model_instance,
                    'preprocessor': preprocessor
                }
                print(
                    f"✅ Enhanced ML Service: Prepared model '{model_name}' with preprocessor.")
            else:
                print(
                    f"⚠️ Enhanced ML Service: Model '{model_name}' not fully loaded or weights missing.")
        except (RuntimeError, ValueError, ImportError) as e:
            print(
                f"❌ Enhanced ML Service: Error initializing model '{model_name}': {e}")


# Initialize on import
initialize_enhanced_service()
