# pylint:  disable=wrong-import-order, unused-import
"""
Enhanced API endpoints with explainability features.
Extends the existing FastAPI backend with SHAP-based model explanations
and improved prediction capabilities.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from backend.models import SpectrumData, PredictionResult
from models.registry import build as build_model
from .preprocessing_fixed import SpectrumPreprocessor

import os


class EnhancedMLService:
    """
    Enhanced ML service with explainability features.
    Provides predictions with feature importance and model confidence.
    """

    def __init__(self):
        self.models = {}
        self.preprocessors = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Enhanced ML Service initialized on {self.device}")

    def load_model(self, model_name: str, model_path: str, target_len: int = 500):
        """
        Load a trained model for inference.

        Args:
            model_name (str): Name of the model architecture
            model_path (str): Path to saved model weights
            target_len (int): Expected input length
        """
        try:
            # Build model architecture
            model = build_model(model_name, target_len)

            # Load weights
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()

                self.models[model_name] = {
                    'model': model,
                    'target_len': target_len,
                    'model_path': model_path
                }

                # Create corresponding preprocessor
                self.preprocessors[model_name] = SpectrumPreprocessor(
                    target_len=target_len,
                    do_baseline=True,
                    do_smooth=True,
                    do_normalize=True
                )

                print(f"✅ Loaded model {model_name} from {model_path}")

            else:
                print(f"⚠️ Model weights not found at {model_path}")

        except FileNotFoundError as e:
            print(f"❌ Model file not found for {model_name}: {e}")
        except RuntimeError as e:
            print(f"❌ Torch error while loading model {model_name}: {e}")
        except OSError as e:
            print(f"❌ An error occurred while loading model {model_name}: {e}")

    def predict_with_explanation(
        self,
        spectrum_data: SpectrumData,
        model_name: str,
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with explainability features.

        Args:
            spectrum_data (SpectrumData): Input spectrum data
            model_name (str): Name of model to use
            include_feature_importance (bool): Whether to compute feature importance

        Returns:
            dict: Prediction results with explanations
        """
        if model_name not in self.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not loaded"
            )

        model_info = self.models[model_name]
        model = model_info['model']
        preprocessor = self.preprocessors[model_name]

        try:
            # Preprocess input data
            x_data = np.array(spectrum_data.x_values)
            y_data = np.array(spectrum_data.y_values)

            # Preprocess spectrum
            processed_spectrum = preprocessor.preprocess_single_spectrum(
                x_data, y_data, use_fitted_stats=False
            )

            # Convert to tensor
            input_tensor = torch.tensor(processed_spectrum, dtype=torch.float32)
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

        except Exception as e:
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
            list: List of model information dictionaries
        """
        model_info = []
        for model_name, info in self.models.items():
            model_info.append({
                'name': model_name,
                'target_length': info['target_len'],
                'model_path': info['model_path'],
                'device': str(self.device),
                'available': True
            })

        return model_info

    def batch_predict_with_explanation(
        self,
        spectra: List[SpectrumData],
        model_name: str,
        include_feature_importance: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction with explanations.

        Args:
            spectra (list): List of spectrum data
            model_name (str): Model to use
            include_feature_importance (bool): Whether to include explanations

        Returns:
            list: List of prediction results
        """
        results = []

        for spectrum in spectra:
            try:
                result = self.predict_with_explanation(
                    spectrum, model_name, include_feature_importance
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

# Load default models if available
def initialize_enhanced_service():
    """Initialize the enhanced ML service with available models."""
    model_paths = [
        ("figure2", "outputs/figure2_model.pth"),
        ("resnet", "outputs/resnet_model.pth"),
        ("resnet18vision", "outputs/resnet18vision_model.pth"),
        ("efficient_cnn", "outputs/efficient_cnn_model.pth"),
        ("enhanced_cnn", "outputs/enhanced_cnn_model.pth"),
        ("hybrid_net", "outputs/hybrid_net_model.pth"),
    ]

    for model_name, model_path in model_paths:
        if os.path.exists(model_path):
            enhanced_ml_service.load_model(model_name, model_path)
        else:
            print(f"⚠️ Model weights not found: {model_path}")

# Initialize on import
initialize_enhanced_service()
