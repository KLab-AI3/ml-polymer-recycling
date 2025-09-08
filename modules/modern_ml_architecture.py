"""
Modern ML Architecture for POLYMEROS
Implements transformer-based models, multi-task learning, and ensemble methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from scipy import stats
import warnings
import json
from pathlib import Path


@dataclass
class ModelPrediction:
    """Structured prediction output with uncertainty quantification"""

    prediction: Union[int, float, np.ndarray]
    confidence: float
    uncertainty_epistemic: float  # Model uncertainty
    uncertainty_aleatoric: float  # Data uncertainty
    class_probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


@dataclass
class MultiTaskTarget:
    """Multi-task learning targets"""

    classification_target: Optional[int] = None  # Polymer type classification
    degradation_level: Optional[float] = None  # Continuous degradation score
    property_predictions: Optional[Dict[str, float]] = None  # Material properties
    aging_rate: Optional[float] = None  # Rate of aging prediction


class SpectralTransformerBlock(nn.Module):
    """Transformer block optimized for spectral data"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.ln1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.ff_network(x)
        x = self.ln2(x + self.dropout(ff_output))

        return x


class SpectralPositionalEncoding(nn.Module):
    """Positional encoding adapted for spectral wavenumber information"""

    def __init__(self, d_model: int, max_seq_length: int = 2000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Use different frequencies for different dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


class SpectralTransformer(nn.Module):
    """Transformer architecture optimized for spectral analysis"""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_length: int = 2000,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = SpectralPositionalEncoding(d_model, max_seq_length)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                SpectralTransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Regression heads for multi-task learning
        self.degradation_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.property_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 5),  # Predict 5 material properties
        )

        # Uncertainty estimation layers
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2),  # Epistemic and aleatoric uncertainty
        )

        # Attention pooling for sequence aggregation
        self.attention_pool = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, input_dim = x.shape

        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Store attention weights if requested
        attention_weights = []

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Attention pooling to get sequence representation
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_output, pool_attention = self.attention_pool(query, x, x)
        pooled_output = pooled_output.squeeze(1)  # (batch, d_model)

        if return_attention:
            attention_weights.append(pool_attention)

        # Multi-task outputs
        outputs = {}

        # Classification output
        classification_logits = self.classification_head(pooled_output)
        outputs["classification_logits"] = classification_logits
        outputs["classification_probs"] = F.softmax(classification_logits, dim=-1)

        # Degradation prediction
        degradation_pred = self.degradation_head(pooled_output)
        outputs["degradation_prediction"] = degradation_pred

        # Property predictions
        property_pred = self.property_head(pooled_output)
        outputs["property_predictions"] = property_pred

        # Uncertainty estimation
        uncertainty_pred = self.uncertainty_head(pooled_output)
        outputs["uncertainty_epistemic"] = torch.nn.Softplus()(uncertainty_pred[:, 0])
        outputs["uncertainty_aleatoric"] = F.softplus(uncertainty_pred[:, 1])

        if return_attention:
            outputs["attention_weights"] = attention_weights

        return outputs


class BayesianUncertaintyEstimator:
    """Bayesian uncertainty quantification using Monte Carlo dropout"""

    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self, model: nn.Module):
        """Enable dropout for uncertainty estimation"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification using Monte Carlo dropout

        Args:
            x: Input tensor

        Returns:
            Predictions with uncertainty estimates
        """
        self.model.eval()
        self.enable_dropout(self.model)

        predictions = []
        classification_probs = []
        degradation_preds = []
        uncertainty_estimates = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(x)
                predictions.append(output["classification_probs"])
                classification_probs.append(output["classification_probs"])
                degradation_preds.append(output["degradation_prediction"])
                uncertainty_estimates.append(
                    torch.stack(
                        [
                            output["uncertainty_epistemic"],
                            output["uncertainty_aleatoric"],
                        ],
                        dim=1,
                    )
                )

        # Stack predictions
        classification_stack = torch.stack(
            classification_probs, dim=0
        )  # (num_samples, batch, classes)
        degradation_stack = torch.stack(degradation_preds, dim=0)
        uncertainty_stack = torch.stack(uncertainty_estimates, dim=0)

        # Calculate statistics
        mean_classification = classification_stack.mean(dim=0)
        std_classification = classification_stack.std(dim=0)

        mean_degradation = degradation_stack.mean(dim=0)
        std_degradation = degradation_stack.std(dim=0)

        mean_uncertainty = uncertainty_stack.mean(dim=0)

        # Calculate epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_classification.mean(dim=1)

        # Calculate aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = mean_uncertainty[:, 1]

        return {
            "mean_classification": mean_classification,
            "std_classification": std_classification,
            "mean_degradation": mean_degradation,
            "std_degradation": std_degradation,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "total_uncertainty": epistemic_uncertainty + aleatoric_uncertainty,
        }


class EnsembleModel:
    """Ensemble model combining multiple approaches"""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_fitted = False

    def add_transformer_model(self, model: SpectralTransformer, weight: float = 1.0):
        """Add transformer model to ensemble"""
        self.models["transformer"] = model
        self.weights["transformer"] = weight

    def add_random_forest(self, n_estimators: int = 100, weight: float = 1.0):
        """Add Random Forest to ensemble"""
        self.models["random_forest_clf"] = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, oob_score=True
        )
        self.models["random_forest_reg"] = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, oob_score=True
        )
        self.weights["random_forest"] = weight

    def add_xgboost(self, weight: float = 1.0):
        """Add XGBoost to ensemble"""
        self.models["xgboost_clf"] = xgb.XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        )
        self.models["xgboost_reg"] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.weights["xgboost"] = weight

    def fit(
        self,
        X: np.ndarray,
        y_classification: np.ndarray,
        y_degradation: Optional[np.ndarray] = None,
    ):
        """
        Fit ensemble models

        Args:
            X: Input features (flattened spectra for traditional ML models)
            y_classification: Classification targets
            y_degradation: Degradation targets (optional)
        """
        # Fit Random Forest
        if "random_forest_clf" in self.models:
            self.models["random_forest_clf"].fit(X, y_classification)
            if y_degradation is not None:
                self.models["random_forest_reg"].fit(X, y_degradation)

        # Fit XGBoost
        if "xgboost_clf" in self.models:
            self.models["xgboost_clf"].fit(X, y_classification)
            if y_degradation is not None:
                self.models["xgboost_reg"].fit(X, y_degradation)

        self.is_fitted = True

    def predict(
        self, X: np.ndarray, X_transformer: Optional[torch.Tensor] = None
    ) -> ModelPrediction:
        """
        Ensemble prediction with uncertainty quantification

        Args:
            X: Input features for traditional ML models
            X_transformer: Input tensor for transformer model

        Returns:
            Ensemble prediction with uncertainty
        """
        if not self.is_fitted and "transformer" not in self.models:
            raise ValueError(
                "Ensemble must be fitted or contain pre-trained transformer"
            )

        predictions = {}
        classification_probs = []
        degradation_preds = []
        model_weights = []

        # Random Forest predictions
        if (
            "random_forest_clf" in self.models
            and self.models["random_forest_clf"] is not None
        ):
            rf_probs = self.models["random_forest_clf"].predict_proba(X)
            classification_probs.append(rf_probs)
            model_weights.append(self.weights["random_forest"])

            if "random_forest_reg" in self.models:
                rf_degradation = self.models["random_forest_reg"].predict(X)
                degradation_preds.append(rf_degradation)

        # XGBoost predictions
        if "xgboost_clf" in self.models and self.models["xgboost_clf"] is not None:
            xgb_probs = self.models["xgboost_clf"].predict_proba(X)
            classification_probs.append(xgb_probs)
            model_weights.append(self.weights["xgboost"])

            if "xgboost_reg" in self.models:
                xgb_degradation = self.models["xgboost_reg"].predict(X)
                degradation_preds.append(xgb_degradation)

        # Transformer predictions
        if "transformer" in self.models and X_transformer is not None:
            transformer_output = self.models["transformer"](X_transformer)
            transformer_probs = (
                transformer_output["classification_probs"].detach().numpy()
            )
            classification_probs.append(transformer_probs)
            model_weights.append(self.weights["transformer"])

            transformer_degradation = (
                transformer_output["degradation_prediction"].detach().numpy()
            )
            degradation_preds.append(transformer_degradation.flatten())

        # Weighted ensemble
        if classification_probs:
            model_weights = np.array(model_weights)
            model_weights = model_weights / np.sum(model_weights)  # Normalize

            # Weighted average of probabilities
            ensemble_probs = np.zeros_like(classification_probs[0])
            for i, probs in enumerate(classification_probs):
                ensemble_probs += model_weights[i] * probs

            # Predicted class
            predicted_class = np.argmax(ensemble_probs, axis=1)[0]
            confidence = np.max(ensemble_probs, axis=1)[0]

            # Calculate uncertainty from model disagreement
            prob_variance = np.var([probs[0] for probs in classification_probs], axis=0)
            epistemic_uncertainty = np.mean(prob_variance)

            # Aleatoric uncertainty (average across models)
            aleatoric_uncertainty = 1.0 - confidence  # Simple estimate

            # Degradation prediction
            ensemble_degradation = None
            if degradation_preds:
                ensemble_degradation = np.average(
                    degradation_preds, weights=model_weights, axis=0
                )[0]

        else:
            raise ValueError("No valid predictions could be made")

        # Feature importance (from Random Forest if available)
        feature_importance = None
        if (
            "random_forest_clf" in self.models
            and self.models["random_forest_clf"] is not None
        ):
            importance = self.models["random_forest_clf"].feature_importances_
            # Convert to wavenumber-based importance (assuming spectral input)
            feature_importance = {
                f"wavenumber_{i}": float(importance[i]) for i in range(len(importance))
            }

        return ModelPrediction(
            prediction=predicted_class,
            confidence=confidence,
            uncertainty_epistemic=epistemic_uncertainty,
            uncertainty_aleatoric=aleatoric_uncertainty,
            class_probabilities=ensemble_probs[0],
            feature_importance=feature_importance,
            explanation=self._generate_explanation(
                predicted_class, confidence, ensemble_degradation
            ),
        )

    def _generate_explanation(
        self,
        predicted_class: int,
        confidence: float,
        degradation: Optional[float] = None,
    ) -> str:
        """Generate human-readable explanation"""
        class_names = {0: "Stable (Unweathered)", 1: "Weathered"}
        class_name = class_names.get(predicted_class, f"Class {predicted_class}")

        explanation = f"Predicted class: {class_name} (confidence: {confidence:.3f})"

        if degradation is not None:
            explanation += f"\nEstimated degradation level: {degradation:.3f}"

        if confidence > 0.8:
            explanation += "\nHigh confidence prediction - strong spectral evidence"
        elif confidence > 0.6:
            explanation += "\nModerate confidence - some uncertainty in classification"
        else:
            explanation += "\nLow confidence - significant uncertainty, consider additional analysis"

        return explanation


class MultiTaskLearningFramework:
    """Framework for multi-task learning in polymer analysis"""

    def __init__(self, model: SpectralTransformer):
        self.model = model
        self.task_weights = {
            "classification": 1.0,
            "degradation": 0.5,
            "properties": 0.3,
        }
        self.optimizer = None
        self.scheduler = None

    def setup_training(self, learning_rate: float = 1e-4):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: MultiTaskTarget,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss

        Args:
            outputs: Model outputs
            targets: Multi-task targets
            batch_size: Batch size

        Returns:
            Loss components
        """
        losses = {}
        total_loss = 0

        # Classification loss
        if targets.classification_target is not None:
            classification_loss = F.cross_entropy(
                outputs["classification_logits"],
                torch.tensor(
                    [targets.classification_target] * batch_size, dtype=torch.long
                ),
            )
            losses["classification"] = classification_loss
            total_loss += self.task_weights["classification"] * classification_loss

        # Degradation regression loss
        if targets.degradation_level is not None:
            degradation_loss = F.mse_loss(
                outputs["degradation_prediction"].squeeze(),
                torch.tensor(
                    [targets.degradation_level] * batch_size, dtype=torch.float
                ),
            )
            losses["degradation"] = degradation_loss
            total_loss += self.task_weights["degradation"] * degradation_loss

        # Property prediction loss
        if targets.property_predictions is not None:
            property_targets = torch.tensor(
                [[targets.property_predictions.get(f"prop_{i}", 0.0) for i in range(5)]]
                * batch_size,
                dtype=torch.float,
            )
            property_loss = F.mse_loss(
                outputs["property_predictions"], property_targets
            )
            losses["properties"] = property_loss
            total_loss += self.task_weights["properties"] * property_loss

        # Uncertainty regularization
        uncertainty_reg = torch.mean(outputs["uncertainty_epistemic"]) + torch.mean(
            outputs["uncertainty_aleatoric"]
        )
        losses["uncertainty_reg"] = uncertainty_reg
        total_loss += 0.01 * uncertainty_reg  # Small weight for regularization

        losses["total"] = total_loss
        return losses

    def train_step(self, x: torch.Tensor, targets: MultiTaskTarget) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        if self.optimizer is None:
            raise ValueError(
                "Optimizer is not initialized. Call setup_training() to initialize it."
            )
        self.optimizer.zero_grad()

        outputs = self.model(x)
        losses = self.compute_loss(outputs, targets, x.size(0))

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if self.optimizer is None:
            raise ValueError(
                "Optimizer is not initialized. Call setup_training() to initialize it."
            )
        self.optimizer.step()

        return {
            k: float(v.item()) if torch.is_tensor(v) else float(v)
            for k, v in losses.items()
        }


class ModernMLPipeline:
    """Complete modern ML pipeline for polymer analysis"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.transformer_model = None
        self.ensemble_model = None
        self.uncertainty_estimator = None
        self.multi_task_framework = None

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "transformer": {
                "d_model": 256,
                "num_heads": 8,
                "num_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
                "num_classes": 2,
            },
            "ensemble": {
                "transformer_weight": 0.4,
                "random_forest_weight": 0.3,
                "xgboost_weight": 0.3,
            },
            "uncertainty": {"num_mc_samples": 50},
            "training": {"learning_rate": 1e-4, "batch_size": 32, "num_epochs": 100},
        }

    def initialize_models(self, input_dim: int = 1, max_seq_length: int = 2000):
        """Initialize all models"""
        # Transformer model
        self.transformer_model = SpectralTransformer(
            input_dim=input_dim,
            d_model=self.config["transformer"]["d_model"],
            num_heads=self.config["transformer"]["num_heads"],
            num_layers=self.config["transformer"]["num_layers"],
            d_ff=self.config["transformer"]["d_ff"],
            max_seq_length=max_seq_length,
            num_classes=self.config["transformer"]["num_classes"],
            dropout=self.config["transformer"]["dropout"],
        )

        # Uncertainty estimator
        self.uncertainty_estimator = BayesianUncertaintyEstimator(
            self.transformer_model,
            num_samples=self.config["uncertainty"]["num_mc_samples"],
        )

        # Multi-task framework
        self.multi_task_framework = MultiTaskLearningFramework(self.transformer_model)

        # Ensemble model
        self.ensemble_model = EnsembleModel()
        self.ensemble_model.add_transformer_model(
            self.transformer_model, self.config["ensemble"]["transformer_weight"]
        )
        self.ensemble_model.add_random_forest(
            weight=self.config["ensemble"]["random_forest_weight"]
        )
        self.ensemble_model.add_xgboost(
            weight=self.config["ensemble"]["xgboost_weight"]
        )

    def train_ensemble(
        self,
        X_flat: np.ndarray,
        X_transformer: torch.Tensor,
        y_classification: np.ndarray,
        y_degradation: Optional[np.ndarray] = None,
    ):
        """Train the ensemble model"""
        if self.ensemble_model is None:
            raise ValueError("Models not initialized. Call initialize_models() first.")

        # Train traditional ML models
        self.ensemble_model.fit(X_flat, y_classification, y_degradation)

        # Setup transformer training
        if self.multi_task_framework is None:
            raise ValueError(
                "Multi-task framework is not initialized. Call initialize_models() first."
            )
        self.multi_task_framework.setup_training(
            self.config["training"]["learning_rate"]
        )

        print(
            "Ensemble training completed (transformer training would require full training loop)"
        )

    def predict_with_all_methods(
        self, X_flat: np.ndarray, X_transformer: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive prediction using all methods

        Args:
            X_flat: Flattened spectral data for traditional ML
            X_transformer: Tensor format for transformer

        Returns:
            Complete prediction results
        """
        results = {}

        # Ensemble prediction
        if self.ensemble_model is None:
            raise ValueError(
                "Ensemble model is not initialized. Call initialize_models() first."
            )
        ensemble_pred = self.ensemble_model.predict(X_flat, X_transformer)
        results["ensemble"] = ensemble_pred

        # Transformer with uncertainty
        if self.transformer_model is not None:
            if self.uncertainty_estimator is None:
                raise ValueError(
                    "Uncertainty estimator is not initialized. Call initialize_models() first."
                )
            uncertainty_pred = self.uncertainty_estimator.predict_with_uncertainty(
                X_transformer
            )
            results["transformer_uncertainty"] = uncertainty_pred

        # Individual model predictions for comparison
        individual_predictions = {}

        if (
            self.ensemble_model is not None
            and "random_forest_clf" in self.ensemble_model.models
        ):
            rf_pred = self.ensemble_model.models["random_forest_clf"].predict_proba(
                X_flat
            )[0]
            individual_predictions["random_forest"] = rf_pred

        if "xgboost_clf" in self.ensemble_model.models:
            xgb_pred = self.ensemble_model.models["xgboost_clf"].predict_proba(X_flat)[
                0
            ]
            individual_predictions["xgboost"] = xgb_pred

        results["individual_models"] = individual_predictions

        return results

    def get_model_insights(
        self, X_flat: np.ndarray, X_transformer: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Generate insights about model behavior and predictions

        Args:
            X_flat: Flattened spectral data
            X_transformer: Transformer input format

        Returns:
            Model insights and explanations
        """
        insights = {}

        # Feature importance from Random Forest
        if "random_forest_clf" in self.ensemble_model.models:
            if (
                self.ensemble_model
                and "random_forest_clf" in self.ensemble_model.models
                and self.ensemble_model.models["random_forest_clf"] is not None
            ):
                rf_importance = self.ensemble_model.models[
                    "random_forest_clf"
                ].feature_importances_
            else:
                rf_importance = None
            if rf_importance is not None:
                top_features = np.argsort(rf_importance)[-10:][::-1]
            else:
                top_features = []
            insights["top_spectral_regions"] = {
                f"wavenumber_{idx}": float(rf_importance[idx])
                for idx in top_features
                if rf_importance is not None
            }

        # Attention weights from transformer
        if self.transformer_model is not None:
            self.transformer_model.eval()
            with torch.no_grad():
                outputs = self.transformer_model(X_transformer, return_attention=True)
                if "attention_weights" in outputs:
                    insights["attention_patterns"] = outputs["attention_weights"]

        # Uncertainty analysis
        predictions = self.predict_with_all_methods(X_flat, X_transformer)
        if "transformer_uncertainty" in predictions:
            uncertainty_data = predictions["transformer_uncertainty"]
            insights["uncertainty_analysis"] = {
                "epistemic_uncertainty": float(
                    uncertainty_data["epistemic_uncertainty"].mean()
                ),
                "aleatoric_uncertainty": float(
                    uncertainty_data["aleatoric_uncertainty"].mean()
                ),
                "total_uncertainty": float(
                    uncertainty_data["total_uncertainty"].mean()
                ),
                "confidence_level": (
                    "high"
                    if uncertainty_data["total_uncertainty"].mean() < 0.1
                    else (
                        "medium"
                        if uncertainty_data["total_uncertainty"].mean() < 0.3
                        else "low"
                    )
                ),
            }

        # Model agreement analysis
        if "individual_models" in predictions:
            individual = predictions["individual_models"]
            agreements = []
            for model1_name, model1_pred in individual.items():
                for model2_name, model2_pred in individual.items():
                    if model1_name != model2_name:
                        # Calculate agreement based on prediction similarity
                        agreement = 1.0 - np.abs(model1_pred - model2_pred).mean()
                        agreements.append(agreement)

            insights["model_agreement"] = {
                "average_agreement": float(np.mean(agreements)) if agreements else 0.0,
                "agreement_level": (
                    "high"
                    if np.mean(agreements) > 0.8
                    else "medium" if np.mean(agreements) > 0.6 else "low"
                ),
            }

        return insights

    def save_models(self, save_path: Path):
        """Save trained models"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save transformer model
        if self.transformer_model is not None:
            torch.save(
                self.transformer_model.state_dict(), save_path / "transformer_model.pth"
            )

        # Save configuration
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        print(f"Models saved to {save_path}")

    def load_models(self, load_path: Path):
        """Load pre-trained models"""
        load_path = Path(load_path)

        # Load configuration
        with open(load_path / "config.json", "r") as f:
            self.config = json.load(f)

        # Initialize and load transformer
        self.initialize_models()
        if (
            self.transformer_model is not None
            and (load_path / "transformer_model.pth").exists()
        ):
            self.transformer_model.load_state_dict(
                torch.load(load_path / "transformer_model.pth", map_location="cpu")
            )
        else:
            raise ValueError(
                "Transformer model is not initialized or model file is missing."
            )

        print(f"Models loaded from {load_path}")


# Utility functions for data preparation
def prepare_transformer_input(
    spectral_data: np.ndarray, max_length: int = 2000
) -> torch.Tensor:
    """
    Prepare spectral data for transformer input

    Args:
        spectral_data: Raw spectral intensities (1D array)
        max_length: Maximum sequence length

    Returns:
        Formatted tensor for transformer
    """
    # Ensure proper length
    if len(spectral_data) > max_length:
        # Downsample
        indices = np.linspace(0, len(spectral_data) - 1, max_length, dtype=int)
        spectral_data = spectral_data[indices]
    elif len(spectral_data) < max_length:
        # Pad with zeros
        padding = np.zeros(max_length - len(spectral_data))
        spectral_data = np.concatenate([spectral_data, padding])

    # Reshape for transformer: (batch_size, sequence_length, features)
    return torch.tensor(spectral_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


def create_multitask_targets(
    classification_label: int,
    degradation_score: Optional[float] = None,
    material_properties: Optional[Dict[str, float]] = None,
) -> MultiTaskTarget:
    """
    Create multi-task learning targets

    Args:
        classification_label: Classification target (0 or 1)
        degradation_score: Continuous degradation score [0, 1]
        material_properties: Dictionary of material properties

    Returns:
        MultiTaskTarget object
    """
    return MultiTaskTarget(
        classification_target=classification_label,
        degradation_level=degradation_score,
        property_predictions=material_properties,
    )
