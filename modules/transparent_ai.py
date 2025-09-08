"""
Transparent AI Reasoning Engine for POLYMEROS
Provides explainable predictions with uncertainty quantification and hypothesis generation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


@dataclass
class PredictionExplanation:
    """Comprehensive explanation for a model prediction"""

    prediction: int
    confidence: float
    confidence_level: str
    probabilities: np.ndarray
    feature_importance: Dict[str, float]
    reasoning_chain: List[str]
    uncertainty_sources: List[str]
    similar_cases: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class Hypothesis:
    """AI-generated scientific hypothesis"""

    statement: str
    confidence: float
    supporting_evidence: List[str]
    testable_predictions: List[str]
    suggested_experiments: List[str]
    related_literature: List[str]


class UncertaintyEstimator:
    """Bayesian uncertainty estimation for model predictions"""

    def __init__(self, model, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples
        self.epistemic_uncertainty = None
        self.aleatoric_uncertainty = None

    def estimate_uncertainty(self, x: torch.Tensor) -> Dict[str, float]:
        """Estimate prediction uncertainty using Monte Carlo dropout"""
        self.model.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = F.softmax(self.model(x), dim=1)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)

        # Calculate uncertainties
        mean_pred = np.mean(predictions, axis=0)
        epistemic = np.var(predictions, axis=0)  # Model uncertainty
        aleatoric = np.mean(predictions * (1 - predictions), axis=0)  # Data uncertainty

        total_uncertainty = epistemic + aleatoric

        return {
            "epistemic": float(np.mean(epistemic)),
            "aleatoric": float(np.mean(aleatoric)),
            "total": float(np.mean(total_uncertainty)),
            "prediction_variance": float(np.var(mean_pred)),
        }

    def confidence_intervals(
        self, x: torch.Tensor, confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = F.softmax(self.model(x), dim=1)
                predictions.append(pred.cpu().numpy().flatten())

        predictions = np.array(predictions)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        intervals = {}
        for i in range(predictions.shape[1]):
            lower = np.percentile(predictions[:, i], lower_percentile)
            upper = np.percentile(predictions[:, i], upper_percentile)
            intervals[f"class_{i}"] = (lower, upper)

        return intervals


class FeatureImportanceAnalyzer:
    """Advanced feature importance analysis for spectral data"""

    def __init__(self, model):
        self.model = model
        self.shap_explainer = None

        if SHAP_AVAILABLE:
            try:
                # Initialize SHAP explainer for the model
                if SHAP_AVAILABLE:
                    if SHAP_AVAILABLE:
                        self.shap_explainer = shap.DeepExplainer(  # type: ignore
                            model, torch.zeros(1, 500)
                        )
                    else:
                        self.shap_explainer = None
                else:
                    self.shap_explainer = None
            except (ValueError, RuntimeError) as e:
                warnings.warn(f"Could not initialize SHAP explainer: {e}")

    def analyze_feature_importance(
        self, x: torch.Tensor, wavenumbers: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Comprehensive feature importance analysis"""
        importance_data = {}

        # SHAP analysis (if available)
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(x)
                importance_data["shap_values"] = shap_values
                importance_data["shap_available"] = True
            except (ValueError, RuntimeError) as e:
                warnings.warn(f"SHAP analysis failed: {e}")
                importance_data["shap_available"] = False
        else:
            importance_data["shap_available"] = False

        # Gradient-based importance
        x.requires_grad_(True)
        self.model.eval()

        output = self.model(x)
        predicted_class = torch.argmax(output, dim=1)

        # Calculate gradients
        self.model.zero_grad()
        output[0, predicted_class].backward()

        if x.grad is not None:
            gradients = x.grad.detach().abs().cpu().numpy().flatten()
        else:
            raise RuntimeError(
                "Gradients were not computed. Ensure x.requires_grad_(True) is set correctly."
            )

        importance_data["gradient_importance"] = gradients

        # Integrated gradients approximation
        integrated_grads = self._integrated_gradients(x, predicted_class)
        importance_data["integrated_gradients"] = integrated_grads

        # Spectral region importance
        if wavenumbers is not None:
            region_importance = self._analyze_spectral_regions(gradients, wavenumbers)
            importance_data["spectral_regions"] = region_importance

        return importance_data

    def _integrated_gradients(
        self, x: torch.Tensor, target_class: torch.Tensor, steps: int = 50
    ) -> np.ndarray:
        """Calculate integrated gradients for feature importance"""
        baseline = torch.zeros_like(x)

        integrated_grads = np.zeros(x.shape[1])

        for i in range(steps):
            alpha = i / steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)

            output = self.model(interpolated)
            self.model.zero_grad()
            output[0, target_class].backward(retain_graph=True)

            if interpolated.grad is not None:
                grads = interpolated.grad.cpu().numpy().flatten()
                integrated_grads += grads

        integrated_grads = (
            integrated_grads * (x - baseline).detach().cpu().numpy().flatten() / steps
        )
        return integrated_grads

    def _analyze_spectral_regions(
        self, importance: np.ndarray, wavenumbers: np.ndarray
    ) -> Dict[str, float]:
        """Analyze importance by common spectral regions"""
        regions = {
            "fingerprint": (400, 1500),
            "ch_stretch": (2800, 3100),
            "oh_stretch": (3200, 3700),
            "carbonyl": (1600, 1800),
            "aromatic": (1450, 1650),
        }

        region_importance = {}

        for region_name, (low, high) in regions.items():
            mask = (wavenumbers >= low) & (wavenumbers <= high)
            if np.any(mask):
                region_importance[region_name] = float(np.mean(importance[mask]))
            else:
                region_importance[region_name] = 0.0

        return region_importance


class HypothesisGenerator:
    """AI-driven scientific hypothesis generation"""

    def __init__(self):
        self.hypothesis_templates = [
            "The spectral differences in the {region} region suggest {mechanism} as a primary degradation pathway",
            "Enhanced intensity at {wavenumber} cm⁻¹ indicates {chemical_change} in weathered samples",
            "The correlation between {feature1} and {feature2} suggests {relationship}",
            "Baseline shifts in {region} region may indicate {structural_change}",
        ]

    def generate_hypotheses(
        self, explanation: PredictionExplanation
    ) -> List[Hypothesis]:
        """Generate testable hypotheses based on model predictions and explanations"""
        hypotheses = []

        # Analyze feature importance for hypothesis generation
        important_features = self._identify_key_features(explanation.feature_importance)

        for feature_info in important_features:
            hypothesis = self._generate_single_hypothesis(feature_info, explanation)
            if hypothesis:
                hypotheses.append(hypothesis)

        return hypotheses

    def _identify_key_features(
        self, feature_importance: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify key features for hypothesis generation"""
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )

        key_features = []
        for feature_name, importance in sorted_features[:5]:  # Top 5 features
            feature_info = {
                "name": feature_name,
                "importance": importance,
                "type": self._classify_feature_type(feature_name),
                "chemical_significance": self._get_chemical_significance(feature_name),
            }
            key_features.append(feature_info)

        return key_features

    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify spectral feature type"""
        if "fingerprint" in feature_name.lower():
            return "fingerprint"
        elif "stretch" in feature_name.lower():
            return "vibrational"
        elif "carbonyl" in feature_name.lower():
            return "functional_group"
        else:
            return "general"

    def _get_chemical_significance(self, feature_name: str) -> str:
        """Get chemical significance of spectral feature"""
        significance_map = {
            "fingerprint": "molecular backbone structure",
            "ch_stretch": "aliphatic chain integrity",
            "oh_stretch": "hydrogen bonding and hydration",
            "carbonyl": "oxidative degradation products",
            "aromatic": "aromatic ring preservation",
        }

        for key, significance in significance_map.items():
            if key in feature_name.lower():
                return significance

        return "structural changes"

    def _generate_single_hypothesis(
        self, feature_info: Dict[str, Any], explanation: PredictionExplanation
    ) -> Optional[Hypothesis]:
        """Generate a single hypothesis from feature information"""
        if feature_info["importance"] < 0.1:  # Skip low-importance features
            return None

        # Create hypothesis statement
        statement = f"Changes in {feature_info['name']} region indicate {feature_info['chemical_significance']} during polymer weathering"

        # Generate supporting evidence
        evidence = [
            f"Feature importance score: {feature_info['importance']:.3f}",
            f"Classification confidence: {explanation.confidence:.3f}",
            f"Chemical significance: {feature_info['chemical_significance']}",
        ]

        # Generate testable predictions
        predictions = [
            f"Controlled weathering experiments should show progressive changes in {feature_info['name']} region",
            f"Different polymer types should exhibit varying {feature_info['name']} responses to weathering",
        ]

        # Suggest experiments
        experiments = [
            f"Time-series weathering study monitoring {feature_info['name']} region",
            f"Comparative analysis across polymer types focusing on {feature_info['chemical_significance']}",
            "Cross-validation with other analytical techniques (DSC, GPC, etc.)",
        ]

        return Hypothesis(
            statement=statement,
            confidence=min(0.9, feature_info["importance"] * explanation.confidence),
            supporting_evidence=evidence,
            testable_predictions=predictions,
            suggested_experiments=experiments,
            related_literature=[],  # Could be populated with literature search
        )


class TransparentAIEngine:
    """Main transparent AI engine combining all reasoning components"""

    def __init__(self, model):
        self.model = model
        self.uncertainty_estimator = UncertaintyEstimator(model)
        self.feature_analyzer = FeatureImportanceAnalyzer(model)
        self.hypothesis_generator = HypothesisGenerator()

    def predict_with_explanation(
        self, x: torch.Tensor, wavenumbers: Optional[np.ndarray] = None
    ) -> PredictionExplanation:
        """Generate comprehensive prediction with full explanation"""
        self.model.eval()

        # Get basic prediction
        with torch.no_grad():
            logits = self.model(x)
            probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
            prediction = int(torch.argmax(logits, dim=1).item())
            confidence = float(np.max(probabilities))

        # Determine confidence level
        if confidence >= 0.80:
            confidence_level = "HIGH"
        elif confidence >= 0.60:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Get uncertainty estimation
        uncertainties = self.uncertainty_estimator.estimate_uncertainty(x)
        confidence_intervals = self.uncertainty_estimator.confidence_intervals(x)

        # Analyze feature importance
        importance_data = self.feature_analyzer.analyze_feature_importance(
            x, wavenumbers
        )

        # Create feature importance dictionary
        if wavenumbers is not None and "spectral_regions" in importance_data:
            feature_importance = importance_data["spectral_regions"]
        else:
            # Use gradient importance
            gradients = importance_data.get("gradient_importance", [])
            feature_importance = {
                f"feature_{i}": float(val) for i, val in enumerate(gradients[:10])
            }

        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(
            prediction, confidence, feature_importance, uncertainties
        )

        # Identify uncertainty sources
        uncertainty_sources = self._identify_uncertainty_sources(uncertainties)

        # Create explanation object
        explanation = PredictionExplanation(
            prediction=prediction,
            confidence=confidence,
            confidence_level=confidence_level,
            probabilities=probabilities,
            feature_importance=feature_importance,
            reasoning_chain=reasoning_chain,
            uncertainty_sources=uncertainty_sources,
            similar_cases=[],  # Could be populated with case-based reasoning
            confidence_intervals=confidence_intervals,
        )

        return explanation

    def generate_hypotheses(
        self, explanation: PredictionExplanation
    ) -> List[Hypothesis]:
        """Generate scientific hypotheses based on prediction explanation"""
        return self.hypothesis_generator.generate_hypotheses(explanation)

    def _generate_reasoning_chain(
        self,
        prediction: int,
        confidence: float,
        feature_importance: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> List[str]:
        """Generate human-readable reasoning chain"""
        reasoning = []

        # Start with prediction
        class_names = ["Stable", "Weathered"]
        reasoning.append(
            f"Model predicts: {class_names[prediction]} (confidence: {confidence:.3f})"
        )

        # Add feature analysis
        top_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        for feature, importance in top_features:
            reasoning.append(
                f"Key evidence: {feature} region shows importance score {importance:.3f}"
            )

        # Add uncertainty analysis
        total_uncertainty = uncertainties.get("total", 0)
        if total_uncertainty > 0.1:
            reasoning.append(
                f"High uncertainty detected ({total_uncertainty:.3f}) - suggests ambiguous case"
            )

        # Add confidence assessment
        if confidence > 0.8:
            reasoning.append(
                "High confidence: Strong spectral signature for classification"
            )
        elif confidence > 0.6:
            reasoning.append("Medium confidence: Some ambiguity in spectral features")
        else:
            reasoning.append("Low confidence: Weak or conflicting spectral evidence")

        return reasoning

    def _identify_uncertainty_sources(
        self, uncertainties: Dict[str, float]
    ) -> List[str]:
        """Identify sources of prediction uncertainty"""
        sources = []

        epistemic = uncertainties.get("epistemic", 0)
        aleatoric = uncertainties.get("aleatoric", 0)

        if epistemic > 0.05:
            sources.append(
                "Model uncertainty: Limited training data for this type of spectrum"
            )

        if aleatoric > 0.05:
            sources.append("Data uncertainty: Noisy or degraded spectral quality")

        if uncertainties.get("prediction_variance", 0) > 0.1:
            sources.append("Prediction instability: Multiple possible interpretations")

        if not sources:
            sources.append("Low uncertainty: Clear and unambiguous classification")

        return sources
