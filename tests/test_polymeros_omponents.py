"""
Test suite for POLYMEROS enhanced components
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from modules.enhanced_data import (
    EnhancedDataManager,
    ContextualSpectrum,
    SpectralMetadata,
)
from modules.transparent_ai import TransparentAIEngine, UncertaintyEstimator
from modules.educational_framework import EducationalFramework


def test_enhanced_data_manager():
    """Test enhanced data management functionality"""
    print("Testing Enhanced Data Manager...")

    # Create data manager
    data_manager = EnhancedDataManager()

    # Create sample spectrum
    x_data = np.linspace(400, 4000, 500)
    y_data = np.exp(-(((x_data - 2900) / 100) ** 2)) + np.random.normal(0, 0.01, 500)

    metadata = SpectralMetadata(
        filename="test_spectrum.txt", instrument_type="Raman", laser_wavelength=785.0
    )

    spectrum = ContextualSpectrum(x_data, y_data, metadata)

    # Test quality assessment
    quality_score = data_manager._assess_data_quality(y_data)
    print(f"Quality score: {quality_score:.3f}")

    # Test preprocessing recommendations
    recommendations = data_manager.get_preprocessing_recommendations(spectrum)
    print(f"Preprocessing recommendations: {recommendations}")

    # Test preprocessing with tracking
    processed_spectrum = data_manager.preprocess_with_tracking(
        spectrum, **recommendations
    )
    print(f"Provenance records: {len(processed_spectrum.provenance)}")

    print("✅ Enhanced Data Manager tests passed!")
    return True


def test_transparent_ai():
    """Test transparent AI functionality"""
    print("Testing Transparent AI Engine...")

    # Create dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(500, 2)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()

    # Test uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(model, n_samples=10)

    # Create test input
    x = torch.randn(1, 500)

    # Test uncertainty estimation
    uncertainties = uncertainty_estimator.estimate_uncertainty(x)
    print(f"Uncertainty metrics: {uncertainties}")

    # Test confidence intervals
    intervals = uncertainty_estimator.confidence_intervals(x)
    print(f"Confidence intervals: {intervals}")

    # Test transparent AI engine
    ai_engine = TransparentAIEngine(model)
    explanation = ai_engine.predict_with_explanation(x)

    print(f"Prediction: {explanation.prediction}")
    print(f"Confidence: {explanation.confidence:.3f}")
    print(f"Reasoning chain: {len(explanation.reasoning_chain)} steps")

    print("✅ Transparent AI tests passed!")
    return True


def test_educational_framework():
    """Test educational framework functionality"""
    print("Testing Educational Framework...")

    # Create educational framework
    framework = EducationalFramework()

    # Initialize user
    user_progress = framework.initialize_user("test_user")
    print(f"User initialized: {user_progress.user_id}")

    # Test competency assessment
    domain = "spectroscopy_basics"
    responses = [2, 1, 0]  # Sample responses

    results = framework.assess_user_competency(domain, responses)
    print(f"Assessment results: {results['score']:.2f}")

    # Test learning path generation
    target_competencies = ["spectroscopy", "polymer_science"]
    learning_path = framework.get_personalized_learning_path(target_competencies)
    print(f"Learning path objectives: {len(learning_path)}")

    # Test virtual experiment
    experiment_result = framework.run_virtual_experiment(
        "polymer_identification", {"polymer_type": "PE"}
    )
    print(f"Virtual experiment success: {experiment_result.get('success', False)}")

    # Test analytics
    analytics = framework.get_learning_analytics()
    print(f"Analytics available: {bool(analytics)}")

    print("✅ Educational Framework tests passed!")
    return True


def run_all_tests():
    """Run all component tests"""
    print("Starting POLYMEROS Component Tests...\n")

    tests = [
        test_enhanced_data_manager,
        test_transparent_ai,
        test_educational_framework,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed: {e}\n")

    print(f"Tests completed: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All POLYMEROS components working correctly!")
    else:
        print("⚠️ Some components need attention")


if __name__ == "__main__":
    run_all_tests()
