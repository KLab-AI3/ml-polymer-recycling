"""
Enhanced Analysis Page for POLYMEROS
Advanced multi-modal spectroscopy analysis with modern ML architecture
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image

# Import POLYMEROS components
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from modules.transparent_ai import TransparentAIEngine, PredictionExplanation
from modules.enhanced_data import (
    EnhancedDataManager,
    ContextualSpectrum,
    SpectralMetadata,
)
from modules.advanced_spectroscopy import MultiModalSpectroscopyEngine
from modules.modern_ml_architecture import (
    ModernMLPipeline,
)
from modules.enhanced_data_pipeline import EnhancedDataPipeline
from core_logic import load_model, parse_spectrum_data
from config import MODEL_CONFIG, TARGET_LEN

# Removed unused preprocess_spectrum import


def init_enhanced_analysis():
    """Initialize enhanced analysis session state with new components"""
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = EnhancedDataManager()

    if "spectroscopy_engine" not in st.session_state:
        st.session_state.spectroscopy_engine = MultiModalSpectroscopyEngine()

    if "ml_pipeline" not in st.session_state:
        st.session_state.ml_pipeline = ModernMLPipeline()
        st.session_state.ml_pipeline.initialize_models()

    if "data_pipeline" not in st.session_state:
        st.session_state.data_pipeline = EnhancedDataPipeline()

    if "transparent_ai" not in st.session_state:
        st.session_state.transparent_ai = None

    if "current_model" not in st.session_state:
        st.session_state.current_model = None

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None


def load_enhanced_model(model_name: str):
    """Load model and initialize transparent AI engine"""
    try:
        model = load_model(model_name)
        if model is not None:
            st.session_state.current_model = model
            st.session_state.transparent_ai = TransparentAIEngine(model)
            return True
        return False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False


def render_enhanced_file_upload():
    """Render enhanced file upload with metadata extraction"""
    st.header("📁 Enhanced Spectrum Analysis")

    uploaded_file = st.file_uploader(
        "Upload spectrum file (.txt)",
        type=["txt"],
        help="Upload a Raman or FTIR spectrum in text format",
    )

    if uploaded_file is not None:
        # Parse spectrum data
        try:
            content = uploaded_file.read().decode("utf-8")
            x_data, y_data = parse_spectrum_data(content)

            # Create enhanced spectrum with metadata
            metadata = SpectralMetadata(
                filename=uploaded_file.name,
                instrument_type="Raman",  # Default, could be detected from filename
                data_quality_score=None,
            )

            spectrum = ContextualSpectrum(x_data, y_data, metadata)

            # Get data quality assessment
            data_manager = st.session_state.data_manager
            quality_score = data_manager._assess_data_quality(y_data)
            spectrum.metadata.data_quality_score = quality_score

            # Display quality assessment
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Points", len(x_data))
            with col2:
                st.metric("Quality Score", f"{quality_score:.2f}")
            with col3:
                quality_color = (
                    "🟢"
                    if quality_score > 0.7
                    else "🟡" if quality_score > 0.4 else "🔴"
                )
                st.metric("Quality", f"{quality_color}")

            # Get preprocessing recommendations
            recommendations = data_manager.get_preprocessing_recommendations(spectrum)

            st.subheader("Intelligent Preprocessing Recommendations")
            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.write("**Recommended settings:**")
                for param, value in recommendations.items():
                    st.write(f"• {param}: {value}")

            with rec_col2:
                st.write("**Manual override:**")
                do_baseline = st.checkbox(
                    "Baseline correction",
                    value=recommendations.get("do_baseline", True),
                )
                do_smooth = st.checkbox(
                    "Smoothing", value=recommendations.get("do_smooth", True)
                )
                do_normalize = st.checkbox(
                    "Normalization", value=recommendations.get("do_normalize", True)
                )

            # Apply preprocessing with tracking
            preprocessing_params = {
                "do_baseline": do_baseline,
                "do_smooth": do_smooth,
                "do_normalize": do_normalize,
                "target_len": TARGET_LEN,
            }

            if st.button("Process and Analyze"):
                with st.spinner("Processing spectrum with provenance tracking..."):
                    # Apply preprocessing with full tracking
                    processed_spectrum = data_manager.preprocess_with_tracking(
                        spectrum, **preprocessing_params
                    )

                    # Store processed spectrum
                    st.session_state.processed_spectrum = processed_spectrum
                    st.success("Spectrum processed with full provenance tracking!")

                    # Display provenance information
                    st.subheader("Processing Provenance")
                    for record in processed_spectrum.provenance:
                        with st.expander(f"Operation: {record.operation}"):
                            st.write(f"**Timestamp:** {record.timestamp}")
                            st.write(f"**Parameters:** {record.parameters}")
                            st.write(f"**Input hash:** {record.input_hash}")
                            st.write(f"**Output hash:** {record.output_hash}")

        except Exception as e:
            st.error(f"Error processing file: {e}")


def render_transparent_analysis():
    """Render transparent AI analysis with explanations"""
    if "processed_spectrum" not in st.session_state:
        st.info("Please upload and process a spectrum first.")
        return

    st.header("🧠 Transparent AI Analysis")

    # Model selection
    model_names = list(MODEL_CONFIG.keys())
    selected_model = st.selectbox("Select AI model:", model_names)

    if st.session_state.current_model is None or st.button("Load Model"):
        with st.spinner(f"Loading {selected_model} model..."):
            if load_enhanced_model(selected_model):
                st.success(f"Model {selected_model} loaded successfully!")
            else:
                st.error("Failed to load model")
                return

    if st.session_state.transparent_ai is not None:
        spectrum = st.session_state.processed_spectrum

        if st.button("Run Transparent Analysis"):
            with st.spinner("Running comprehensive analysis..."):
                # Prepare input tensor
                y_processed = spectrum.y_data
                x_input = torch.tensor(y_processed, dtype=torch.float32).unsqueeze(0)

                # Get transparent explanation
                explanation = st.session_state.transparent_ai.predict_with_explanation(
                    x_input, wavenumbers=spectrum.x_data
                )

                # Generate hypotheses
                hypotheses = st.session_state.transparent_ai.generate_hypotheses(
                    explanation
                )

                # Store results
                st.session_state.analysis_results = {
                    "explanation": explanation,
                    "hypotheses": hypotheses,
                }

                # Display results
                render_analysis_results(explanation, hypotheses)


def render_analysis_results(explanation: PredictionExplanation, hypotheses: list):
    """Render comprehensive analysis results"""
    st.subheader("🎯 Prediction Results")

    # Main prediction
    class_names = ["Stable", "Weathered"]
    predicted_class = class_names[explanation.prediction]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", predicted_class)
    with col2:
        st.metric("Confidence", f"{explanation.confidence:.3f}")
    with col3:
        confidence_emoji = (
            "🟢"
            if explanation.confidence_level == "HIGH"
            else "🟡" if explanation.confidence_level == "MEDIUM" else "🔴"
        )
        st.metric("Level", f"{confidence_emoji} {explanation.confidence_level}")

    # Probability distribution
    st.subheader("📊 Probability Distribution")
    prob_data = {"Class": class_names, "Probability": explanation.probabilities}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(prob_data["Class"], prob_data["Probability"])
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    ax.set_ylim(0, 1)

    # Color bars based on prediction
    for i, bar in enumerate(bars):
        if i == explanation.prediction:
            bar.set_color("steelblue")
        else:
            bar.set_color("lightgray")

    st.pyplot(fig)

    # Reasoning chain
    st.subheader("🔍 AI Reasoning Chain")
    for i, reasoning in enumerate(explanation.reasoning_chain):
        st.write(f"{i+1}. {reasoning}")

    # Feature importance
    if explanation.feature_importance:
        st.subheader("🎯 Feature Importance Analysis")

        # Create feature importance plot
        features = list(explanation.feature_importance.keys())
        importances = list(explanation.feature_importance.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importances)
        ax.set_xlabel("Importance Score")
        ax.set_title("Spectral Region Importance")

        # Color bars based on importance
        for bar, importance in zip(bars, importances):
            if abs(importance) > 0.5:
                bar.set_color("red")
            elif abs(importance) > 0.3:
                bar.set_color("orange")
            else:
                bar.set_color("lightblue")

        plt.tight_layout()
        st.pyplot(fig)

    # Uncertainty analysis
    st.subheader("🤔 Uncertainty Analysis")
    for source in explanation.uncertainty_sources:
        st.write(f"• {source}")

    # Confidence intervals
    if explanation.confidence_intervals:
        st.subheader("📈 Confidence Intervals")
        for class_name, (lower, upper) in explanation.confidence_intervals.items():
            st.write(f"**{class_name}:** [{lower:.3f}, {upper:.3f}]")

    # AI-generated hypotheses
    if hypotheses:
        st.subheader("🧪 AI-Generated Scientific Hypotheses")

        for i, hypothesis in enumerate(hypotheses):
            with st.expander(f"Hypothesis {i+1}: {hypothesis.statement}"):
                st.write(f"**Confidence:** {hypothesis.confidence:.3f}")

                st.write("**Supporting Evidence:**")
                for evidence in hypothesis.supporting_evidence:
                    st.write(f"• {evidence}")

                st.write("**Testable Predictions:**")
                for prediction in hypothesis.testable_predictions:
                    st.write(f"• {prediction}")

                st.write("**Suggested Experiments:**")
                for experiment in hypothesis.suggested_experiments:
                    st.write(f"• {experiment}")


def render_data_provenance():
    """Render data provenance and quality information"""
    if "processed_spectrum" not in st.session_state:
        st.info("No processed spectrum available.")
        return

    st.header("📋 Data Provenance & Quality")

    spectrum = st.session_state.processed_spectrum

    # Metadata display
    st.subheader("📄 Spectrum Metadata")
    metadata = spectrum.metadata

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Filename:** {metadata.filename}")
        st.write(f"**Instrument:** {metadata.instrument_type}")
        st.write(f"**Quality Score:** {metadata.data_quality_score:.3f}")

    with col2:
        if metadata.laser_wavelength:
            st.write(f"**Laser Wavelength:** {metadata.laser_wavelength} nm")
        if metadata.acquisition_date:
            st.write(f"**Acquisition Date:** {metadata.acquisition_date}")
        st.write(f"**Data Hash:** {spectrum.data_hash}")

    # Provenance timeline
    st.subheader("🕒 Processing Timeline")

    if spectrum.provenance:
        for i, record in enumerate(spectrum.provenance):
            with st.expander(
                f"Step {i+1}: {record.operation} ({record.timestamp[:19]})"
            ):
                st.write(f"**Operation:** {record.operation}")
                st.write(f"**Operator:** {record.operator}")
                st.write(f"**Parameters:**")
                for param, value in record.parameters.items():
                    st.write(f"  - {param}: {value}")
                st.write(f"**Input Hash:** {record.input_hash}")
                st.write(f"**Output Hash:** {record.output_hash}")
    else:
        st.info("No processing operations recorded yet.")

    # Quality assessment details
    st.subheader("🔍 Quality Assessment Details")

    if hasattr(spectrum, "quality_metrics"):
        metrics = spectrum.quality_metrics
        for metric, value in metrics.items():
            st.write(f"**{metric}:** {value}")
    else:
        st.info("Run quality assessment to see detailed metrics.")


def main():
    """Main enhanced analysis interface"""
    st.set_page_config(
        page_title="POLYMEROS Enhanced Analysis", page_icon="🔬", layout="wide"
    )

    st.title("🔬 POLYMEROS Enhanced Analysis")
    st.markdown("**Transparent AI with Explainability and Hypothesis Generation**")

    # Initialize session
    init_enhanced_analysis()

    # Sidebar navigation
    st.sidebar.title("🧪 Analysis Tools")
    analysis_mode = st.sidebar.selectbox(
        "Select analysis mode:",
        [
            "Spectrum Upload & Processing",
            "Transparent AI Analysis",
            "Data Provenance & Quality",
        ],
    )

    # Render selected mode
    if analysis_mode == "Spectrum Upload & Processing":
        render_enhanced_file_upload()
    elif analysis_mode == "Transparent AI Analysis":
        render_transparent_analysis()
    elif analysis_mode == "Data Provenance & Quality":
        render_data_provenance()

    # Additional information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Enhanced Features:**")
    st.sidebar.markdown("• Complete provenance tracking")
    st.sidebar.markdown("• Intelligent preprocessing")
    st.sidebar.markdown("• Uncertainty quantification")
    st.sidebar.markdown("• AI hypothesis generation")
    st.sidebar.markdown("• Explainable predictions")

    # Display current analysis status
    if st.session_state.analysis_results:
        st.sidebar.success("✅ Analysis completed")
    elif "processed_spectrum" in st.session_state:
        st.sidebar.info("📊 Spectrum processed")
    else:
        st.sidebar.info("📁 Ready for upload")


if __name__ == "__main__":
    main()
