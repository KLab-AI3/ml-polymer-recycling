"""Session results management for multi-file inference.
Handles in-memory results table and export functionality.
Supports multi-model comparison and statistical analysis."""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import io
from collections import defaultdict
import matplotlib.pyplot as plt


def local_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


class ResultsManager:
    """Manages session-wide results for multi-file inference"""

    RESULTS_KEY = "inference_results"

    @staticmethod
    def init_results_table() -> None:
        """Initialize the results table in session state"""
        if ResultsManager.RESULTS_KEY not in st.session_state:
            st.session_state[ResultsManager.RESULTS_KEY] = []

    @staticmethod
    def add_results(
        filename: str,
        model_name: str,
        prediction: int,
        predicted_class: str,
        confidence: float,
        logits: List[float],
        ground_truth: Optional[int] = None,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single inference result to the results table"""
        ResultsManager.init_results_table()

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "model": model_name,
            "prediction": prediction,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "logits": logits,
            "ground_truth": ground_truth,
            "processing_time": processing_time,
            "metadata": metadata or {},
        }

        st.session_state[ResultsManager.RESULTS_KEY].append(result)

    @staticmethod
    def get_results() -> List[Dict[str, Any]]:
        """Get all inference results"""
        ResultsManager.init_results_table()
        return st.session_state[ResultsManager.RESULTS_KEY]

    @staticmethod
    def get_results_count() -> int:
        """Get the number of stored results"""
        return len(ResultsManager.get_results())

    @staticmethod
    def clear_results() -> None:
        """Clear all stored results"""
        st.session_state[ResultsManager.RESULTS_KEY] = []

    @staticmethod
    def get_spectrum_data_for_file(filename: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieves raw and resampled spectrum data for a given filename.
        Returns None if no data is found for the filename or if data is incomplete.
        """
        results = ResultsManager.get_results()
        for r in results:
            if r["filename"] == filename:
                # Ensure all required keys are present and not None
                if all(
                    r.get(k) is not None
                    for k in ["x_raw", "y_raw", "x_resampled", "y_resampled"]
                ):
                    return {
                        "x_raw": r["x_raw"],
                        "y_raw": r["y_raw"],
                        "x_resampled": r["x_resampled"],
                        "y_resampled": r["y_resampled"],
                    }
                else:
                    # If the metadata exists but spectrum data is missing for this entry,
                    # it means it was processed before we started storing spectrums.
                    return None
        return None  # Return None if filename not found

    @staticmethod
    def get_results_dataframe() -> pd.DataFrame:
        """Convert results to pandas DataFrame for display and export"""
        results = ResultsManager.get_results()
        if not results:
            return pd.DataFrame()

        # ===Flatten the results for DataFrame===
        df_data = []
        for result in results:
            row = {
                "Timestamp": result["timestamp"],
                "Filename": result["filename"],
                "Model": result["model"],
                "Prediction": result["prediction"],
                "Predicted Class": result["predicted_class"],
                "Confidence": f"{result['confidence']:.3f}",
                "Stable Logit": (
                    f"{result['logits'][0]:.3f}" if len(result["logits"]) > 0 else "N/A"
                ),
                "Weathered Logit": (
                    f"{result['logits'][1]:.3f}" if len(result["logits"]) > 1 else "N/A"
                ),
                "Ground Truth": (
                    result["ground_truth"]
                    if result["ground_truth"] is not None
                    else "Unknown"
                ),
                "Processing Time (s)": f"{result['processing_time']:.3f}",
            }
            df_data.append(row)

        return pd.DataFrame(df_data)

    @staticmethod
    def export_to_csv() -> bytes:
        """Export results to CSV format"""
        df = ResultsManager.get_results_dataframe()
        if df.empty:
            return b""

        # ===Use StringIO to create CSV in memory===
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode("utf-8")

    @staticmethod
    def export_to_json() -> str:
        """Export results to JSON format"""
        results = ResultsManager.get_results()
        return json.dumps(results, indent=2, default=str)

    @staticmethod
    def get_summary_stats() -> Dict[str, Any]:
        """Get summary statistics for the results"""
        results = ResultsManager.get_results()
        if not results:
            return {}

        df = ResultsManager.get_results_dataframe()

        stats = {
            "total_files": len(results),
            "models_used": list(set(r["model"] for r in results)),
            "stable_predictions": sum(1 for r in results if r["prediction"] == 0),
            "weathered_predictions": sum(1 for r in results if r["prediction"] == 1),
            "avg_confidence": sum(r["confidence"] for r in results) / len(results),
            "avg_processing_time": sum(r["processing_time"] for r in results)
            / len(results),
            "files_with_ground_truth": sum(
                1 for r in results if r["ground_truth"] is not None
            ),
        }
        # ===Calculate accuracy if ground truth is available===
        correct_predictions = sum(
            1
            for r in results
            if r["ground_truth"] is not None and r["prediction"] == r["ground_truth"]
        )
        total_with_gt = stats["files_with_ground_truth"]
        if total_with_gt > 0:
            stats["accuracy"] = correct_predictions / total_with_gt
        else:
            stats["accuracy"] = None

        return stats

    @staticmethod
    def remove_result_by_filename(filename: str) -> bool:
        """Remove a result by filename. Returns True if removed, False if not found."""
        results = ResultsManager.get_results()
        original_length = len(results)

        # Filter out results with matching filename
        st.session_state[ResultsManager.RESULTS_KEY] = [
            r for r in results if r["filename"] != filename
        ]

        return len(st.session_state[ResultsManager.RESULTS_KEY]) < original_length

    @staticmethod
    def add_multi_model_results(
        filename: str,
        model_results: Dict[str, Dict[str, Any]],
        ground_truth: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add results from multiple models for the same file.

        Args:
            filename: Name of the processed file
            model_results: Dict with model_name -> result dict
            ground_truth: True label if available
            metadata: Additional file metadata
        """
        for model_name, result in model_results.items():
            ResultsManager.add_results(
                filename=filename,
                model_name=model_name,
                prediction=result["prediction"],
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                logits=result["logits"],
                ground_truth=ground_truth,
                processing_time=result.get("processing_time", 0.0),
                metadata=metadata,
            )

    @staticmethod
    def get_comparison_stats() -> Dict[str, Any]:
        """Get comparative statistics across all models."""
        results = ResultsManager.get_results()
        if not results:
            return {}

        # Group results by model
        model_stats = defaultdict(list)
        for result in results:
            model_stats[result["model"]].append(result)

        comparison = {}
        for model_name, model_results in model_stats.items():
            stats = {
                "total_predictions": len(model_results),
                "avg_confidence": np.mean([r["confidence"] for r in model_results]),
                "std_confidence": np.std([r["confidence"] for r in model_results]),
                "avg_processing_time": np.mean(
                    [r["processing_time"] for r in model_results]
                ),
                "stable_predictions": sum(
                    1 for r in model_results if r["prediction"] == 0
                ),
                "weathered_predictions": sum(
                    1 for r in model_results if r["prediction"] == 1
                ),
            }

            # Calculate accuracy if ground truth available
            with_gt = [r for r in model_results if r["ground_truth"] is not None]
            if with_gt:
                correct = sum(
                    1 for r in with_gt if r["prediction"] == r["ground_truth"]
                )
                stats["accuracy"] = correct / len(with_gt)
                stats["num_with_ground_truth"] = len(with_gt)
            else:
                stats["accuracy"] = None
                stats["num_with_ground_truth"] = 0

            comparison[model_name] = stats

        return comparison

    @staticmethod
    def get_agreement_matrix() -> pd.DataFrame:
        """
        Calculate agreement matrix between models for the same files.

        Returns:
            DataFrame showing model agreement rates
        """
        results = ResultsManager.get_results()
        if not results:
            return pd.DataFrame()

        # Group by filename
        file_results = defaultdict(dict)
        for result in results:
            file_results[result["filename"]][result["model"]] = result["prediction"]

        # Get unique models
        all_models = list(set(r["model"] for r in results))

        if len(all_models) < 2:
            return pd.DataFrame()

        # Calculate agreement matrix
        agreement_matrix = np.zeros((len(all_models), len(all_models)))

        for i, model1 in enumerate(all_models):
            for j, model2 in enumerate(all_models):
                if i == j:
                    agreement_matrix[i, j] = 1.0  # Perfect self-agreement
                else:
                    agreements = 0
                    comparisons = 0

                    for filename, predictions in file_results.items():
                        if model1 in predictions and model2 in predictions:
                            comparisons += 1
                            if predictions[model1] == predictions[model2]:
                                agreements += 1

                    if comparisons > 0:
                        agreement_matrix[i, j] = agreements / comparisons

        return pd.DataFrame(agreement_matrix, index=all_models, columns=all_models)

    @staticmethod
    def create_comparison_visualization() -> plt.Figure:
        """Create visualization comparing model performance."""
        comparison_stats = ResultsManager.get_comparison_stats()

        if not comparison_stats:
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        models = list(comparison_stats.keys())

        # 1. Average Confidence
        confidences = [comparison_stats[m]["avg_confidence"] for m in models]
        conf_stds = [comparison_stats[m]["std_confidence"] for m in models]
        ax1.bar(models, confidences, yerr=conf_stds, capsize=5)
        ax1.set_title("Average Confidence by Model")
        ax1.set_ylabel("Confidence")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Processing Time
        proc_times = [comparison_stats[m]["avg_processing_time"] for m in models]
        ax2.bar(models, proc_times)
        ax2.set_title("Average Processing Time")
        ax2.set_ylabel("Time (seconds)")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Prediction Distribution
        stable_counts = [comparison_stats[m]["stable_predictions"] for m in models]
        weathered_counts = [
            comparison_stats[m]["weathered_predictions"] for m in models
        ]

        x = np.arange(len(models))
        width = 0.35
        ax3.bar(x - width / 2, stable_counts, width, label="Stable", alpha=0.8)
        ax3.bar(x + width / 2, weathered_counts, width, label="Weathered", alpha=0.8)
        ax3.set_title("Prediction Distribution")
        ax3.set_ylabel("Count")
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()

        # 4. Accuracy (if available)
        accuracies = []
        models_with_acc = []
        for model in models:
            if comparison_stats[model]["accuracy"] is not None:
                accuracies.append(comparison_stats[model]["accuracy"])
                models_with_acc.append(model)

        if accuracies:
            ax4.bar(models_with_acc, accuracies)
            ax4.set_title("Model Accuracy (where ground truth available)")
            ax4.set_ylabel("Accuracy")
            ax4.set_ylim(0, 1)
            ax4.tick_params(axis="x", rotation=45)
        else:
            ax4.text(
                0.5,
                0.5,
                "No ground truth\navailable",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Model Accuracy")

        plt.tight_layout()
        return fig

    @staticmethod
    def export_comparison_report() -> str:
        """Export comprehensive comparison report as JSON."""
        comparison_stats = ResultsManager.get_comparison_stats()
        agreement_matrix = ResultsManager.get_agreement_matrix()

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_comparison": comparison_stats,
            "agreement_matrix": (
                agreement_matrix.to_dict() if not agreement_matrix.empty else {}
            ),
            "summary": {
                "total_models_compared": len(comparison_stats),
                "total_files_processed": len(
                    set(r["filename"] for r in ResultsManager.get_results())
                ),
                "overall_statistics": ResultsManager.get_summary_stats(),
            },
        }

        return json.dumps(report, indent=2, default=str)

    @staticmethod
    # ==UTILITY FUNCTIONS==
    def init_session_state():
        """Keep a persistent session state"""
        defaults = {
            "status_message": "Ready to analyze polymer spectra 🔬",
            "status_type": "info",
            "input_text": None,
            "filename": None,
            "input_source": None,  # "upload", "batch" or "sample"
            "sample_select": "-- Select Sample --",
            "input_mode": "Upload File",  # controls which pane is visible
            "inference_run_once": False,
            "x_raw": None,
            "y_raw": None,
            "y_resampled": None,
            "log_messages": [],
            "uploader_version": 0,
            "current_upload_key": "upload_txt_0",
            "active_tab": "Details",
            "batch_mode": False,
        }

        # Init session state with defaults
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def reset_ephemeral_state():
        """Comprehensive reset for the entire app state."""

        current_version = st.session_state.get("uploader_version", 0)

        # Define keys that should NOT be cleared by a full reset
        keep_keys = {"model_select", "input_mode"}

        for k in list(st.session_state.keys()):
            if k not in keep_keys:
                st.session_state.pop(k, None)

        st.session_state["status_message"] = "Ready to analyze polymer spectra"
        st.session_state["status_type"] = "info"
        st.session_state["batch_files"] = []
        st.session_state["inference_run_once"] = True
        st.session_state[""] = ""

        # CRITICAL: Increment the preserved version and re-assign it
        st.session_state["uploader_version"] = current_version + 1
        st.session_state["current_upload_key"] = (
            f"upload_txt_{st.session_state['uploader_version']}"
        )

    @staticmethod
    def display_results_table() -> None:
        """Display the results table in Streamlit UI"""
        df = ResultsManager.get_results_dataframe()

        if df.empty:
            st.info(
                "No inference results yet. Upload files and run analysis to see results here."
            )
            return

        local_css("static/style.css")
        st.subheader(f"Inference Results ({len(df)} files)")

        # ==Summary stats==
        stats = ResultsManager.get_summary_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", stats["total_files"])
            with col2:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
            with col3:
                st.metric(
                    "Stable/Weathered",
                    f"{stats['stable_predictions']}/{stats['weathered_predictions']}",
                )
            with col4:
                if stats["accuracy"] is not None:
                    st.metric("Accuracy", f"{stats['accuracy']:.3f}")
                else:
                    st.metric("Accuracy", "N/A")

            # ==Results Table==
            st.dataframe(df, use_container_width=True)
            with st.container(border=None, key="page-link-container"):
                st.page_link(
                    "pages/2_Dashboard.py",
                    label="Inference Analysis Dashboard",
                    help="Dive deeper into your batch results.",
                    use_container_width=False,
                )

            # ==Export Button==
            with st.container(border=None, key="buttons-container"):
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    csv_data = ResultsManager.export_to_csv()
                    if csv_data:
                        with st.container(border=None, key="csv-button"):
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"polymer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Export Results to CSV",
                                use_container_width=True,
                                type="tertiary",
                            )

                with col2:
                    json_data = ResultsManager.export_to_json()
                    if json_data:
                        with st.container(border=None, key="json-button"):
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"polymer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                help="Export Results to JSON",
                                type="tertiary",
                                use_container_width=True,
                            )

                with col3:
                    with st.container(border=None, key="clearall-button"):
                        st.button(
                            label="Clear All Results",
                            help="Clear all stored results",
                            on_click=ResultsManager.reset_ephemeral_state,
                            use_container_width=True,
                            type="tertiary",
                        )
