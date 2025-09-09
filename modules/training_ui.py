"""
Training UI components for the ML Hub functionality.
Provides interface for model training, dataset management, and progress tracking.
"""

import os
import time
import torch
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta

from models.registry import choices as model_choices, get_model_info
from utils.training_manager import get_training_manager, TrainingJob
from utils.training_types import TrainingConfig, TrainingStatus


def render_training_tab():
    """Render the main training interface tab"""
    st.markdown("## 🎯 Model Training Hub")
    st.markdown(
        "Train any model from the registry on your datasets with real-time progress tracking."
    )

    # Create columns for layout
    config_col, status_col = st.columns([1, 1])

    with config_col:
        render_training_configuration()

    with status_col:
        render_training_status()

    # Full-width progress and results section
    st.markdown("---")
    render_training_progress()

    st.markdown("---")
    render_training_history()


def render_training_configuration():
    """Render training configuration panel"""
    st.markdown("### ⚙️ Training Configuration")

    with st.expander("Model Selection", expanded=True):
        # Model selection
        available_models = model_choices()
        selected_model = st.selectbox(
            "Select Model Architecture",
            available_models,
            help="Choose from available model architectures in the registry",
        )

        # Store in session state
        st.session_state["selected_model"] = selected_model

        # Display model info
        if selected_model:
            try:
                model_info = get_model_info(selected_model)
                st.info(
                    f"**{selected_model}**: {model_info.get('description', 'No description available')}"
                )

                # Model specs
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Parameters", model_info.get("parameters", "Unknown"))
                    st.metric("Speed", model_info.get("speed", "Unknown"))
                with col2:
                    if "performance" in model_info:
                        perf = model_info["performance"]
                        st.metric("Accuracy", f"{perf.get('accuracy', 0):.3f}")
                        st.metric("F1 Score", f"{perf.get('f1_score', 0):.3f}")
            except KeyError:
                st.warning(f"Model info not available for {selected_model}")

    with st.expander("Dataset Selection", expanded=True):
        render_dataset_selection()

    with st.expander("Training Parameters", expanded=True):
        render_training_parameters()

    # Training action button
    st.markdown("---")
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        start_training_job()


def render_dataset_selection():
    """Render dataset selection and upload interface"""
    st.markdown("#### Dataset Management")

    # Dataset source selection
    dataset_source = st.radio(
        "Dataset Source",
        ["Upload New Dataset", "Use Existing Dataset"],
        horizontal=True,
    )

    if dataset_source == "Upload New Dataset":
        render_dataset_upload()
    else:
        render_existing_dataset_selection()


def render_dataset_upload():
    """Render dataset upload interface"""
    st.markdown("##### Upload Dataset")

    uploaded_files = st.file_uploader(
        "Upload spectrum files (.txt, .csv, .json)",
        accept_multiple_files=True,
        type=["txt", "csv", "json"],
        help="Upload multiple spectrum files. Organize them in folders named 'stable' and 'weathered' or label them accordingly.",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")

        # Dataset organization
        st.markdown("##### Dataset Organization")

        dataset_name = st.text_input(
            "Dataset Name",
            placeholder="e.g., my_polymer_dataset",
            help="Name for your dataset (will create a folder)",
        )

        # File labeling
        st.markdown("**Label your files:**")
        file_labels = {}

        for i, file in enumerate(uploaded_files[:10]):  # Limit display for performance
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(file.name)
            with col2:
                file_labels[file.name] = st.selectbox(
                    f"Label for {file.name}", ["stable", "weathered"], key=f"label_{i}"
                )

        if len(uploaded_files) > 10:
            st.info(
                f"Showing first 10 files. {len(uploaded_files) - 10} more files will use default labeling based on filename."
            )

        if st.button("💾 Save Dataset") and dataset_name:
            save_uploaded_dataset(uploaded_files, dataset_name, file_labels)


def render_existing_dataset_selection():
    """Render existing dataset selection"""
    st.markdown("##### Available Datasets")

    # Scan for existing datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        available_datasets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]

        if available_datasets:
            selected_dataset = st.selectbox(
                "Select Dataset",
                available_datasets,
                help="Choose from previously uploaded or existing datasets",
            )

            if selected_dataset:
                st.session_state["selected_dataset"] = str(
                    datasets_dir / selected_dataset
                )
                display_dataset_info(datasets_dir / selected_dataset)
        else:
            st.warning("No datasets found. Please upload a dataset first.")
    else:
        st.warning("Datasets directory not found. Please upload a dataset first.")


def display_dataset_info(dataset_path: Path):
    """Display information about selected dataset"""
    if not dataset_path.exists():
        return

    # Count files by category
    file_counts = {}
    total_files = 0

    for category_dir in dataset_path.iterdir():
        if category_dir.is_dir():
            count = (
                len(list(category_dir.glob("*.txt")))
                + len(list(category_dir.glob("*.csv")))
                + len(list(category_dir.glob("*.json")))
            )
            file_counts[category_dir.name] = count
            total_files += count

    if file_counts:
        st.info(f"**Dataset**: {dataset_path.name}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Categories", len(file_counts))

        # Display breakdown
        for category, count in file_counts.items():
            st.text(f"• {category}: {count} files")


def render_training_parameters():
    """Render training parameter configuration with enhanced options"""
    st.markdown("#### Training Parameters")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )

    with col2:
        num_folds = st.number_input(
            "Cross-Validation Folds", min_value=3, max_value=10, value=10
        )
        target_len = st.number_input(
            "Target Length", min_value=100, max_value=1000, value=500
        )
        modality = st.selectbox("Modality", ["raman", "ftir"], index=0)

    # Advanced Cross-Validation Options
    st.markdown("**Cross-Validation Strategy**")
    cv_strategy = st.selectbox(
        "CV Strategy",
        ["stratified_kfold", "kfold", "time_series_split"],
        index=0,
        help="Choose CV strategy: Stratified K-Fold (recommended for balanced datasets), K-Fold (for any dataset), Time Series Split (for temporal data)",
    )

    # Data Augmentation Options
    st.markdown("**Data Augmentation**")
    col1, col2 = st.columns(2)

    with col1:
        enable_augmentation = st.checkbox(
            "Enable Spectral Augmentation",
            value=False,
            help="Add realistic noise and variations to improve model robustness",
        )
    with col2:
        noise_level = st.slider(
            "Noise Level",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
            disabled=not enable_augmentation,
            help="Amount of Gaussian noise to add for augmentation",
        )

    # Spectroscopy-Specific Options
    st.markdown("**Spectroscopy-Specific Settings**")
    spectral_weight = st.slider(
        "Spectral Metrics Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Weight for spectroscopy-specific metrics (cosine similarity, peak matching)",
    )

    # Preprocessing options
    st.markdown("**Preprocessing Options**")
    col1, col2, col3 = st.columns(3)

    with col1:
        baseline_correction = st.checkbox("Baseline Correction", value=True)
    with col2:
        smoothing = st.checkbox("Smoothing", value=True)
    with col3:
        normalization = st.checkbox("Normalization", value=True)

    # Device selection
    device_options = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")

    device = st.selectbox("Device", device_options, index=0)

    # Store parameters in session state
    st.session_state.update(
        {
            "train_epochs": epochs,
            "train_batch_size": batch_size,
            "train_learning_rate": learning_rate,
            "train_num_folds": num_folds,
            "train_target_len": target_len,
            "train_modality": modality,
            "train_cv_strategy": cv_strategy,
            "train_enable_augmentation": enable_augmentation,
            "train_noise_level": noise_level,
            "train_spectral_weight": spectral_weight,
            "train_baseline_correction": baseline_correction,
            "train_smoothing": smoothing,
            "train_normalization": normalization,
            "train_device": device,
        }
    )


def render_training_status():
    """Render training status and active jobs"""
    st.markdown("### 📊 Training Status")

    training_manager = get_training_manager()

    # Active jobs
    active_jobs = training_manager.list_jobs(TrainingStatus.RUNNING)
    pending_jobs = training_manager.list_jobs(TrainingStatus.PENDING)

    if active_jobs or pending_jobs:
        st.markdown("#### Active Jobs")
        for job in active_jobs + pending_jobs:
            render_job_status_card(job)

    # Recent completed jobs
    completed_jobs = training_manager.list_jobs(TrainingStatus.COMPLETED)[
        :3
    ]  # Show last 3
    if completed_jobs:
        st.markdown("#### Recent Completed")
        for job in completed_jobs:
            render_job_status_card(job, compact=True)


def render_job_status_card(job: TrainingJob, compact: bool = False):
    """Render a status card for a training job"""
    status_color = {
        TrainingStatus.PENDING: "🟡",
        TrainingStatus.RUNNING: "🔵",
        TrainingStatus.COMPLETED: "🟢",
        TrainingStatus.FAILED: "🔴",
        TrainingStatus.CANCELLED: "⚫",
    }

    with st.expander(
        f"{status_color[job.status]} {job.config.model_name} - {job.job_id[:8]}",
        expanded=not compact,
    ):
        if not compact:
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Model: {job.config.model_name}")
                st.text(f"Dataset: {Path(job.config.dataset_path).name}")
                st.text(f"Status: {job.status.value}")
            with col2:
                st.text(f"Created: {job.created_at.strftime('%H:%M:%S')}")
                if job.status == TrainingStatus.RUNNING:
                    st.text(
                        f"Fold: {job.progress.current_fold}/{job.progress.total_folds}"
                    )
                    st.text(
                        f"Epoch: {job.progress.current_epoch}/{job.progress.total_epochs}"
                    )

        if job.status == TrainingStatus.RUNNING:
            # Progress bars
            fold_progress = job.progress.current_fold / job.progress.total_folds
            epoch_progress = job.progress.current_epoch / job.progress.total_epochs

            st.progress(fold_progress)
            st.caption(
                f"Overall: {fold_progress:.1%} | Current Loss: {job.progress.current_loss:.4f}"
            )

        elif job.status == TrainingStatus.COMPLETED and job.progress.fold_accuracies:
            mean_acc = np.mean(job.progress.fold_accuracies)
            std_acc = np.std(job.progress.fold_accuracies)
            st.success(f"✅ Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

        elif job.status == TrainingStatus.FAILED:
            st.error(f"❌ Error: {job.error_message}")


def render_training_progress():
    """Render detailed training progress visualization"""
    st.markdown("### 📈 Training Progress")

    training_manager = get_training_manager()
    active_jobs = training_manager.list_jobs(TrainingStatus.RUNNING)

    if not active_jobs:
        st.info("No active training jobs. Start a training job to see progress here.")
        return

    # Job selector for multiple active jobs
    if len(active_jobs) > 1:
        selected_job_id = st.selectbox(
            "Select Job to Monitor",
            [job.job_id for job in active_jobs],
            format_func=lambda x: f"{x[:8]} - {next(job.config.model_name for job in active_jobs if job.job_id == x)}",
        )
        selected_job = next(job for job in active_jobs if job.job_id == selected_job_id)
    else:
        selected_job = active_jobs[0]

    # Real-time progress visualization
    render_job_progress_details(selected_job)


def render_job_progress_details(job: TrainingJob):
    """Render detailed progress for a specific job with enhanced metrics"""
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Fold", f"{job.progress.current_fold}/{job.progress.total_folds}"
        )
        st.metric(
            "Current Epoch", f"{job.progress.current_epoch}/{job.progress.total_epochs}"
        )

    with col2:
        st.metric("Current Loss", f"{job.progress.current_loss:.4f}")
        st.metric("Current Accuracy", f"{job.progress.current_accuracy:.3f}")

    # Progress bars
    fold_progress = (
        job.progress.current_fold / job.progress.total_folds
        if job.progress.total_folds > 0
        else 0
    )
    epoch_progress = (
        job.progress.current_epoch / job.progress.total_epochs
        if job.progress.total_epochs > 0
        else 0
    )

    st.progress(fold_progress)
    st.caption(f"Overall Progress: {fold_progress:.1%}")

    st.progress(epoch_progress)
    st.caption(f"Current Fold Progress: {epoch_progress:.1%}")

    # Enhanced metrics visualization
    if job.progress.fold_accuracies and job.progress.spectroscopy_metrics:
        col1, col2 = st.columns(2)

        with col1:
            # Standard accuracy chart
            fig_acc = go.Figure(
                data=go.Bar(
                    x=[f"Fold {i+1}" for i in range(len(job.progress.fold_accuracies))],
                    y=job.progress.fold_accuracies,
                    name="Validation Accuracy",
                    marker_color="lightblue",
                )
            )
            fig_acc.update_layout(
                title="Cross-Validation Accuracies by Fold",
                yaxis_title="Accuracy",
                height=300,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            # Spectroscopy-specific metrics
            if len(job.progress.spectroscopy_metrics) > 0:
                # Extract metrics across folds
                f1_scores = [
                    m.get("f1_score", 0) for m in job.progress.spectroscopy_metrics
                ]
                cosine_sim = [
                    m.get("cosine_similarity", 0)
                    for m in job.progress.spectroscopy_metrics
                ]
                dist_sim = [
                    m.get("distribution_similarity", 0)
                    for m in job.progress.spectroscopy_metrics
                ]

                fig_spectro = go.Figure()

                # Add traces for different metrics
                fig_spectro.add_trace(
                    go.Scatter(
                        x=[f"Fold {i+1}" for i in range(len(f1_scores))],
                        y=f1_scores,
                        mode="lines+markers",
                        name="F1 Score",
                        line=dict(color="green"),
                    )
                )

                if any(c > 0 for c in cosine_sim):
                    fig_spectro.add_trace(
                        go.Scatter(
                            x=[f"Fold {i+1}" for i in range(len(cosine_sim))],
                            y=cosine_sim,
                            mode="lines+markers",
                            name="Cosine Similarity",
                            line={"color": "orange"},
                        )
                    )

                fig_spectro.add_trace(
                    go.Scatter(
                        x=[f"Fold {i+1}" for i in range(len(dist_sim))],
                        y=dist_sim,
                        mode="lines+markers",
                        name="Distribution Similarity",
                        line=dict(color="purple"),
                    )
                )

                fig_spectro.update_layout(
                    title="Spectroscopy-Specific Metrics by Fold",
                    yaxis_title="Score",
                    height=300,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig_spectro, use_container_width=True)

    elif job.progress.fold_accuracies:
        # Fallback to standard accuracy chart only
        fig = go.Figure(
            data=go.Bar(
                x=[f"Fold {i+1}" for i in range(len(job.progress.fold_accuracies))],
                y=job.progress.fold_accuracies,
                name="Validation Accuracy",
            )
        )
        fig.update_layout(
            title="Cross-Validation Accuracies by Fold",
            yaxis_title="Accuracy",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_training_history():
    """Render training history and results"""
    st.markdown("### 📚 Training History")

    training_manager = get_training_manager()
    all_jobs = training_manager.list_jobs()

    if not all_jobs:
        st.info("No training history available. Start training some models!")
        return

    # Convert to DataFrame for display
    history_data = []
    for job in all_jobs:
        row = {
            "Job ID": job.job_id[:8],
            "Model": job.config.model_name,
            "Dataset": Path(job.config.dataset_path).name,
            "Status": job.status.value,
            "Created": job.created_at.strftime("%Y-%m-%d %H:%M"),
            "Duration": "",
            "Accuracy": "",
        }

        if job.completed_at and job.started_at:
            duration = job.completed_at - job.started_at
            row["Duration"] = str(duration).split(".")[0]  # Remove microseconds

        if job.status == TrainingStatus.COMPLETED and job.progress.fold_accuracies:
            mean_acc = np.mean(job.progress.fold_accuracies)
            std_acc = np.std(job.progress.fold_accuracies)
            row["Accuracy"] = f"{mean_acc:.3f} ± {std_acc:.3f}"

        history_data.append(row)

    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)

    # Job details
    if st.checkbox("Show detailed results"):
        completed_jobs = [
            job for job in all_jobs if job.status == TrainingStatus.COMPLETED
        ]
        if completed_jobs:
            selected_job_id = st.selectbox(
                "Select job for details",
                [job.job_id for job in completed_jobs],
                format_func=lambda x: f"{x[:8]} - {next(job.config.model_name for job in completed_jobs if job.job_id == x)}",
            )

            selected_job = next(
                job for job in completed_jobs if job.job_id == selected_job_id
            )
            render_training_results(selected_job)


def render_training_results(job: TrainingJob):
    """Render detailed training results for a completed job with enhanced metrics"""
    st.markdown(f"#### Results for {job.config.model_name} - {job.job_id[:8]}")

    if not job.progress.fold_accuracies:
        st.warning("No results available for this job.")
        return

    # Summary metrics
    mean_acc = np.mean(job.progress.fold_accuracies)
    std_acc = np.std(job.progress.fold_accuracies)

    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Accuracy", f"{mean_acc:.3f}")
    with col2:
        st.metric("Std Deviation", f"{std_acc:.3f}")
    with col3:
        st.metric("Best Fold", f"{max(job.progress.fold_accuracies):.3f}")
    with col4:
        st.metric("CV Strategy", job.config.cv_strategy.replace("_", " ").title())

    # Spectroscopy-specific metrics summary
    if job.progress.spectroscopy_metrics:
        st.markdown("**Spectroscopy-Specific Metrics Summary**")
        spectro_summary = {}

        for metric_name in ["f1_score", "cosine_similarity", "distribution_similarity"]:
            values = [
                m.get(metric_name, 0)
                for m in job.progress.spectroscopy_metrics
                if m.get(metric_name, 0) > 0
            ]
            if values:
                spectro_summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "best": max(values),
                }

        if spectro_summary:
            cols = st.columns(len(spectro_summary))
            for i, (metric, stats) in enumerate(spectro_summary.items()):
                with cols[i]:
                    metric_display = metric.replace("_", " ").title()
                    st.metric(
                        f"{metric_display}",
                        f"{stats['mean']:.3f} ± {stats['std']:.3f}",
                        f"Best: {stats['best']:.3f}",
                    )

    # Configuration summary
    with st.expander("Training Configuration"):
        config_display = {
            "Model": job.config.model_name,
            "Dataset": Path(job.config.dataset_path).name,
            "Epochs": job.config.epochs,
            "Batch Size": job.config.batch_size,
            "Learning Rate": job.config.learning_rate,
            "CV Folds": job.config.num_folds,
            "CV Strategy": job.config.cv_strategy,
            "Augmentation": "Enabled" if job.config.enable_augmentation else "Disabled",
            "Noise Level": (
                job.config.noise_level if job.config.enable_augmentation else "N/A"
            ),
            "Spectral Weight": job.config.spectral_weight,
            "Device": job.config.device,
        }

        config_df = pd.DataFrame(
            list(config_display.items()), columns=["Parameter", "Value"]
        )
        st.dataframe(config_df, use_container_width=True)

    # Enhanced visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Accuracy distribution
        fig_acc = go.Figure(
            data=go.Box(y=job.progress.fold_accuracies, name="Fold Accuracies")
        )
        fig_acc.update_layout(
            title="Cross-Validation Accuracy Distribution", yaxis_title="Accuracy"
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        # Metrics comparison if available
        if (
            job.progress.spectroscopy_metrics
            and len(job.progress.spectroscopy_metrics) > 0
        ):
            metrics_df = pd.DataFrame(job.progress.spectroscopy_metrics)

            if not metrics_df.empty:
                fig_metrics = go.Figure()

                for col in metrics_df.columns:
                    if col in [
                        "accuracy",
                        "f1_score",
                        "cosine_similarity",
                        "distribution_similarity",
                    ]:
                        fig_metrics.add_trace(
                            go.Scatter(
                                x=list(range(1, len(metrics_df) + 1)),
                                y=metrics_df[col],
                                mode="lines+markers",
                                name=col.replace("_", " ").title(),
                            )
                        )

                fig_metrics.update_layout(
                    title="All Metrics Across Folds",
                    xaxis_title="Fold",
                    yaxis_title="Score",
                    height=300,
                )
                st.plotly_chart(fig_metrics, use_container_width=True)

    # Download options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Download Weights", key=f"weights_{job.job_id}"):
            if job.weights_path and os.path.exists(job.weights_path):
                with open(job.weights_path, "rb") as f:
                    st.download_button(
                        "Download Model Weights",
                        f.read(),
                        file_name=f"{job.config.model_name}_{job.job_id[:8]}.pth",
                        mime="application/octet-stream",
                    )

    with col2:
        if st.button("📄 Download Logs", key=f"logs_{job.job_id}"):
            if job.logs_path and os.path.exists(job.logs_path):
                with open(job.logs_path, "r") as f:
                    st.download_button(
                        "Download Training Logs",
                        f.read(),
                        file_name=f"training_log_{job.job_id[:8]}.json",
                        mime="application/json",
                    )

    with col3:
        if st.button("📊 Download Metrics CSV", key=f"metrics_{job.job_id}"):
            # Create comprehensive metrics CSV
            metrics_data = []
            for i, (acc, spectro) in enumerate(
                zip(
                    job.progress.fold_accuracies,
                    job.progress.spectroscopy_metrics or [],
                )
            ):
                row = {"fold": i + 1, "accuracy": acc}
                if spectro:
                    row.update(spectro)
                metrics_data.append(row)

            metrics_df = pd.DataFrame(metrics_data)
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                "Download Metrics CSV",
                csv,
                file_name=f"metrics_{job.job_id[:8]}.csv",
                mime="text/csv",
            )

    # Interpretability section
    if st.checkbox("🔍 Show Model Interpretability", key=f"interpret_{job.job_id}"):
        render_model_interpretability(job)


def render_model_interpretability(job: TrainingJob):
    """Render model interpretability features"""
    st.markdown("##### 🔍 Model Interpretability")

    try:
        # Try to load the trained model for interpretation
        if not job.weights_path or not os.path.exists(job.weights_path):
            st.warning("Model weights not available for interpretation.")
            return

        # Simple feature importance visualization
        st.markdown("**Feature Importance Analysis**")

        # Generate mock feature importance for demonstration
        # In a real implementation, this would use SHAP, Captum, or gradient-based methods
        wavenumbers = np.linspace(400, 4000, job.config.target_len)

        # Simulate feature importance (peaks at common polymer bands)
        importance = np.zeros_like(wavenumbers)

        # Simulate important regions for polymer degradation
        # C-H stretch (2800-3000 cm⁻¹)
        ch_region = (wavenumbers >= 2800) & (wavenumbers <= 3000)
        importance[ch_region] = np.random.normal(0.8, 0.1, (np.sum(ch_region),))

        # C=O stretch (1600-1800 cm⁻¹) - often changes with degradation
        co_region = (wavenumbers >= 1600) & (wavenumbers <= 1800)
        importance[co_region] = np.random.normal(0.9, 0.1, int(np.sum(co_region)))

        # Fingerprint region (400-1500 cm⁻¹)
        fingerprint_region = (wavenumbers >= 400) & (wavenumbers <= 1500)
        importance[fingerprint_region] = np.random.normal(
            0.3, 0.2, int(np.sum(fingerprint_region))
        )

        # Normalize importance
        importance = np.abs(importance)
        importance = (
            importance / np.max(importance) if np.max(importance) > 0 else importance
        )

        # Create interpretability plot
        fig_interpret = go.Figure()

        # Add feature importance
        fig_interpret.add_trace(
            go.Scatter(
                x=wavenumbers,
                y=importance,
                mode="lines",
                name="Feature Importance",
                fill="tonexty",
                line=dict(color="red", width=2),
            )
        )

        # Add annotations for important regions
        fig_interpret.add_annotation(
            x=2900,
            y=0.8,
            text="C-H Stretch<br>(Polymer backbone)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="blue",
            bgcolor="lightblue",
            bordercolor="blue",
        )

        fig_interpret.add_annotation(
            x=1700,
            y=0.9,
            text="C=O Stretch<br>(Degradation marker)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="lightcoral",
            bordercolor="red",
        )

        fig_interpret.update_layout(
            title="Model Feature Importance for Polymer Degradation Classification",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Feature Importance",
            height=400,
            showlegend=False,
        )

        st.plotly_chart(fig_interpret, use_container_width=True)

        # Interpretation insights
        st.markdown("**Key Insights:**")
        col1, col2 = st.columns(2)

        with col1:
            st.info(
                "🔬 **High Importance Regions:**\n"
                "- C=O stretch (1600-1800 cm⁻¹): Critical for degradation detection\n"
                "- C-H stretch (2800-3000 cm⁻¹): Polymer backbone changes"
            )

        with col2:
            st.info(
                "📊 **Model Behavior:**\n"
                "- Focuses on spectral regions known to change with polymer degradation\n"
                "- Fingerprint region provides molecular specificity"
            )

        # Attention heatmap simulation
        st.markdown("**Spectral Attention Heatmap**")

        # Create a 2D heatmap showing attention across different samples
        n_samples = 10
        attention_matrix = np.random.beta(2, 5, (n_samples, len(wavenumbers)))

        # Enhance attention in important regions
        for i in range(n_samples):
            attention_matrix[i, ch_region] *= np.random.uniform(2, 4)
            attention_matrix[i, co_region] *= np.random.uniform(3, 5)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=attention_matrix,
                x=wavenumbers[::10],  # Subsample for display
                y=[f"Sample {i+1}" for i in range(n_samples)],
                colorscale="Viridis",
                colorbar=dict(title="Attention Score"),
            )
        )

        fig_heatmap.update_layout(
            title="Model Attention Across Different Samples",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Sample",
            height=300,
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown(
            "**Note:** *This interpretability analysis is simulated for demonstration. "
            "In production, this would use actual gradient-based attribution methods "
            "(SHAP, Integrated Gradients, etc.) on the trained model.*"
        )

    except Exception as e:
        st.error(f"Error generating interpretability analysis: {e}")
        st.info("Interpretability features require the trained model to be available.")


def start_training_job():
    """Start a new training job with current configuration"""
    # Validate configuration
    if "selected_dataset" not in st.session_state:
        st.error("❌ Please select a dataset first.")
        return

    if not Path(st.session_state["selected_dataset"]).exists():
        st.error("❌ Selected dataset path does not exist.")
        return

    # Create training configuration
    config = TrainingConfig(
        model_name=st.session_state.get("selected_model", "figure2"),
        dataset_path=st.session_state["selected_dataset"],
        target_len=st.session_state.get("train_target_len", 500),
        batch_size=st.session_state.get("train_batch_size", 16),
        epochs=st.session_state.get("train_epochs", 10),
        learning_rate=st.session_state.get("train_learning_rate", 1e-3),
        num_folds=st.session_state.get("train_num_folds", 10),
        baseline_correction=st.session_state.get("train_baseline_correction", True),
        smoothing=st.session_state.get("train_smoothing", True),
        normalization=st.session_state.get("train_normalization", True),
        modality=st.session_state.get("train_modality", "raman"),
        device=st.session_state.get("train_device", "auto"),
        cv_strategy=st.session_state.get("train_cv_strategy", "stratified_kfold"),
        enable_augmentation=st.session_state.get("train_enable_augmentation", False),
        noise_level=st.session_state.get("train_noise_level", 0.01),
        spectral_weight=st.session_state.get("train_spectral_weight", 0.1),
    )

    # Submit job
    training_manager = get_training_manager()
    job_id = training_manager.submit_training_job(config)

    st.success(f"✅ Training job started! Job ID: {job_id[:8]}")
    st.info("Monitor progress in the Training Status section above.")

    # Auto-refresh to show new job
    time.sleep(1)
    st.rerun()


def save_uploaded_dataset(
    uploaded_files, dataset_name: str, file_labels: Dict[str, str]
):
    """Save uploaded dataset to local storage"""
    try:
        # Create dataset directory
        dataset_dir = Path("datasets") / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create label directories
        (dataset_dir / "stable").mkdir(exist_ok=True)
        (dataset_dir / "weathered").mkdir(exist_ok=True)

        # Save files
        saved_count = 0
        for file in uploaded_files:
            # Determine label
            label = file_labels.get(file.name, "stable")  # Default to stable
            if "weathered" in file.name.lower() or "degraded" in file.name.lower():
                label = "weathered"

            # Save file
            target_path = dataset_dir / label / file.name
            with open(target_path, "wb") as f:
                f.write(file.getbuffer())
            saved_count += 1

        st.success(
            f"✅ Dataset '{dataset_name}' saved successfully! {saved_count} files processed."
        )
        st.session_state["selected_dataset"] = str(dataset_dir)

        # Display saved dataset info
        display_dataset_info(dataset_dir)

    except Exception as e:
        st.error(f"❌ Error saving dataset: {str(e)}")


# Auto-refresh for active training jobs
def setup_training_auto_refresh():
    """Set up auto-refresh for training progress"""
    if "training_auto_refresh" not in st.session_state:
        st.session_state.training_auto_refresh = True

    training_manager = get_training_manager()
    active_jobs = training_manager.list_jobs(TrainingStatus.RUNNING)

    if active_jobs and st.session_state.training_auto_refresh:
        # Auto-refresh every 5 seconds if there are active jobs
        time.sleep(5)
        st.rerun()
