import os
import torch
import streamlit as st
import hashlib
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import time
from config import TARGET_LEN, LABEL_MAP, MODEL_WEIGHTS_DIR
from models.registry import choices, get_model_info
from modules.callbacks import (
    on_model_change,
    on_input_mode_change,
    on_sample_change,
    reset_results,
    reset_ephemeral_state,
    log_message,
)
from core_logic import (
    get_sample_files,
    load_model,
    run_inference,
    parse_spectrum_data,
    label_file,
)
from utils.results_manager import ResultsManager
from utils.multifile import process_multiple_files
from utils.preprocessing import resample_spectrum
from utils.confidence import calculate_softmax_confidence


def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def create_spectrum_plot(x_raw, y_raw, x_resampled, y_resampled, _cache_key=None):
    """Create spectrum visualization plot"""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

    # Raw spectrum
    ax[0].plot(x_raw, y_raw, label="Raw", color="dimgray", linewidth=1)
    ax[0].set_title("Raw Input Spectrum")
    ax[0].set_xlabel("Wavenumber (cm⁻¹)")
    ax[0].set_ylabel("Intensity")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Resampled spectrum
    ax[1].plot(
        x_resampled, y_resampled, label="Resampled", color="steelblue", linewidth=1
    )
    ax[1].set_title(f"Resampled ({len(y_resampled)} points)")
    ax[1].set_xlabel("Wavenumber (cm⁻¹)")
    ax[1].set_ylabel("Intensity")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)  # Prevent memory leaks

    return Image.open(buf)


# //////////////////////////////////////////


def render_confidence_progress(
    probs: np.ndarray,
    labels: list[str] = ["Stable", "Weathered"],
    highlight_idx: Union[int, None] = None,
    side_by_side: bool = True,
):
    """Render Streamlit native progress bars with scientific formatting."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    if side_by_side:
        cols = st.columns(len(labels))
        for i, (lbl, val, col) in enumerate(zip(labels, p, cols)):
            with col:
                is_highlighted = highlight_idx is not None and i == highlight_idx
                label_text = f"**{lbl}**" if is_highlighted else lbl
                st.markdown(f"{label_text}: {val*100:.1f}%")
                st.progress(int(round(val * 100)))
    else:
        # Vertical layout for better readability
        for i, (lbl, val) in enumerate(zip(labels, p)):
            is_highlighted = highlight_idx is not None and i == highlight_idx

            # Create a container for each probability
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    if is_highlighted:
                        st.markdown(f"**{lbl}** ← Predicted")
                    else:
                        st.markdown(f"{lbl}")
                with col2:
                    st.metric(label="", value=f"{val*100:.1f}%", delta=None)

                # Progress bar with conditional styling
                if is_highlighted:
                    st.progress(int(round(val * 100)))
                    st.caption("🎯 **Model Prediction**")
                else:
                    st.progress(int(round(val * 100)))

                if i < len(labels) - 1:  # Add spacing between items
                    st.markdown("")


from typing import Optional


def render_kv_grid(d: Optional[dict] = None, ncols: int = 2):
    if d is None:
        d = {}
    if not d:
        return
    items = list(d.items())
    cols = st.columns(ncols)
    for i, (k, v) in enumerate(items):
        with cols[i % ncols]:
            st.caption(f"**{k}:** {v}")


# //////////////////////////////////////////


def render_model_meta(model_choice: str):
    info = get_model_info(model_choice)
    emoji = info.get("emoji", "")
    desc = info.get("description", "").strip()
    acc = info.get("performance", {}).get("accuracy", "-")
    f1 = info.get("performance", {}).get("f1_score", "-")

    st.caption(f"{emoji} **Model Snapshot** - {model_choice}")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Accuracy", acc)
    with cols[1]:
        st.metric("F1 Score", f1)
    if desc:
        st.caption(desc)


# //////////////////////////////////////////


def get_confidence_description(logit_margin):
    """Get human-readable confidence description"""
    if logit_margin > 1000:
        return "VERY HIGH", "🟢"
    elif logit_margin > 250:
        return "HIGH", "🟡"
    elif logit_margin > 100:
        return "MODERATE", "🟠"
    else:
        return "LOW", "🔴"


# //////////////////////////////////////////


def render_sidebar():
    with st.sidebar:
        # Header
        st.header("AI-Driven Polymer Classification")
        st.caption(
            "Predict polymer degradation (Stable vs Weathered) from Raman/FTIR spectra using validated CNN models. — v0.01"
        )

        # Modality Selection
        st.markdown("##### Spectroscopy Modality")
        modality = st.selectbox(
            "Choose Modality",
            ["raman", "ftir"],
            index=0,
            key="modality_select",
            format_func=lambda x: f"{'Raman' if x == 'raman' else 'FTIR'}",
        )

        # Display modality info
        if modality == "ftir":
            st.info("FTIR mode: 400-4000 cm-1 range with atmospheric correction")
        else:
            st.info("Raman mode: 200-4000 cm-1 range with standard preprocessing")

        # Model selection
        st.markdown("##### AI Model Selection")

        model_emojis = {
            "figure2": "📄",
            "resnet": "🧠",
            "resnet18vision": "👁️",
            "enhanced_cnn": "✨",
            "efficient_cnn": "⚡",
            "hybrid_net": "🧬",
        }

        available_models = choices()
        model_labels = [
            f"{model_emojis.get(name, '🤖')} {name}" for name in available_models
        ]

        selected_label = st.selectbox(
            "Choose AI Model",
            model_labels,
            key="model_select",
            on_change=on_model_change,
        )
        model_choice = selected_label.split(" ", 1)[1]

        # Compact metadata directly under dropdown
        render_model_meta(model_choice)

        # Collapsed info to reduce clutter
        with st.expander("About This App", icon=":material/info:", expanded=False):
            st.markdown(
                """
            **AI-Driven Polymer Aging Prediction and Classification**

            **Purpose**: Classify polymer degradation using AI<br>
            **Input**: Raman spectroscopy .txt files<br>
            **Models**: CNN architectures for classification<br>
            **Modalities**: Raman and FTIR spectroscopy support<br>
            **Features**: Multi-model comparison and analysis<br>


            **Contributors**<br>
            - Dr. Sanmukh Kuppannagari (Mentor)<br>
            - Dr. Metin Karailyan (Mentor)<br>
            - Jaser Hasan (Author)<br>


            **Links**<br>
            [HF Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)<br>
            [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)


            **Citation Figure2CNN (baseline)**
            Neo et al., 2023, *Resour. Conserv. Recycl.*, 188, 106718.
            [https://doi.org/10.1016/j.resconrec.2022.106718](https://doi.org/10.1016/j.resconrec.2022.106718)
            """,
                unsafe_allow_html=True,
            )


# //////////////////////////////////////////
def render_input_column():
    st.markdown("##### Data Input")

    mode = st.radio(
        "Input mode",
        ["Upload File", "Batch Upload", "Sample Data"],
        key="input_mode",
        horizontal=True,
        on_change=on_input_mode_change,
    )

    # == Input Mode Logic ==
    if mode == "Upload File":
        upload_key = st.session_state["current_upload_key"]
        up = st.file_uploader(
            "Upload spectrum file (.txt, .csv, .json)",
            type=["txt", "csv", "json"],
            help="Upload spectroscopy data: TXT (2-column), CSV (with headers), or JSON format",
            key=upload_key,  # ← versioned key
        )

        # Process change immediately
        if up is not None:
            raw = up.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            # only reparse if its a different file|source
            if (
                st.session_state.get("filename") != getattr(up, "name", None)
                or st.session_state.get("input_source") != "upload"
            ):
                st.session_state["input_text"] = text
                st.session_state["filename"] = getattr(up, "name", None)
                st.session_state["input_source"] = "upload"
                # Ensure single file mode
                st.session_state["batch_mode"] = False
                st.session_state["status_message"] = (
                    f"File '{st.session_state['filename']}' ready for analysis"
                )
                st.session_state["status_type"] = "success"
                reset_results("New file uploaded")

    # Batch Upload tab
    elif mode == "Batch Upload":
        st.session_state["batch_mode"] = True
        # Use a versioned key to ensure the file uploader resets properly.
        batch_upload_key = f"batch_upload_{st.session_state['uploader_version']}"
        uploaded_files = st.file_uploader(
            "Upload multiple spectrum files (.txt, .csv, .json)",
            type=["txt", "csv", "json"],
            accept_multiple_files=True,
            help="Upload spectroscopy files in TXT, CSV, or JSON format.",
            key=batch_upload_key,
        )

        if uploaded_files:
            # Use a dictionary to keep only unique files based on name and size
            unique_files = {(file.name, file.size): file for file in uploaded_files}
            unique_file_list = list(unique_files.values())

            num_uploaded = len(uploaded_files)
            num_unique = len(unique_file_list)

            # Optionally, inform the user that duplicates were removed
            if num_uploaded > num_unique:
                st.info(f"{num_uploaded - num_unique} duplicate file(s) were removed.")

            # Use the unique list
            st.session_state["batch_files"] = unique_file_list
            st.session_state["status_message"] = (
                f"{num_unique} ready for batch analysis"
            )
            st.session_state["status_type"] = "success"
        else:
            st.session_state["batch_files"] = []
            # This check prevents resetting the status if files are already staged
            if not st.session_state.get("batch_files"):
                st.session_state["status_message"] = (
                    "No files selected for batch processing"
                )
                st.session_state["status_type"] = "info"

    # Sample tab
    elif mode == "Sample Data":
        st.session_state["batch_mode"] = False
        sample_files = get_sample_files()
        if sample_files:
            options = ["-- Select Sample --"] + [p.name for p in sample_files]
            sel = st.selectbox(
                "Choose sample spectrum:",
                options,
                key="sample_select",
                on_change=on_sample_change,
            )
            if sel != "-- Select Sample --":
                st.session_state["status_message"] = (
                    f"📁 Sample '{sel}' ready for analysis"
                )
                st.session_state["status_type"] = "success"
        else:
            st.info("No sample data available")
    # == Status box (displays the message) ==
    msg = st.session_state.get("status_message", "Ready")
    typ = st.session_state.get("status_type", "info")
    if typ == "success":
        st.success(msg)
    elif typ == "error":
        st.error(msg)
    else:
        st.info(msg)

    # Safely get model choice from session state
    model_choice = st.session_state.get("model_select", " ").split(" ", 1)[1]
    model = load_model(model_choice)

    # Determine if the app is ready for inference
    is_batch_ready = st.session_state.get("batch_mode", False) and st.session_state.get(
        "batch_files"
    )
    is_single_ready = not st.session_state.get(
        "batch_mode", False
    ) and st.session_state.get("input_text")
    inference_ready = (is_batch_ready or is_single_ready) and model is not None
    # Store for other modules to access
    st.session_state["inference_ready"] = inference_ready

    # Render buttons
    with st.form("analysis_form", clear_on_submit=False):
        submitted = st.form_submit_button(
            "Run Analysis", type="primary", disabled=not inference_ready
        )
    st.button(
        "Reset All",
        on_click=reset_ephemeral_state,
        help="Clear all uploaded files and results.",
    )

    # Handle form submission
    if submitted and inference_ready:
        if st.session_state.get("batch_mode"):
            batch_files = st.session_state.get("batch_files", [])
            with st.spinner(f"Processing {len(batch_files)} files ..."):
                st.session_state["batch_results"] = process_multiple_files(
                    uploaded_files=batch_files,
                    model_choice=model_choice,
                    run_inference_func=run_inference,
                    label_file_func=label_file,
                    modality=st.session_state.get("modality_select", "raman"),
                )
        else:
            try:
                x_raw, y_raw = parse_spectrum_data(st.session_state["input_text"])
                x_resampled, y_resampled = resample_spectrum(x_raw, y_raw, TARGET_LEN)
                st.session_state.update(
                    {
                        "x_raw": x_raw,
                        "y_raw": y_raw,
                        "x_resampled": x_resampled,
                        "y_resampled": y_resampled,
                        "inference_run_once": True,
                    }
                )
            except (ValueError, TypeError) as e:
                st.error(f"Error processing spectrum data: {e}")


# //////////////////////////////////////////


def render_results_column():
    # Get the current mode and check for batch results
    is_batch_mode = st.session_state.get("batch_mode", False)
    has_batch_results = "batch_results" in st.session_state

    if is_batch_mode and has_batch_results:
        # THEN render the main interactive dashboard from ResultsManager
        ResultsManager.display_results_table()

    elif st.session_state.get("inference_run_once", False) and not is_batch_mode:
        st.markdown("##### Analysis Results")
        # Get data from session state
        x_raw = st.session_state.get("x_raw")
        y_raw = st.session_state.get("y_raw")
        x_resampled = st.session_state.get("x_resampled")  # ← NEW
        y_resampled = st.session_state.get("y_resampled")
        filename = st.session_state.get("filename", "Unknown")

        if all(v is not None for v in [x_raw, y_raw, y_resampled]):
            # Run inference
            if y_resampled is None:
                raise ValueError(
                    "y_resampled is None. Ensure spectrum data is properly resampled before proceeding."
                )
            cache_key = hashlib.md5(
                f"{y_resampled.tobytes()}{st.session_state.get('model_select', 'Unknown').split(' ', 1)[1]}".encode()
            ).hexdigest()
            # MODIFIED: Pass modality to run_inference
            prediction, logits_list, probs, inference_time, logits = run_inference(
                y_resampled,
                (
                    st.session_state.get("model_select", "").split(" ", 1)[1]
                    if "model_select" in st.session_state
                    else None
                ),
                modality=st.session_state.get("modality_select", "raman"),
                _cache_key=cache_key,
            )
            if prediction is None:
                st.error(
                    "❌ Inference failed: Model not loaded. Please check that weights are available."
                )
                st.stop()  # prevents the rest of the code in this block from executing

            log_message(
                f"Inference completed in {inference_time:.2f}s, prediction: {prediction}"
            )

            # Get ground truth
            true_label_idx = label_file(filename)
            true_label_str = (
                LABEL_MAP.get(true_label_idx, "Unknown")
                if true_label_idx is not None
                else "Unknown"
            )
            # Get prediction
            predicted_class = LABEL_MAP.get(int(prediction), f"Class {int(prediction)}")

            # Enhanced confidence calculation
            if logits is not None:
                # Use new softmax-based confidence
                probs_np, max_confidence, confidence_level, confidence_emoji = (
                    calculate_softmax_confidence(logits)
                )
                confidence_desc = confidence_level
            else:
                # Fallback to legacy method
                logit_margin = abs(
                    (logits_list[0] - logits_list[1])
                    if logits_list is not None and len(logits_list) >= 2
                    else 0
                )
                confidence_desc, confidence_emoji = get_confidence_description(
                    logit_margin
                )
                max_confidence = logit_margin / 10.0  # Normalize for display
                probs_np = np.array([])

            # Store result in results manager for single file too
            ResultsManager.add_results(
                filename=filename,
                model_name=(
                    st.session_state.get("model_select", "").split(" ", 1)[1]
                    if "model_select" in st.session_state
                    else "Unknown"
                ),
                prediction=int(prediction),
                predicted_class=predicted_class,
                confidence=max_confidence,
                logits=logits_list if logits_list else [],
                ground_truth=true_label_idx if true_label_idx >= 0 else None,
                processing_time=inference_time if inference_time is not None else 0.0,
                metadata={
                    "confidence_level": confidence_desc,
                    "confidence_emoji": confidence_emoji,
                },
            )

            # Precompute Stats
            model_choice = (
                st.session_state.get("model_select", "").split(" ", 1)[1]
                if "model_select" in st.session_state
                else None
            )
            if not model_choice:
                st.error(
                    "⚠️ Model choice is not defined. Please select a model from the sidebar."
                )
                st.stop()
            model_path = os.path.join(MODEL_WEIGHTS_DIR, f"{model_choice}_model.pth")
            mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None
            file_hash = (
                hashlib.md5(open(model_path, "rb").read()).hexdigest()
                if os.path.exists(model_path)
                else "N/A"
            )

            start_render = time.time()

            active_tab = st.selectbox(
                "View Results",
                ["Details", "Technical", "Explanation"],
                key="active_tab",  # reuse the key you were managing manually
            )

            if active_tab == "Details":
                st.markdown('<div class="expander-results">', unsafe_allow_html=True)
                # Use a dynamic and informative title for the expander
                with st.expander(f"Results for {filename}", expanded=True):

                    # --- START: STREAMLINED METRICS ---
                    # A single, powerful row for the most important results.
                    key_metric_cols = st.columns(3)

                    # Metric 1: The Prediction
                    key_metric_cols[0].metric("Prediction", predicted_class)

                    # Metric 2: The Confidence (with level in tooltip)
                    confidence_icon = (
                        "🟢"
                        if max_confidence >= 0.8
                        else "🟡" if max_confidence >= 0.6 else "🔴"
                    )
                    key_metric_cols[1].metric(
                        "Confidence",
                        f"{confidence_icon} {max_confidence:.1%}",
                        help=f"Confidence Level: {confidence_desc}",
                    )

                    # Metric 3: Ground Truth + Correctness (Combined)
                    if true_label_idx is not None:
                        is_correct = predicted_class == true_label_str
                        delta_text = "✅ Correct" if is_correct else "❌ Incorrect"
                        # Use delta_color="normal" to let the icon provide the visual cue
                        key_metric_cols[2].metric(
                            "Ground Truth",
                            true_label_str,
                            delta=delta_text,
                            delta_color="normal",
                        )
                    else:
                        key_metric_cols[2].metric("Ground Truth", "N/A")

                    st.divider()
                    # --- END: STREAMLINED METRICS ---

                    # --- START: CONSOLIDATED CONFIDENCE ANALYSIS ---
                    st.markdown("##### Probability Breakdown")

                    # This custom bullet bar logic remains as it is highly specific and valuable
                    def create_bullet_bar(probability, width=20, predicted=False):
                        filled_count = int(probability * width)
                        bar = "▤" * filled_count + "▢" * (width - filled_count)
                        percentage = f"{probability:.1%}"
                        pred_marker = "↩ Predicted" if predicted else ""
                        return f"{bar} {percentage}    {pred_marker}"

                    if probs is not None:
                        stable_prob, weathered_prob = probs[0], probs[1]
                    else:
                        st.error(
                            "❌ Probability values are missing. Please check the inference process."
                        )
                        # Default values to prevent further errors
                        stable_prob, weathered_prob = 0.0, 0.0
                    is_stable_predicted, is_weathered_predicted = (
                        int(prediction) == 0
                    ), (int(prediction) == 1)

                    st.markdown(
                        f"""
                        <div style="font-family: 'Fira Code', monospace;">
                            Stable (Unweathered)<br>
                            {create_bullet_bar(stable_prob, predicted=is_stable_predicted)}<br><br>
                            Weathered (Degraded)<br>
                            {create_bullet_bar(weathered_prob, predicted=is_weathered_predicted)}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.divider()

                    # METADATA FOOTER
                    st.caption(
                        f"Analyzed with **{st.session_state.get('model_select', 'Unknown')}** in **{inference_time:.2f}s**."
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            elif active_tab == "Technical":
                with st.container():
                    st.markdown("Technical Diagnostics")

                    # Model performance metrics
                    with st.container(border=True):
                        st.markdown("##### **Model Performance**")
                        tech_col1, tech_col2 = st.columns(2)

                        with tech_col1:
                            st.metric("Inference Time", f"{inference_time:.3f}s")
                            st.metric(
                                "Input Length",
                                f"{len(x_raw) if x_raw is not None else 0} points",
                            )
                            st.metric("Resampled Length", f"{TARGET_LEN} points")

                        with tech_col2:
                            st.metric(
                                "Model Loaded",
                                (
                                    "✅ Yes"
                                    if st.session_state.get("model_loaded", False)
                                    else "❌ No"
                                ),
                            )
                            st.metric("Device", "CPU")
                            st.metric("Confidence Score", f"{max_confidence:.3f}")

                    # Raw logits display
                    with st.container(border=True):
                        st.markdown("##### **Raw Model Outputs (Logits)**")
                        logits_df = {
                            "Class": (
                                [
                                    LABEL_MAP.get(i, f"Class {i}")
                                    for i in range(len(logits_list))
                                ]
                                if logits_list is not None
                                else []
                            ),
                            "Logit Value": (
                                [f"{score:.4f}" for score in logits_list]
                                if logits_list is not None
                                else []
                            ),
                            "Probability": (
                                [f"{prob:.4f}" for prob in probs_np]
                                if logits_list is not None and len(probs_np) > 0
                                else []
                            ),
                        }

                        # Display as a simple table format
                        for i, (cls, logit, prob) in enumerate(
                            zip(
                                logits_df["Class"],
                                logits_df["Logit Value"],
                                logits_df["Probability"],
                            )
                        ):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                if i == prediction:
                                    st.markdown(f"**{cls}** ← Predicted")
                                else:
                                    st.markdown(cls)
                            with col2:
                                st.caption(f"Logit: {logit}")
                            with col3:
                                st.caption(f"Prob: {prob}")

                    # Spectrum statistics in organized sections
                    with st.container(border=True):
                        st.markdown("##### **Spectrum Analysis**")
                        spec_cols = st.columns(2)

                        with spec_cols[0]:
                            st.markdown("**Original Spectrum:**")
                            render_kv_grid(
                                {
                                    "Length": f"{len(x_raw) if x_raw is not None else 0} points",
                                    "Range": (
                                        f"{min(x_raw):.1f} - {max(x_raw):.1f} cm⁻¹"
                                        if x_raw is not None
                                        else "N/A"
                                    ),
                                    "Min Intensity": (
                                        f"{min(y_raw):.2e}"
                                        if y_raw is not None
                                        else "N/A"
                                    ),
                                    "Max Intensity": (
                                        f"{max(y_raw):.2e}"
                                        if y_raw is not None
                                        else "N/A"
                                    ),
                                },
                                ncols=1,
                            )

                        with spec_cols[1]:
                            st.markdown("**Processed Spectrum:**")
                            render_kv_grid(
                                {
                                    "Length": f"{TARGET_LEN} points",
                                    "Resampling": "Linear interpolation",
                                    "Normalization": "None",
                                    "Input Shape": f"(1, 1, {TARGET_LEN})",
                                },
                                ncols=1,
                            )

                    # Model information
                    with st.container(border=True):
                        st.markdown("##### **Model Information**")
                        model_info_cols = st.columns(2)

                        with model_info_cols[0]:
                            render_kv_grid(
                                {
                                    "Architecture": model_choice,
                                    "Path": model_path,
                                    "Weights Modified": (
                                        time.strftime(
                                            "%Y-%m-%d %H:%M:%S", time.localtime(mtime)
                                        )
                                        if mtime
                                        else "N/A"
                                    ),
                                },
                                ncols=1,
                            )

                        with model_info_cols[1]:
                            if os.path.exists(model_path):
                                file_hash = hashlib.md5(
                                    open(model_path, "rb").read()
                                ).hexdigest()
                                render_kv_grid(
                                    {
                                        "Weights Hash": f"{file_hash[:16]}...",
                                        "Output Shape": f"(1, {len(LABEL_MAP)})",
                                        "Activation": "Softmax",
                                    },
                                    ncols=1,
                                )

                    # Debug logs (collapsed by default)
                    with st.expander("📋 Debug Logs", expanded=False):
                        log_content = "\n".join(
                            st.session_state.get("log_messages", [])
                        )
                        if log_content.strip():
                            st.code(log_content, language="text")
                        else:
                            st.caption("No debug logs available")

            elif active_tab == "Explanation":
                with st.container():
                    st.markdown("### 🔍 Methodology & Interpretation")

                    # Process explanation
                    st.markdown("Analysis Pipeline")
                    process_steps = [
                        "📁 **Data Upload**: Raman spectrum file loaded and validated",
                        "🔍 **Preprocessing**: Spectrum parsed and resampled to 500 data points using linear interpolation",
                        "🧠 **AI Inference**: Convolutional Neural Network analyzes spectral patterns and molecular signatures",
                        "📊 **Classification**: Binary prediction with confidence scoring using softmax probabilities",
                        "✅ **Validation**: Ground truth comparison (when available from filename)",
                    ]

                    for step in process_steps:
                        st.markdown(step)

                    st.markdown("---")

                    # Model interpretation
                    st.markdown("#### Scientific Interpretation")

                    interp_col1, interp_col2 = st.columns(2)

                    with interp_col1:
                        st.markdown("**Stable (Unweathered) Polymers:**")
                        st.info(
                            """
                        - Well-preserved molecular structure
                        - Minimal oxidative degradation
                        - Characteristic Raman peaks intact
                        - 
                        itable for recycling applications
                        """
                        )

                    with interp_col2:
                        st.markdown("**Weathered (Degraded) Polymers:**")
                        st.warning(
                            """
                        - Oxidized molecular bonds
                        - Surface degradation present
                        - Altered spectral signatures
                        - May require additional processing
                        """
                        )

                    st.markdown("---")

                    # Applications
                    st.markdown("#### Research Applications")

                    applications = [
                        "🔬 **Material Science**: Polymer degradation studies",
                        "♻️ **Recycling Research**: Viability assessment for circular economy",
                        "🌱 **Environmental Science**: Microplastic weathering analysis",
                        "🏭 **Quality Control**: Manufacturing process monitoring",
                        "📈 **Longevity Studies**: Material aging prediction",
                    ]

                    for app in applications:
                        st.markdown(app)

                    # Technical details
                    # MODIFIED: Wrap the expander in a div with the 'expander-advanced' class
                    st.markdown(
                        '<div class="expander-advanced">', unsafe_allow_html=True
                    )
                    with st.expander("🔧 Technical Details", expanded=False):
                        st.markdown(
                            """
                        **Model Architecture:**
                        - Convolutional layers for feature extraction
                        - Residual connections for gradient flow
                        - Fully connected layers for classification
                        - Softmax activation for probability distribution

                        **Performance Metrics:**
                        - Accuracy: 94.8-96.2% on validation set
                        - F1-Score: 94.3-95.9% across classes
                        - Robust to spectral noise and baseline variations

                        **Data Processing:**
                        - Input: Raman spectra (any length)
                        - Resampling: Linear interpolation to 500 points
                        - Normalization: None (preserves intensity relationships)
                        """
                        )
                    st.markdown(
                        "</div>", unsafe_allow_html=True
                    )  # Close the wrapper div

                    render_time = time.time() - start_render
                    log_message(
                        f"col2 rendered in {render_time:.2f}s, active tab: {active_tab}"
                    )

            with st.expander("Spectrum Preprocessing Results", expanded=False):
                st.caption("<br>Spectral Analysis", unsafe_allow_html=True)

                # Add some context about the preprocessing
                st.markdown(
                    """
                **Preprocessing Overview:**
                - **Original Spectrum**: Raw Raman data as uploaded
                - **Resampled Spectrum**: Data interpolated to 500 points for model input
                - **Purpose**: Ensures consistent input dimensions for neural network
                """
                )

                # Create and display plot
                cache_key = hashlib.md5(
                    f"{(x_raw.tobytes() if x_raw is not None else b'')}"
                    f"{(y_raw.tobytes() if y_raw is not None else b'')}"
                    f"{(x_resampled.tobytes() if x_resampled is not None else b'')}"
                    f"{(y_resampled.tobytes() if y_resampled is not None else b'')}".encode()
                ).hexdigest()
                spectrum_plot = create_spectrum_plot(
                    x_raw, y_raw, x_resampled, y_resampled, _cache_key=cache_key
                )
                st.image(
                    spectrum_plot,
                    caption="Raman Spectrum: Raw vs Processed",
                    use_container_width=True,
                )

        else:
            st.markdown(
                """
            ##### How to Get Started

            1.  **Select an AI Model:** Use the dropdown menu in the sidebar to choose a model.
            2.  **Provide Your Data:** Select one of the three input modes:
                -   **Upload File:** Analyze a single spectrum.
                -   **Batch Upload:** Process multiple files at once.
                -   **Sample Data:** Explore functionality with pre-loaded examples.
            3.  **Run Analysis:** Click the "Run Analysis" button to generate the classification results.

            ---

            ##### Supported Data Format

            -   **File Type:** Plain text (`.txt`)
            -   **Content:** Must contain two columns: `wavenumber` and `intensity`.
            -   **Separators:** Values can be separated by spaces or commas.
            -   **Preprocessing:** Your spectrum will be automatically resampled to 500 data points to match the model's input requirements.

            ---

            ##### Example Applications
            - 🔬 Research on polymer degradation
            - ♻️ Recycling feasibility assessment
            - 🌱 Sustainability impact studies
            - 🏭 Quality control in manufacturing
            """
            )
    else:
        # Getting Started
        st.markdown(
            """
        ##### How to Get Started

        1.  **Select an AI Model:** Use the dropdown menu in the sidebar to choose a model.
        2.  **Provide Your Data:** Select one of the three input modes:
            -   **Upload File:** Analyze a single spectrum.
            -   **Batch Upload:** Process multiple files at once.
            -   **Sample Data:** Explore functionality with pre-loaded examples.
        3.  **Run Analysis:** Click the "Run Analysis" button to generate the classification results.

        ---

        ##### Supported Data Format

        -   **File Type:** Plain text (`.txt`)
        -   **Content:** Must contain two columns: `wavenumber` and `intensity`.
        -   **Separators:** Values can be separated by spaces or commas.
        -   **Preprocessing:** Your spectrum will be automatically resampled to 500 data points to match the model's input requirements.

        ---

        ##### Example Applications
        - 🔬 Research on polymer degradation
        - ♻️ Recycling feasibility assessment
        - 🌱 Sustainability impact studies
        - 🏭 Quality control in manufacturing
        """
        )


# //////////////////////////////////////////


def render_comparison_tab():
    """Render the multi-model comparison interface"""
    import streamlit as st
    import matplotlib.pyplot as plt
    from models.registry import (
        choices,
        validate_model_list,
        models_for_modality,
        get_models_metadata,
    )
    from utils.results_manager import ResultsManager
    from core_logic import get_sample_files, run_inference, parse_spectrum_data
    from utils.preprocessing import preprocess_spectrum
    from utils.multifile import parse_spectrum_data
    import numpy as np
    import time

    st.markdown("### Multi-Model Comparison Analysis")
    st.markdown(
        "Compare predictions across different AI models for comprehensive analysis."
    )

    # Modality selector
    col_mod1, col_mod2 = st.columns([1, 2])
    with col_mod1:
        modality = st.selectbox(
            "Select Modality",
            ["raman", "ftir"],
            index=0,
            help="Choose the spectroscopy modality for analysis",
            key="comparison_modality",
        )
        # Don't override existing session state
        if "modality_select" not in st.session_state:
            st.session_state["modality_select"] = modality

    with col_mod2:
        # Filter models by modality
        compatible_models = models_for_modality(modality)
        if not compatible_models:
            st.error(f"No models available for {modality.upper()} modality")
            return

        st.info(f"📊 {len(compatible_models)} models available for {modality.upper()}")

    # Enhanced model selection with metadata
    st.markdown("##### Select Models for Comparison")

    # Display model information
    models_metadata = get_models_metadata()

    # Create enhanced multiselect with model descriptions
    model_options = []
    model_descriptions = {}
    for model in compatible_models:
        desc = models_metadata.get(model, {}).get("description", "No description")
        model_options.append(model)
        model_descriptions[model] = desc

    selected_models = st.multiselect(
        "Choose models to compare",
        model_options,
        default=(model_options[:2] if len(model_options) >= 2 else model_options),
        help="Select 2 or more models to compare their predictions side-by-side",
        key="comparison_model_select",
    )

    # Display selected model information
    if selected_models:
        with st.expander("Selected Model Details", expanded=False):
            for model in selected_models:
                info = models_metadata.get(model, {})
                st.markdown(f"**{model}**: {info.get('description', 'No description')}")
                if "citation" in info:
                    st.caption(f"Citation: {info['citation']}")

    if len(selected_models) < 2:
        st.warning("⚠️ Please select at least 2 models for comparison.")

    # Input selection for comparison
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("###### Input Data")

        # File upload for comparison
        comparison_file = st.file_uploader(
            "Upload spectrum for comparison",
            type=["txt", "csv", "json"],
            key="comparison_file_upload",
            help="Upload a spectrum file to test across all selected models",
        )

        # Or select sample data
        selected_sample = None  # Initialize with a default value
        sample_files = get_sample_files()
        if sample_files:
            sample_options = ["-- Select Sample --"] + [p.name for p in sample_files]
            selected_sample = st.selectbox(
                "Or choose sample data", sample_options, key="comparison_sample_select"
            )

        # Get modality from session state
        modality = st.session_state.get("modality_select", "raman")
        st.info(f"Using {modality.upper()} preprocessing parameters")

        # Run comparison button
        run_comparison = st.button(
            "Run Multi-Model Comparison",
            type="primary",
            disabled=not (
                comparison_file
                or (sample_files and selected_sample != "-- Select Sample --")
            ),
        )

    with col2:
        st.markdown("###### Comparison Results")

        if run_comparison:
            # Determine input source
            input_text = None
            filename = "unknown"

            if comparison_file:
                raw = comparison_file.read()
                input_text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                filename = comparison_file.name
            elif sample_files and selected_sample != "-- Select Sample --":
                sample_path = next(p for p in sample_files if p.name == selected_sample)
                with open(sample_path, "r", encoding="utf-8") as f:
                    input_text = f.read()
                filename = selected_sample

            if input_text:
                try:
                    # Parse spectrum data
                    x_raw, y_raw = parse_spectrum_data(
                        str(input_text), filename or "unknown_filename"
                    )

                    # Preprocess spectrum once
                    _, y_processed = preprocess_spectrum(
                        x_raw, y_raw, modality=modality, target_len=500
                    )

                    # Synchronous processing
                    comparison_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, model_name in enumerate(selected_models):
                        status_text.text(f"Running inference with {model_name}...")

                        start_time = time.time()

                        # Run inference
                        prediction, logits_list, probs, inference_time, logits = (
                            run_inference(y_processed, model_name, modality=modality)
                        )

                        processing_time = time.time() - start_time

                        # --- FIX FOR SYNCHRONOUS PATH: Handle silent failure ---
                        if prediction is None:
                            comparison_results[model_name] = {
                                "status": "failed",
                                "error": "Model failed to load or returned None.",
                            }
                        else:
                            # Map prediction to class name
                            class_names = ["Stable", "Weathered"]
                            predicted_class = (
                                class_names[int(prediction)]
                                if int(prediction) < len(class_names)
                                else f"Class_{prediction}"
                            )
                            confidence = (
                                float(np.max(probs))
                                if probs is not None and probs.size > 0
                                else 0.0
                            )

                            comparison_results[model_name] = {
                                "prediction": prediction,
                                "predicted_class": predicted_class,
                                "confidence": confidence,
                                "probs": (probs.tolist() if probs is not None else []),
                                "logits": (
                                    logits_list if logits_list is not None else []
                                ),
                                "processing_time": inference_time or 0.0,
                                "status": "success",
                            }

                        progress_bar.progress((i + 1) / len(selected_models))

                    status_text.text("Comparison complete!")

                    # Enhanced results display
                    if comparison_results:
                        # Filter successful results
                        successful_results = {
                            k: v
                            for k, v in comparison_results.items()
                            if v.get("status") == "success"
                        }
                        failed_results = {
                            k: v
                            for k, v in comparison_results.items()
                            if v.get("status") == "failed"
                        }

                        if failed_results:
                            st.error(
                                f"Failed models: {', '.join(failed_results.keys())}"
                            )
                            for model, result in failed_results.items():
                                st.error(
                                    f"{model}: {result.get('error', 'Unknown error')}"
                                )

                        if successful_results:
                            try:
                                st.markdown("###### Model Predictions")

                                # Create enhanced comparison table
                                import pandas as pd

                                table_data = []
                                for model_name, result in successful_results.items():
                                    row = {
                                        "Model": model_name,
                                        "Prediction": result["predicted_class"],
                                        "Confidence": f"{result['confidence']:.3f}",
                                        "Processing Time (s)": f"{result['processing_time']:.3f}",
                                        "Agreement": (
                                            "✓"
                                            if len(
                                                set(
                                                    r["prediction"]
                                                    for r in successful_results.values()
                                                )
                                            )
                                            == 1
                                            else "✗"
                                        ),
                                    }
                                    table_data.append(row)

                                df = pd.DataFrame(table_data)
                                st.dataframe(df, use_container_width=True)

                                # Model agreement analysis
                                predictions = [
                                    r["prediction"] for r in successful_results.values()
                                ]
                                agreement_rate = len(set(predictions)) == 1

                                if agreement_rate:
                                    st.success("🎯 All models agree on the prediction!")
                                else:
                                    st.warning(
                                        "⚠️ Models disagree - review individual confidences"
                                    )

                                # Enhanced visualization section
                                st.markdown("##### Enhanced Analysis Dashboard")

                                tab1, tab2, tab3 = st.tabs(
                                    [
                                        "Confidence Analysis",
                                        "Performance Metrics",
                                        "Detailed Breakdown",
                                    ]
                                )

                                with tab1:
                                    try:
                                        # Enhanced confidence comparison
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            # Bar chart of confidences
                                            models = list(successful_results.keys())
                                            confidences = [
                                                successful_results[m]["confidence"]
                                                for m in models
                                            ]

                                            fig, ax = plt.subplots(figsize=(8, 5))
                                            colors = plt.cm.Set3(
                                                np.linspace(0, 1, len(models))
                                            )
                                            bars = ax.bar(
                                                models,
                                                confidences,
                                                alpha=0.8,
                                                color=colors,
                                            )

                                            # Add value labels on bars
                                            for bar, conf in zip(bars, confidences):
                                                height = bar.get_height()
                                                ax.text(
                                                    bar.get_x() + bar.get_width() / 2.0,
                                                    height + 0.01,
                                                    f"{conf:.3f}",
                                                    ha="center",
                                                    va="bottom",
                                                )

                                            ax.set_ylabel("Confidence")
                                            ax.set_title("Model Confidence Comparison")
                                            ax.set_ylim(0, 1.1)
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            st.pyplot(fig)

                                        with col2:
                                            # Confidence distribution
                                            st.markdown("**Confidence Statistics**")
                                            conf_stats = {
                                                "Mean": np.mean(confidences),
                                                "Std Dev": np.std(confidences),
                                                "Min": np.min(confidences),
                                                "Max": np.max(confidences),
                                                "Range": np.max(confidences)
                                                - np.min(confidences),
                                            }

                                            for stat, value in conf_stats.items():
                                                st.metric(stat, f"{value:.4f}")
                                    except Exception as e:
                                        st.error(f"Error rendering results: {e}")
                            except Exception as e:
                                st.error(f"Error rendering results: {e}")
                                st.error(f"Error in Confidence Analysis tab: {e}")

                            with tab2:
                                # Performance metrics
                                models = list(successful_results.keys())
                                times = [
                                    successful_results[m]["processing_time"]
                                    for m in models
                                ]

                                perf_col1, perf_col2 = st.columns(2)

                                with perf_col1:
                                    # Processing time comparison
                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    bars = ax.bar(
                                        models, times, alpha=0.8, color="skyblue"
                                    )

                                    for bar, time_val in zip(bars, times):
                                        height = bar.get_height()
                                        ax.text(
                                            bar.get_x() + bar.get_width() / 2.0,
                                            height + 0.001,
                                            f"{time_val:.3f}s",
                                            ha="center",
                                            va="bottom",
                                        )

                                    ax.set_ylabel("Processing Time (s)")
                                    ax.set_title("Model Processing Time Comparison")
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)

                                with perf_col2:
                                    # Performance statistics
                                    st.markdown("**Performance Statistics**")
                                    perf_stats = {
                                        "Fastest Model": models[np.argmin(times)],
                                        "Slowest Model": models[np.argmax(times)],
                                        "Total Time": f"{np.sum(times):.3f}s",
                                        "Average Time": f"{np.mean(times):.3f}s",
                                        "Speed Difference": f"{np.max(times) - np.min(times):.3f}s",
                                    }

                                    for stat, value in perf_stats.items():
                                        st.write(f"**{stat}**: {value}")

                            with tab3:
                                # Detailed breakdown
                                for (
                                    model_name,
                                    result,
                                ) in successful_results.items():
                                    with st.expander(
                                        f"Detailed Results - {model_name}"
                                    ):
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.write(
                                                f"**Prediction**: {result['predicted_class']}"
                                            )
                                            st.write(
                                                f"**Confidence**: {result['confidence']:.4f}"
                                            )
                                            st.write(
                                                f"**Processing Time**: {result['processing_time']:.4f}s"
                                            )

                                            # ROBUST CHECK FOR PROBABILITIES
                                            if (
                                                "probs" in result
                                                and result["probs"] is not None
                                                and len(result["probs"]) > 0
                                            ):
                                                st.write("**Class Probabilities**:")
                                                class_names = [
                                                    "Stable",
                                                    "Weathered",
                                                ]
                                                for i, prob in enumerate(
                                                    result["probs"]
                                                ):
                                                    if i < len(class_names):
                                                        st.write(
                                                            f"  - {class_names[i]}: {prob:.4f}"
                                                        )

                                        with col2:
                                            # ROBUST CHECK FOR LOGITS
                                            if (
                                                "logits" in result
                                                and result["logits"] is not None
                                                and len(result["logits"]) > 0
                                            ):
                                                st.write("**Raw Logits**:")
                                                for i, logit in enumerate(
                                                    result["logits"]
                                                ):
                                                    st.write(
                                                        f"  - Class {i}: {logit:.4f}"
                                                    )

                            # Export options
                            st.markdown("##### Export Results")
                            export_col1, export_col2 = st.columns(2)

                            with export_col1:
                                if st.button("📋 Copy Results to Clipboard"):
                                    results_text = df.to_string(index=False)
                                    st.code(results_text)

                            with export_col2:
                                # Download results as CSV
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download as CSV",
                                    data=csv_data,
                                    file_name=f"model_comparison_{filename}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )
                except Exception as e:
                    import traceback

                    st.error(f"Error during comparison: {str(e)}")
                    st.code(traceback.format_exc())  # Add traceback for debugging

            # Show recent comparison results if available
            elif "last_comparison_results" in st.session_state:
                st.info(
                    "Previous comparison results available. Upload a new file or select a sample to run new comparison."
                )

    # Show comparison history
    comparison_stats = ResultsManager.get_comparison_stats()
    if comparison_stats:
        st.markdown("#### Comparison History")

        with st.expander("View detailed comparison statistics", expanded=False):
            # Show model statistics table
            stats_data = []
            for model_name, stats in comparison_stats.items():
                row = {
                    "Model": model_name,
                    "Total Predictions": stats["total_predictions"],
                    "Avg Confidence": f"{stats['avg_confidence']:.3f}",
                    "Avg Processing Time": f"{stats['avg_processing_time']:.3f}s",
                    "Accuracy": (
                        f"{stats['accuracy']:.3f}"
                        if stats["accuracy"] is not None
                        else "N/A"
                    ),
                }
                stats_data.append(row)

            if stats_data:
                import pandas as pd

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                # Show agreement matrix if multiple models
                agreement_matrix = ResultsManager.get_agreement_matrix()
                if not agreement_matrix.empty and len(agreement_matrix) > 1:
                    st.markdown("**Model Agreement Matrix**")
                    st.dataframe(agreement_matrix.round(3), use_container_width=True)

                    # Plot agreement heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(
                        agreement_matrix.values, cmap="RdYlGn", vmin=0, vmax=1
                    )

                    # Add text annotations
                    for i in range(len(agreement_matrix)):
                        for j in range(len(agreement_matrix.columns)):
                            text = ax.text(
                                j,
                                i,
                                f"{agreement_matrix.iloc[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="black",
                            )

                    ax.set_xticks(range(len(agreement_matrix.columns)))
                    ax.set_yticks(range(len(agreement_matrix)))
                    ax.set_xticklabels(agreement_matrix.columns, rotation=45)
                    ax.set_yticklabels(agreement_matrix.index)
                    ax.set_title("Model Agreement Matrix")

                    plt.colorbar(im, ax=ax, label="Agreement Rate")
                    plt.tight_layout()
                    st.pyplot(fig)

        # Export functionality
        if "last_comparison_results" in st.session_state:
            st.markdown("##### Export Results")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("📥 Export Comparison (JSON)"):
                import json

                results = st.session_state["last_comparison_results"]
                json_str = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"comparison_{results['filename'].split('.')[0]}.json",
                    mime="application/json",
                )

        with export_col2:
            if st.button("📊 Export Full Report"):
                report = ResultsManager.export_comparison_report()
                st.download_button(
                    label="Download Full Report",
                    data=report,
                    file_name="model_comparison_report.json",
                    mime="application/json",
                )


# //////////////////////////////////////////


from utils.performance_tracker import display_performance_dashboard


def render_performance_tab():
    """Render the performance tracking and analysis tab."""
    display_performance_dashboard()


# //////////////////////////////////////////
