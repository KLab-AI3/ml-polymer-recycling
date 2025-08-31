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
from config import MODEL_CONFIG, TARGET_LEN, LABEL_MAP
from modules.callbacks import (
    on_model_change,
    on_input_mode_change,
    on_sample_change,
    reset_ephemeral_state,
    log_message,
    clear_batch_results,
)
from core_logic import (
    get_sample_files,
    load_model,
    run_inference,
    parse_spectrum_data,
    label_file,
)
from modules.callbacks import reset_results
from utils.results_manager import ResultsManager
from utils.confidence import calculate_softmax_confidence
from utils.multifile import process_multiple_files, display_batch_results
from utils.preprocessing import resample_spectrum


def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def create_spectrum_plot(x_raw, y_raw, x_resampled, y_resampled, _cache_key=None):
    """Create spectrum visualization plot"""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

    # == Raw spectrum ==
    ax[0].plot(x_raw, y_raw, label="Raw", color="dimgray", linewidth=1)
    ax[0].set_title("Raw Input Spectrum")
    ax[0].set_xlabel("Wavenumber (cm⁻¹)")
    ax[0].set_ylabel("Intensity")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # == Resampled spectrum ==
    ax[1].plot(
        x_resampled, y_resampled, label="Resampled", color="steelblue", linewidth=1
    )
    ax[1].set_title(f"Resampled ({len(y_resampled)} points)")
    ax[1].set_xlabel("Wavenumber (cm⁻¹)")
    ax[1].set_ylabel("Intensity")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    # == Convert to image ==
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)  # Prevent memory leaks

    return Image.open(buf)


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


def render_kv_grid(d: dict = {}, ncols: int = 2):
    if d is None:
        d = {}
    if not d:
        return
    items = list(d.items())
    cols = st.columns(ncols)
    for i, (k, v) in enumerate(items):
        with cols[i % ncols]:
            st.caption(f"**{k}:** {v}")


def render_model_meta(model_choice: str):
    info = MODEL_CONFIG.get(model_choice, {})
    emoji = info.get("emoji", "")
    desc = info.get("description", "").strip()
    acc = info.get("accuracy", "-")
    f1 = info.get("f1", "-")

    st.caption(f"{emoji} **Model Snapshot** - {model_choice}")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Accuracy", acc)
    with cols[1]:
        st.metric("F1 Score", f1)
    if desc:
        st.caption(desc)


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


def render_sidebar():
    with st.sidebar:
        # Header
        st.header("AI-Driven Polymer Classification")
        st.caption(
            "Predict polymer degradation (Stable vs Weathered) from Raman spectra using validated CNN models. — v0.1"
        )
        model_labels = [
            f"{MODEL_CONFIG[name]['emoji']} {name}" for name in MODEL_CONFIG.keys()
        ]
        selected_label = st.selectbox(
            "Choose AI Model",
            model_labels,
            key="model_select",
            on_change=on_model_change,
        )
        model_choice = selected_label.split(" ", 1)[1]

        # ===Compact metadata directly under dropdown===
        render_model_meta(model_choice)

        # ===Collapsed info to reduce clutter===
        with st.expander("About This App", icon=":material/info:", expanded=False):
            st.markdown(
                """
            **AI-Driven Polymer Aging Prediction and Classification**

            **Purpose**: Classify polymer degradation using AI<br>
            **Input**: Raman spectroscopy .txt files<br>
            **Models**: CNN architectures for binary classification<br>
            **Next**: More trained CNNs in evaluation pipeline<br>


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


# col1 goes here

# In modules/ui_components.py


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
    # ... (The if/elif/else block for Upload, Batch, and Sample modes remains exactly the same) ...
    # ==Upload tab==
    if mode == "Upload File":
        upload_key = st.session_state["current_upload_key"]
        up = st.file_uploader(
            "Upload Raman spectrum (.txt)",
            type="txt",
            help="Upload a text file with wavenumber and intensity columns",
            key=upload_key,  # ← versioned key
        )

        # ==Process change immediately (no on_change; simpler & reliable)==
        if up is not None:
            raw = up.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            # == only reparse if its a different file|source ==
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

    # ==Batch Upload tab==
    elif mode == "Batch Upload":
        st.session_state["batch_mode"] = True
        # --- START: BUG 1 & 3 FIX ---
        # Use a versioned key to ensure the file uploader resets properly.
        batch_upload_key = f"batch_upload_{st.session_state['uploader_version']}"
        uploaded_files = st.file_uploader(
            "Upload multiple Raman spectrum files (.txt)",
            type="txt",
            accept_multiple_files=True,
            help="Upload one or more text files with wavenumber and intensity columns.",
            key=batch_upload_key,
        )
        # --- END: BUG 1 & 3 FIX ---

        if uploaded_files:
            # --- START: Bug 1 Fix ---
            # Use a dictionary to keep only unique files based on name and size
            unique_files = {(file.name, file.size): file for file in uploaded_files}
            unique_file_list = list(unique_files.values())

            num_uploaded = len(uploaded_files)
            num_unique = len(unique_file_list)

            # Optionally, inform the user that duplicates were removed
            if num_uploaded > num_unique:
                st.info(
                    f"ℹ️ {num_uploaded - num_unique} duplicate file(s) were removed."
                )

            # Use the unique list
            st.session_state["batch_files"] = unique_file_list
            st.session_state["status_message"] = (
                f"{num_unique} ready for batch analysis"
            )
            st.session_state["status_type"] = "success"
            # --- END: Bug 1 Fix ---
        else:
            st.session_state["batch_files"] = []
            # This check prevents resetting the status if files are already staged
            if not st.session_state.get("batch_files"):
                st.session_state["status_message"] = (
                    "No files selected for batch processing"
                )
                st.session_state["status_type"] = "info"

    # ==Sample tab==
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

    # --- DE-NESTED LOGIC STARTS HERE ---
    # This code now runs on EVERY execution, guaranteeing the buttons will appear.

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
                    load_model_func=load_model,
                    run_inference_func=run_inference,
                    label_file_func=label_file,
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


# col2 goes here


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
            # ===Run inference===
            if y_resampled is None:
                raise ValueError(
                    "y_resampled is None. Ensure spectrum data is properly resampled before proceeding."
                )
            cache_key = hashlib.md5(
                f"{y_resampled.tobytes()}{st.session_state.get('model_select', 'Unknown').split(' ', 1)[1]}".encode()
            ).hexdigest()
            prediction, logits_list, probs, inference_time, logits = run_inference(
                y_resampled,
                (
                    st.session_state.get("model_select", "").split(" ", 1)[1]
                    if "model_select" in st.session_state
                    else None
                ),
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

            # ===Get ground truth===
            true_label_idx = label_file(filename)
            true_label_str = (
                LABEL_MAP.get(true_label_idx, "Unknown")
                if true_label_idx is not None
                else "Unknown"
            )
            # ===Get prediction===
            predicted_class = LABEL_MAP.get(int(prediction), f"Class {int(prediction)}")

            # Enhanced confidence calculation
            if logits is not None:
                # Use new softmax-based confidence
                probs_np, max_confidence, confidence_level, confidence_emoji = (
                    calculate_softmax_confidence(logits)
                )
                confidence_desc = confidence_level
            else:
                # Fallback to legace method
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

            # ===Precompute Stats===
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
            model_path = MODEL_CONFIG[model_choice]["path"]
            mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None
            file_hash = (
                hashlib.md5(open(model_path, "rb").read()).hexdigest()
                if os.path.exists(model_path)
                else "N/A"
            )
            # Removed unused variable 'input_tensor'

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
                    # --- END: CONSOLIDATED CONFIDENCE ANALYSIS ---

                    st.divider()

                    # --- START: CLEAN METADATA FOOTER ---
                    # Secondary info is now a clean, single-line caption
                    st.caption(
                        f"Analyzed with **{st.session_state.get('model_select', 'Unknown')}** in **{inference_time:.2f}s**."
                    )
                    # --- END: CLEAN METADATA FOOTER ---

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
                                    "Path": MODEL_CONFIG[model_choice]["path"],
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
        # ===Getting Started===
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
