import os
import torch
import streamlit as st
import hashlib
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import uuid
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
from core_logic import get_sample_files, load_model, run_inference, label_file
from utils.results_manager import ResultsManager
from utils.multifile import process_multiple_files, parse_spectrum_data
from utils.preprocessing import (
    validate_spectrum_modality,
    preprocess_spectrum,
)
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
            "Analyze and classify polymer degradation with a suite of explainable AI models for Raman & FTIR spectroscopy. — v0.02"
        )

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
            width="stretch",
        )
        model_choice = selected_label.split(" ", 1)[1]

        # Compact metadata directly under dropdown
        render_model_meta(model_choice)

        # Collapsed info to reduce clutter
        with st.expander("About This App", icon=":material/info:", expanded=False):
            st.markdown(
                """
            **AI-Driven Polymer Analysis Platform**

            **Purpose**: Classify, analyze, and understand polymer degradation using explainable AI.

            **Input**: Raman & FTIR spectra in `.txt`, `.csv`, or `.json` formats.

            **Features**:
            - Single & Batch Spectrum Analysis
            - Multi-Model Performance Comparison
            - Interactive Model Training Hub
            - Explainable AI (XAI) with feature importance
            - Modality-Aware Preprocessing

            **Links**  
            [HF Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)  
            [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)

            **Contributors**
            - Dr. Sanmukh Kuppannagari (Mentor)
            - Dr. Metin Karailyan (Mentor)
            - Jaser Hasan (Author)


            **Citation (Baseline Model)**
            Neo et al., 2023, *Resour. Conserv. Recycl.*, 188, 106718.
            https://doi.org/10.1016/j.resconrec.2022.106718
            """
            )


def render_input_column():
    st.markdown("##### Data Input")

    # Modality Selection - Moved from sidebar to be the primary context setter
    st.markdown("###### 1. Choose Spectroscopy Modality")
    modality = st.selectbox(
        "Choose Modality",
        ["raman", "ftir"],
        index=0,
        key="modality_select",
        format_func=lambda x: f"{'Raman' if x == 'raman' else 'FTIR'}",
        help="Select the type of spectroscopy data you are analyzing. This choice affects preprocessing steps.",
        width=325,
    )

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
                width=350,
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

    # Handle form submission
    if submitted and inference_ready:
        st.session_state["run_uuid"] = uuid.uuid4().hex[:8]
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
                x_raw, y_raw = parse_spectrum_data(
                    st.session_state["input_text"],
                    filename=st.session_state.get("filename", "unknown"),
                )

                # QC Summary
                st.session_state["qc_summary"] = {
                    "n_points": len(x_raw),
                    "x_min": f"{np.min(x_raw):.1f}",
                    "x_max": f"{np.max(x_raw):.1f}",
                    "monotonic_x": bool(np.all(np.diff(x_raw) > 0)),
                    "nan_free": not (
                        np.any(np.isnan(x_raw)) or np.any(np.isnan(y_raw))
                    ),
                    "variance_proxy": f"{np.var(y_raw):.2e}",
                }

                # Preprocessing parameters
                preproc_params = {
                    "target_len": TARGET_LEN,
                    "modality": st.session_state.get("modality_select", "raman"),
                    "do_baseline": True,
                    "do_smooth": True,
                    "do_normalize": True,
                }

                # Validate that spectrum matches selected modality
                selected_modality = st.session_state.get("modality_select", "raman")
                is_valid, issues = validate_spectrum_modality(
                    x_raw, y_raw, selected_modality
                )

                if not is_valid:
                    st.warning("⚠️ **Spectrum-Modality Mismatch Detected**")
                    for issue in issues:
                        st.warning(f"• {issue}")

                    # Ask user if they want to continue
                    st.info(
                        "💡 **Suggestion**: Check if the correct modality is selected in the sidebar, or verify your data file."
                    )

                    if st.button("⚠️ Continue Anyway", key="continue_with_mismatch"):
                        st.warning(
                            "Proceeding with potentially mismatched data. Results may be unreliable."
                        )
                    else:
                        st.stop()  # Stop processing until user confirms

                x_resampled, y_resampled = preprocess_spectrum(
                    x_raw, y_raw, **preproc_params
                )
                st.session_state["preproc_params"] = preproc_params
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
                cache_key=cache_key,
            )
            if prediction is None:
                st.error(
                    "❌ Inference failed: Model not loaded. Please check that weights are available."
                )
                st.stop()  # prevents the rest of the code in this block from executing

            # Store results in session state for the Details tab
            st.session_state["prediction"] = prediction
            st.session_state["probs"] = probs
            st.session_state["inference_time"] = inference_time

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
            model_info = get_model_info(model_choice)
            st.session_state["model_info"] = model_info
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
                # Use a dynamic and informative title for the expander
                with st.expander(f"Results for {filename}", expanded=True):

                    # ...inside the Details tab, after metrics...

                    import json, math, uuid

                    st.subheader("Probability Breakdown")

                    def _entropy(ps):
                        ps = [max(min(float(p), 1.0), 1e-12) for p in ps]
                        return -sum(p * math.log(p) for p in ps)

                    def _badge(text, kind="info"):
                        # This function now relies on CSS classes defined in style.css
                        # for better separation of concerns and maintainability.
                        st.markdown(
                            f"<span class='badge badge-{kind}'>{text}</span>",
                            unsafe_allow_html=True,
                        )

                    def _render_prob_row(label: str, prob: float, is_pred: bool):
                        c1, c2, c3 = st.columns([2, 7, 3])
                        with c1:
                            st.write(label)
                        with c2:
                            st.progress(min(max(prob, 0.0), 1.0))
                        with c3:
                            suffix = "  \u2190 Predicted" if is_pred else ""
                            st.write(f"{prob:.1%}{suffix}")

                    probs = st.session_state.get("probs")
                    prediction = st.session_state.get("prediction")
                    inference_time = float(st.session_state.get("inference_time", 0.0))

                    if probs is None or len(probs) != 2:
                        st.error(
                            "❌ Probability values are missing or invalid. Check the inference process."
                        )
                        stable_prob, weathered_prob = 0.0, 0.0
                    else:
                        stable_prob, weathered_prob = float(probs[0]), float(probs[1])

                    is_stable_predicted = (
                        (int(prediction) == 0)
                        if prediction is not None
                        else (stable_prob >= weathered_prob)
                    )
                    is_weathered_predicted = (
                        (int(prediction) == 1)
                        if prediction is not None
                        else (weathered_prob > stable_prob)
                    )

                    margin = abs(stable_prob - weathered_prob)
                    entropy = _entropy([stable_prob, weathered_prob])
                    thresh = float(st.session_state.get("decision_threshold", 0.5))
                    cal = st.session_state.get("calibration", {}) or {}
                    cal_enabled = bool(cal.get("enabled", False))
                    ece = cal.get("ece", None)

                    ABSTAIN_TAU = 0.10
                    OOD_MAX_SOFT = 0.60
                    max_softmax = max(stable_prob, weathered_prob)

                    colA, colB, colC, colD = st.columns([3, 3, 3, 3])
                    with colA:
                        st.metric(
                            "Predicted",
                            "Stable" if is_stable_predicted else "Weathered",
                        )
                    with colB:
                        st.metric("Decision Margin", f"{margin:.2f}")
                    with colC:
                        st.metric("Entropy", f"{entropy:.3f}")
                    with colD:
                        st.metric("Threshold", f"{thresh:.2f}")

                    row = st.columns([3, 3, 6])
                    with row[0]:
                        if margin < ABSTAIN_TAU:
                            _badge("Low margin — consider abstain / re-measure", "warn")
                    with row[1]:
                        if max_softmax < OOD_MAX_SOFT:
                            _badge("Low confidence — possible OOD", "bad")
                    with row[2]:
                        if cal_enabled:
                            _badge(
                                (
                                    f"Calibrated (ECE={ece:.2%})"
                                    if isinstance(ece, (int, float))
                                    else "Calibrated"
                                ),
                                "good",
                            )
                        else:
                            _badge(
                                "Uncalibrated — probabilities may be miscalibrated",
                                "info",
                            )

                    st.write("")

                    _render_prob_row(
                        "Stable (Unweathered)", stable_prob, is_stable_predicted
                    )
                    _render_prob_row(
                        "Weathered (Degraded)", weathered_prob, is_weathered_predicted
                    )

                    qc = st.session_state.get("qc_summary", {}) or {}
                    pp = st.session_state.get("preproc_params", {}) or {}
                    model_info = st.session_state.get("model_info", {}) or {}
                    run_info = {
                        "model": model_choice,
                        "inference_time_s": inference_time,
                        "run_uuid": st.session_state.get("run_uuid", ""),
                        "app_commit": st.session_state.get("app_commit", "unknown"),
                    }

                    with st.expander("Input QC"):
                        st.write(
                            {
                                "n_points": qc.get("n_points", "N/A"),
                                "x_min_cm-1": qc.get("x_min", "N/A"),
                                "x_max_cm-1": qc.get("x_max", "N/A"),
                                "monotonic_x": qc.get("monotonic_x", "N/A"),
                                "nan_free": qc.get("nan_free", "N/A"),
                                "variance_proxy": qc.get("variance_proxy", "N/A"),
                            }
                        )

                    with st.expander("Preprocessing (applied)"):
                        st.write(pp)

                    with st.expander("Model & Run"):
                        st.write(
                            {
                                "model_name": model_info.get("name", model_choice),
                                "version": model_info.get("version", "n/a"),
                                "weights_mtime": model_info.get("weights_mtime", "n/a"),
                                "cv_accuracy": model_info.get("cv_accuracy", "n/a"),
                                "class_priors": model_info.get("class_priors", "n/a"),
                                **run_info,
                            }
                        )

                    export_payload = {
                        "prediction": "stable" if is_stable_predicted else "weathered",
                        "probs": {"stable": stable_prob, "weathered": weathered_prob},
                        "margin": margin,
                        "entropy": entropy,
                        "threshold": thresh,
                        "calibration": {
                            "enabled": cal_enabled,
                            "ece": ece,
                            "method": cal.get("method"),
                            "T": cal.get("T"),
                        },
                        "qc": qc,
                        "preprocessing": pp,
                        "model_info": model_info,
                        "run_info": run_info,
                    }
                    fname = f"result_{run_info['run_uuid'] or uuid.uuid4().hex}.json"
                    st.download_button(
                        "Download result JSON",
                        json.dumps(export_payload, indent=2),
                        file_name=fname,
                        mime="application/json",
                    )

                    # METADATA FOOTER
                    st.caption(
                        f"Analyzed with **{run_info['model']}** in **{inference_time:.2f}s**."
                    )

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

                    st.markdown("#### Analysis Pipeline")
                    process_steps = [
                        "📁 **Data Input**: Upload a spectrum file (`.txt`, `.csv`, `.json`) and select the spectroscopy modality (Raman or FTIR).",
                        "🔬 **Modality-Aware Preprocessing**: The spectrum is automatically processed with steps tailored to the selected modality, including baseline correction, smoothing, normalization, and resampling to a fixed length (500 points).",
                        "🧠 **AI Inference**: A selected model from the registry (e.g., `Figure2CNN`, `ResNet`, `EnhancedCNN`) analyzes the processed spectrum to identify key patterns.",
                        "📊 **Classification & Confidence**: The model outputs a binary prediction (Stable vs. Weathered) along with a detailed probability breakdown and confidence score.",
                        "✅ **Validation & Explainability**: Results are presented with technical diagnostics, and where possible, explainability metrics to interpret the model's decision.",
                    ]

                    for step in process_steps:
                        st.markdown(f"- {step}")

                    st.markdown("---")

                    # Model interpretation
                    st.markdown("#### Scientific Interpretation")

                    interp_col1, interp_col2 = st.columns(2)

                    with interp_col1:
                        st.markdown("**Stable (Unweathered) Polymers:**")
                        st.info(
                            """
                        - **Spectral Signature**: Sharp, well-defined peaks corresponding to the polymer's known vibrational modes.
                        - **Chemical State**: Minimal evidence of oxidation or chain scission. The polymer backbone is intact.
                        - **Model Behavior**: The AI identifies a strong match with the spectral fingerprint of a non-degraded reference material.
                        - **Implication**: Suitable for high-quality recycling applications.
                        """
                        )

                    with interp_col2:
                        st.markdown("**Weathered (Degraded) Polymers:**")
                        st.warning(
                            """
                        - **Spectral Signature**: Peak broadening, baseline shifts, and the emergence of new peaks (e.g., carbonyl group at ~1715 cm⁻¹).
                        - **Chemical State**: Evidence of oxidation, hydrolysis, or other degradation pathways.
                        - **Model Behavior**: The AI detects features that deviate significantly from the reference fingerprint, indicating chemical alteration.
                        - **Implication**: May require more intensive processing or be unsuitable for certain recycling streams.
                        """
                        )

                    st.markdown("---")

                    # Applications
                    st.markdown("#### Research & Industrial Applications")

                    applications = [
                        " **Material Science**: Quantify degradation rates and study aging mechanisms in novel polymers.",
                        "♻️ **Circular Economy**: Automate the quality control and sorting of post-consumer plastics for recycling.",
                        "🌱 **Environmental Science**: Analyze the weathering of microplastics in various environmental conditions.",
                        "🏭 **Industrial QC**: Monitor material integrity and predict product lifetime in manufacturing processes.",
                        "🤖 **AI-Driven Discovery**: Use explainability features to generate new hypotheses about material behavior.",
                    ]

                    for app in applications:
                        st.markdown(f"- {app}")

                    # Technical details
                    with st.expander(
                        "🔧 Technical Architecture Details", expanded=False
                    ):
                        st.markdown(
                            """
                        **Model Architectures:**
                        - The app features a registry of models, including the `Figure2CNN` baseline, `ResNet` variants, and more advanced custom architectures like `EnhancedCNN` and `HybridSpectralNet`.
                        - Each model is trained on a comprehensive dataset of stable and weathered polymer spectra.

                        **Unified Training Engine:**
                        - A central `TrainingEngine` ensures that all models are trained and validated using a consistent, reproducible 10-fold cross-validation strategy.
                        - This engine can be accessed via the **CLI** (`scripts/train_model.py`) for automated experiments or the **UI** ("Model Training Hub") for interactive use.

                        **Explainability & Transparency (XAI):**
                        - **Feature Importance**: The system is designed to incorporate SHAP and gradient-based methods to highlight which spectral regions most influence a prediction.
                        - **Uncertainty Quantification**: Advanced models can estimate both model (epistemic) and data (aleatoric) uncertainty.
                        - **Data Provenance**: The enhanced data pipeline tracks every preprocessing step, ensuring full traceability from raw data to final prediction.
                        """
                        )

                    render_time = time.time() - start_render
                    log_message(
                        f"col2 rendered in {render_time:.2f}s, active tab: {active_tab}"
                    )

            with st.expander("Spectrum Preprocessing Results", expanded=False):
                st.markdown("---")
                st.markdown("##### Spectral Analysis")

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

            -   **File Type(s):** `.txt`, `.csv`, `.json`
            -   **Content:** Must contain two columns: `wavenumber` and `intensity`.
            -   **Separators:** Values can be separated by spaces or commas.
            -   **Preprocessing:** Your spectrum will be automatically resampled to 500 data points to match the model's input requirements.
            -   **Examples:** Use the "Sample Data" input mode to see examples, or find public data on sites like Open Specy.
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

        -   **File Type(s):** `.txt`, `.csv`, `.json`
        -   **Content:** Must contain two columns: `wavenumber` and `intensity`.
        -   **Separators:** Values can be separated by spaces or commas.
        -   **Preprocessing:** Your spectrum will be automatically resampled to 500 data points to match the model's input requirements.
        -   **Examples:** Use the "Sample Data" input mode to see examples, or find public data on sites like Open Specy.
        """
        )


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
    from core_logic import get_sample_files, run_inference
    from utils.preprocessing import preprocess_spectrum
    from utils.multifile import parse_spectrum_data
    import numpy as np
    import time

    st.markdown("### Multi-Model Comparison Analysis")
    st.markdown(
        "Compare predictions across different AI models for comprehensive analysis."
    )

    # Use the global modality selector from the main page
    modality = st.session_state.get("modality_select", "raman")
    st.info(
        f"Comparing models using **{modality.upper()}** preprocessing parameters. You can change this on the 'Upload and Run' page."
    )

    compatible_models = models_for_modality(modality)
    if not compatible_models:
        st.error(f"No models available for {modality.upper()} modality")
        return

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

                    # Validate spectrum modality
                    is_valid, issues = validate_spectrum_modality(
                        x_raw, y_raw, modality
                    )
                    if not is_valid:
                        st.error("**Spectrum-Modality Mismatch in Comparison**")
                        for issue in issues:
                            st.error(f"• {issue}")
                        st.info(
                            "Please check the selected modality or verify your data file."
                        )
                        return  # Exit comparison if validation fails

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
                        cache_key = hashlib.md5(
                            f"{y_processed.tobytes()}{model_name}".encode()
                        ).hexdigest()
                        prediction, logits_list, probs, inference_time, logits = (
                            run_inference(
                                y_processed,
                                model_name,
                                modality=modality,
                                cache_key=cache_key,
                            )
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

                                            if len(confidences) == 0:
                                                st.warning(
                                                    "No confidence data available for visualization."
                                                )
                                            else:
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
                                                        bar.get_x()
                                                        + bar.get_width() / 2.0,
                                                        height + 0.01,
                                                        f"{conf:.3f}",
                                                        ha="center",
                                                        va="bottom",
                                                    )

                                                ax.set_ylabel("Confidence")
                                                ax.set_title(
                                                    "Model Confidence Comparison"
                                                )
                                                ax.set_ylim(0, 1.1)
                                                plt.xticks(rotation=45)
                                                plt.tight_layout()
                                                st.pyplot(fig)

                                        with col2:
                                            # Confidence distribution
                                            st.markdown("**Confidence Statistics**")
                                            if len(confidences) == 0:
                                                st.warning(
                                                    "No confidence data available for statistics."
                                                )
                                            else:
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

                                    except ValueError as e:
                                        st.error(f"Error rendering results: {e}")

                            except ValueError as e:
                                st.error(f"Error rendering results: {e}")
                                st.error(f"Error in Confidence Analysis tab: {e}")

                            with tab2:
                                # Performance metrics
                                models = list(successful_results.keys())
                                times = [
                                    successful_results[m]["processing_time"]
                                    for m in models
                                ]
                                if len(times) == 0:
                                    st.warning(
                                        "No performance data available for visualization"
                                    )
                                else:

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


from utils.performance_tracker import display_performance_dashboard


def render_performance_tab():
    """Render the performance tracking and analysis tab."""
    display_performance_dashboard()
