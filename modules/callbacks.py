import streamlit as st
from pathlib import Path
from utils.results_manager import ResultsManager
from utils.errors import ErrorHandler
from config import SAMPLE_DATA_DIR


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

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # ==Initialize results table==
    ResultsManager.init_results_table()


def log_message(msg: str):
    """Append a timestamped line to the in-app log, creating the buffer if needed."""
    ErrorHandler.log_info(msg)


def on_sample_change():
    """Read selected sample once and persist as text."""
    sel = st.session_state.get("sample_select", "-- Select Sample --")
    if sel == "-- Select Sample --":
        return
    try:
        text = Path(SAMPLE_DATA_DIR / sel).read_text(encoding="utf-8")
        st.session_state["input_text"] = text
        st.session_state["filename"] = sel
        st.session_state["input_source"] = "sample"
        # 🔧 Clear previous results so right column resets immediately
        reset_results("New sample selected")
        st.session_state["status_message"] = f"📁 Sample '{sel}' ready for analysis"
        st.session_state["status_type"] = "success"
    except (FileNotFoundError, IOError) as e:
        st.session_state["status_message"] = f"❌ Error loading sample: {e}"
        st.session_state["status_type"] = "error"


def on_input_mode_change():
    """Reset sample when switching to Upload"""
    if st.session_state["input_mode"] == "Upload File":
        st.session_state["sample_select"] = "-- Select Sample --"
        st.session_state["batch_mode"] = False  # Reset batch mode
    elif st.session_state["input_mode"] == "Sample Data":
        st.session_state["batch_mode"] = False  # Reset batch mode
    # 🔧 Reset when switching modes to prevent stale right-column visuals
    reset_results("Switched input mode")


def reset_ephemeral_state():
    """Comprehensive reset for the entire app state."""
    # Define keys that should NOT be cleared by a full reset
    keep_keys = {"model_select", "input_mode"}

    for k in list(st.session_state.keys()):
        if k not in keep_keys:
            st.session_state.pop(k, None)

    # Re-initialize the core state after clearing
    init_session_state()

    # CRITICAL: Bump the uploader version to force a widget reset
    st.session_state["uploader_version"] += 1
    st.session_state["current_upload_key"] = (
        f"upload_txt_{st.session_state['uploader_version']}"
    )


def on_model_change():
    """Force the right column back to init state when the model changes"""
    reset_results("Model changed")


def reset_results(reason: str = ""):
    """Clear previous inference artifacts so the right column returns to initial state."""
    st.session_state["inference_run_once"] = False
    st.session_state["x_raw"] = None
    st.session_state["y_raw"] = None
    st.session_state["y_resampled"] = None
    # ||== Clear batch results when resetting ==||
    if "batch_results" in st.session_state:
        del st.session_state["batch_results"]
    # ||== Clear logs between runs ==||
    st.session_state["log_messages"] = []
    # ||== Always reset the status box ==||
    st.session_state["status_message"] = (
        f"ℹ️ {reason}" if reason else "Ready to analyze polymer spectra 🔬"
    )
    st.session_state["status_type"] = "info"


def clear_batch_results():
    """Callback to clear only the batch results and the results log table."""
    if "batch_results" in st.session_state:
        del st.session_state["batch_results"]
    # Also clear the persistent table from the ResultsManager utility
    ResultsManager.clear_results()


# --- END: BUG 2 FIX (Callback Function) ---


def reset_all():
    # Increment the key to force the file uploader to re-render
    st.session_state.uploader_key += 1


def trigger_run():
    """Set a flag so we can detect button press reliably across reruns"""
    st.session_state["run_requested"] = True
