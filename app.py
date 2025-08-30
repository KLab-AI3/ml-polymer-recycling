from typing import Union
from utils.multifile import create_batch_uploader, process_multiple_files, display_batch_results
from utils.confidence import calculate_softmax_confidence, get_confidence_badge, create_confidence_progress_html
from utils.results_manager import ResultsManager
from utils.errors import ErrorHandler, safe_execute
from utils.preprocessing import resample_spectrum
from models.resnet_cnn import ResNet1D
from models.figure2_cnn import Figure2CNN
import hashlib
import gc
import time
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import os
import sys
from pathlib import Path

# Ensure 'utils' directory is in the Python path
utils_path = Path(__file__).resolve().parent / "utils"
if utils_path.is_dir() and str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))
matplotlib.use("Agg")  # ensure headless rendering in Spaces

# ==Import local modules + new modules==

KEEP_KEYS = {
    # ==global UI context we want to keep after "Reset"==
    "model_select",     # sidebar model key
    "input_mode",       # radio for Upload|Sample
    "uploader_version",  # version counter for file uploader
    "input_registry",   # radio controlling Upload vs Sample
}

# ==Page Configuration==
st.set_page_config(
    page_title="ML Polymer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/KLab-AI3/ml-polymer-recycling"}
)


# ==============================================================================
# THEME-AWARE CUSTOM CSS
# ==============================================================================
# This CSS block has been refactored to use Streamlit's internal theme
# variables. This ensures that all custom components will automatically adapt
# to both light and dark themes selected by the user in the settings menu.
st.markdown("""
<style>
/* ====== Font Imports (Optional but Recommended) ====== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Fira+Code:wght@400&display=swap');

/* ====== Base & Typography ====== */
.stApp,
section[data-testid="stSidebar"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"] {
  font-family: 'Inter', sans-serif;
  /* Uses the main text color from the current theme (light or dark) */
  color: var(--text-color);
}

.kv-val {
  font-family: 'Fira Code', monospace;
}

/* ====== Custom Containers: Tabs & Info Boxes ====== */
div[data-testid="stTabs"] > div[role="tablist"] + div {
  min-height: 400px;
  /* Uses the secondary background color, which is different in light and dark modes */
  background-color: var(--secondary-background-color);
  /* Border color uses a semi-transparent version of the text color for a subtle effect that works on any background */
  border: 10px solid rgba(128, 128, 128, 0.2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.info-box {
  font-size: 0.9rem;
  padding: 12px 16px;
  border: 1px solid rgba(128, 128, 128, 0.2);
  border-radius: 10px;
  background-color: var(--secondary-background-color);
}

/* ====== Key-Value Pair Styling ====== */
.kv-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(128, 128, 128, 0.2);
}
.kv-row:last-child {
  border-bottom: none;
}
.kv-key {
  opacity: 0.7;
  font-size: 0.9rem;
  white-space: nowrap;
}
.kv-val {
  font-size: 0.9rem;
  overflow-wrap: break-word;
  text-align: right;
}

/* ====== Custom Expander Styling ====== */
div.stExpander > details > summary::-webkit-details-marker,
div.stExpander > details > summary::marker,
div[data-testid="stExpander"] summary svg {
  display: none !important;
}

div.stExpander > details > summary::after {
  content: 'DETAILS';
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  padding: 4px 12px;
  border-radius: 999px;
  /* The primary color is set in config.toml and adapted by Streamlit */
  background-color: var(--primary);
  /* Text on the primary color needs high contrast. White works well for our chosen purple. */

  transition: background-color 0.2s ease-in-out;
}

div.stExpander > details > summary:hover::after {
  /* Using a fixed darker shade on hover. A more advanced solution could use color-mix() in CSS. */
  filter: brightness(90%);
}

/* Specialized Expander Labels */
.expander-results div[data-testid="stExpander"] summary::after {
  content: "RESULTS";
  background-color: #16A34A; /* Green is universal for success */

}
div[data-testid="stExpander"] details {
  content: "RESULTS";
  background-color: var(--primary);
  border-radius: 10px;
  padding: 10px

}
.expander-advanced div[data-testid="stExpander"] summary::after {
  content: "ADVANCED";
  background-color: #D97706; /* Amber is universal for warning/technical */

}

[data-testid="stExpanderDetails"] {
  padding: 16px 4px 4px 4px;
  background-color: transparent;
  border-top: 1px solid rgba(128, 128, 128, 0.2);
  margin-top: 12px;
}

/* ====== Sidebar & Metrics ====== */
section[data-testid="stSidebar"] > div:first-child {
  background-color: var(--secondary-background-color);
  border-right: 1px solid rgba(128, 128, 128, 0.2);
}

div[data-testid="stMetricValue"] {
  font-size: 1.1rem !important;
  font-weight: 500;
}
div[data-testid="stMetricLabel"] {
  font-size: 0.85rem !important;
  opacity: 0.8;
}

/* ====== Interactivity & Accessibility ====== */
:focus-visible {
  /* The focus outline now uses the theme's primary color */
  outline: 2px solid var(--primary);
  outline-offset: 2px;
  border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ==CONSTANTS==
TARGET_LEN = 500
SAMPLE_DATA_DIR = Path("sample_data")
# Prefer env var, else 'model_weights' if present; else canonical 'outputs'
MODEL_WEIGHTS_DIR = (
    os.getenv("WEIGHTS_DIR")
    or ("model_weights" if os.path.isdir("model_weights") else "outputs")
)

# Model configuration
MODEL_CONFIG = {
    "Figure2CNN (Baseline)": {
        "class": Figure2CNN,
        "path": f"{MODEL_WEIGHTS_DIR}/figure2_model.pth",
        "emoji": "",
        "description": "Baseline CNN with standard filters",
        "accuracy": "94.80%",
        "f1": "94.30%"
    },
    "ResNet1D (Advanced)": {
        "class": ResNet1D,
        "path": f"{MODEL_WEIGHTS_DIR}/resnet_model.pth",
        "emoji": "",
        "description": "Residual CNN with deeper feature learning",
        "accuracy": "96.20%",
        "f1": "95.90%"
    }
}

# ==Label mapping==
LABEL_MAP = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}


# ==UTILITY FUNCTIONS==
def init_session_state():
    """Keep a persistent session state"""
    defaults = {
        "status_message": "Ready to analyze polymer spectra 🔬",
        "status_type": "info",
        "input_text": None,
        "filename": None,
        "input_source": None,     # "upload", "batch" or "sample"
        "sample_select": "-- Select Sample --",
        "input_mode": "Upload File",   # controls which pane is visible
        "inference_run_once": False,
        "x_raw": None, "y_raw": None, "y_resampled": None,
        "log_messages": [],
        "uploader_version": 0,
        "current_upload_key": "upload_txt_0",
        "active_tab": "Details",
        "batch_mode": False,
    }

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # ==Initialize results table==
    ResultsManager.init_results_table()


def label_file(filename: str) -> int:
    """Extract label from filename based on naming convention"""
    name = Path(filename).name.lower()
    if name.startswith("sta"):
        return 0
    elif name.startswith("wea"):
        return 1
    else:
        # Return None for unknown patterns instead of raising error
        return -1  # Default value for unknown patterns


@st.cache_data
def load_state_dict(_mtime, model_path):
    """Load state dict with mtime in cache key to detect file changes"""
    try:
        return torch.load(model_path, map_location="cpu")
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"Error loading state dict: {e}")
        return None


@st.cache_resource
def load_model(model_name):
    """Load and cache the specified model with error handling"""
    try:
        config = MODEL_CONFIG[model_name]
        model_class = config["class"]
        model_path = config["path"]

        # Initialize model
        model = model_class(input_length=TARGET_LEN)

        # Check if model file exists
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model weights not found: {model_path}")
            st.info("Using randomly initialized model for demonstration purposes.")
            return model, False

        # Get mtime for cache invalidation
        mtime = os.path.getmtime(model_path)

        # Load weights
        state_dict = load_state_dict(mtime, model_path)
        if state_dict:
            model.load_state_dict(state_dict, strict=True)
            if model is None:
                raise ValueError(
                    "Model is not loaded. Please check the model configuration or weights.")
            model.eval()
            return model, True
        else:
            return model, False

    except (FileNotFoundError, KeyError, RuntimeError) as e:
        st.error(f"❌ Error loading model {model_name}: {str(e)}")
        return None, False


def cleanup_memory():
    """Clean up memory after inference"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_data
def run_inference(y_resampled, model_choice, _cache_key=None):
    """Run model inference and cache results"""
    model, model_loaded = load_model(model_choice)
    if not model_loaded:
        return None, None, None, None, None

    input_tensor = torch.tensor(
        y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        if model is None:
            raise ValueError(
                "Model is not loaded. Please check the model configuration or weights.")
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
        logits_list = logits.detach().numpy().tolist()[0]
        probs = F.softmax(logits.detach(), dim=1).cpu().numpy().flatten()
    inference_time = time.time() - start_time
    cleanup_memory()
    return prediction, logits_list, probs, inference_time,  logits


@st.cache_data
def get_sample_files():
    """Get list of sample files if available"""
    sample_dir = Path(SAMPLE_DATA_DIR)
    if sample_dir.exists():
        return sorted(list(sample_dir.glob("*.txt")))
    return []


def parse_spectrum_data(raw_text):
    """Parse spectrum data from text with robust error handling and validation"""
    x_vals, y_vals = [], []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue

        try:
            # Handle different separators
            parts = line.replace(",", " ").split()
            numbers = [p for p in parts if p.replace('.', '', 1).replace(
                '-', '', 1).replace('+', '', 1).isdigit()]

            if len(numbers) >= 2:
                x, y = float(numbers[0]), float(numbers[1])
                x_vals.append(x)
                y_vals.append(y)

        except ValueError:
            # Skip problematic lines but don't fail completely
            continue

    if len(x_vals) < 10:  # Minimum reasonable spectrum length
        raise ValueError(
            f"Insufficient data points: {len(x_vals)}. Need at least 10 points.")

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Check for NaNs
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")

    # Check monotonic increasing x
    if not np.all(np.diff(x) > 0):
        raise ValueError("Wavenumbers must be strictly increasing")

    # Check reasonable range for Raman spectroscopy
    if min(x) < 0 or max(x) > 10000 or (max(x) - min(x)) < 100:
        raise ValueError(
            f"Invalid wavenumber range: {min(x)} - {max(x)}. Expected ~400-4000 cm⁻¹ with span >100")

    return x, y


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
    ax[1].plot(x_resampled, y_resampled, label="Resampled",
               color="steelblue", linewidth=1)
    ax[1].set_title(f"Resampled ({len(y_resampled)} points)")
    ax[1].set_xlabel("Wavenumber (cm⁻¹)")
    ax[1].set_ylabel("Intensity")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    # == Convert to image ==
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)  # Prevent memory leaks

    return Image.open(buf)


def render_confidence_progress(
    probs: np.ndarray,
    labels: list[str] = ["Stable", "Weathered"],
    highlight_idx: Union[int, None] = None,
    side_by_side: bool = True
):
    """Render Streamlit native progress bars with scientific formatting."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    if side_by_side:
        cols = st.columns(len(labels))
        for i, (lbl, val, col) in enumerate(zip(labels, p, cols)):
            with col:
                is_highlighted = (
                    highlight_idx is not None and i == highlight_idx)
                label_text = f"**{lbl}**" if is_highlighted else lbl
                st.markdown(f"{label_text}: {val*100:.1f}%")
                st.progress(int(round(val * 100)))
    else:
        # Vertical layout for better readability
        for i, (lbl, val) in enumerate(zip(labels, p)):
            is_highlighted = (highlight_idx is not None and i == highlight_idx)

            # Create a container for each probability
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    if is_highlighted:
                        st.markdown(f"**{lbl}** ← Predicted")
                    else:
                        st.markdown(f"{lbl}")
                with col2:
                    st.metric(
                        label="",
                        value=f"{val*100:.1f}%",
                        delta=None
                    )

                # Progress bar with conditional styling
                if is_highlighted:
                    st.progress(int(round(val * 100)))
                    st.caption("🎯 **Model Prediction**")
                else:
                    st.progress(int(round(val * 100)))

                if i < len(labels) - 1:  # Add spacing between items
                    st.markdown("")


def render_kv_grid(d: dict, ncols: int = 2):
    """Display dict as a clean grid of key/value rows using native Streamlit components."""
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


def log_message(msg: str):
    """Append a timestamped line to the in-app log, creating the buffer if needed."""
    ErrorHandler.log_info(msg)


def trigger_run():
    """Set a flag so we can detect button press reliably across reruns"""
    st.session_state['run_requested'] = True


def on_sample_change():
    """Read selected sample once and persist as text."""
    sel = st.session_state.get("sample_select", "-- Select Sample --")
    if sel == "-- Select Sample --":
        return
    try:
        text = (Path(SAMPLE_DATA_DIR / sel).read_text(encoding="utf-8"))
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
        f"ℹ️ {reason}"
        if reason else "Ready to analyze polymer spectra 🔬"
    )
    st.session_state["status_type"] = "info"


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
    st.session_state["current_upload_key"] = f"upload_txt_{st.session_state['uploader_version']}"


# --- START: BUG 2 FIX (Callback Function) ---


def clear_batch_results():
    """Callback to clear only the batch results and the results log table."""
    if "batch_results" in st.session_state:
        del st.session_state["batch_files"]
    # Also clear the persistent table from the ResultsManager utility
    ResultsManager.clear_results()
    st.rerun()
# --- END: BUG 2 FIX (Callback Function) ---


def reset_all():
    # Increment the key to force the file uploader to re-render
    st.session_state.uploader_key += 1


# Main app
def main():
    init_session_state()

    # Sidebar
    with st.sidebar:
        # Header
        st.header("AI-Driven Polymer Classification")
        st.caption(
            "Predict polymer degradation (Stable vs Weathered) from Raman spectra using validated CNN models. — v0.1")
        model_labels = [
            f"{MODEL_CONFIG[name]['emoji']} {name}" for name in MODEL_CONFIG.keys()]
        selected_label = st.selectbox(
            "Choose AI Model", model_labels, key="model_select", on_change=on_model_change)
        model_choice = selected_label.split(" ", 1)[1]

        # ===Compact metadata directly under dropdown===
        render_model_meta(model_choice)

        # ===Collapsed info to reduce clutter===
        with st.expander("About This App", icon=":material/info:", expanded=False):
            st.markdown("""
            AI-Driven Polymer Aging Prediction and Classification

            **Purpose**: Classify polymer degradation using AI
            **Input**: Raman spectroscopy `.txt` files
            **Models**: CNN architectures for binary classification
            **Next**: More trained CNNs in evaluation pipeline


            **Contributors**
            Dr. Sanmukh Kuppannagari (Mentor)
            Dr. Metin Karailyan (Mentor)
            Jaser Hasan (Author)


            **Links**
            [Live HF Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)
            [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)


            **Citation Figure2CNN (baseline)**
            Neo et al., 2023, *Resour. Conserv. Recycl.*, 188, 106718.
            [https://doi.org/10.1016/j.resconrec.2022.106718](https://doi.org/10.1016/j.resconrec.2022.106718)
            """, )

    # Main content area
    col1, col2 = st.columns([1, 1.35], gap="small")

    with col1:
        st.markdown("##### Data Input")

        mode = st.radio(
            "Input mode",
            ["Upload File", "Batch Upload", "Sample Data"],
            key="input_mode",
            horizontal=True,
            on_change=on_input_mode_change
        )

        # ==Upload tab==
        if mode == "Upload File":
            upload_key = st.session_state["current_upload_key"]
            up = st.file_uploader(
                "Upload Raman spectrum (.txt)",
                type="txt",
                help="Upload a text file with wavenumber and intensity columns",
                key=upload_key,     # ← versioned key
            )

            # ==Process change immediately (no on_change; simpler & reliable)==
            if up is not None:
                raw = up.read()
                text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                # == only reparse if its a different file|source ==
                if st.session_state.get("filename") != getattr(up, "name", None) or st.session_state.get("input_source") != "upload":
                    st.session_state["input_text"] = text
                    st.session_state["filename"] = getattr(up, "name", None)
                    st.session_state["input_source"] = "upload"
                    # Ensure single file mode
                    st.session_state["batch_mode"] = False
                    st.session_state["status_message"] = f"File '{st.session_state['filename']}' ready for analysis"
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
                key=batch_upload_key
            )
            # --- END: BUG 1 & 3 FIX ---

            if uploaded_files:
                # --- START: Bug 1 Fix ---
                # Use a dictionary to keep only unique files based on name and size
                unique_files = {(file.name, file.size)
                                 : file for file in uploaded_files}
                unique_file_list = list(unique_files.values())

                num_uploaded = len(uploaded_files)
                num_unique = len(unique_file_list)

                # Optionally, inform the user that duplicates were removed
                if num_uploaded > num_unique:
                    st.info(
                        f"ℹ️ {num_uploaded - num_unique} duplicate file(s) were removed.")

                # Use the unique list
                st.session_state["batch_files"] = unique_file_list
                st.session_state["status_message"] = f"{num_unique} ready for batch analysis"
                st.session_state["status_type"] = "success"
                # --- END: Bug 1 Fix ---
            else:
                st.session_state["batch_files"] = []
                # This check prevents resetting the status if files are already staged
                if not st.session_state.get("batch_files"):
                    st.session_state["status_message"] = "No files selected for batch processing"
                    st.session_state["status_type"] = "info"

        # ==Sample tab==
        elif mode == "Sample Data":
            st.session_state["batch_mode"] = False
            sample_files = get_sample_files()
            if sample_files:
                options = ["-- Select Sample --"] + \
                    [p.name for p in sample_files]
                sel = st.selectbox(
                    "Choose sample spectrum:",
                    options,
                    key="sample_select",
                    on_change=on_sample_change,
                )
                if sel != "-- Select Sample --":
                    st.session_state["status_message"] = f"📁 Sample '{sel}' ready for analysis"
                    st.session_state["status_type"] = "success"
            else:
                st.info("No sample data available")

        # ==Status box==
        msg = st.session_state.get("status_message", "Ready")
        typ = st.session_state.get("status_type", "info")
        if typ == "success":
            st.success(msg)
        elif typ == "error":
            st.error(msg)
        else:
            st.info(msg)

        # ==Model load==
        model, model_loaded = load_model(model_choice)
        if not model_loaded:
            st.warning("⚠️ Model weights not available - using demo mode")

        # ==Ready to run if we have text (single) or files (batch) and a model==|
        is_batch_mode = st.session_state.get("batch_mode", False)
        batch_files = st.session_state.get("batch_files", [])

        inference_ready = False  # Initialize with a default value
        if is_batch_mode:
            inference_ready = len(batch_files) > 0 and (model is not None)
        else:
            inference_ready = st.session_state.get(
                "input_text") is not None and (model is not None)

        # === Run Analysis (form submit batches state) ===
        with st.form("analysis_form", clear_on_submit=False):
            submitted = st.form_submit_button(
                "Run Analysis",
                type="primary",
                disabled=not inference_ready,
            )

        # Renamed for clarity and uses the robust on_click callback
        st.button("Reset All", on_click=reset_ephemeral_state,
                  help="Clear all uploaded files and results.")

        if submitted and inference_ready:
            if is_batch_mode:
                with st.spinner(f"Processing {len(batch_files)} files ..."):
                    try:
                        batch_results = process_multiple_files(
                            uploaded_files=batch_files,
                            model_choice=model_choice,
                            load_model_func=load_model,
                            run_inference_func=run_inference,
                            label_file_func=label_file
                        )
                        st.session_state["batch_results"] = batch_results
                        st.success(
                            f"Successfully processed {len([r for r in batch_results if r.get('success', False)])}/{len(batch_files)} files")
                    except Exception as e:
                        st.error(f"Error during batch processing: {e}")
            else:
                try:
                    x_raw, y_raw = parse_spectrum_data(
                        st.session_state["input_text"])
                    x_resampled, y_resampled = resample_spectrum(
                        x_raw, y_raw, TARGET_LEN)
                    st.session_state["x_raw"] = x_raw
                    st.session_state["y_raw"] = y_raw
                    st.session_state["x_resampled"] = x_resampled
                    st.session_state["y_resampled"] = y_resampled
                    st.session_state["inference_run_once"] = True
                except (ValueError, TypeError) as e:
                    st.error(f"Error processing spectrum data: {e}")
                    st.session_state["status_message"] = f"❌ Error: {e}"
                    st.session_state["status_type"] = "error"

    # Results column
    with col2:

        # Check if we're in batch more or have batch results
        is_batch_mode = st.session_state.get("batch_mode", False)
        has_batch_results = "batch_results" in st.session_state

        if is_batch_mode and has_batch_results:
            # Display batch results
            st.markdown("##### Batch Analysis Results")
            batch_results = st.session_state["batch_results"]
            display_batch_results(batch_results)

            # Add session results table
            st.markdown("---")

            # --- START: BUG 2 FIX (Button) ---
            # This button will clear all results from col2 correctly.
            # st.button("Clear Results", on_click=clear_batch_results,
            #           help="Clear all uploaded files and results.")
            # --- END: BUG 2 FIX (Button) ---

            ResultsManager.display_results_table()

        elif st.session_state.get("inference_run_once", False) and not is_batch_mode:
            st.markdown("##### Analysis Results")

            # Get data from session state
            x_raw = st.session_state.get('x_raw')
            y_raw = st.session_state.get('y_raw')
            x_resampled = st.session_state.get('x_resampled')   # ← NEW
            y_resampled = st.session_state.get('y_resampled')
            filename = st.session_state.get('filename', 'Unknown')

            if all(v is not None for v in [x_raw, y_raw, y_resampled]):
                # ===Run inference===
                if y_resampled is None:
                    raise ValueError(
                        "y_resampled is None. Ensure spectrum data is properly resampled before proceeding.")
                cache_key = hashlib.md5(
                    f"{y_resampled.tobytes()}{model_choice}".encode()).hexdigest()
                prediction, logits_list, probs, inference_time, logits = run_inference(
                    y_resampled, model_choice, _cache_key=cache_key
                )
                if prediction is None:
                    st.error(
                        "❌ Inference failed: Model not loaded. Please check that weights are available.")
                    st.stop()  # prevents the rest of the code in this block from executing

                log_message(
                    f"Inference completed in {inference_time:.2f}s, prediction: {prediction}")

                # ===Get ground truth===
                true_label_idx = label_file(filename)
                true_label_str = LABEL_MAP.get(
                    true_label_idx, "Unknown") if true_label_idx is not None else "Unknown"
                # ===Get prediction===
                predicted_class = LABEL_MAP.get(
                    int(prediction), f"Class {int(prediction)}")

                # Enhanced confidence calculation
                if logits is not None:
                    # Use new softmax-based confidence
                    probs_np, max_confidence, confidence_level, confidence_emoji = calculate_softmax_confidence(
                        logits)
                    confidence_desc = confidence_level
                else:
                    # Fallback to legace method
                    logit_margin = abs(
                        (logits_list[0] - logits_list[1]) if logits_list is not None and len(logits_list) >= 2 else 0)
                    confidence_desc, confidence_emoji = get_confidence_description(
                        logit_margin)
                    max_confidence = logit_margin / 10.0  # Normalize for display
                    probs_np = np.array([])

                # Store result in results manager for single file too
                ResultsManager.add_results(
                    filename=filename,
                    model_name=model_choice,
                    prediction=int(prediction),
                    predicted_class=predicted_class,
                    confidence=max_confidence,
                    logits=logits_list if logits_list else [],
                    ground_truth=true_label_idx if true_label_idx >= 0 else None,
                    processing_time=inference_time if inference_time is not None else 0.0,
                    metadata={
                        "confidence_level": confidence_desc,
                        "confidence_emoji": confidence_emoji
                    }
                )

                # ===Precompute Stats===
                spec_stats = {
                    "Original Length": len(x_raw) if x_raw is not None else 0,
                    "Resampled Length": TARGET_LEN,
                    "Wavenumber Range": f"{min(x_raw):.1f}-{max(x_raw):.1f} cm⁻¹" if x_raw is not None else "N/A",
                    "Intensity Range": f"{min(y_raw):.1f}-{max(y_raw):.1f} au" if y_raw is not None else "N/A",
                    "Confidence Bucket": confidence_desc,
                }
                model_path = MODEL_CONFIG[model_choice]["path"]
                mtime = os.path.getmtime(
                    model_path) if os.path.exists(model_path) else None
                file_hash = (
                    hashlib.md5(open(model_path, 'rb').read()).hexdigest()
                    if os.path.exists(model_path) else "N/A"
                )
                input_tensor = torch.tensor(
                    y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                model_stats = {
                    "Architecture": model_choice,
                    "Model Path": model_path,
                    "Weights Last Modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)) if mtime else "N/A",
                    "Weights Hash (md5)": file_hash,
                    "Input Shape": list(input_tensor.shape),
                    "Output Shape": list(logits.shape) if logits is not None else "N/A",
                    "Inference Time": f"{inference_time:.3f}s",
                    "Device": "CPU",
                    "Model Loaded": model_loaded,
                }

                start_render = time.time()

                active_tab = st.selectbox(
                    "View Results",
                    ["Details", "Technical", "Explanation"],
                    key="active_tab",   # reuse the key you were managing manually
                )

                if active_tab == "Details":
                    st.markdown('<div class="expander-results">',
                                unsafe_allow_html=True)
                    # Use a dynamic and informative title for the expander
                    with st.expander(f"Results for {filename}", expanded=True):

                        # --- START: STREAMLINED METRICS ---
                        # A single, powerful row for the most important results.
                        key_metric_cols = st.columns(3)

                        # Metric 1: The Prediction
                        key_metric_cols[0].metric(
                            "Prediction", predicted_class)

                        # Metric 2: The Confidence (with level in tooltip)
                        confidence_icon = "🟢" if max_confidence >= 0.8 else "🟡" if max_confidence >= 0.6 else "🔴"
                        key_metric_cols[1].metric(
                            "Confidence",
                            f"{confidence_icon} {max_confidence:.1%}",
                            help=f"Confidence Level: {confidence_desc}"
                        )

                        # Metric 3: Ground Truth + Correctness (Combined)
                        if true_label_idx is not None:
                            is_correct = (predicted_class == true_label_str)
                            delta_text = "✅ Correct" if is_correct else "❌ Incorrect"
                            # Use delta_color="normal" to let the icon provide the visual cue
                            key_metric_cols[2].metric(
                                "Ground Truth", true_label_str, delta=delta_text, delta_color="normal")
                        else:
                            key_metric_cols[2].metric("Ground Truth", "N/A")

                        st.divider()
                        # --- END: STREAMLINED METRICS ---

                        # --- START: CONSOLIDATED CONFIDENCE ANALYSIS ---
                        st.markdown("##### Probability Breakdown")

                        # This custom bullet bar logic remains as it is highly specific and valuable
                        def create_bullet_bar(probability, width=20, predicted=False):
                            filled_count = int(probability * width)
                            bar = "▤" * filled_count + \
                                "▢" * (width - filled_count)
                            percentage = f"{probability:.1%}"
                            pred_marker = "↩ Predicted" if predicted else ""
                            return f"{bar} {percentage}    {pred_marker}"

                        stable_prob, weathered_prob = probs[0], probs[1]
                        is_stable_predicted, is_weathered_predicted = (
                            int(prediction) == 0), (int(prediction) == 1)

                        st.markdown(f"""
                            <div style="font-family: 'Fira Code', monospace;">
                                Stable (Unweathered)<br>
                                {create_bullet_bar(stable_prob, predicted=is_stable_predicted)}<br><br>
                                Weathered (Degraded)<br>
                                {create_bullet_bar(weathered_prob, predicted=is_weathered_predicted)}
                            </div>
                        """, unsafe_allow_html=True)
                        # --- END: CONSOLIDATED CONFIDENCE ANALYSIS ---

                        st.divider()

                        # --- START: CLEAN METADATA FOOTER ---
                        # Secondary info is now a clean, single-line caption
                        st.caption(
                            f"Analyzed with **{model_choice}** in **{inference_time:.2f}s**.")
                        # --- END: CLEAN METADATA FOOTER ---

                    st.markdown('</div>', unsafe_allow_html=True)

                elif active_tab == "Technical":
                    with st.container():
                        st.markdown("Technical Diagnostics")

                        # Model performance metrics
                        with st.container(border=True):
                            st.markdown("##### **Model Performance**")
                            tech_col1, tech_col2 = st.columns(2)

                            with tech_col1:
                                st.metric("Inference Time",
                                          f"{inference_time:.3f}s")
                                st.metric(
                                    "Input Length", f"{len(x_raw) if x_raw is not None else 0} points")
                                st.metric("Resampled Length",
                                          f"{TARGET_LEN} points")

                            with tech_col2:
                                st.metric("Model Loaded",
                                          "✅ Yes" if model_loaded else "❌ No")
                                st.metric("Device", "CPU")
                                st.metric("Confidence Score",
                                          f"{max_confidence:.3f}")

                        # Raw logits display
                        with st.container(border=True):
                            st.markdown("##### **Raw Model Outputs (Logits)**")
                            if logits_list is not None:
                                logits_df = {
                                    "Class": [LABEL_MAP.get(i, f"Class {i}") for i in range(len(logits_list))],
                                    "Logit Value": [f"{score:.4f}" for score in logits_list],
                                    "Probability": [f"{prob:.4f}" for prob in probs_np] if len(probs_np) > 0 else ["N/A"] * len(logits_list)
                                }

                            # Display as a simple table format
                            for i, (cls, logit, prob) in enumerate(zip(logits_df["Class"], logits_df["Logit Value"], logits_df["Probability"])):
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
                                render_kv_grid({
                                    "Length": f"{len(x_raw) if x_raw is not None else 0} points",
                                    "Range": f"{min(x_raw):.1f} - {max(x_raw):.1f} cm⁻¹" if x_raw is not None else "N/A",
                                    "Min Intensity": f"{min(y_raw):.2e}" if y_raw is not None else "N/A",
                                    "Max Intensity": f"{max(y_raw):.2e}" if y_raw is not None else "N/A"
                                }, ncols=1)

                            with spec_cols[1]:
                                st.markdown("**Processed Spectrum:**")
                                render_kv_grid({
                                    "Length": f"{TARGET_LEN} points",
                                    "Resampling": "Linear interpolation",
                                    "Normalization": "None",
                                    "Input Shape": f"(1, 1, {TARGET_LEN})"
                                }, ncols=1)

                        # Model information
                        with st.container(border=True):
                            st.markdown("##### **Model Information**")
                            model_info_cols = st.columns(2)

                            with model_info_cols[0]:
                                render_kv_grid({
                                    "Architecture": model_choice,
                                    "Path": MODEL_CONFIG[model_choice]["path"],
                                    "Weights Modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)) if mtime else "N/A"
                                }, ncols=1)

                            with model_info_cols[1]:
                                if os.path.exists(model_path):
                                    file_hash = hashlib.md5(
                                        open(model_path, 'rb').read()).hexdigest()
                                    render_kv_grid({
                                        "Weights Hash": f"{file_hash[:16]}...",
                                        "Output Shape": f"(1, {len(LABEL_MAP)})",
                                        "Activation": "Softmax"
                                    }, ncols=1)

                        # Debug logs (collapsed by default)
                        with st.expander("📋 Debug Logs", expanded=False):
                            log_content = "\n".join(
                                st.session_state.get("log_messages", []))
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
                            "✅ **Validation**: Ground truth comparison (when available from filename)"
                        ]

                        for step in process_steps:
                            st.markdown(step)

                        st.markdown("---")

                        # Model interpretation
                        st.markdown("#### Scientific Interpretation")

                        interp_col1, interp_col2 = st.columns(2)

                        with interp_col1:
                            st.markdown("**Stable (Unweathered) Polymers:**")
                            st.info("""
                            - Well-preserved molecular structure
                            - Minimal oxidative degradation
                            - Characteristic Raman peaks intact
                            - Suitable for recycling applications
                            """)

                        with interp_col2:
                            st.markdown("**Weathered (Degraded) Polymers:**")
                            st.warning("""
                            - Oxidized molecular bonds
                            - Surface degradation present
                            - Altered spectral signatures
                            - May require additional processing
                            """)

                        st.markdown("---")

                        # Applications
                        st.markdown("#### Research Applications")

                        applications = [
                            "🔬 **Material Science**: Polymer degradation studies",
                            "♻️ **Recycling Research**: Viability assessment for circular economy",
                            "🌱 **Environmental Science**: Microplastic weathering analysis",
                            "🏭 **Quality Control**: Manufacturing process monitoring",
                            "📈 **Longevity Studies**: Material aging prediction"
                        ]

                        for app in applications:
                            st.markdown(app)

                        # Technical details
                        # MODIFIED: Wrap the expander in a div with the 'expander-advanced' class
                        st.markdown('<div class="expander-advanced">',
                                    unsafe_allow_html=True)
                        with st.expander("🔧 Technical Details", expanded=False):
                            st.markdown("""
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
                            """)
                        st.markdown(
                            '</div>', unsafe_allow_html=True)  # Close the wrapper div

                        render_time = time.time() - start_render
                        log_message(
                            f"col2 rendered in {render_time:.2f}s, active tab: {active_tab}")

                with st.expander("Spectrum Preprocessing Results", expanded=False):
                    st.caption("<br>Spectral Analysis", unsafe_allow_html=True)

                    # Add some context about the preprocessing
                    st.markdown("""
                    **Preprocessing Overview:**
                    - **Original Spectrum**: Raw Raman data as uploaded
                    - **Resampled Spectrum**: Data interpolated to 500 points for model input
                    - **Purpose**: Ensures consistent input dimensions for neural network
                    """)

                    # Create and display plot
                    cache_key = hashlib.md5(
                        f"{(x_raw.tobytes() if x_raw is not None else b'')}"
                        f"{(y_raw.tobytes() if y_raw is not None else b'')}"
                        f"{(x_resampled.tobytes() if x_resampled is not None else b'')}"
                        f"{(y_resampled.tobytes() if y_resampled is not None else b'')}".encode()
                    ).hexdigest()
                    spectrum_plot = create_spectrum_plot(
                        x_raw, y_raw, x_resampled, y_resampled, _cache_key=cache_key)
                    st.image(
                        spectrum_plot, caption="Raman Spectrum: Raw vs Processed", use_container_width=True)

            else:
                st.error(
                    "❌ Missing spectrum data. Please upload a file and run analysis.")
        else:
            # ===Getting Started===
            st.markdown("""
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
            """)


# Run the application
main()
