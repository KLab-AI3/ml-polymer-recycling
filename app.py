"""
AI-Driven Polymer Aging Prediction and Classification
Hugging Face Spaces Deployment

This Streamlit app provides an interface for classifying polymer degradation
using deep learning models trained on Raman spectroscopy data.

Features:
- Single file and batch upload processing
- Multiple CNN model architectures
- Results export to CSV/JSON
- Enhanced confidence visualization
- Session-wide results management
"""

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

# Import local modules
from utils.preprocessing import resample_spectrum

# Import new utilities
from utils.errors import ErrorHandler, safe_execute
from utils.results_manager import ResultsManager
from utils.confidence import calculate_softmax_confidence, get_confidence_badge, create_confidence_progress_html
from utils.multifile import create_batch_uploader, process_multiple_files, display_batch_results

KEEP_KEYS = {
    # === global UI context we want to keep after "Reset" ===
    "model_select",     # sidebar model key
    "input_mode",       # radio for Upload|Sample
    "uploader_version", # version counter for file uploader
    "input_registry",   # radio controlling Upload vs Sample
}

# Configuration
st.set_page_config(
    page_title="ML Polymer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Keep only scoped utility styles; no .block-container edits */

/* Tabs content area height (your original intent) */
div[data-testid="stTabs"] > div[role="tablist"] + div { min-height: 420px; }

/* Compact info box for confidence bar */
.confbox {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.95rem;
  padding: 8px 10px; border: 1px solid rgba(0,0,0,.07);
  border-radius: 8px; background: rgba(0,0,0,.02);
}

/* Clean key–value rows for technical info */
.kv-row { display:flex; justify-content:space-between;
  border-bottom: 1px dotted rgba(0,0,0,.10); padding: 3px 0; gap: 12px; }
.kv-key { opacity:.75; font-size: 0.95rem; white-space: nowrap; }
.kv-val { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  overflow-wrap: anywhere; }

/* Ensure markdown h5 headings remain visible after layout shifts */
:where(h5, .stMarkdown h5) { margin-top: 0.25rem; }

/* === Base Expander Header === */
div.stExpander > details > summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  list-style: none;             /* remove default arrow */
  cursor: pointer;
  border: 1px solid rgba(0,0,0,.15);
  border-left: 4px solid #9ca3af;   /* default gray accent */
  border-radius: 6px;
  padding: 6px 12px;
  margin: 6px 0;
  background: rgba(0,0,0,0.04);
  font-weight: 600;
  font-size: 0.95rem;
}

/* Remove ugly default disclosure triangle */
div.stExpander > details > summary::-webkit-details-marker {
  display: none;
}
div.stExpander > details > summary::marker {
  display: none;
}

/* Hover/active subtlety */
div.stExpander > details[open] > summary {
  background: rgba(0,0,0,0.06);
}

/* Hide Streamlit's custom arrow icon inside expanders */
div[data-testid="stExpander"] summary svg {
  display: none !important;
}

/* === Right Badge === */
div.stExpander > details > summary::after {
  content: "MORE ↓";
  font-size: 0.70rem;
  font-weight: 600;
  letter-spacing: .04em;
  padding: 2px 8px;
  border-radius: 999px;
  margin-left: auto;
  background: #e5e7eb;
  color: #111827;
}

/* === Stable cross-browser expander behavior  === */
.expander-marker + div[data-testid="stExpander"] summary {
  border-left-color: #2e7d32;
  background: rgba(46,125,50,0.08);
}
.expander-marker + div[data-testid="stExpander"] summary::after {
  content: "RESULTS";
  background: rgba(46,125,50,0.15);
  color: #184a1d;
}


div.stExpander:has(summary:contains("Technical")) > details > summary {
  border-left-color: #ed6c02;
  background: rgba(237,108,2,0.08);
}
div.stExpander:has(summary:contains("Technical")) > details > summary::after {
  content: "ADVANCED";
  background: rgba(237,108,2,0.18); color: #7a3d00;
}

/* === FONT SIZE STANDARDIZATION === */

/* Sidebar metrics (Accuracy, F1 Score) */
div[data-testid="stMetricValue"] {
  font-size: 0.95rem !important;  /* uniform body size */
}
div[data-testid="stMetricLabel"] {
  font-size: 0.85rem !important;
  opacity: 0.85;
}

/* Sidebar expander text */
section[data-testid="stSidebar"] .stMarkdown p {
  font-size: 0.95rem !important;
  line-height: 1.4;
}

/* Diagnostics tab metrics (Logits) */
div[data-testid="stMetricValue"] {
  font-size: 0.95rem !important;
}
div[data-testid="stMetricLabel"] {
  font-size: 0.85rem !important;
}


</style>
""", unsafe_allow_html=True)


# Constants
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

# Label mapping
LABEL_MAP = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}


# === UTILITY FUNCTIONS ===
def init_session_state():
    """Keep a persistent session state"""
    defaults = {
        "status_message": "Ready to analyze polymer spectra 🔬",
        "status_type": "info",
        "input_text": None,
        "filename": None,
        "input_source": None,     # "upload" or "sample"
        "sample_select": "-- Select Sample --",
        "input_mode": "Upload File",   # controls which pane is visible
        "inference_run_once": False,
        "x_raw": None, "y_raw": None, "y_resampled": None,
        "log_messages": [],
        "uploader_version": 0,
        "current_upload_key": "upload_txt_0",
        "active_tab": "Details",
        "batch_mode": False,  # Track if in batch mode
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Initialize results table
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
        return torch.load(model_path, map_location="cpu", weights_only=True)
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

    except (FileNotFoundError, KeyError) as e:
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

    input_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        if model is None:
            raise ValueError("Model is not loaded. Please check the model configuration or weights.")
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
    ax[1].plot(x_resampled, y_resampled, label="Resampled", color="steelblue", linewidth=1)
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

from typing import Union

def render_confidence_progress(
    probs: np.ndarray,
    labels: list[str] = ["Stable", "Weathered"],
    highlight_idx: Union[int, None] = None,
    side_by_side: bool = True
):
    """Render Streamlit native progress bars (0 - 100). Optionally bold the winning class
    and place the two bars side-by-side for compactness."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    def _title(i: int, lbl: str, val: float) -> str:
        t = f"{lbl} - {val*100:.1f}%"
        return f"**{t}**" if (highlight_idx is not None and i == highlight_idx) else t

    if side_by_side:
        cols = st.columns(len(labels))
        for i, (lbl, val, col) in enumerate(zip(labels, p, cols)):
            with col:
                st.markdown(_title(i, lbl, float(val)))
                st.progress(int(round(val * 100)))
    else:
        for i, (lbl, val) in enumerate(zip(labels, p)):
            st.markdown(_title(i, lbl, float(val)))
            st.progress(int(round(val * 100)))


def render_kv_grid(d: dict, ncols: int = 2):
    """Display dict as a clean grid of key/value rows."""
    if not d: 
        return
    items = list(d.items())
    cols = st.columns(ncols)
    for i, (k, v) in enumerate(items):
        with cols[i % ncols]:
            st.markdown(
                f"<div class='kv-row'><span class='kv-key'>{k}</span>"
                f"<span class='kv-val'>{v}</span></div>",
                unsafe_allow_html=True
            )




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
    # ||== Clear logs between runs ==||
    st.session_state["log_messages"] = []
    # ||== Always reset the status box ==||
    st.session_state["status_message"] = (
        f"ℹ️ {reason}"
        if reason else "Ready to analyze polymer spectra 🔬"
    )
    st.session_state["status_type"] = "info"

def reset_ephemeral_state():
    """remove everything except KEPT global UI context"""
    for k in list(st.session_state.keys()):
        if k not in KEEP_KEYS:
            st.session_state.pop(k, None)

    # == bump the uploader version → new widget instance with empty value ==
    st.session_state["uploader_version"] += 1
    st.session_state["current_upload_key"] = f"upload_txt_{st.session_state['uploader_version']}"
    
    # == reseed other emphemeral state ==
    st.session_state["input_text"] =  None
    st.session_state["filename"] = None
    st.session_state["input_source"] = None
    st.session_state["sample_select"] = "-- Select Sample --"
    # == return the UI to a clean state ==
    st.session_state["inference_run_once"] = False
    st.session_state["x_raw"] = None
    st.session_state["y_raw"] = None
    st.session_state["y_resampled"] = None
    st.session_state["log_messages"] = []
    st.session_state["status_message"] = "Ready to analyze polymer spectra 🔬"
    st.session_state["status_type"] = "info"
    
    st.rerun()

# Main app
def main():
    init_session_state()

    # Sidebar
    with st.sidebar:
        # Header
        st.header("AI-Driven Polymer Classification")
        st.caption("Predict polymer degradation (Stable vs Weathered) from Raman spectra using validated CNN models. — v0.1")
        model_labels = [f"{MODEL_CONFIG[name]['emoji']} {name}" for name in MODEL_CONFIG.keys()]
        selected_label = st.selectbox("Choose AI Model", model_labels, key="model_select", on_change=on_model_change)
        model_choice = selected_label.split(" ", 1)[1]

        # ===Compact metadata directly under dropdown===
        render_model_meta(model_choice)

        # ===Collapsed info to reduce clutter===
        with st.expander("About This App",icon=":material/info:", expanded=False):
            st.markdown("""
            AI-Driven Polymer Aging Prediction and Classification

            **Purpose**: Classify polymer degradation using AI  
            **Input**: Raman spectroscopy `.txt` files  
            **Models**: CNN architectures for binary classification  
            **Next**: More trained CNNs in evaluation pipeline

            ---

            **Contributors**  
            Dr. Sanmukh Kuppannagari (Mentor)  
            Dr. Metin Karailyan (Mentor)  
            👨‍💻 Jaser Hasan (Author)

            ---

            **Links**  
            🔗 [Live HF Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)  
            📂 [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)

            ---

            **Citation Figure2CNN (baseline)**  
            Neo et al., 2023, *Resour. Conserv. Recycl.*, 188, 106718.
            [https://doi.org/10.1016/j.resconrec.2022.106718](https://doi.org/10.1016/j.resconrec.2022.106718)
            """)

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

        # ---- Upload tab ----
        if mode == "Upload File":
            upload_key = st.session_state["current_upload_key"]
            up = st.file_uploader(
                "Upload Raman spectrum (.txt)",
                type="txt",
                help="Upload a text file with wavenumber and intensity columns",
                key=upload_key,     # ← versioned key
            )

            # == process change immediately (no on_change; simpler & reliable) ==
            if up is not None:
                raw = up.read()
                text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                # == only reparse if its a different file|source ==
                if st.session_state.get("filename") != getattr(up, "name", None) or st.session_state.get("input_source") != "upload":
                    st.session_state["input_text"] = text 
                    st.session_state["filename"] = getattr(up, "name", "uploaded.txt")
                    st.session_state["input_source"] = "upload"
                    st.session_state["batch_mode"] = False

                    # == clear right column immediately ==
                    reset_results("New file selected")
                    st.session_state["status_message"] = f"📁 File '{st.session_state['filename']}' ready for analysis"
                    st.session_state["status_type"] = "success"

        # ---- Batch Upload tab ----
        elif mode == "Batch Upload":
            st.session_state["batch_mode"] = True
            uploaded_files = create_batch_uploader()
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files selected for batch processing")
                st.session_state["batch_files"] = uploaded_files
                st.session_state["status_message"] = f"📁 {len(uploaded_files)} files ready for batch analysis"
                st.session_state["status_type"] = "success"
            else:
                st.session_state["batch_files"] = []

        # ---- Sample tab ----
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
                    on_change=on_sample_change,  # <-- critical
                )
                if sel != "-- Select Sample --":
                    st.markdown(f"✅ Loaded sample: {sel}")
            else:
                st.info("No sample data available")

        # ---- Status box ----
        msg = st.session_state.get("status_message", "Ready")
        typ = st.session_state.get("status_type", "info")
        if typ == "success":
            st.success(msg)
        elif typ == "error":
            st.error(msg)
        else:
            st.info(msg)

        # ---- Model load ----
        model, model_loaded = load_model(model_choice)
        if not model_loaded:
            st.warning("⚠️ Model weights not available - using demo mode")

        # Ready to run if we have text (single) or files (batch) and a model
        is_batch_mode = st.session_state.get("batch_mode", False)
        batch_files = st.session_state.get("batch_files", [])
        
        if is_batch_mode:
            inference_ready = len(batch_files) > 0 and (model is not None)
            button_text = f"Run Batch Analysis ({len(batch_files)} files)"
        else:
            inference_ready = bool(st.session_state.get("input_text")) and (model is not None)
            button_text = "Run Analysis"

        # === Run Analysis (form submit batches state) ===
        with st.form("analysis_form", clear_on_submit=False):
            submitted = st.form_submit_button(
                button_text,
                type="primary",
                disabled=not inference_ready,
            )

        if st.button("Reset", help="Clear current file(s), plots, and results"):
            reset_ephemeral_state()

            

        if submitted and inference_ready:
            if is_batch_mode:
                # === Batch Mode Processing ===
                with st.spinner(f"Processing {len(batch_files)} files..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, filename):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {filename} ({current}/{total})")
                    
                    # Process all files
                    batch_results = process_multiple_files(
                        batch_files,
                        model_choice,
                        load_model,
                        run_inference,
                        label_file,
                        progress_callback
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Batch processing complete!")
                    
                    # Update session state
                    st.session_state["batch_results"] = batch_results
                    st.session_state["inference_run_once"] = True
                    successful_count = sum(1 for r in batch_results if r.get("success", False))
                    st.session_state["status_message"] = f"🔍 Batch analysis completed: {successful_count}/{len(batch_files)} successful"
                    st.session_state["status_type"] = "success"
                    
                st.rerun()
            else:
                # === Single File Mode Processing ===
                # parse → preprocess → predict → render
                # Handles the submission of the analysis form and performs spectrum data processing
                try:
                    raw_text = st.session_state["input_text"]
                    filename = st.session_state.get("filename") or "unknown.txt"

                    # Parse
                    with st.spinner("Parsing spectrum data..."):
                        x_raw, y_raw = parse_spectrum_data(raw_text)

                    # Resample
                    with st.spinner("Resampling spectrum..."):
                        # ===Resample Unpack===
                        r1, r2 = resample_spectrum(x_raw, y_raw, TARGET_LEN)

                        def _is_strictly_increasing(a):
                            a = np.asarray(a)
                            return a.ndim == 1  and a.size >= 2 and np.all(np.diff(a) > 0)

                        if _is_strictly_increasing(r1) and not _is_strictly_increasing(r2):
                            x_resampled, y_resampled = np.asarray(r1), np.asarray(r2)
                        elif _is_strictly_increasing(r2) and not _is_strictly_increasing(r1):
                            x_resampled, y_resampled = np.asarray(r2), np.asarray(r1)
                        else:
                            # == Ambigous; assume (x, y) and log
                            x_resampled, y_resampled = np.asarray(r1), np.asarray(r2)
                            log_message("Resample outputs ambigous; assumed (x, y).")

                        # ===Persists for plotting + inference===
                        st.session_state["x_raw"] = x_raw
                        st.session_state["y_raw"] = y_raw
                        st.session_state["x_resampled"] = x_resampled   #  ←-- NEW 
                        st.session_state["y_resampled"] = y_resampled

                    # Persist results (drives right column)
                    st.session_state["x_raw"] = x_raw
                    st.session_state["y_raw"] = y_raw
                    st.session_state["y_resampled"] = y_resampled
                    st.session_state["inference_run_once"] = True
                    st.session_state["status_message"] = f"🔍 Analysis completed for: {filename}"
                    st.session_state["status_type"] = "success"

                    st.rerun()

                except (ValueError, TypeError) as e:
                    ErrorHandler.log_error(e, "Single file analysis")
                    st.error(f"❌ Analysis failed: {e}")
                    st.session_state["status_message"] = f"❌ Error: {e}"
                    st.session_state["status_type"] = "error"

    # Results column
    with col2:
        # Check if we're in batch mode or have batch results
        is_batch_mode = st.session_state.get("batch_mode", False)
        has_batch_results = "batch_results" in st.session_state
        
        if is_batch_mode and has_batch_results:
            # === Display Batch Results ===
            st.markdown("##### Batch Analysis Results")
            batch_results = st.session_state["batch_results"]
            display_batch_results(batch_results)
            
            # Add session results table
            st.markdown("---")
            ResultsManager.display_results_table()
            
        elif st.session_state.get("inference_run_once", False) and not is_batch_mode:
            # === Display Single File Results ===
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
                    raise ValueError("y_resampled is None. Ensure spectrum data is properly resampled before proceeding.")
                cache_key = hashlib.md5(f"{y_resampled.tobytes()}{model_choice}".encode()).hexdigest()
                prediction, logits_list, probs, inference_time, logits = run_inference(
                    y_resampled, model_choice, _cache_key=cache_key
                )
                if prediction is None:
                    st.error("❌ Inference failed: Model not loaded. Please check that weights are available.")
                    st.stop()  # prevents the rest of the code in this block from executing

                log_message(f"Inference completed in {inference_time:.2f}s, prediction: {prediction}")

                # ===Get ground truth===
                true_label_idx = label_file(filename)
                true_label_str = LABEL_MAP.get(
                    true_label_idx, "Unknown") if true_label_idx is not None else "Unknown"
                
                # ===Get prediction===
                predicted_class = LABEL_MAP.get(
                    int(prediction), f"Class {int(prediction)}")
                
                # === Enhanced confidence calculation ===
                if logits is not None:
                    # Use new softmax-based confidence
                    probs_np, max_confidence, confidence_level, confidence_emoji = calculate_softmax_confidence(logits)
                    confidence_desc = confidence_level
                else:
                    # Fallback to legacy method
                    logit_margin = abs(
                        (logits_list[0] - logits_list[1]) if logits_list is not None and len(logits_list) >= 2 else 0
                    )
                    confidence_desc, confidence_emoji = get_confidence_description(logit_margin)
                    max_confidence = logit_margin / 10.0  # Normalize for display
                    probs_np = np.array([])

                # Store result in results manager for single file too
                ResultsManager.add_result(
                    filename=filename,
                    model_name=model_choice,
                    prediction=prediction,
                    predicted_class=predicted_class,
                    confidence=max_confidence,
                    logits=logits_list if logits_list else [],
                    ground_truth=true_label_idx if true_label_idx >= 0 else None,
                    processing_time=inference_time,
                    metadata={
                        "confidence_level": confidence_desc,
                        "confidence_emoji": confidence_emoji
                    }
                )

                #===Precompute Stats===
                spec_stats = {
                    "Original Length": len(x_raw) if x_raw is not None else 0,
                    "Resampled Length": TARGET_LEN,
                    "Wavenumber Range": f"{min(x_raw):.1f}-{max(x_raw):.1f} cm⁻¹" if x_raw is not None else "N/A",
                    "Intensity Range": f"{min(y_raw):.1f}-{max(y_raw):.1f} cm⁻¹" if y_raw is not None else "N/A",
                    "Confidence Bucket": confidence_desc,
                }
                model_path = MODEL_CONFIG[model_choice]["path"]
                mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None
                file_hash = (
                    hashlib.md5(open(model_path, 'rb').read()).hexdigest()
                    if os.path.exists(model_path) else "N/A"
                )
                input_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                model_stats =  {
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
                    with st.container():
                        st.markdown(f"""
                        **Sample**: `{filename}`  
                        **Model**: `{model_choice}`  
                        **Processing Time**: `{inference_time:.2f}s`
                        """)
                        st.markdown("<div class='expander-marker expander-success'></div>", unsafe_allow_html=True)
                        with st.expander("Prediction/Ground Truth & Model Confidence Margin", expanded=True):
                            if predicted_class == "Stable (Unweathered)":
                                st.markdown(f"🟢 **Prediction**: {predicted_class}")
                            else:
                                st.markdown(f"🟡 **Prediction**: {predicted_class}")
                            st.markdown(
                                f"**{confidence_emoji} Confidence**: {confidence_desc} ({max_confidence:.1%})")
                            if true_label_idx is not None:
                                if predicted_class == true_label_str:
                                    st.markdown(
                                        f"✅ **Ground Truth**: {true_label_str} - **Correct!**")
                                else:
                                    st.markdown(
                                        f"❌ **Ground Truth**: {true_label_str} - **Incorrect**")
                            else:
                                st.markdown(
                                    "**Ground Truth**: Unknown (filename doesn't follow naming convention)")

                            st.markdown("###### Confidence Overview")
                            if len(probs_np) > 0:
                                confidence_html = create_confidence_progress_html(
                                    probs_np,
                                    labels=["Stable", "Weathered"],
                                    highlight_idx=int(prediction)
                                )
                                st.markdown(confidence_html, unsafe_allow_html=True)
                            else:
                                # Fallback to legacy method
                                render_confidence_progress(
                                    probs if probs is not None else np.array([]),
                                    labels=["Stable", "Weathered"],
                                    highlight_idx=int(prediction),
                                    side_by_side=True, # Set false for stacked <<
                                )

                elif active_tab == "Technical":
                    with st.container():
                        st.markdown("<div class='expander-marker expander-success'></div>", unsafe_allow_html=True)
                        with st.expander("Diagnostics/Technical Info (advanced)", expanded=True):
                            st.markdown("###### Model Output (Logits)")
                            cols = st.columns(2)
                            if logits_list is not None:
                                for i, score in enumerate(logits_list):
                                    label = LABEL_MAP.get(i, f"Class {i}")
                                    cols[i % 2].metric(label, f"{score:.2f}")
                            st.markdown("###### Spectrum Statistics")
                            render_kv_grid(spec_stats, ncols=2)
                            st.markdown("---")
                            st.markdown("###### Model Statistics")
                            render_kv_grid(model_stats, ncols=2)
                            st.markdown("---")
                            st.markdown("###### Debug Log")
                            st.text_area("Logs", "\n".join(st.session_state.get("log_messages", [])), height=110)

                elif active_tab == "Explanation":
                    with st.container():
                            st.markdown("""
                            **🔍 Analysis Process**
                        
                            1. **Data Upload**: Raman spectrum file loaded
                            2. **Preprocessing**: Data parsed and resampled to 500 points
                            3. **AI Inference**: CNN model analyzes spectral patterns
                            4. **Classification**: Binary prediction with confidence scores
                            
                            **🧠 Model Interpretation**
                            
                            The AI model identifies spectral features indicative of:
                            - **Stable polymers**: Well-preserved molecular structure
                            - **Weathered polymers**: Degraded/oxidized molecular bonds
                            
                            **🎯 Applications**
                            
                            - Material longevity assessment
                            - Recycling viability evaluation  
                            - Quality control in manufacturing
                            - Environmental impact studies
                            """)

                    render_time = time.time() - start_render
                    log_message(f"col2 rendered in {render_time:.2f}s, active tab: {active_tab}")

                st.markdown("<div class='expander-marker expander-success'></div>", unsafe_allow_html=True)
                with st.expander("Spectrum Preprocessing Results", expanded=False):
                    # Create and display plot
                    cache_key = hashlib.md5(
                        f"{(x_raw.tobytes() if x_raw is not None else b'')}"
                        f"{(y_raw.tobytes() if y_raw is not None else b'')}"
                        f"{(x_resampled.tobytes() if x_resampled is not None else b'')}"
                        f"{(y_resampled.tobytes() if y_resampled is not None else b'')}".encode()
                    ).hexdigest()
                    spectrum_plot = create_spectrum_plot(x_raw, y_raw, x_resampled, y_resampled, _cache_key=cache_key)
                    st.image(spectrum_plot, caption="Spectrum Preprocessing Results", use_container_width=True)

            else:
                st.error(
                    "❌ Missing spectrum data. Please upload a file and run analysis.")
        else:
            # ===Getting Started===
            st.markdown("""
            ##### Get started by:
            1. Select an AI model in the sidebar
            2. Upload a Raman spectrum file or choose a sample
            3. Click "Run Analysis" to get predictions
            
            ##### Supported formats:
            - Text files (.txt) with wavenumber and intensity columns
            - Space or comma-separated values
            - Any length (automatically resampled to 500 points)
            
            ##### Example applications:
            - 🔬 Research on polymer degradation
            - ♻️ Recycling feasibility assessment
            - 🌱 Sustainability impact studies
            - 🏭 Quality control in manufacturing
            """)


# Run the application
main()
