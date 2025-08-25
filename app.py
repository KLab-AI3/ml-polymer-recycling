"""
AI-Driven Polymer Aging Prediction and Classification
Hugging Face Spaces Deployment
This is an adapted version of the Streamlit app optimized for Hugging Face Spaces deployment.
It maintains all the functionality of the original app while being self-contained and cloud-ready.
"""

# BUILD_LABEL = "proof-2025-08-24-01"
# import os, streamlit as st, sys
# st.sidebar.caption(
#     f"Build: {BUILD_LABEL} | __file__: {__file__} | cwd: {os.getcwd()} | py: {sys.version.split()[0]}"
# )

import os
import sys
from pathlib import Path

# Ensure 'utils' directory is in the Python path
utils_path = Path(__file__).resolve().parent / "utils"
if utils_path.is_dir() and str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))
import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # ensure headless rendering in Spaces
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path
import time
import gc
import hashlib
import logging

# Import local modules
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
# Prefer canonical script; fallback to local utils for HF hard-copy scenario
try:
    from scripts.preprocess_dataset import resample_spectrum
except ImportError:
    from utils.preprocessing import resample_spectrum

# Configuration
st.set_page_config(
    page_title="ML Polymer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stabilize tab panel height on HF Spaces to prevent visible column jitter.
# This sets a minimum height for the content area under the tab headers.
st.markdown("""
<style>
/*  Tabs content area: the sibling after the tablist */
    div[data-testid="stTabs"] > div[role="tablist"] + div { min-height: 420px;}
</style>
""", unsafe_allow_html=True)

# Constants
TARGET_LEN = 500
SAMPLE_DATA_DIR = "sample_data"
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
        "emoji": "🔬",
        "description": "Baseline CNN with standard filters",
        "accuracy": "94.80%",
        "f1": "94.30%"
    },
    "ResNet1D (Advanced)": {
        "class": ResNet1D,
        "path": f"{MODEL_WEIGHTS_DIR}/resnet_model.pth",
        "emoji": "🧠", 
        "description": "Residual CNN with deeper feature learning",
        "accuracy": "96.20%",
        "f1": "95.90%"
    }
}

# Label mapping
LABEL_MAP = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}

# Utility functions
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
    except Exception as e:
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
            model.eval()
            return model, True
        else:
            return model, False

    except Exception as e:
        st.error(f"❌ Error loading model {model_name}: {str(e)}")
        return None, False

def cleanup_memory():
    """Clean up memory after inference"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            numbers = [p for p in parts if p.replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).isdigit()]

            if len(numbers) >= 2:
                x, y = float(numbers[0]), float(numbers[1])
                x_vals.append(x)
                y_vals.append(y)

        except ValueError:
            # Skip problematic lines but don't fail completely
            continue

    if len(x_vals) < 10:  # Minimum reasonable spectrum length
        raise ValueError(f"Insufficient data points: {len(x_vals)}. Need at least 10 points.")

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
        raise ValueError(f"Invalid wavenumber range: {min(x)} - {max(x)}. Expected ~400-4000 cm⁻¹ with span >100")

    return x, y

def create_spectrum_plot(x_raw, y_raw, y_resampled):
    """Create spectrum visualization plot"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

    # Raw spectrum
    ax[0].plot(x_raw, y_raw, label="Raw", color="dimgray", linewidth=1)
    ax[0].set_title("Raw Input Spectrum")
    ax[0].set_xlabel("Wavenumber (cm⁻¹)")
    ax[0].set_ylabel("Intensity")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Resampled spectrum  
    x_resampled = np.linspace(min(x_raw), max(x_raw), TARGET_LEN)
    ax[1].plot(x_resampled, y_resampled, label="Resampled", color="steelblue", linewidth=1)
    ax[1].set_title(f"Resampled ({TARGET_LEN} points)")
    ax[1].set_xlabel("Wavenumber (cm⁻¹)")
    ax[1].set_ylabel("Intensity")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)  # Prevent memory leaks

    return Image.open(buf)

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

def init_session_state():
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
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def log_message(msg: str):
    """Append a timestamped line to the in-app log, creating the buffer if needed."""
    if "log_messages" not in st.session_state or st.session_state["log_messages"] is None:
        st.session_state["log_messages"] = []
    st.session_state["log_messages"].append(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    )

def trigger_run():
    """Set a flag so we can detect button press reliably across reruns"""
    st.session_state['run_requested'] = True

def on_upload_change():
    """Read uploaded file once and persist as text."""
    up = st.session_state.get("upload_txt")  # the uploader's key
    if not up:
        return
    raw = up.read()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    st.session_state["input_text"] = text
    st.session_state["filename"] = getattr(up, "name", "uploaded.txt")
    st.session_state["input_source"] = "upload"
    st.session_state["status_message"] = f"📁 File '{st.session_state['filename']}' ready for analysis"
    st.session_state["status_type"] = "success"

def on_sample_change():
    """Read selected sample once and persist as text."""
    sel = st.session_state.get("sample_select", "-- Select Sample --")
    if sel == "-- Select Sample --":
        # Do nothing; leave current input intact (prevents clobbering uploads)
        return
    try:
        text = (Path(SAMPLE_DATA_DIR) / sel).read_text(encoding="utf-8")
        st.session_state["input_text"] = text
        st.session_state["filename"] = sel
        st.session_state["input_source"] = "sample"
        st.session_state["status_message"] = f"📁 Sample '{sel}' ready for analysis"
        st.session_state["status_type"] = "success"
    except Exception as e:
        st.session_state["status_message"] = f"❌ Error loading sample: {e}"
        st.session_state["status_type"] = "error"

def on_input_mode_change():
    if st.session_state["input_mode"] == "Upload File":
        # reset sample when switching to Upload
        st.session_state["sample_select"] = "-- Select Sample --"


# Main app
def main():
    init_session_state()
    # Header
    st.title("🔬 AI-Driven Polymer Classification")
    st.markdown("**Predict polymer degradation states using Raman spectroscopy and deep learning**")
    st.info(
        "⚠️ **Prototype Notice:** v0.1 Raman-only. "
        "Multi-model CNN evaluation in progress. "
        "FTIR support planned.",
        icon="⚡"
    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.sidebar.markdown("""
        AI-Driven Polymer Aging Prediction and Classification

        🎯 **Purpose**: Classify polymer degradation using AI  
        📊 **Input**: Raman spectroscopy `.txt` files  
        🧠 **Models**: CNN architectures for binary classification  
        💾 **Current**: Figure2CNN (baseline)  
        📈 **Next**: More trained CNNs in evaluation pipeline

        ---

        **Team**  
        👨‍🏫 Dr. Sanmukh Kuppannagari (Mentor)  
        👨‍🏫 Dr. Metin Karailyan (Mentor)  
        👨‍💻 Jaser Hasan (Author)

        ---

        **Links**  
        🔗 [Live HF Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)  
        📂 [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)

        ---

        **Model Credit**  
        Baseline model inspired by *Figure 2 CNN* from:  
        Neo, E.R.K., Low, J.S.C., Goodship, V., Debattista, K. (2023).  
        *Deep learning for chemometric analysis of plastic spectral data from infrared and Raman databases*.  
        _Resources, Conservation & Recycling_, **188**, 106718.  
        [https://doi.org/10.1016/j.resconrec.2022.106718](https://doi.org/10.1016/j.resconrec.2022.106718)
        """)

        st.markdown("---")

        # Model selection
        st.subheader("🧠 Model Selection")
        model_labels = [f"{MODEL_CONFIG[name]['emoji']} {name}" for name in MODEL_CONFIG.keys()]
        selected_label = st.selectbox("Choose AI model:", model_labels)
        model_choice = selected_label.split(" ", 1)[1]

        # Model info
        config = MODEL_CONFIG[model_choice]
        st.markdown(f"""
        **📈 {config['emoji']} Model Details**
        
        *{config['description']}*
        
        - **Accuracy**: `{config['accuracy']}`
        - **F1 Score**: `{config['f1']}`
        """)

    # Main content area
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.subheader("📁 Data Input")

        mode = st.radio(
            "Input mode",
            ["Upload File", "Sample Data"],
            key="input_mode",
            horizontal=True,
            on_change=on_input_mode_change
        )

        # ---- Upload tab ----
        if mode == "Upload File":
            up = st.file_uploader(
                "Upload Raman spectrum (.txt)",
                type="txt",
                help="Upload a text file with wavenumber and intensity columns",
                key="upload_txt",
                on_change=on_upload_change,  # <-- critical
            )
            if up:
                st.success(f"✅ Loaded: {up.name}")

        # ---- Sample tab ----
        else:
            sample_files = get_sample_files()
            if sample_files:
                options = ["-- Select Sample --"] + [p.name for p in sample_files]
                sel = st.selectbox(
                    "Choose sample spectrum:",
                    options,
                    key="sample_select",
                    on_change=on_sample_change,  # <-- critical
                )
                if sel != "-- Select Sample --":
                    st.success(f"✅ Loaded sample: {sel}")
            else:
                st.info("No sample data available")

        # ---- Status box ----
        st.subheader("🚦 Status")
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

        # Ready to run if we have text and a model
        inference_ready = bool(st.session_state.get("input_text")) and (model is not None)

        # ---- Run Analysis (form submit batches state + submit atomically) ----
        with st.form("analysis_form", clear_on_submit=False):
            submitted = st.form_submit_button(
                "▶️ Run Analysis",
                type="primary",
                disabled=not inference_ready,
            )

        if submitted and inference_ready:
            try:
                raw_text = st.session_state["input_text"]
                filename = st.session_state.get("filename") or "unknown.txt"

                # Parse
                with st.spinner("Parsing spectrum data..."):
                    x_raw, y_raw = parse_spectrum_data(raw_text)

                # Resample
                with st.spinner("Resampling spectrum..."):
                    y_resampled = resample_spectrum(x_raw, y_raw, TARGET_LEN)

                # Persist results (drives right column)
                st.session_state["x_raw"] = x_raw
                st.session_state["y_raw"] = y_raw
                st.session_state["y_resampled"] = y_resampled
                st.session_state["inference_run_once"] = True
                st.session_state["status_message"] = f"🔍 Analysis completed for: {filename}"
                st.session_state["status_type"] = "success"

                st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.session_state["status_message"] = f"❌ Error: {e}"
                st.session_state["status_type"] = "error"


    # Results column
    with col2:
        if st.session_state.get("inference_run_once", False):
            st.subheader("📊 Analysis Results")

            # Get data from session state
            x_raw = st.session_state.get('x_raw')
            y_raw = st.session_state.get('y_raw')
            y_resampled = st.session_state.get('y_resampled')
            filename = st.session_state.get('filename', 'Unknown')

            if all(v is not None for v in [x_raw, y_raw, y_resampled]):

                # Create and display plot
                try:
                    spectrum_plot = create_spectrum_plot(x_raw, y_raw, y_resampled)
                    st.image(spectrum_plot, caption="Spectrum Preprocessing Results", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate plot: {e}")
                    log_message(f"Plot generation error: {e}")

                # Run inference
                try:
                    with st.spinner("Running AI inference..."):
                        start_time = time.time()

                        # Prepare input tensor
                        input_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                        # Run inference
                        model.eval()
                        with torch.no_grad():
                            if model is None:
                                raise ValueError("Model is not loaded. Please check the model configuration or weights.")
                            logits = model(input_tensor)
                            prediction = torch.argmax(logits, dim=1).item()
                            logits_list = logits.detach().numpy().tolist()[0]

                        inference_time = time.time() - start_time
                        log_message(f"Inference completed in {inference_time:.2f}s, prediction: {prediction}")

                        # Clean up memory
                        cleanup_memory()

                    # Get ground truth if available
                    true_label_idx = label_file(filename)
                    true_label_str = LABEL_MAP.get(true_label_idx, "Unknown") if true_label_idx is not None else "Unknown"

                    # Get prediction
                    predicted_class = LABEL_MAP.get(int(prediction), f"Class {int(prediction)}")

                    # Calculate confidence metrics
                    logit_margin = abs(logits_list[0] - logits_list[1]) if len(logits_list) >= 2 else 0
                    confidence_desc, confidence_emoji = get_confidence_description(logit_margin)

                    # Display results
                    st.markdown("### 🎯 Prediction Results")

                    # Main prediction
                    st.markdown(f"""
                    **🔬 Sample**: `{filename}`  
                    **🧠 Model**: `{model_choice}`  
                    **⏱️ Processing Time**: `{inference_time:.2f}s`
                    """)

                    # Prediction box
                    if predicted_class == "Stable (Unweathered)":
                        st.success(f"🟢 **Prediction**: {predicted_class}")
                    else:
                        st.warning(f"🟡 **Prediction**: {predicted_class}")

                    # Confidence
                    st.markdown(f"**{confidence_emoji} Confidence**: {confidence_desc} (margin: {logit_margin:.1f})")

                    # Ground truth comparison
                    if true_label_idx is not None:
                        if predicted_class == true_label_str:
                            st.success(f"✅ **Ground Truth**: {true_label_str} - **Correct!**")
                        else:
                            st.error(f"❌ **Ground Truth**: {true_label_str} - **Incorrect**")
                    else:
                        st.info("ℹ️ **Ground Truth**: Unknown (filename doesn't follow naming convention)")

                    # Detailed results tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Details", "🔬 Technical", "📘 Explanation"])

                    with tab1:
                        st.markdown("**Model Output (Logits)**")
                        for i, score in enumerate(logits_list):
                            label = LABEL_MAP.get(i, f"Class {i}")
                            st.metric(label, f"{score:.2f}")

                        st.markdown("**Spectrum Statistics**")
                        st.json({
                            "Original Length": len(x_raw) if x_raw is not None else 0,
                            "Resampled Length": TARGET_LEN,
                            "Wavenumber Range": f"{min(x_raw):.1f} - {max(x_raw):.1f} cm⁻¹" if x_raw is not None else "N/A",
                            "Intensity Range": f"{min(y_raw):.1f} - {max(y_raw):.1f}" if y_raw is not None else "N/A",
                            "Model Confidence": confidence_desc
                        })

                    with tab2:
                        st.markdown("**Technical Information**")
                        model_path = MODEL_CONFIG[model_choice]["path"]
                        mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else "N/A"
                        file_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest() if os.path.exists(model_path) else "N/A"
                        st.json({
                            "Model Architecture": model_choice,
                            "Model Path": model_path,
                            "Weights Last Modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)) if mtime != "N/A" else "N/A",
                            "Weights Hash": file_hash,
                            "Input Shape": list(input_tensor.shape),
                            "Output Shape": list(logits.shape),
                            "Inference Time": f"{inference_time:.3f}s",
                            "Device": "CPU",
                            "Model Loaded": model_loaded
                        })

                        if not model_loaded:
                            st.warning("⚠️ Demo mode: Using randomly initialized weights")

                        # Debug log
                        st.markdown("**Debug Log**")
                        st.text_area("Logs", "\n".join(st.session_state.get("log_messages", [])), height=200)

                    with tab3:
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

                except Exception as e:
                    st.error(f"❌ Inference failed: {str(e)}")
                    log_message(f"Inference error: {str(e)}")

            else:
                st.error("❌ Missing spectrum data. Please upload a file and run analysis.")
        else:
            # Welcome message
            st.markdown("""
            ### 👋 Welcome to AI Polymer Classification
            
            **Get started by:**
            1. 🧠 Select an AI model in the sidebar
            2. 📁 Upload a Raman spectrum file or choose a sample
            3. ▶️ Click "Run Analysis" to get predictions
            
            **Supported formats:**
            - Text files (.txt) with wavenumber and intensity columns
            - Space or comma-separated values
            - Any length (automatically resampled to 500 points)
            
            **Example applications:**
            - 🔬 Research on polymer degradation
            - ♻️ Recycling feasibility assessment
            - 🌱 Sustainability impact studies
            - 🏭 Quality control in manufacturing
            """)

# Run the application
main()