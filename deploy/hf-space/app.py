"""
AI-Driven Polymer Aging Prediction and Classification
Hugging Face Spaces Deployment
This is an adapted version of the Streamlit app optimized for Hugging Face Spaces deployment.
It maintains all the functionality of the original app while being self-contained and cloud-ready.
"""

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
from io import StringIO

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

        # Load weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        if model is not None:
            model.eval()
        else:
            raise ValueError("Model is not loaded. Please check the model configuration or weights.")

        return model, True

    except (FileNotFoundError, ValueError, RuntimeError) as e:
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
    """Parse spectrum data from text with robust error handling"""
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

    return np.array(x_vals), np.array(y_vals)

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

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'status_message': "Ready to analyze polymer spectra 🔬",
        'status_type': "info",
        'uploaded_file': None,
        'filename': None,
        'inference_run_once': False,
        'x_raw': None,
        'y_raw': None,
        'y_resampled': None
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_app():
    """Hard reset: clear cache and session state, then rerun."""
    try:
        st.cache_resource.clear()
    except RuntimeError:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Main app
def main():
    init_session_state()

    # Header
    st.title("🔬 AI-Driven Polymer Classification")
    st.markdown("**Predict polymer degradation states using Raman spectroscopy and deep learning**")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        AI-Driven Polymer Aging Prediction and Classification
        
        🎯 **Purpose**: Classify polymer degradation using ML  
        📊 **Input**: Raman/FTIR spectroscopy data  
        🧠 **Models**: CNN architectures for binary classification  
        
        **Team**: 
        - **Mentor**: Dr. Sanmukh Kuppannagari
        - **Mentor**: Dr. Metin Karailyan
        - **Author**: Jaser Hasan 
        
        🔗 [GitHub Repository](https://github.com/KLab-AI3/ml-polymer-recycling)
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

        # Weights source indicator (resolved path + existence + size)
        try:
            resolved_path = os.path.abspath(config["path"])
            exists = os.path.exists(resolved_path)
            size_mb = (os.path.getsize(resolved_path) / (1024 * 1024)) if exists else None
            env_src = os.getenv("WEIGHTS_DIR")
            with st.expander("Weights source", expanded=False):
                st.code(resolved_path, language="bash")
                st.write(
                    ("✅ **Found**" + (f" • {size_mb:.2f} MB" if size_mb is not None else ""))
                    if exists else "⚠️ **Missing**"
                )
                if env_src:
                    st.caption(f"WEIGHTS_DIR env: `{env_src}`")
                else:
                    st.caption(f"WEIGHTS_DIR env not set; using fallback directory `{MODEL_WEIGHTS_DIR}`")
        except (FileNotFoundError, PermissionError, OSError) as _e:
            st.caption("Weights source: (could not resolve path)")

        # Reset session controls
        st.markdown("---")
        if st.button("↩️ Reset Session", help="Clear caches and session state, then rerun"):
            reset_app()

    # Main content area
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.subheader("📁 Data Input")

        # File upload tabs
        tab1, tab2 = st.tabs(["📤 Upload File", "🧪 Sample Data"])

        uploaded_file = None

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload Raman spectrum (.txt)", 
                type="txt",
                help="Upload a text file with wavenumber and intensity columns"
            )

            if uploaded_file:
                st.success(f"✅ Loaded: {uploaded_file.name}")

        with tab2:
            sample_files = get_sample_files()
            if sample_files:
                sample_options = ["-- Select Sample --"] + [f.name for f in sample_files]
                selected_sample = st.selectbox("Choose sample spectrum:", sample_options)

                if selected_sample != "-- Select Sample --":
                    selected_path = Path(SAMPLE_DATA_DIR) / selected_sample
                    try:
                        with open(selected_path, "r", encoding="utf-8") as f:
                            file_contents = f.read()
                        uploaded_file = StringIO(file_contents)
                        uploaded_file.name = selected_sample
                        st.success(f"✅ Loaded sample: {selected_sample}")
                    except (FileNotFoundError, IOError) as e:
                        st.error(f"Error loading sample: {e}")
            else:
                st.info("No sample data available")

        # Update session state
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['filename'] = uploaded_file.name
            st.session_state['status_message'] = f"📁 File '{uploaded_file.name}' ready for analysis"
            st.session_state['status_type'] = "success"

        # Status display
        st.subheader("🚦 Status")
        status_msg = st.session_state.get("status_message", "Ready")
        status_type = st.session_state.get("status_type", "info")

        if status_type == "success":
            st.success(status_msg)
        elif status_type == "error":
            st.error(status_msg)
        else:
            st.info(status_msg)

        # Load model
        model, model_loaded = load_model(model_choice)

        # Inference button
        inference_ready = (
            'uploaded_file' in st.session_state and 
            st.session_state['uploaded_file'] is not None and
            model is not None
        )

        if not model_loaded:
            st.warning("⚠️ Model weights not available - using demo mode")

        if st.button("▶️ Run Analysis", disabled=not inference_ready, type="primary"):
            if inference_ready:
                try:
                    # Get file data
                    uploaded_file = st.session_state['uploaded_file']
                    filename = st.session_state['filename']

                    # Read file content
                    uploaded_file.seek(0)
                    raw_data = uploaded_file.read()
                    raw_text = raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data

                    # Parse spectrum
                    with st.spinner("Parsing spectrum data..."):
                        x_raw, y_raw = parse_spectrum_data(raw_text)

                    # Resample spectrum
                    with st.spinner("Resampling spectrum..."):
                        y_resampled = resample_spectrum(x_raw, y_raw, TARGET_LEN)

                    # Store in session state
                    st.session_state['x_raw'] = x_raw
                    st.session_state['y_raw'] = y_raw  
                    st.session_state['y_resampled'] = y_resampled
                    st.session_state['inference_run_once'] = True
                    st.session_state['status_message'] = f"🔍 Analysis completed for: {filename}"
                    st.session_state['status_type'] = "success"

                    st.rerun()

                except (ValueError, IOError) as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.session_state['status_message'] = f"❌ Error: {str(e)}"
                    st.session_state['status_type'] = "error"

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
                    st.image(spectrum_plot, caption="Spectrum Preprocessing Results", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not generate plot: {e}")

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
                        st.json({
                            "Model Architecture": model_choice,
                            "Input Shape": list(input_tensor.shape),
                            "Output Shape": list(logits.shape),
                            "Inference Time": f"{inference_time:.3f}s",
                            "Device": "CPU",
                            "Model Loaded": model_loaded
                        })

                        if not model_loaded:
                            st.warning("⚠️ Demo mode: Using randomly initialized weights")

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

                except (RuntimeError, ValueError) as e:
                    st.error(f"❌ Inference failed: {str(e)}")

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
