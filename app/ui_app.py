import os
import sys

# Project base path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
from scripts.preprocess_dataset import resample_spectrum

from io import StringIO
from glob import glob
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt



# Label map and label extractor
label_map = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}

def label_file(filename: str) -> int:
    name = Path(filename).name.lower()
    if name.startswith("sta"):
        return 0
    elif name.startswith("wea"):
        return 1
    else:
        raise ValueError("Unknown label pattern")

# Page configuration
st.set_page_config(
    page_title="Polymer Aging Inference",
    initial_sidebar_state="collapsed",
    page_icon="🔬",
    layout="wide")


# Reset status if nothing is uploaded
if 'uploaded_file' not in st.session_state:
    st.session_state.status_message = "Awaiting input..."
    st.session_state.status_type = "info"

# Title and caption
st.markdown("**🧪 Raman Spectrum Classifier**")
st.caption("AI-driven classification of polymer degradation using Raman spectroscopy.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    Part of the **AIRE 2025 Internship Project**:
    `AI-Driven Polymer Aging Prediction and Classification`

    Uses Raman spectra and deep learning to predict material degradation.

    **Author**: Jaser Hasan  
    **Mentor**: Dr. Sanmukh Kuppannagari  
    [🔗 GitHub](https://github.com/dev-jaser/ai-ml-polymer-aging-prediction)
    """)

# Metadata for visual badges and metrics
model_metadata = {
    "Figure2CNN (Baseline)": {
        "emoji": "🔬",
        "description": "Baseline CNN with standard filters",
        "accuracy": "94.80%",
        "f1": "94.30%"
    },
    "ResNet1D (Advanced)": {
        "emoji": "🧠",
        "description": "Residual CNN with deeper feature learning",
        "accuracy": "96.20%",
        "f1": "95.90%"
    }
}

model_config = {
    "Figure2CNN (Baseline)": {
        "model_class": Figure2CNN,
        "model_path": "outputs/figure2_model.pth"
    },
    "ResNet1D (Advanced)": {
        "model_class": ResNet1D,
        "model_path": "outputs/resnet_model.pth"
    }
}

col1, col2 = st.columns([1.1, 2], gap="large")  # optional for cleaner spacing

try:
    with col1:
        # 📊 Upload + Model Selection
        st.markdown("**📁 Upload Spectrum**")

        # [NEW POSITION] 🧠 Model Selection grounded near data input
        with st.container():
            st.markdown("**🧠 Model Selection**")
            # Enhanced model selector
            model_labels = [
                f"{model_metadata[name]['emoji']} {name}" for name in model_config.keys()
            ]
            selected_label = st.selectbox(
                "Choose model architecture:",
                model_labels,
                key="model_selector"
            )
            model_choice = selected_label.split(" ", 1)[1]
            with st.container():
                meta = model_metadata[model_choice]
                st.markdown(f"""
                **📈 Model Overview**
                *{meta['description']}*

                - **Accuracy**: `{meta['accuracy']}`
                - **F1 Score**: `{meta['f1']}`
                """)

            
            # Model path & check
            # [PATCH] Use selected model config
            MODEL_PATH = model_config[model_choice]["model_path"]
            MODEL_EXISTS = Path(MODEL_PATH).exists()
            TARGET_LEN = 500

            if not MODEL_EXISTS:
                st.error("🚫 Model file not found. Please train the model first.")
        tab1, tab2 = st.tabs(["Upload File", "Use Sample"])
        with tab1:
            uploaded_file = st.file_uploader("Upload Raman `.txt` spectrum", type="txt")
        with tab2:
            sample_files = sorted(glob("app/sample_spectra/*.txt"))
            sample_options = ["-- Select --"] + sample_files
            selected_sample = st.selectbox("Choose a sample:", sample_options)
            if selected_sample != "-- Select --":
                with open(selected_sample, "r", encoding="utf-8") as f:
                    file_contents = f.read()
                uploaded_file = StringIO(file_contents)
                uploaded_file.name = os.path.basename(selected_sample)

        # Capture file in session
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['filename'] = uploaded_file.name
            st.session_state.status_message = f"📁 File `{uploaded_file.name}` loaded. Ready to infer."
            st.session_state.status_type = "success"
            st.session_state.inference_run_once = False

        # Status banner
        st.markdown("**🚦 Pipeline Status**")
        status_msg = st.session_state.get("status_message", "Awaiting input...")
        status_typ = st.session_state.get("status_type", "info")
        if status_typ == "success":
            st.success(status_msg)
        elif status_typ == "error":
            st.error(status_msg)
        else:
            st.info(status_msg)

        # Inference trigger
        if st.button("▶️ Run Inference") and 'uploaded_file' in st.session_state and MODEL_EXISTS:
            spectrum_name = st.session_state['filename']
            uploaded_file = st.session_state['uploaded_file']
            uploaded_file.seek(0)
            raw_data = uploaded_file.read()
            raw_text = raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data

            # Parse spectrum
            x_vals, y_vals = [], []
            for line in raw_text.splitlines():
                parts = line.strip().replace(",", " ").split()
                numbers = [p for p in parts if p.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if len(numbers) >= 2:
                    try:
                        x, y = float(numbers[0]), float(numbers[1])
                        x_vals.append(x)
                        y_vals.append(y)
                    except ValueError:
                        continue

            x_raw = np.array(x_vals)
            y_raw = np.array(y_vals)
            y_resampled = resample_spectrum(x_raw, y_raw, TARGET_LEN)
            st.session_state['x_raw'] = x_raw
            st.session_state['y_raw'] = y_raw
            st.session_state['y_resampled'] = y_resampled

            # ---

            # Update banner for inference
            st.session_state.status_message = f"🔍 Inference running on: `{spectrum_name}`"
            st.session_state.status_type = "info"
            st.session_state.inference_run_once = True


    # Inference
    
    with col2:
        if st.session_state.get("inference_run_once", False):
            # Plot: Raw + Resampled
            x_raw = st.session_state.get("x_raw", None)
            y_raw = st.session_state.get("y_raw", None)
            y_resampled = st.session_state.get("y_resampled", None)
            if x_raw is not None and y_raw is not None and y_resampled is not None:
                st.subheader("📉 Spectrum Overview")
                st.write("")  # Spacer line for visual breathing room
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                from PIL import Image
                import io 

                # Create smaller figure
                fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), dpi=150)
                ax[0].plot(x_raw, y_raw, label="Raw", color="dimgray")
                ax[0].set_title("Raw Input")
                ax[0].set_xlabel("Wavenumber")
                ax[0].set_ylabel("Intensity")
                ax[0].legend()

                ax[1].plot(np.linspace(min(x_raw), max(x_raw), TARGET_LEN), y_resampled, label="Resampled", color="steelblue")
                ax[1].set_title("Resampled")
                ax[1].set_xlabel("Wavenumber")
                ax[1].set_ylabel("Intensity")
                ax[1].legend()

                plt.tight_layout()

                # Render to image buffer
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)

                # Display fixed-size image
                st.image(Image.open(buf), caption="Raw vs. Resampled Spectrum", width=880)


            st.session_state['x_raw'] = x_raw
            st.session_state['y_raw'] = y_raw

            y_resampled = st.session_state.get('y_resampled', None)
            if y_resampled is None:
                st.error("❌ Error: Missing resampled spectrum. Please upload and run inference.")
                st.stop()
            input_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # [PATCH] Load selected model
            ModelClass = model_config[model_choice]["model_class"]
            model = ModelClass(input_length=TARGET_LEN)

            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
            model.eval()
            with torch.no_grad():
                logits = model(input_tensor)
                prediction = torch.argmax(logits, dim=1).item()
                logits_list = logits.numpy().tolist()[0]
            try:
                true_label_idx = label_file(spectrum_name)
                true_label_str = label_map[true_label_idx]
            except Exception:
                true_label_idx = None
                true_label_str = "Unknown"
            predicted_class = label_map.get(prediction, f"Class {prediction}")

            import torch.nn.functional as F
            probs = F.softmax(torch.tensor(logits_list), dim=0).numpy()

            
            # 🔬 Redesigned Prediction Block – Distinguishing Model vs Classification
            tab_summary, tab_logits, tab_system, tab_explainer = st.tabs([
            "🧠 Model Summary", "🔬 Logits", "⚙️ System Info", "📘 Explanation"])

            
            with tab_summary:
                st.markdown("### 🧠 AI Model Decision Summary")
                st.markdown(f"""
                **📃 File Analyzed:** `{spectrum_name}`
                
                **🛠️ Model Chosen:** `{model_choice}`
                """)
                st.markdown("**🔍 Internal Model Prediction**")
                st.write(f"The model believes this sample best matches: **`{predicted_class}`**")
                if true_label_idx is not None:
                    st.caption(f"Ground Truth Label: `{true_label_str}`")
            
                logit_margin = abs(logits_list[0] - logits_list[1])
                if logit_margin > 1000:
                    strength_desc = "VERY STRONG"
                elif logit_margin > 250:
                    strength_desc = "STRONG"
                elif logit_margin > 100:
                    strength_desc = "MODERATE"
                else:
                    strength_desc = "UNCERTAIN"
            
                st.markdown("🧪 Final Classification")
                st.markdown("**📊 Model Confidence Estimate**")
                st.write(f"**Decision Confidence:** `{strength_desc}` (margin = `{logit_margin:.1f}`)")
                st.success(f"This spectrum is classified as: **`{predicted_class}`**")
            
            with tab_logits:
                st.markdown("🔬 View Internal Model Output (Logits)")
                st.markdown("""
                    These are the **raw output scores** from the model before making a final prediction.
            
                    Higher scores indicate stronger alignment between the input spectrum and that class.
                """)
                st.json({
                    label_map.get(i, f"Class {i}"): float(score)
                    for i, score in enumerate(logits_list)
                })
            
            with tab_system:
                st.markdown("⚙️ View System Info")
                st.json({
                    "Model Chosen": model_choice,
                    "Spectrum Length": TARGET_LEN,
                    "Processing Steps": "Raw Signal → Resampled → Inference"
                })
            
            with tab_explainer:
                st.markdown("📘 What Just Happened?")
                st.markdown("""
                **🔍 Process Overview**
                1. 🗂 A Raman spectrum was uploaded  
                2. 📏 Data was standardized  
                3. 🤖 AI model analyzed the spectrum  
                4. 📌 A classification was made  
            
                ---
                **🧠 How the Model Operates**
            
                Trained on known polymer conditions, the system detects spectral patterns  
                indicative of stable or weathered polymers.
            
                ---
                **✅ Why It Matters**
            
                Enables:
                - 🔬 Material longevity research  
                - 🔁 Recycling assessments  
                - 🌱 Sustainability decisions  
                """)
            
except (ValueError, TypeError, RuntimeError) as e:
        st.error(f"❌ Inference error: {e}")