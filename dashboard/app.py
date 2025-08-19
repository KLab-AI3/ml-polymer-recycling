"""
Streamlit UI for ml-polymer-recycling — Step 3a: Multi-Modality Interface Skeleton
- Introduces top-level modality selector (Raman | Image | FTIR)
- Modular routing for each modality with scoped navigation
- Only Raman pages are active (Model Management, Inference)
- Image/FTIR pages show 'Coming Soon' placeholders
- All routing and session state preserved cleanly
"""

import streamlit as st
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ML Polymer Recycling",
    page_icon="🧪",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Ready."
    if "status_type" not in st.session_state:
        st.session_state.status_type = "ok"
    if "modality" not in st.session_state:
        st.session_state.modality = "Raman"

# --- STATUS BANNER ---
def display_status():
    style_map = {
        "ok": ("#e8f5e9", "#2e7d32"),
        "warn": ("#fff8e1", "#f9a825"),
        "err": ("#ffebee", "#c62828")
    }
    bg_color, text_color = style_map.get(st.session_state.status_type, ("#f0f0f0", "#333"))
    st.markdown(f"""
        <div style='background-color:{bg_color}; padding:0.75em 1em; border-radius:8px; color:{text_color};'>
            <b>Status:</b> {st.session_state.status_message}
        </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
def display_sidebar():
    with st.sidebar:
        st.header("🧪 ML Polymer Dashboard")

        modality = st.radio("Modality", ["Raman", "Image", "FTIR"])
        st.session_state.modality = modality

        if modality == "Raman":
            page = st.radio("Raman Pages", ["Dashboard", "Model Management", "Inference"])
        elif modality == "Image":
            page = st.radio("Image Pages", ["Model Management", "Inference"])
        elif modality == "FTIR":
            page = st.radio("FTIR Pages", ["Model Management", "Inference"])

        return modality, page

# --- HEADER BAR ---
def display_header(title: str):
    st.title(title)
    display_status()
    st.markdown("---")

# --- MODEL DISCOVERY (Raman only for now) ---
def discover_models(outputs_dir="outputs"):
    out = []
    root = Path(outputs_dir)
    if not root.exists():
        return []
    for p in sorted(root.rglob("*.pth")):
        out.append(p)
    return out

# --- RAMAN PAGES ---
def raman_dashboard():
    display_header("Raman Dashboard")
    st.write("This will house future metrics, model count, and version history.")

def raman_model_management():
    display_header("Raman Model Management")
    models = discover_models()
    if not models:
        st.info("No model weights found in outputs/. Place .pth files there to make them discoverable.")
    else:
        st.markdown(f"**Discovered {len(models)} model weight file(s):**")
        for m in models:
            st.code(str(m), language="text")

def raman_inference():
    display_header("Raman Inference")
    st.write("This will house the real-time spectrum upload and model prediction interface.")

# --- IMAGE + FTIR PLACEHOLDERS ---
def image_model_management():
    display_header("Image Model Management")
    st.info("Image-based model integration is coming soon.")

def image_inference():
    display_header("Image Inference")
    st.info("This page will allow batch image upload and multi-model prediction.")

def ftir_model_management():
    display_header("FTIR Model Management")
    st.info("FTIR model support is planned and will be developed after clarification with Dr. K.")

def ftir_inference():
    display_header("FTIR Inference")
    st.info("FTIR input and prediction support will be added in a future phase.")

# --- MAIN ENTRY POINT ---
def main():
    init_session_state()
    modality, page = display_sidebar()

    if modality == "Raman":
        if page == "Dashboard":
            raman_dashboard()
        elif page == "Model Management":
            raman_model_management()
        elif page == "Inference":
            raman_inference()

    elif modality == "Image":
        if page == "Model Management":
            image_model_management()
        elif page == "Inference":
            image_inference()

    elif modality == "FTIR":
        if page == "Model Management":
            ftir_model_management()
        elif page == "Inference":
            ftir_inference()

if __name__ == "__main__":
    main()
