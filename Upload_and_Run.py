# In Upload_and_Run.py
import streamlit as st

from modules.callbacks import init_session_state

from modules.ui_components import (
    render_sidebar,
    render_results_column,
    render_input_column,
    render_comparison_tab,
    render_performance_tab,
    load_css,
)

from modules.training_ui import render_training_tab

from utils.image_processing import render_image_upload_interface

st.set_page_config(
    page_title="ML Polymer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)


def main():
    """Modularized main content to other scripts to clean the main app"""
    load_css("static/style.css")
    init_session_state()

    render_sidebar()

    # Create main tabs for different analysis modes
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Standard Analysis",
            "Model Comparison",
            "Model Training",
            "Image Analysis",
            "Performance Tracking",
        ]
    )

    with tab1:
        # Standard single-model analysis
        col1, col2 = st.columns([1, 1.35], gap="small")
        with col1:
            render_input_column()
        with col2:
            render_results_column()

    with tab2:
        # Multi-model comparison interface
        render_comparison_tab()

    with tab3:
        # Model training interface
        render_training_tab()

    with tab4:
        # Image analysis interface
        render_image_upload_interface()

    with tab5:
        # Performance tracking interface
        render_performance_tab()


if __name__ == "__main__":
    main()
