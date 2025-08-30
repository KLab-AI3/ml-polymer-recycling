"""Streamlit main entrance; modularized for clarity"""

import streamlit as st

from modules.callbacks import init_session_state

from modules.ui_components import (
    render_sidebar,
    render_results_column,
    render_input_column,
    load_css,
)


# --- Page Setup (Called only ONCE) ---
st.set_page_config(
    page_title="ML Polymer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": "https://github.com/KLab-AI3/ml-polymer-recycling"},
)


def main():
    """Modularized main content to other scripts to clean the main app"""
    load_css("static/style.css")
    init_session_state()

    # Render UI components
    render_sidebar()

    col1, col2 = st.columns([1, 1.35], gap="small")
    with col1:
        render_input_column()
    with col2:
        render_results_column()


if __name__ == "__main__":
    main()
