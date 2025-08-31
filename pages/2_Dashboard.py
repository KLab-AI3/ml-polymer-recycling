# In pages/2_Dashboard.py

import streamlit as st
from utils.results_manager import ResultsManager
from modules.analyzer import BatchAnalysis  # Adjusted import path

st.set_page_config(page_title="Analysis Dashboard", layout="wide")

# --- INITIALIZE SESSION STATE FOR THIS PAGE ---
if "cm_filter_active" not in st.session_state:
    st.session_state["cm_filter_active"] = False
if "selected_spectrum_file" not in st.session_state:
    st.session_state["selected_spectrum_file"] = (
        None  # Stores the filename of the clicked row
    )
# --- END INITIALIZATION ---

st.title("Interactive Analysis Dashboard")
st.markdown(
    "Dive deeper into your batch results. Use the charts below to analyze model performance."
)
st.divider()

# --- Initialize session state for CM filter ---
if "cm_filter_active" not in st.session_state:
    st.session_state["cm_filter_active"] = False


# Get the results from the session state
results_df = ResultsManager.get_results_dataframe()

# Create an instance of our analyzer with the results
analyzer = BatchAnalysis(results_df)

# Render the entire dashboard with one line!
analyzer.render()
