# In modules/analyzer.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import contextmanager  # Correctly imported for use with @contextmanager

from config import LABEL_MAP  # Assuming LABEL_MAP is correctly defined in config.py

# --- ADD THESE IMPORTS AT THE TOP OF THE FILE ---
from utils.results_manager import ResultsManager
from modules.ui_components import create_spectrum_plot
import hashlib


# --- NEW HELPER FUNCTION for theme-aware plots ---
@contextmanager
def theme_aware_plot():
    """A context manager to make Matplotlib plots respect Streamlit's theme."""
    # Get the current theme from Streamlit's config with error handling
    try:
        theme_opts = st.get_option("theme") or {}
    except RuntimeError:
        # Fallback to empty dict if theme config is not available
        theme_opts = {}

    text_color = theme_opts.get("textColor", "#000000")
    bg_color = theme_opts.get("backgroundColor", "#FFFFFF")

    # Set Matplotlib's rcParams to match the theme
    with plt.rc_context(
        {
            "figure.facecolor": bg_color,
            "axes.facecolor": bg_color,
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "grid.color": text_color,
            "axes.edgecolor": text_color,
        }
    ):
        yield


# --- END HELPER FUNCTION ---


class BatchAnalysis:
    def __init__(self, df: pd.DataFrame):
        """Initializes the analysis object with the results DataFrame."""
        self.df = df
        if self.df.empty:
            return

        self.total_files = len(self.df)
        self.has_ground_truth = (
            "Ground Truth" in self.df.columns
            and not self.df["Ground Truth"].isnull().all()
        )
        self._prepare_data()
        self.kpis = self._calculate_kpis()

    def _prepare_data(self):
        """Ensures data types are correct for analysis."""
        self.df["Confidence"] = pd.to_numeric(self.df["Confidence"], errors="coerce")
        if self.has_ground_truth:
            self.df["Ground Truth"] = pd.to_numeric(
                self.df["Ground Truth"], errors="coerce"
            )

    def _calculate_kpis(self) -> dict:
        """A private method to compute all the key performance indicators."""
        stable_count = self.df[
            self.df["Predicted Class"] == "Stable (Unweathered)"
        ].shape[0]
        accuracy = "N/A"
        if self.has_ground_truth:
            valid_gt = self.df.dropna(subset=["Ground Truth", "Prediction"])
            accuracy = (valid_gt["Prediction"] == valid_gt["Ground Truth"]).mean()

        return {
            "Total Files": self.total_files,
            "Avg. Confidence": self.df["Confidence"].mean(),
            "Stable/Weathered": f"{stable_count}/{self.total_files - stable_count}",
            "Accuracy": accuracy,
        }

    def render_kpis(self):
        """Renders the top-level KPI metrics."""
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Files", f"{self.kpis['Total Files']}")
        kpi_cols[1].metric("Avg. Confidence", f"{self.kpis['Avg. Confidence']:.3f}")
        kpi_cols[2].metric("Stable/Weathered", self.kpis["Stable/Weathered"])
        kpi_cols[3].metric(
            "Accuracy",
            (
                f"{self.kpis['Accuracy']:.3f}"
                if isinstance(self.kpis["Accuracy"], float)
                else "N/A"
            ),
        )

    # In modules/analyzer.py

    def render_visual_diagnostics(self):
        """
        Renders the main diagnostic plots with improved aesthetics, layout,
        and automatic theme adaptation.
        """
        st.markdown("##### Visual Analysis")
        if not self.has_ground_truth:
            st.info("Visual analysis requires Ground Truth data for this batch.")
            return

        valid_gt_df = self.df.dropna(subset=["Ground Truth"])

        # Use a single row of columns for the two main plots
        plot_col1, plot_col2 = st.columns(2)

        # --- Chart 1: Confusion Matrix ---
        with plot_col1:  # Content for the first column
            with st.container(border=True):  # Group plot and buttons visually
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(
                    valid_gt_df["Ground Truth"],
                    valid_gt_df["Prediction"],
                    labels=list(LABEL_MAP.keys()),
                )

                with theme_aware_plot():  # Apply theme-aware styling
                    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="g",
                        ax=ax,
                        cmap="Blues",
                        xticklabels=list(LABEL_MAP.values()),
                        yticklabels=list(LABEL_MAP.values()),
                    )
                    ax.set_ylabel("Actual Class", fontsize=12)
                    ax.set_xlabel("Predicted Class", fontsize=12)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    st.pyplot(fig, use_container_width=True)  # Render the plot

                st.caption("Click a cell below to filter the data grid:")

                # Render CM filter buttons directly below the plot in the same column
                cm_labels = list(LABEL_MAP.values())
                for i, actual_label in enumerate(cm_labels):
                    btn_cols_row = st.columns(
                        len(cm_labels)
                    )  # Create a row of columns for buttons
                    for j, predicted_label in enumerate(cm_labels):
                        cell_value = cm[i, j]
                        btn_cols_row[j].button(  # Button for each cell
                            f"Actual: {actual_label}\nPred: {predicted_label} ({cell_value})",
                            key=f"cm_cell_{i}_{j}",
                            on_click=self._set_cm_filter,
                            args=(i, j, actual_label, predicted_label),
                            use_container_width=True,
                        )
                # Clear filter button for CM
                if st.session_state.get("cm_filter_active", False):
                    st.button(
                        "Clear Matrix Filter",
                        on_click=self._clear_cm_filter,
                        key="clear_cm_filter_btn_below",
                    )

        # --- Chart 2: Confidence vs. Correctness Box Plot ---
        with plot_col2:  # Content for the second column
            with st.container(border=True):  # Group plot visually
                st.markdown("**Confidence Analysis**")
                valid_gt_df["Result"] = np.where(
                    valid_gt_df["Prediction"] == valid_gt_df["Ground Truth"],
                    "Correct",
                    "Incorrect",
                )

                with theme_aware_plot():  # Apply theme-aware styling
                    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
                    sns.boxplot(
                        x="Result",
                        y="Confidence",
                        data=valid_gt_df,
                        ax=ax,
                        palette={"Correct": "#64C764", "Incorrect": "#E57373"},
                    )
                    ax.set_ylabel("Model Confidence", fontsize=12)
                    ax.set_xlabel("Prediction Result", fontsize=12)
                    st.pyplot(fig, use_container_width=True)
        st.divider()  # Divider after the entire visual section

    def _set_cm_filter(
        self,
        actual_idx: int,
        predicted_idx: int,
        actual_label: str,
        predicted_label: str,
    ):
        """Callback to set the confusion matrix filter in session state."""
        st.session_state["cm_actual_filter"] = actual_idx
        st.session_state["cm_predicted_filter"] = predicted_idx
        st.session_state["cm_filter_label"] = (
            f"Actual: {actual_label}, Predicted: {predicted_label}"
        )
        st.session_state["cm_filter_active"] = True
        # Streamlit will rerun automatically

    def _clear_cm_filter(self):
        """Callback to clear the confusion matrix filter from session state."""
        if "cm_actual_filter" in st.session_state:
            del st.session_state["cm_actual_filter"]
        if "cm_predicted_filter" in st.session_state:
            del st.session_state["cm_predicted_filter"]
        if "cm_filter_label" in st.session_state:
            del st.session_state["cm_filter_label"]
        if "cm_filter_active" in st.session_state:
            del st.session_state["cm_filter_active"]

    def render_interactive_grid(self):
        """
        Renders the filterable, detailed data grid with robust handling for
        row selection to prevent KeyError.
        """
        st.markdown("##### Detailed Results Explorer")

        # Start with a full copy of the dataframe to apply filters to
        filtered_df = self.df.copy()

        # --- Filter Section ---
        st.markdown("**Filters**")
        filter_cols = st.columns([2, 2, 3])  # Allocate more space for the slider

        # Filter 1: By Predicted Class
        selected_classes = filter_cols[0].multiselect(
            "Filter by Prediction:",
            options=self.df["Predicted Class"].unique(),
            default=self.df["Predicted Class"].unique(),
        )
        filtered_df = filtered_df[filtered_df["Predicted Class"].isin(selected_classes)]

        # Filter 2: By Ground Truth Correctness (if available)
        if self.has_ground_truth:
            filtered_df["Correct"] = (
                filtered_df["Prediction"] == filtered_df["Ground Truth"]
            )
            correctness_options = ["✅ Correct", "❌ Incorrect"]

            # Create a temporary column for display in multiselect
            filtered_df["Result_Display"] = np.where(
                filtered_df["Correct"], "✅ Correct", "❌ Incorrect"
            )

            selected_correctness = filter_cols[1].multiselect(
                "Filter by Result:",
                options=correctness_options,
                default=correctness_options,
            )
            # Filter based on the boolean 'Correct' column
            filter_correctness_bools = [
                True if c == "✅ Correct" else False for c in selected_correctness
            ]
            filtered_df = filtered_df[
                filtered_df["Correct"].isin(filter_correctness_bools)
            ]

        # --- NEW: Filter 3: By Confidence Range ---
        min_conf, max_conf = filter_cols[2].slider(
            "Filter by Confidence Range:",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),  # Default to the full range
            step=0.01,
        )
        filtered_df = filtered_df[
            (filtered_df["Confidence"] >= min_conf)
            & (filtered_df["Confidence"] <= max_conf)
        ]
        # --- END NEW FILTER ---

        # Apply Confusion Matrix Drill-Down Filter (if active)
        if st.session_state.get("cm_filter_active", False):
            actual_idx = st.session_state["cm_actual_filter"]
            predicted_idx = st.session_state["cm_predicted_filter"]
            filter_label = st.session_state["cm_filter_label"]

            st.info(f"Filtering results for: **{filter_label}**")
            filtered_df = filtered_df[
                (filtered_df["Ground Truth"] == actual_idx)
                & (filtered_df["Prediction"] == predicted_idx)
            ]

        # --- Display the Filtered Data Table ---
        if filtered_df.empty:
            st.warning("No files match the current filter criteria.")
            st.session_state.selected_spectrum_file = None
        else:
            display_df = filtered_df.drop(
                columns=["Correct", "Result_Display"], errors="ignore"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="results_grid_selection",
            )

            # --- ROBUST SELECTION HANDLING (THE FIX) ---
            selection_state = st.session_state.get("results_grid_selection")

            # Check if selection_state is a dictionary AND if it contains the 'rows' key
            if (
                isinstance(selection_state, dict)
                and "rows" in selection_state
                and selection_state["rows"]
            ):
                selected_index = selection_state["rows"][0]

                if selected_index < len(filtered_df):
                    st.session_state.selected_spectrum_file = filtered_df.iloc[
                        selected_index
                    ]["Filename"]
                else:
                    # This can happen if the table is re-filtered and the old index is now out of bounds
                    st.session_state.selected_spectrum_file = None
            else:
                # If the selection is empty or in an unexpected format, clear the selection
                st.session_state.selected_spectrum_file = None
            # --- END ROBUST HANDLING ---

    # --- ADD THIS ENTIRE NEW METHOD ---
    def render_selected_spectrum(self):
        """
        Renders an expander with the spectrum plot for the currently selected file.
        This is called after the data grid.
        """
        selected_file = st.session_state.get("selected_spectrum_file")

        # Only render if a file has been selected in the current session
        if selected_file:
            with st.expander(
                f"🔬 View Spectrum for: **{selected_file}**", expanded=True
            ):
                # Retrieve the full, detailed record for the selected file
                spectrum_data = ResultsManager.get_spectrum_data_for_file(selected_file)

                # Check if the detailed data was successfully retrieved and contains all necessary arrays
                if spectrum_data and all(
                    spectrum_data.get(k) is not None
                    for k in ["x_raw", "y_raw", "x_resampled", "y_resampled"]
                ):
                    # Generate a unique cache key for the plot to avoid re-generating it unnecessarily
                    cache_key = hashlib.md5(
                        (
                            f"{spectrum_data['x_raw'].tobytes()}"
                            f"{spectrum_data['y_raw'].tobytes()}"
                            f"{spectrum_data['x_resampled'].tobytes()}"
                            f"{spectrum_data['y_resampled'].tobytes()}"
                        ).encode()
                    ).hexdigest()

                    # Call the plotting function from ui_components
                    plot_image = create_spectrum_plot(
                        spectrum_data["x_raw"],
                        spectrum_data["y_raw"],
                        spectrum_data["x_resampled"],
                        spectrum_data["y_resampled"],
                        _cache_key=cache_key,
                    )
                    st.image(
                        plot_image,
                        caption=f"Raw vs. Resampled Spectrum for {selected_file}",
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        f"Could not retrieve spectrum data for '{selected_file}'. The data might not have been stored during the initial run."
                    )

    # --- END NEW METHOD ---

    def render(self):
        """
        The main public method to render the entire dashboard using a more
        organized and streamlined tab-based layout.
        """
        if self.df.empty:
            st.info(
                "The results table is empty. Please run an analysis on the 'Upload and Run' page."
            )
            return

        # --- Tier 1: KPIs (Always visible at the top) ---
        self.render_kpis()
        st.divider()

        # --- Tier 2: Tabbed Interface for Deeper Analysis ---
        tab1, tab2 = st.tabs(["📊 Visual Diagnostics", "🗂️ Results Explorer"])

        with tab1:
            # The visual diagnostics (Confusion Matrix, etc.) go here.
            self.render_visual_diagnostics()

        with tab2:
            # The interactive grid AND the spectrum viewer it controls go here.
            self.render_interactive_grid()
            self.render_selected_spectrum()
