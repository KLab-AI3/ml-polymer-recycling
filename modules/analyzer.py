# In modules/analyzer.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

from config import LABEL_MAP  # Assuming LABEL_MAP is correctly defined in config.py


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

    def render_visual_diagnostics(self):
        """
        Renders the main diagnostic plots with improved aesthetics and layout.
        """
        st.markdown("##### Visual Analysis")
        if not self.has_ground_truth:
            st.info(
                "Visual analysis requires Ground Truth data, which is not available for this batch."
            )
            return

        valid_gt_df = self.df.dropna(subset=["Ground Truth"])

        viz_cols = st.columns(2)

        # --- Chart 1: Confusion Matrix (Aesthetically Improved) ---
        with viz_cols[0]:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(
                valid_gt_df["Ground Truth"],
                valid_gt_df["Prediction"],
                labels=list(LABEL_MAP.keys()),
            )

            # Use Matplotlib's constrained_layout for better sizing
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

            # Improve label readability and appearance
            ax.set_ylabel("Actual Class", fontsize=12)
            ax.set_xlabel("Predicted Class", fontsize=12)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right"
            )  # Rotate labels to prevent overlap
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            # Use `use_container_width=True` to let Streamlit manage the plot's width
            st.pyplot(fig, use_container_width=True)

        # --- Chart 2: Confidence vs. Correctness Box Plot (Aesthetically Improved) ---
        with viz_cols[1]:
            st.markdown("**Confidence Analysis**")
            valid_gt_df["Result"] = np.where(
                valid_gt_df["Prediction"] == valid_gt_df["Ground Truth"],
                "Correct",
                "Incorrect",
            )

            fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

            sns.boxplot(
                x="Result",
                y="Confidence",
                data=valid_gt_df,
                ax=ax,
                palette={"Correct": "#64C764", "Incorrect": "#E57373"},
            )  # Use softer colors

            ax.set_ylabel("Model Confidence", fontsize=12)
            ax.set_xlabel("Prediction Result", fontsize=12)

            st.pyplot(fig, use_container_width=True)

        # ... (The interactive button grid for the confusion matrix remains the same) ...
        st.markdown("Click on a cell below to filter the results grid:")
        cm_labels = list(LABEL_MAP.values())
        for i, actual_label in enumerate(cm_labels):
            cols = st.columns(len(cm_labels))
            for j, predicted_label in enumerate(cm_labels):
                cell_value = cm[i, j]
                cols[j].button(
                    f"Actual: {actual_label}\nPred: {predicted_label} ({cell_value})",
                    key=f"cm_cell_{i}_{j}",
                    on_click=self._set_cm_filter,
                    args=(i, j, actual_label, predicted_label),
                    use_container_width=True,
                )

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

    def render(self):
        """The main public method to render the entire dashboard."""
        if self.df.empty:
            st.info(
                "The results table is empty. Please run an analysis on the 'Upload and Run' page."
            )
            return

        self.render_kpis()
        st.divider()
        self.render_visual_diagnostics()
        st.divider()
        self.render_interactive_grid()
