"""
Educational Interface Page for POLYMEROS
Interactive learning system with adaptive progression and virtual laboratory
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any

# Import POLYMEROS educational components
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from modules.educational_framework import EducationalFramework


def init_educational_session():
    """Initialize educational session state"""
    if "educational_framework" not in st.session_state:
        st.session_state.educational_framework = EducationalFramework()

    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = "demo_user"

    if "user_progress" not in st.session_state:
        st.session_state.user_progress = (
            st.session_state.educational_framework.initialize_user(
                st.session_state.current_user_id
            )
        )


def render_competency_assessment():
    """Render interactive competency assessment"""
    st.header("🧪 Knowledge Assessment")

    domains = ["spectroscopy_basics", "polymer_aging", "ai_ml_concepts"]
    selected_domain = st.selectbox(
        "Select assessment domain:",
        domains,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    framework = st.session_state.educational_framework
    assessor = framework.competency_assessor

    if selected_domain in assessor.assessment_tasks:
        tasks = assessor.assessment_tasks[selected_domain]

        st.subheader(f"Assessment: {selected_domain.replace('_', ' ').title()}")

        responses = []
        for i, task in enumerate(tasks):
            st.write(f"**Question {i+1}:** {task['question']}")

            response = st.radio(
                f"Select answer for question {i+1}:",
                options=range(len(task["options"])),
                format_func=lambda x, task=task: task["options"][x],
                key=f"q_{selected_domain}_{i}",
                index=0,
            )
            responses.append(response)

        if st.button("Submit Assessment", key=f"submit_{selected_domain}"):
            results = framework.assess_user_competency(selected_domain, responses)

            st.success(f"Assessment completed! Score: {results['score']:.1%}")
            st.write(f"**Your level:** {results['level']}")

            st.subheader("Detailed Feedback:")
            for feedback in results["feedback"]:
                st.write(feedback)

            st.subheader("Recommendations:")
            for rec in results["recommendations"]:
                st.write(f"• {rec}")


def render_learning_path():
    """Render personalized learning path"""
    st.header("🎯 Your Learning Path")

    user_progress = st.session_state.user_progress
    framework = st.session_state.educational_framework

    # Display current progress
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Completed Objectives", len(user_progress.completed_objectives))

    with col2:
        avg_score = (
            np.mean(list(user_progress.competency_scores.values()))
            if user_progress.competency_scores
            else 0
        )
        st.metric("Average Score", f"{avg_score:.1%}")

    with col3:
        st.metric("Current Level", user_progress.current_level.title())

    # Learning style selection
    st.subheader("Learning Preferences")
    learning_styles = ["visual", "hands-on", "theoretical", "collaborative"]

    current_style = user_progress.preferred_learning_style
    new_style = st.selectbox(
        "Preferred learning style:",
        learning_styles,
        index=(
            learning_styles.index(current_style)
            if current_style in learning_styles
            else 0
        ),
    )

    if new_style != current_style:
        user_progress.preferred_learning_style = new_style
        framework.save_user_progress()
        st.success("Learning style updated!")

    # Target competencies
    st.subheader("Learning Goals")
    target_competencies = st.multiselect(
        "Select areas you want to focus on:",
        ["spectroscopy", "polymer_science", "machine_learning", "data_analysis"],
        default=["spectroscopy", "polymer_science"],
    )

    if st.button("Generate Learning Path"):
        learning_path = framework.get_personalized_learning_path(target_competencies)

        if learning_path:
            st.subheader("Recommended Learning Path:")

            for i, item in enumerate(learning_path):
                objective = item["objective"]

                with st.expander(
                    f"{i+1}. {objective['title']} (Level {objective['difficulty_level']})"
                ):
                    st.write(f"**Description:** {objective['description']}")
                    st.write(
                        f"**Estimated time:** {objective['estimated_time']} minutes"
                    )
                    st.write(
                        f"**Recommended approach:** {item['recommended_approach']}"
                    )

                    if item["priority_resources"]:
                        st.write("**Priority resources:**")
                        for resource in item["priority_resources"]:
                            st.write(f"- {resource['type']}: {resource['url']}")
        else:
            st.info("Complete an assessment to get personalized recommendations!")


def render_virtual_laboratory():
    """Render virtual laboratory interface"""
    st.header("🔬 Virtual Laboratory")

    framework = st.session_state.educational_framework
    virtual_lab = framework.virtual_lab

    # Select experiment
    experiments = list(virtual_lab.experiments.keys())
    selected_experiment = st.selectbox(
        "Select experiment:",
        experiments,
        format_func=lambda x: virtual_lab.experiments[x]["title"],
    )

    experiment_info = virtual_lab.experiments[selected_experiment]

    st.subheader(experiment_info["title"])
    st.write(f"**Description:** {experiment_info['description']}")
    st.write(f"**Difficulty:** {experiment_info['difficulty']}/5")
    st.write(f"**Estimated time:** {experiment_info['estimated_time']} minutes")

    # Experiment-specific inputs
    if selected_experiment == "polymer_identification":
        st.subheader("Polymer Identification Challenge")
        polymer_type = st.selectbox(
            "Select polymer to analyze:", ["PE", "PP", "PS", "PVC"]
        )

        if st.button("Generate Spectrum"):
            result = framework.run_virtual_experiment(
                selected_experiment, {"polymer_type": polymer_type}
            )

            if result.get("success"):
                # Plot the spectrum
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(result["wavenumbers"], result["spectrum"])
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity")
                ax.set_title(f"Unknown Polymer Spectrum")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.subheader("Analysis Hints:")
                for hint in result["hints"]:
                    st.write(f"💡 {hint}")

                # User identification
                user_guess = st.selectbox(
                    "Your identification:", ["PE", "PP", "PS", "PVC"]
                )
                if st.button("Submit Identification"):
                    if user_guess == polymer_type:
                        st.success("🎉 Correct! Well done!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer is {polymer_type}")

    elif selected_experiment == "aging_simulation":
        st.subheader("Polymer Aging Simulation")
        aging_time = st.slider("Aging time (hours):", 0, 200, 50)

        if st.button("Run Aging Simulation"):
            result = framework.run_virtual_experiment(
                selected_experiment, {"aging_time": aging_time}
            )

            if result.get("success"):
                # Plot comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Initial spectrum
                ax1.plot(result["wavenumbers"], result["initial_spectrum"])
                ax1.set_title("Initial Spectrum")
                ax1.set_xlabel("Wavenumber (cm⁻¹)")
                ax1.set_ylabel("Intensity")
                ax1.grid(True, alpha=0.3)

                # Aged spectrum
                ax2.plot(result["wavenumbers"], result["aged_spectrum"])
                ax2.set_title(f"After {aging_time} hours")
                ax2.set_xlabel("Wavenumber (cm⁻¹)")
                ax2.set_ylabel("Intensity")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Observations:")
                for obs in result["observations"]:
                    st.write(f"📊 {obs}")

    elif selected_experiment == "model_training":
        st.subheader("Train Your Own Model")

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model type:", ["CNN", "ResNet", "Transformer"])
        with col2:
            epochs = st.slider("Training epochs:", 5, 50, 10)

        if st.button("Start Training"):
            with st.spinner("Training model..."):
                result = framework.run_virtual_experiment(
                    selected_experiment, {"model_type": model_type, "epochs": epochs}
                )

            if result.get("success"):
                # Plot training metrics
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Training loss
                ax1.plot(result["train_losses"])
                ax1.set_title("Training Loss")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.grid(True, alpha=0.3)

                # Validation accuracy
                ax2.plot(result["val_accuracies"])
                ax2.set_title("Validation Accuracy")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Accuracy")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                st.success(
                    f"Training completed! Final accuracy: {result['final_accuracy']:.3f}"
                )

                st.subheader("Training Insights:")
                for insight in result["insights"]:
                    st.write(f"🎯 {insight}")


def render_progress_analytics():
    """Render learning analytics dashboard"""
    st.header("📊 Your Progress Analytics")

    framework = st.session_state.educational_framework
    analytics = framework.get_learning_analytics()

    if analytics:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Completed Objectives", analytics["completed_objectives"])

        with col2:
            st.metric("Study Time", f"{analytics['total_study_time']} min")

        with col3:
            st.metric("Current Level", analytics["current_level"].title())

        with col4:
            st.metric("Sessions", analytics["session_count"])

        # Competency scores
        if analytics["competency_scores"]:
            st.subheader("Competency Scores")

            domains = list(analytics["competency_scores"].keys())
            scores = list(analytics["competency_scores"].values())

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(domains, scores)
            ax.set_ylabel("Score")
            ax.set_title("Competency Assessment Results")
            ax.set_ylim(0, 1)

            # Color bars based on score
            for bar, score in zip(bars, scores):
                if score >= 0.8:
                    bar.set_color("green")
                elif score >= 0.6:
                    bar.set_color("orange")
                else:
                    bar.set_color("red")

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        # Learning style
        st.subheader("Learning Profile")
        st.write(f"**Preferred learning style:** {analytics['learning_style'].title()}")

        # Recommendations
        recommendations = framework.get_learning_recommendations()
        if recommendations:
            st.subheader("Next Steps")
            for rec in recommendations:
                st.write(f"• {rec}")
    else:
        st.info("Complete assessments to see your progress analytics!")


def main():
    """Main educational interface"""
    st.set_page_config(
        page_title="POLYMEROS Educational Interface", page_icon="🎓", layout="wide"
    )

    st.title("🎓 POLYMEROS Educational Interface")
    st.markdown("**Interactive Learning System for Polymer Science and AI**")

    # Initialize session
    init_educational_session()

    # Sidebar navigation
    st.sidebar.title("📚 Learning Modules")
    page = st.sidebar.selectbox(
        "Select module:",
        [
            "Knowledge Assessment",
            "Learning Path",
            "Virtual Laboratory",
            "Progress Analytics",
        ],
    )

    # Render selected page
    if page == "Knowledge Assessment":
        render_competency_assessment()
    elif page == "Learning Path":
        render_learning_path()
    elif page == "Virtual Laboratory":
        render_virtual_laboratory()
    elif page == "Progress Analytics":
        render_progress_analytics()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**POLYMEROS Educational Framework**")
    st.sidebar.markdown("*Adaptive learning for polymer science*")


if __name__ == "__main__":
    main()
