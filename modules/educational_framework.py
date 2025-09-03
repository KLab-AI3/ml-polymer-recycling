"""
Educational Framework for POLYMEROS
Interactive learning system with adaptive progression and competency tracking
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import streamlit as st


@dataclass
class LearningObjective:
    """Individual learning objective with assessment criteria"""

    id: str
    title: str
    description: str
    prerequisite_ids: List[str]
    difficulty_level: int  # 1-5 scale
    estimated_time: int  # minutes
    assessment_criteria: List[str]
    resources: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningObjective":
        return cls(**data)


@dataclass
class UserProgress:
    """Track user progress and competency"""

    user_id: str
    completed_objectives: List[str]
    competency_scores: Dict[str, float]  # objective_id -> score
    learning_path: List[str]
    session_history: List[Dict[str, Any]]
    preferred_learning_style: str
    current_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProgress":
        return cls(**data)


class CompetencyAssessment:
    """Assess user competency through interactive tasks"""

    def __init__(self):
        self.assessment_tasks = {
            "spectroscopy_basics": [
                {
                    "type": "spectrum_identification",
                    "question": "Which spectral region typically shows C-H stretching vibrations?",
                    "options": [
                        "400-1500 cm⁻¹",
                        "1500-1700 cm⁻¹",
                        "2800-3100 cm⁻¹",
                        "3200-3600 cm⁻¹",
                    ],
                    "correct": 2,
                    "explanation": "C-H stretching vibrations appear in the 2800-3100 cm⁻¹ region",
                },
                {
                    "type": "peak_interpretation",
                    "question": "A peak at 1715 cm⁻¹ in a polymer spectrum most likely indicates:",
                    "options": [
                        "C-H bending",
                        "C=O stretching",
                        "O-H stretching",
                        "C-C stretching",
                    ],
                    "correct": 1,
                    "explanation": "C=O stretching typically appears around 1715 cm⁻¹, indicating carbonyl groups",
                },
            ],
            "polymer_aging": [
                {
                    "type": "degradation_mechanism",
                    "question": "Which process is most commonly responsible for polymer degradation?",
                    "options": [
                        "Hydrolysis",
                        "Oxidation",
                        "Thermal decomposition",
                        "UV radiation",
                    ],
                    "correct": 1,
                    "explanation": "Oxidation is the most common degradation mechanism in polymers",
                }
            ],
            "ai_ml_concepts": [
                {
                    "type": "model_interpretation",
                    "question": "What does a confidence score of 0.95 indicate?",
                    "options": [
                        "95% accuracy",
                        "95% probability",
                        "95% certainty",
                        "95% training success",
                    ],
                    "correct": 1,
                    "explanation": "Confidence score represents the model's estimated probability of the prediction",
                }
            ],
        }

    def assess_competency(self, domain: str, user_responses: List[int]) -> float:
        """Assess user competency in a specific domain"""
        if domain not in self.assessment_tasks:
            return 0.0

        tasks = self.assessment_tasks[domain]
        if len(user_responses) != len(tasks):
            # Handle mismatched response count gracefully
            min_len = min(len(user_responses), len(tasks))
            user_responses = user_responses[:min_len]
            tasks = tasks[:min_len]

            if not tasks:  # No tasks to assess
                return 0.0

        correct_count = sum(
            1
            for i, response in enumerate(user_responses)
            if response == tasks[i]["correct"]
        )

        return correct_count / len(tasks)

    def get_personalized_feedback(
        self, domain: str, user_responses: List[int]
    ) -> List[str]:
        """Provide personalized feedback based on assessment results"""
        feedback = []

        if domain not in self.assessment_tasks:
            return ["Domain not found"]

        tasks = self.assessment_tasks[domain]

        # Handle mismatched response count
        min_len = min(len(user_responses), len(tasks))
        user_responses = user_responses[:min_len]
        tasks = tasks[:min_len]

        for i, response in enumerate(user_responses):
            if i < len(tasks):
                task = tasks[i]
                if response == task["correct"]:
                    feedback.append(f"✅ Correct! {task['explanation']}")
                else:
                    feedback.append(f"❌ Incorrect. {task['explanation']}")

        return feedback


class AdaptiveLearningPath:
    """Generate personalized learning paths based on user competency and goals"""

    def __init__(self):
        self.learning_objectives = self._initialize_objectives()
        self.learning_styles = ["visual", "hands-on", "theoretical", "collaborative"]

    def _initialize_objectives(self) -> Dict[str, LearningObjective]:
        """Initialize learning objectives database"""
        objectives = {}

        # Basic spectroscopy objectives
        objectives["spec_001"] = LearningObjective(
            id="spec_001",
            title="Introduction to Vibrational Spectroscopy",
            description="Understand the principles of Raman and FTIR spectroscopy",
            prerequisite_ids=[],
            difficulty_level=1,
            estimated_time=15,
            assessment_criteria=[
                "Identify spectral regions",
                "Explain molecular vibrations",
            ],
            resources=[
                {"type": "tutorial", "url": "interactive_spectroscopy_tutorial"},
                {"type": "video", "url": "spectroscopy_basics_video"},
            ],
        )

        objectives["spec_002"] = LearningObjective(
            id="spec_002",
            title="Spectral Interpretation",
            description="Learn to interpret peaks and identify functional groups",
            prerequisite_ids=["spec_001"],
            difficulty_level=2,
            estimated_time=25,
            assessment_criteria=[
                "Identify functional groups",
                "Interpret peak intensities",
            ],
            resources=[
                {"type": "interactive", "url": "peak_identification_tool"},
                {"type": "practice", "url": "spectral_analysis_exercises"},
            ],
        )

        # Polymer science objectives
        objectives["poly_001"] = LearningObjective(
            id="poly_001",
            title="Polymer Structure and Properties",
            description="Understand polymer chemistry and structure-property relationships",
            prerequisite_ids=[],
            difficulty_level=2,
            estimated_time=20,
            assessment_criteria=[
                "Explain polymer structures",
                "Relate structure to properties",
            ],
            resources=[
                {"type": "tutorial", "url": "polymer_basics_tutorial"},
                {"type": "simulation", "url": "molecular_structure_viewer"},
            ],
        )

        objectives["poly_002"] = LearningObjective(
            id="poly_002",
            title="Polymer Degradation Mechanisms",
            description="Learn about polymer aging and degradation pathways",
            prerequisite_ids=["poly_001"],
            difficulty_level=3,
            estimated_time=30,
            assessment_criteria=[
                "Identify degradation mechanisms",
                "Predict aging effects",
            ],
            resources=[
                {"type": "case_study", "url": "degradation_case_studies"},
                {"type": "interactive", "url": "aging_simulation"},
            ],
        )

        # AI/ML objectives
        objectives["ai_001"] = LearningObjective(
            id="ai_001",
            title="Introduction to Machine Learning",
            description="Basic concepts of ML for scientific applications",
            prerequisite_ids=[],
            difficulty_level=2,
            estimated_time=20,
            assessment_criteria=["Explain ML concepts", "Understand model types"],
            resources=[
                {"type": "tutorial", "url": "ml_basics_tutorial"},
                {"type": "interactive", "url": "model_playground"},
            ],
        )

        objectives["ai_002"] = LearningObjective(
            id="ai_002",
            title="Model Interpretation and Validation",
            description="Understanding model outputs and reliability assessment",
            prerequisite_ids=["ai_001"],
            difficulty_level=3,
            estimated_time=25,
            assessment_criteria=["Interpret model outputs", "Assess model reliability"],
            resources=[
                {"type": "hands-on", "url": "model_interpretation_lab"},
                {"type": "case_study", "url": "validation_examples"},
            ],
        )

        return objectives

    def generate_learning_path(
        self, user_progress: UserProgress, target_competencies: List[str]
    ) -> List[str]:
        """Generate personalized learning path"""
        available_objectives = []

        # Find objectives that meet prerequisites
        for obj_id, objective in self.learning_objectives.items():
            if obj_id not in user_progress.completed_objectives:
                prerequisites_met = all(
                    prereq in user_progress.completed_objectives
                    for prereq in objective.prerequisite_ids
                )
                if prerequisites_met:
                    available_objectives.append(obj_id)

        # Sort by difficulty and relevance to target competencies
        def objective_priority(obj_id):
            obj = self.learning_objectives[obj_id]
            relevance = (
                1.0
                if any(comp in obj.title.lower() for comp in target_competencies)
                else 0.5
            )
            difficulty_penalty = obj.difficulty_level * 0.1
            return relevance - difficulty_penalty

        sorted_objectives = sorted(
            available_objectives, key=objective_priority, reverse=True
        )

        return sorted_objectives[:5]  # Return top 5 recommendations

    def adapt_to_learning_style(
        self, objective_id: str, learning_style: str
    ) -> Dict[str, Any]:
        """Adapt content delivery to user's learning style"""
        objective = self.learning_objectives[objective_id]
        adapted_content = {
            "objective": objective.to_dict(),
            "recommended_approach": "",
            "priority_resources": [],
        }

        if learning_style == "visual":
            adapted_content["recommended_approach"] = (
                "Start with visualizations and diagrams"
            )
            adapted_content["priority_resources"] = [
                r for r in objective.resources if r["type"] in ["video", "simulation"]
            ]

        elif learning_style == "hands-on":
            adapted_content["recommended_approach"] = "Begin with interactive exercises"
            adapted_content["priority_resources"] = [
                r
                for r in objective.resources
                if r["type"] in ["interactive", "hands-on"]
            ]

        elif learning_style == "theoretical":
            adapted_content["recommended_approach"] = (
                "Focus on conceptual understanding"
            )
            adapted_content["priority_resources"] = [
                r
                for r in objective.resources
                if r["type"] in ["tutorial", "case_study"]
            ]

        elif learning_style == "collaborative":
            adapted_content["recommended_approach"] = (
                "Engage with community discussions"
            )
            adapted_content["priority_resources"] = [
                r
                for r in objective.resources
                if r["type"] in ["practice", "case_study"]
            ]

        return adapted_content


class VirtualLaboratory:
    """Simulated laboratory environment for hands-on learning"""

    def __init__(self):
        self.experiments = {
            "polymer_identification": {
                "title": "Polymer Identification Challenge",
                "description": "Identify unknown polymers using spectroscopic analysis",
                "difficulty": 2,
                "estimated_time": 20,
                "learning_objectives": ["spec_002", "poly_001"],
            },
            "aging_simulation": {
                "title": "Polymer Aging Simulation",
                "description": "Observe spectral changes during accelerated aging",
                "difficulty": 3,
                "estimated_time": 30,
                "learning_objectives": ["poly_002", "spec_002"],
            },
            "model_training": {
                "title": "Train Your Own Model",
                "description": "Build and train a classification model",
                "difficulty": 4,
                "estimated_time": 45,
                "learning_objectives": ["ai_001", "ai_002"],
            },
        }

    def run_experiment(
        self, experiment_id: str, user_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run virtual experiment with user inputs"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}

        # The experiment details are not used directly here
        # Removed unused variable assignment

        if experiment_id == "polymer_identification":
            return self._run_identification_experiment(user_inputs)
        elif experiment_id == "aging_simulation":
            return self._run_aging_simulation(user_inputs)
        elif experiment_id == "model_training":
            return self._run_model_training(user_inputs)

        return {"error": "Experiment not implemented"}

    def _run_identification_experiment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate polymer identification experiment"""
        # Generate synthetic spectrum for learning
        wavenumbers = np.linspace(400, 4000, 500)

        # Simple synthetic spectrum generation
        polymer_type = inputs.get("polymer_type", "PE")
        if polymer_type == "PE":
            # Polyethylene-like spectrum
            spectrum = (
                np.exp(-(((wavenumbers - 2920) / 50) ** 2)) * 0.8
                + np.exp(-(((wavenumbers - 2850) / 30) ** 2)) * 0.6
                + np.random.normal(0, 0.02, len(wavenumbers))
            )
        else:
            # Generic polymer spectrum
            spectrum = np.exp(
                -(((wavenumbers - 1600) / 100) ** 2)
            ) * 0.5 + np.random.normal(0, 0.02, len(wavenumbers))

        return {
            "wavenumbers": wavenumbers.tolist(),
            "spectrum": spectrum.tolist(),
            "hints": [
                "Look for C-H stretching around 2900 cm⁻¹",
                "Check the fingerprint region for characteristic patterns",
            ],
            "success": True,
        }

    def _run_aging_simulation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate polymer aging experiment"""
        aging_time = inputs.get("aging_time", 0)

        # Generate time-series data showing spectral changes
        wavenumbers = np.linspace(400, 4000, 500)

        # Base spectrum
        base_spectrum = np.exp(-(((wavenumbers - 2900) / 100) ** 2)) * 0.8

        # Add aging effects
        oxidation_peak = np.exp(-(((wavenumbers - 1715) / 20) ** 2)) * (
            aging_time / 100
        )
        degraded_spectrum = base_spectrum + oxidation_peak
        degraded_spectrum += np.random.normal(0, 0.01, len(wavenumbers))

        return {
            "wavenumbers": wavenumbers.tolist(),
            "initial_spectrum": base_spectrum.tolist(),
            "aged_spectrum": degraded_spectrum.tolist(),
            "aging_time": aging_time,
            "observations": [
                "New peak emerging at 1715 cm⁻¹ (carbonyl)",
                f"Aging time: {aging_time} hours",
                "Oxidative degradation pathway activated",
            ],
            "success": True,
        }

    def _run_model_training(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate model training experiment"""
        model_type = inputs.get("model_type", "CNN")
        epochs = inputs.get("epochs", 10)

        # Simulate training metrics
        train_losses = [
            1.0 - i * 0.08 + np.random.normal(0, 0.02) for i in range(epochs)
        ]
        val_accuracies = [
            0.5 + i * 0.04 + np.random.normal(0, 0.01) for i in range(epochs)
        ]

        return {
            "model_type": model_type,
            "epochs": epochs,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "final_accuracy": val_accuracies[-1],
            "insights": [
                "Model converged after 8 epochs",
                "Validation accuracy plateau suggests good generalization",
                "Consider data augmentation for further improvement",
            ],
            "success": True,
        }


class EducationalFramework:
    """Main educational framework interface"""

    def __init__(self, user_data_dir: str = "user_data"):
        self.user_data_dir = Path(user_data_dir)
        self.user_data_dir.mkdir(exist_ok=True)

        self.competency_assessor = CompetencyAssessment()
        self.learning_path_generator = AdaptiveLearningPath()
        self.virtual_lab = VirtualLaboratory()

        self.current_user: Optional[UserProgress] = None

    def initialize_user(self, user_id: str) -> UserProgress:
        """Initialize or load user progress"""
        user_file = self.user_data_dir / f"{user_id}.json"

        if user_file.exists():
            with open(user_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_progress = UserProgress.from_dict(data)
        else:
            user_progress = UserProgress(
                user_id=user_id,
                completed_objectives=[],
                competency_scores={},
                learning_path=[],
                session_history=[],
                preferred_learning_style="visual",
                current_level="beginner",
            )

        self.current_user = user_progress
        return user_progress

    def assess_user_competency(
        self, domain: str, responses: List[int]
    ) -> Dict[str, Any]:
        """Assess user competency and update progress"""
        if not self.current_user:
            return {"error": "No user initialized"}

        score = self.competency_assessor.assess_competency(domain, responses)
        feedback = self.competency_assessor.get_personalized_feedback(domain, responses)

        # Update user progress
        self.current_user.competency_scores[domain] = score

        # Determine user level based on overall competency
        avg_score = np.mean(list(self.current_user.competency_scores.values()))
        if avg_score >= 0.8:
            self.current_user.current_level = "advanced"
        elif avg_score >= 0.6:
            self.current_user.current_level = "intermediate"
        else:
            self.current_user.current_level = "beginner"

        self.save_user_progress()

        return {
            "score": score,
            "feedback": feedback,
            "level": self.current_user.current_level,
            "recommendations": self._get_learning_recommendations(),
        }

    def get_personalized_learning_path(
        self, target_competencies: List[str]
    ) -> List[Dict[str, Any]]:
        """Get personalized learning path for user"""
        if not self.current_user:
            return []

        path_ids = self.learning_path_generator.generate_learning_path(
            self.current_user, target_competencies
        )

        adapted_path = []
        for obj_id in path_ids:
            adapted_content = self.learning_path_generator.adapt_to_learning_style(
                obj_id, self.current_user.preferred_learning_style
            )
            adapted_path.append(adapted_content)

        return adapted_path

    def run_virtual_experiment(
        self, experiment_id: str, user_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run virtual laboratory experiment"""
        result = self.virtual_lab.run_experiment(experiment_id, user_inputs)

        # Track experiment in user history
        if self.current_user and result.get("success"):
            experiment_record = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "inputs": user_inputs,
                "completed": True,
            }
            self.current_user.session_history.append(experiment_record)
            self.save_user_progress()

        return result

    def _get_learning_recommendations(self) -> List[str]:
        """Get learning recommendations based on current progress"""
        recommendations = []

        if not self.current_user or not self.current_user.competency_scores:
            recommendations.append("Start with basic spectroscopy concepts")
            recommendations.append("Complete the introductory assessment")
        else:
            weak_areas = [
                domain
                for domain, score in (
                    self.current_user.competency_scores.items()
                    if self.current_user
                    else {}
                )
                if score < 0.6
            ]

            for area in weak_areas:
                recommendations.append(f"Review {area} concepts")

            if not weak_areas:
                recommendations.append(
                    "Explore advanced topics in your areas of interest"
                )
                recommendations.append("Try hands-on virtual experiments")

        return recommendations

    def save_user_progress(self):
        """Save user progress to file"""
        if self.current_user:
            user_file = self.user_data_dir / f"{self.current_user.user_id}.json"
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(self.current_user.to_dict(), f, indent=2)

    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get learning analytics for the current user"""
        if not self.current_user:
            return {}

        total_time = sum(
            obj.estimated_time
            for obj_id in self.current_user.completed_objectives
            for obj in [self.learning_path_generator.learning_objectives.get(obj_id)]
            if obj
        )

        return {
            "completed_objectives": len(self.current_user.completed_objectives),
            "total_study_time": total_time,
            "competency_scores": self.current_user.competency_scores,
            "current_level": self.current_user.current_level,
            "learning_style": self.current_user.preferred_learning_style,
            "session_count": len(self.current_user.session_history),
        }
