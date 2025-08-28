"""
Session results management for multi-file inference.
Handles in-memory results table and export functionality.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import io


class ResultsManager:
    """Manages session-wide results for multi-file inference"""
    
    RESULTS_KEY = "inference_results"
    
    @staticmethod
    def init_results_table() -> None:
        """Initialize the results table in session state"""
        if ResultsManager.RESULTS_KEY not in st.session_state:
            st.session_state[ResultsManager.RESULTS_KEY] = []
    
    @staticmethod
    def add_result(
        filename: str,
        model_name: str,
        prediction: int,
        predicted_class: str,
        confidence: float,
        logits: List[float],
        ground_truth: Optional[int] = None,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single inference result to the results table"""
        ResultsManager.init_results_table()
        
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "model": model_name,
            "prediction": prediction,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "logits": logits,
            "ground_truth": ground_truth,
            "processing_time": processing_time,
            "metadata": metadata or {}
        }
        
        st.session_state[ResultsManager.RESULTS_KEY].append(result)
    
    @staticmethod
    def get_results() -> List[Dict[str, Any]]:
        """Get all inference results"""
        ResultsManager.init_results_table()
        return st.session_state[ResultsManager.RESULTS_KEY]
    
    @staticmethod
    def get_results_count() -> int:
        """Get the number of stored results"""
        return len(ResultsManager.get_results())
    
    @staticmethod
    def clear_results() -> None:
        """Clear all stored results"""
        st.session_state[ResultsManager.RESULTS_KEY] = []
    
    @staticmethod
    def get_results_dataframe() -> pd.DataFrame:
        """Convert results to pandas DataFrame for display and export"""
        results = ResultsManager.get_results()
        if not results:
            return pd.DataFrame()
        
        # Flatten the results for DataFrame
        df_data = []
        for result in results:
            row = {
                "Timestamp": result["timestamp"],
                "Filename": result["filename"],
                "Model": result["model"],
                "Prediction": result["prediction"],
                "Predicted Class": result["predicted_class"],
                "Confidence": f"{result['confidence']:.3f}",
                "Stable Logit": f"{result['logits'][0]:.3f}" if len(result['logits']) > 0 else "N/A",
                "Weathered Logit": f"{result['logits'][1]:.3f}" if len(result['logits']) > 1 else "N/A",
                "Ground Truth": result["ground_truth"] if result["ground_truth"] is not None else "Unknown",
                "Processing Time (s)": f"{result['processing_time']:.3f}",
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    @staticmethod
    def export_to_csv() -> bytes:
        """Export results to CSV format"""
        df = ResultsManager.get_results_dataframe()
        if df.empty:
            return b""
        
        # Use StringIO to create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')
    
    @staticmethod
    def export_to_json() -> str:
        """Export results to JSON format"""
        results = ResultsManager.get_results()
        return json.dumps(results, indent=2, default=str)
    
    @staticmethod
    def get_summary_stats() -> Dict[str, Any]:
        """Get summary statistics for the results"""
        results = ResultsManager.get_results()
        if not results:
            return {}
        
        df = ResultsManager.get_results_dataframe()
        
        stats = {
            "total_files": len(results),
            "models_used": list(set(r["model"] for r in results)),
            "stable_predictions": sum(1 for r in results if r["prediction"] == 0),
            "weathered_predictions": sum(1 for r in results if r["prediction"] == 1),
            "avg_confidence": sum(r["confidence"] for r in results) / len(results),
            "avg_processing_time": sum(r["processing_time"] for r in results) / len(results),
            "files_with_ground_truth": sum(1 for r in results if r["ground_truth"] is not None),
        }
        
        # Calculate accuracy if ground truth is available
        correct_predictions = sum(
            1 for r in results 
            if r["ground_truth"] is not None and r["prediction"] == r["ground_truth"]
        )
        total_with_gt = stats["files_with_ground_truth"]
        if total_with_gt > 0:
            stats["accuracy"] = correct_predictions / total_with_gt
        else:
            stats["accuracy"] = None
        
        return stats
    
    @staticmethod
    def remove_result_by_filename(filename: str) -> bool:
        """Remove a result by filename. Returns True if removed, False if not found."""
        results = ResultsManager.get_results()
        original_length = len(results)
        
        # Filter out results with matching filename
        st.session_state[ResultsManager.RESULTS_KEY] = [
            r for r in results if r["filename"] != filename
        ]
        
        return len(st.session_state[ResultsManager.RESULTS_KEY]) < original_length
    
    @staticmethod
    def display_results_table() -> None:
        """Display the results table in Streamlit UI"""
        df = ResultsManager.get_results_dataframe()
        
        if df.empty:
            st.info("No inference results yet. Upload files and run analysis to see results here.")
            return
        
        st.subheader(f"📊 Inference Results ({len(df)} files)")
        
        # Summary stats
        stats = ResultsManager.get_summary_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", stats["total_files"])
            with col2:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
            with col3:
                st.metric("Stable/Weathered", f"{stats['stable_predictions']}/{stats['weathered_predictions']}")
            with col4:
                if stats["accuracy"] is not None:
                    st.metric("Accuracy", f"{stats['accuracy']:.3f}")
                else:
                    st.metric("Accuracy", "N/A")
        
        # Results table
        st.dataframe(df, use_container_width=True)
        
        # Export buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            csv_data = ResultsManager.export_to_csv()
            if csv_data:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"polymer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            json_data = ResultsManager.export_to_json()
            if json_data:
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"polymer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("🗑️ Clear All Results", help="Clear all stored results"):
                ResultsManager.clear_results()
                st.rerun()