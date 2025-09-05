"""Performance tracking and logging utilities for POLYMEROS platform."""

import time
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""

    model_name: str
    prediction_time: float
    preprocessing_time: float
    total_time: float
    memory_usage_mb: float
    accuracy: Optional[float]
    confidence: float
    timestamp: str
    input_size: int
    modality: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracker:
    """Automatic performance tracking and logging system."""

    def __init__(self, db_path: str = "outputs/performance_tracking.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_time REAL NOT NULL,
                    preprocessing_time REAL NOT NULL,
                    total_time REAL NOT NULL,
                    memory_usage_mb REAL,
                    accuracy REAL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    input_size INTEGER NOT NULL,
                    modality TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO performance_metrics 
                (model_name, prediction_time, preprocessing_time, total_time, 
                 memory_usage_mb, accuracy, confidence, timestamp, input_size, modality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.model_name,
                    metrics.prediction_time,
                    metrics.preprocessing_time,
                    metrics.total_time,
                    metrics.memory_usage_mb,
                    metrics.accuracy,
                    metrics.confidence,
                    metrics.timestamp,
                    metrics.input_size,
                    metrics.modality,
                ),
            )
            conn.commit()

    @contextmanager
    def track_inference(self, model_name: str, modality: str = "raman"):
        """Context manager for automatic performance tracking."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        tracking_data = {
            "model_name": model_name,
            "modality": modality,
            "start_time": start_time,
            "start_memory": start_memory,
            "preprocessing_time": 0.0,
        }

        try:
            yield tracking_data
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            total_time = end_time - start_time
            memory_usage = max(end_memory - start_memory, 0)

            # Create metrics object if not provided
            if "metrics" not in tracking_data:
                metrics = PerformanceMetrics(
                    model_name=model_name,
                    prediction_time=tracking_data.get("prediction_time", total_time),
                    preprocessing_time=tracking_data.get("preprocessing_time", 0.0),
                    total_time=total_time,
                    memory_usage_mb=memory_usage,
                    accuracy=tracking_data.get("accuracy"),
                    confidence=tracking_data.get("confidence", 0.0),
                    timestamp=datetime.now().isoformat(),
                    input_size=tracking_data.get("input_size", 0),
                    modality=modality,
                )
                self.log_performance(metrics)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.execute(
                """
                SELECT * FROM performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_model_statistics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistical summary of model performance."""
        where_clause = "WHERE model_name = ?" if model_name else ""
        params = (model_name,) if model_name else ()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT 
                    model_name,
                    COUNT(*) as total_inferences,
                    AVG(prediction_time) as avg_prediction_time,
                    AVG(preprocessing_time) as avg_preprocessing_time,
                    AVG(total_time) as avg_total_time,
                    AVG(memory_usage_mb) as avg_memory_usage,
                    AVG(confidence) as avg_confidence,
                    MIN(total_time) as fastest_inference,
                    MAX(total_time) as slowest_inference
                FROM performance_metrics 
                {where_clause}
                GROUP BY model_name
            """,
                params,
            )

            results = cursor.fetchall()
            if model_name and results:
                # Return single model stats as dict
                row = results[0]
                return {
                    "model_name": row[0],
                    "total_inferences": row[1],
                    "avg_prediction_time": row[2],
                    "avg_preprocessing_time": row[3],
                    "avg_total_time": row[4],
                    "avg_memory_usage": row[5],
                    "avg_confidence": row[6],
                    "fastest_inference": row[7],
                    "slowest_inference": row[8],
                }
            elif not model_name:
                # Return all models stats as dict of dicts
                return {
                    row[0]: {
                        "model_name": row[0],
                        "total_inferences": row[1],
                        "avg_prediction_time": row[2],
                        "avg_preprocessing_time": row[3],
                        "avg_total_time": row[4],
                        "avg_memory_usage": row[5],
                        "avg_confidence": row[6],
                        "fastest_inference": row[7],
                        "slowest_inference": row[8],
                    }
                    for row in results
                }
            else:
                return {}

    def create_performance_visualization(self) -> plt.Figure:
        """Create performance visualization charts."""
        metrics = self.get_recent_metrics(50)

        if not metrics:
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Convert to convenient format
        models = [m["model_name"] for m in metrics]
        times = [m["total_time"] for m in metrics]
        confidences = [m["confidence"] for m in metrics]
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metrics]

        # 1. Inference Time Over Time
        ax1.plot(timestamps, times, "o-", alpha=0.7)
        ax1.set_title("Inference Time Over Time")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Performance by Model
        model_stats = self.get_model_statistics()
        if model_stats:
            model_names = list(model_stats.keys())
            avg_times = [model_stats[m]["avg_total_time"] for m in model_names]

            ax2.bar(model_names, avg_times, alpha=0.7)
            ax2.set_title("Average Inference Time by Model")
            ax2.set_ylabel("Time (seconds)")
            ax2.tick_params(axis="x", rotation=45)

        # 3. Confidence Distribution
        ax3.hist(confidences, bins=20, alpha=0.7)
        ax3.set_title("Confidence Score Distribution")
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Frequency")

        # 4. Memory Usage if available
        memory_usage = [
            m["memory_usage_mb"] for m in metrics if m["memory_usage_mb"] is not None
        ]
        if memory_usage:
            ax4.plot(range(len(memory_usage)), memory_usage, "o-", alpha=0.7)
            ax4.set_title("Memory Usage")
            ax4.set_xlabel("Inference Number")
            ax4.set_ylabel("Memory (MB)")
        else:
            ax4.text(
                0.5,
                0.5,
                "Memory tracking\nnot available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Memory Usage")

        plt.tight_layout()
        return fig

    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics in specified format."""
        metrics = self.get_recent_metrics(1000)  # Get more for export

        if format == "json":
            return json.dumps(metrics, indent=2, default=str)
        elif format == "csv":
            import pandas as pd

            df = pd.DataFrame(metrics)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global tracker instance
_tracker = None


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker


def display_performance_dashboard():
    """Display performance tracking dashboard in Streamlit."""
    tracker = get_performance_tracker()

    st.markdown("### 📈 Performance Dashboard")

    # Recent metrics summary
    recent_metrics = tracker.get_recent_metrics(20)

    if not recent_metrics:
        st.info(
            "No performance data available yet. Run some inferences to see metrics."
        )
        return

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    total_inferences = len(recent_metrics)
    avg_time = np.mean([m["total_time"] for m in recent_metrics])
    avg_confidence = np.mean([m["confidence"] for m in recent_metrics])
    unique_models = len(set(m["model_name"] for m in recent_metrics))

    with col1:
        st.metric("Total Inferences", total_inferences)
    with col2:
        st.metric("Avg Time", f"{avg_time:.3f}s")
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    with col4:
        st.metric("Models Used", unique_models)

    # Performance visualization
    fig = tracker.create_performance_visualization()
    if fig:
        st.pyplot(fig)

    # Model comparison table
    st.markdown("#### Model Performance Comparison")
    model_stats = tracker.get_model_statistics()

    if model_stats:
        import pandas as pd

        stats_data = []
        for model_name, stats in model_stats.items():
            stats_data.append(
                {
                    "Model": model_name,
                    "Total Inferences": stats["total_inferences"],
                    "Avg Time (s)": f"{stats['avg_total_time']:.3f}",
                    "Avg Confidence": f"{stats['avg_confidence']:.3f}",
                    "Fastest (s)": f"{stats['fastest_inference']:.3f}",
                    "Slowest (s)": f"{stats['slowest_inference']:.3f}",
                }
            )

        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True)

    # Export options
    with st.expander("📥 Export Performance Data"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export JSON"):
                json_data = tracker.export_metrics("json")
                st.download_button(
                    "Download JSON",
                    json_data,
                    "performance_metrics.json",
                    "application/json",
                )

        with col2:
            if st.button("Export CSV"):
                csv_data = tracker.export_metrics("csv")
                st.download_button(
                    "Download CSV", csv_data, "performance_metrics.csv", "text/csv"
                )


if __name__ == "__main__":
    # Test the performance tracker
    tracker = PerformanceTracker()

    # Simulate some metrics
    for i in range(5):
        metrics = PerformanceMetrics(
            model_name=f"test_model_{i%2}",
            prediction_time=0.1 + i * 0.01,
            preprocessing_time=0.05,
            total_time=0.15 + i * 0.01,
            memory_usage_mb=100 + i * 10,
            accuracy=0.8 + i * 0.02,
            confidence=0.7 + i * 0.05,
            timestamp=datetime.now().isoformat(),
            input_size=500,
            modality="raman",
        )
        tracker.log_performance(metrics)

    print("Performance tracking test completed!")
    print(f"Recent metrics: {len(tracker.get_recent_metrics())}")
    print(f"Model stats: {tracker.get_model_statistics()}")
