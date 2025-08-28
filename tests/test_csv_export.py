"""
Test CSV export and session results functionality
"""

import pytest
import tempfile
import csv
import json
from io import StringIO

from utils.results_manager import ResultsManager


class TestCSVExport:
    """Test CSV export functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        ResultsManager.clear_results()
    
    def test_csv_export_structure(self):
        """Test CSV export has correct structure"""
        # Add some test results
        ResultsManager.add_result(
            filename="stable_sample.txt",
            model_name="Figure2CNN",
            prediction=0,
            predicted_class="Stable (Unweathered)",
            confidence=0.92,
            logits=[3.2, 1.1],
            ground_truth=0,
            processing_time=0.145,
            metadata={"confidence_level": "HIGH", "confidence_emoji": "🟢"}
        )
        
        ResultsManager.add_result(
            filename="weathered_sample.txt",
            model_name="ResNet1D",
            prediction=1,
            predicted_class="Weathered (Degraded)",
            confidence=0.78,
            logits=[1.5, 2.8],
            ground_truth=1,
            processing_time=0.223,
            metadata={"confidence_level": "MEDIUM", "confidence_emoji": "🟡"}
        )
        
        # Export to CSV
        csv_data = ResultsManager.export_to_csv()
        assert len(csv_data) > 0
        
        # Parse CSV
        csv_reader = csv.DictReader(StringIO(csv_data.decode('utf-8')))
        rows = list(csv_reader)
        
        assert len(rows) == 2
        
        # Check first row
        row1 = rows[0]
        assert row1["Filename"] == "stable_sample.txt"
        assert row1["Model"] == "Figure2CNN"
        assert row1["Predicted Class"] == "Stable (Unweathered)"
        assert float(row1["Confidence"]) == pytest.approx(0.92, abs=0.01)
        assert row1["Ground Truth"] == "0"
        
        # Check second row
        row2 = rows[1]
        assert row2["Filename"] == "weathered_sample.txt"
        assert row2["Model"] == "ResNet1D"
        assert row2["Predicted Class"] == "Weathered (Degraded)"
        assert float(row2["Confidence"]) == pytest.approx(0.78, abs=0.01)
    
    def test_json_export_structure(self):
        """Test JSON export has correct structure"""
        ResultsManager.add_result(
            filename="test_sample.txt",
            model_name="TestModel",
            prediction=1,
            predicted_class="Weathered (Degraded)",
            confidence=0.85,
            logits=[1.2, 2.7],
            ground_truth=1,
            processing_time=0.180
        )
        
        # Export to JSON
        json_data = ResultsManager.export_to_json()
        
        # Parse JSON
        results = json.loads(json_data)
        assert len(results) == 1
        
        result = results[0]
        assert result["filename"] == "test_sample.txt"
        assert result["model"] == "TestModel"
        assert result["prediction"] == 1
        assert result["confidence"] == pytest.approx(0.85, abs=0.01)
        assert result["logits"] == [1.2, 2.7]
        assert result["ground_truth"] == 1
    
    def test_empty_results_export(self):
        """Test export behavior with no results"""
        csv_data = ResultsManager.export_to_csv()
        assert csv_data == b""
        
        json_data = ResultsManager.export_to_json()
        results = json.loads(json_data)
        assert results == []
    
    def test_results_with_unknown_ground_truth(self):
        """Test export with unknown ground truth values"""
        ResultsManager.add_result(
            filename="unknown_sample.txt",
            model_name="TestModel",
            prediction=0,
            predicted_class="Stable (Unweathered)",
            confidence=0.67,
            logits=[2.1, 1.8],
            ground_truth=None,  # Unknown ground truth
            processing_time=0.156
        )
        
        csv_data = ResultsManager.export_to_csv()
        csv_reader = csv.DictReader(StringIO(csv_data.decode('utf-8')))
        row = next(csv_reader)
        
        assert row["Ground Truth"] == "Unknown"
    
    def test_accuracy_calculation_with_mixed_results(self):
        """Test accuracy calculation with correct and incorrect predictions"""
        # Correct prediction
        ResultsManager.add_result(
            filename="correct1.txt",
            model_name="TestModel",
            prediction=0,
            predicted_class="Stable",
            confidence=0.9,
            logits=[3.0, 1.0],
            ground_truth=0,
            processing_time=0.1
        )
        
        # Incorrect prediction
        ResultsManager.add_result(
            filename="incorrect1.txt",
            model_name="TestModel",
            prediction=1,
            predicted_class="Weathered",
            confidence=0.7,
            logits=[1.0, 2.0],
            ground_truth=0,  # Actually stable but predicted weathered
            processing_time=0.12
        )
        
        # No ground truth
        ResultsManager.add_result(
            filename="unknown1.txt",
            model_name="TestModel",
            prediction=0,
            predicted_class="Stable",
            confidence=0.8,
            logits=[2.5, 1.5],
            ground_truth=None,
            processing_time=0.11
        )
        
        stats = ResultsManager.get_summary_stats()
        
        assert stats["total_files"] == 3
        assert stats["files_with_ground_truth"] == 2
        assert stats["accuracy"] == 0.5  # 1 correct out of 2 with ground truth


if __name__ == "__main__":
    pytest.main([__file__])