"""
Test suite for multi-file inference functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
import tempfile
import os
from pathlib import Path

# Import the utilities to test
from utils.multifile import parse_spectrum_data, process_single_file, process_multiple_files
from utils.results_manager import ResultsManager
from utils.confidence import calculate_softmax_confidence, get_confidence_badge
from utils.errors import ErrorHandler


class TestSpectrumParsing:
    """Test spectrum data parsing functionality"""
    
    def test_parse_spectrum_data_comma_separated(self):
        """Test parsing comma-separated spectrum data"""
        # Generate enough data points (at least 10)
        test_data = "\n".join([f"{100.0 + i*10},{0.5 + i*0.1}" for i in range(15)])
        x, y = parse_spectrum_data(test_data, "test.txt")
        
        assert len(x) == 15
        assert len(y) == 15
        np.testing.assert_array_equal(x[:3], [100.0, 110.0, 120.0])
        np.testing.assert_array_almost_equal(y[:3], [0.5, 0.6, 0.7], decimal=1)
    
    def test_parse_spectrum_data_space_separated(self):
        """Test parsing space-separated spectrum data"""
        # Generate enough data points
        test_data = "\n".join([f"{100.0 + i*10} {0.5 + i*0.1}" for i in range(12)])
        x, y = parse_spectrum_data(test_data, "test.txt")
        
        assert len(x) == 12
        assert len(y) == 12
        np.testing.assert_array_equal(x[:3], [100.0, 110.0, 120.0])
    
    def test_parse_spectrum_data_with_comments(self):
        """Test parsing with comment lines"""
        # Generate data with comments
        lines = ["# Comment line"]
        lines.extend([f"{100.0 + i*10},{0.5 + i*0.1}" for i in range(10)])
        lines.append("% Another comment")
        lines.extend([f"{200.0 + i*10},{1.5 + i*0.1}" for i in range(5)])
        test_data = "\n".join(lines)
        
        x, y = parse_spectrum_data(test_data, "test.txt")
        
        assert len(x) == 15  # 10 + 5 data lines
        assert len(y) == 15
    
    def test_parse_spectrum_data_insufficient_points(self):
        """Test error handling for insufficient data points"""
        test_data = "100.0,0.5\n200.0,1.0"  # Only 2 points
        with pytest.raises(ValueError, match="Insufficient data points"):
            parse_spectrum_data(test_data, "test.txt")
    
    def test_parse_spectrum_data_empty_file(self):
        """Test error handling for empty file"""
        test_data = ""
        with pytest.raises(ValueError, match="No data lines found"):
            parse_spectrum_data(test_data, "test.txt")
    
    def test_parse_spectrum_data_invalid_format(self):
        """Test error handling for invalid data format"""
        # Start with some valid data
        valid_lines = [f"{100.0 + i*10},{0.5 + i*0.1}" for i in range(12)]
        test_data = "invalid,data,format\n" + "\n".join(valid_lines)
        # Should handle the invalid line and continue if there are enough valid points
        x, y = parse_spectrum_data(test_data, "test.txt")
        assert len(x) >= 10  # Should have enough points despite the invalid line


class TestResultsManager:
    """Test results management functionality"""
    
    def setup_method(self):
        """Setup for each test - clear results"""
        ResultsManager.clear_results()
    
    def test_add_and_get_result(self):
        """Test adding and retrieving results"""
        ResultsManager.add_result(
            filename="test.txt",
            model_name="TestModel",
            prediction=1,
            predicted_class="Weathered",
            confidence=0.85,
            logits=[2.0, 3.5],
            ground_truth=1,
            processing_time=0.123
        )
        
        results = ResultsManager.get_results()
        assert len(results) == 1
        assert results[0]["filename"] == "test.txt"
        assert results[0]["prediction"] == 1
        assert results[0]["confidence"] == 0.85
    
    def test_results_dataframe(self):
        """Test converting results to DataFrame"""
        ResultsManager.add_result(
            filename="test1.txt",
            model_name="TestModel",
            prediction=0,
            predicted_class="Stable",
            confidence=0.90,
            logits=[3.0, 1.0],
            processing_time=0.1
        )
        
        df = ResultsManager.get_results_dataframe()
        assert not df.empty
        assert len(df) == 1
        assert "Filename" in df.columns
        assert "Confidence" in df.columns
    
    def test_export_csv(self):
        """Test CSV export functionality"""
        ResultsManager.add_result(
            filename="test.txt",
            model_name="TestModel",
            prediction=1,
            predicted_class="Weathered",
            confidence=0.75,
            logits=[1.5, 2.5],
            processing_time=0.2
        )
        
        csv_data = ResultsManager.export_to_csv()
        assert len(csv_data) > 0
        assert b"Filename" in csv_data  # Header should be present
        assert b"test.txt" in csv_data
    
    def test_export_json(self):
        """Test JSON export functionality"""
        ResultsManager.add_result(
            filename="test.txt",
            model_name="TestModel",
            prediction=0,
            predicted_class="Stable",
            confidence=0.95,
            logits=[4.0, 1.0],
            processing_time=0.15
        )
        
        json_data = ResultsManager.export_to_json()
        assert "test.txt" in json_data
        assert "TestModel" in json_data
        assert "0.95" in json_data
    
    def test_summary_stats(self):
        """Test summary statistics calculation"""
        # Add multiple results
        ResultsManager.add_result("test1.txt", "Model", 0, "Stable", 0.9, [3,1], 0, 0.1)
        ResultsManager.add_result("test2.txt", "Model", 1, "Weathered", 0.8, [1,3], 1, 0.2)
        ResultsManager.add_result("test3.txt", "Model", 0, "Stable", 0.7, [2,1], None, 0.15)
        
        stats = ResultsManager.get_summary_stats()
        assert stats["total_files"] == 3
        assert stats["stable_predictions"] == 2
        assert stats["weathered_predictions"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.8, abs=0.01)
        assert stats["files_with_ground_truth"] == 2
        assert stats["accuracy"] == 1.0  # Both with ground truth are correct


class TestConfidenceCalculation:
    """Test confidence calculation and visualization"""
    
    def test_softmax_confidence_high(self):
        """Test softmax confidence calculation for high confidence"""
        logits = torch.tensor([[3.0, 1.0]])  # High confidence for class 0
        probs, max_conf, level, emoji = calculate_softmax_confidence(logits)
        
        assert len(probs) == 2
        assert max_conf > 0.8  # Should be high confidence
        assert level == "HIGH"
        assert emoji == "🟢"
    
    def test_softmax_confidence_medium(self):
        """Test softmax confidence calculation for medium confidence"""
        logits = torch.tensor([[1.5, 1.0]])  # Medium confidence
        probs, max_conf, level, emoji = calculate_softmax_confidence(logits)
        
        assert 0.6 <= max_conf < 0.8
        assert level == "MEDIUM"
        assert emoji == "🟡"
    
    def test_softmax_confidence_low(self):
        """Test softmax confidence calculation for low confidence"""
        logits = torch.tensor([[1.1, 1.0]])  # Low confidence
        probs, max_conf, level, emoji = calculate_softmax_confidence(logits)
        
        assert max_conf < 0.6
        assert level == "LOW"
        assert emoji == "🔴"
    
    def test_confidence_badge(self):
        """Test confidence badge generation"""
        emoji, level = get_confidence_badge(0.9)
        assert emoji == "🟢"
        assert level == "HIGH"
        
        emoji, level = get_confidence_badge(0.7)
        assert emoji == "🟡"
        assert level == "MEDIUM"
        
        emoji, level = get_confidence_badge(0.5)
        assert emoji == "🔴"
        assert level == "LOW"


class TestErrorHandling:
    """Test error handling functionality"""
    
    def test_error_logging(self):
        """Test error logging to session state"""
        # Clear existing logs
        ErrorHandler.clear_logs()
        
        test_error = ValueError("Test error")
        ErrorHandler.log_error(test_error, "test context")
        
        logs = ErrorHandler.get_logs()
        assert len(logs) > 0
        assert "Test error" in logs[-1]
        assert "test context" in logs[-1]
    
    def test_file_error_handling(self):
        """Test file processing error handling"""
        test_error = FileNotFoundError("File not found")
        message = ErrorHandler.handle_file_error("test.txt", test_error)
        
        assert "File not found" in message
        assert "test.txt" in message
    
    def test_inference_error_handling(self):
        """Test inference error handling"""
        test_error = RuntimeError("CUDA out of memory")
        message = ErrorHandler.handle_inference_error("TestModel", test_error)
        
        assert "Device error" in message
        assert "TestModel" in message
    
    def test_parsing_error_handling(self):
        """Test spectrum parsing error handling"""
        test_error = ValueError("could not convert string to float")
        message = ErrorHandler.handle_parsing_error("test.txt", test_error)
        
        assert "Invalid data format" in message
        assert "test.txt" in message


class TestMultiFileProcessing:
    """Test multi-file processing functionality"""
    
    def create_mock_file(self, name, content):
        """Create a mock uploaded file"""
        mock_file = MagicMock()
        mock_file.name = name
        mock_file.read.return_value = content.encode('utf-8')
        return mock_file
    
    def test_process_single_file_success(self):
        """Test successful single file processing"""
        # Create test spectrum data with enough points
        test_content = "\n".join([f"{500 + i*2},{i*0.1}" for i in range(50)])
        
        # Mock functions with proper return values
        mock_load_model = MagicMock(return_value=(MagicMock(), True))
        
        # Mock run_inference to return the expected tuple format: (prediction, logits_list, probs, inference_time, logits)
        mock_run_inference = MagicMock(return_value=(1, [1.0, 2.0], None, 0.1, torch.tensor([[1.0, 2.0]])))
        mock_label_file = MagicMock(return_value=1)
        
        # Mock safe_execute to avoid session state issues
        with patch('utils.multifile.safe_execute') as mock_safe_execute:
            # Setup safe_execute to return successful results
            def safe_execute_side_effect(func, *args, **kwargs):
                if func.__name__ == 'parse_spectrum_data':
                    return parse_spectrum_data(test_content, "test.txt"), True
                elif func.__name__ == 'resample_spectrum':
                    x_raw = np.array([500 + i*2 for i in range(50)])
                    y_raw = np.array([i*0.1 for i in range(50)])
                    from utils.preprocessing import resample_spectrum
                    return resample_spectrum(x_raw, y_raw, 500), True
                else:
                    return func(*args, **kwargs), True
            
            mock_safe_execute.side_effect = safe_execute_side_effect
            
            result = process_single_file(
                "test.txt",
                test_content,
                "TestModel",
                mock_load_model,
                mock_run_inference,
                mock_label_file
            )
            
            assert result is not None
            assert result["success"] is True
            assert result["filename"] == "test.txt"
            assert result["prediction"] == 1
    
    def test_process_single_file_parsing_error(self):
        """Test single file processing with parsing error"""
        # Invalid content
        test_content = "invalid data"
        
        mock_load_model = MagicMock()
        mock_run_inference = MagicMock()
        mock_label_file = MagicMock()
        
        result = process_single_file(
            "test.txt",
            test_content,
            "TestModel",
            mock_load_model,
            mock_run_inference,
            mock_label_file
        )
        
        assert result is not None
        assert result["success"] is False
        assert "error" in result
    
    @patch('utils.multifile.process_single_file')
    def test_process_multiple_files(self, mock_process_single):
        """Test processing multiple files"""
        # Setup mocks
        mock_process_single.side_effect = [
            {"filename": "file1.txt", "success": True, "prediction": 0},
            {"filename": "file2.txt", "success": True, "prediction": 1},
        ]
        
        # Create mock files with sufficient data
        files = [
            self.create_mock_file("file1.txt", "\n".join([f"{100+i},{0.5+i*0.01}" for i in range(20)])),
            self.create_mock_file("file2.txt", "\n".join([f"{150+i},{0.7+i*0.01}" for i in range(20)])),
        ]
        
        mock_load_model = MagicMock()
        mock_run_inference = MagicMock()
        mock_label_file = MagicMock()
        
        results = process_multiple_files(
            files,
            "TestModel",
            mock_load_model,
            mock_run_inference,
            mock_label_file
        )
        
        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert mock_process_single.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])