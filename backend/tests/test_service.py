"""
Unit tests for backend.service.run_inference and /api/v1/analyze endpoint.
"""

import unittest
from unittest.mock import patch
from backend.service import ml_service
from backend.pydantic_models import SpectrumData  # Adjust import if needed
from fastapi.testclient import TestClient
from backend.main import app

class TestService(unittest.TestCase):
    """Tests for ml_service.run_inference."""

    @patch('backend.service.log_model_performance')
    def test_run_inference_calls_log_model_performance(self, mock_log_model_performance):
        """Test that run_inference calls log_model_performance with valid input."""
        # Build a real SpectrumData instance with required fields only
        dummy_spectrum = SpectrumData(
            x_values=[200, 210],
            y_values=[0.5, 0.6],
            filename="dummy.txt"
        )
        model_name = "figure2"
        modality = "raman"

        # Call with separate model_name and modality args (not as SpectrumData attributes)
        ml_service.run_inference(dummy_spectrum, model_name, modality)

        mock_log_model_performance.assert_called_once()

class TestAPI(unittest.TestCase):
    """Tests for /api/v1/analyze endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_analyze_spectrum_valid_payload(self):
        """Test /api/v1/analyze with valid payload."""
        payload = {
            "spectrum": {
                "x_values": [200, 210],
                "y_values": [0.5, 0.6],
                "filename": "dummy.txt"
            },
            "modality": "raman",
            "model_name": "figure2"
        }
        response = self.client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 200
        # Optionally, check response.json() for expected keys

if __name__ == "__main__":
    unittest.main()
