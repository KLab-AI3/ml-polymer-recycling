from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_analyze_spectrum():
    payload = {
        "spectrum": {
            "x_values": [200, 210],           # At least 2 points
            "y_values": [0.5, 0.6],           # At least 2 points
            "filename": "test.txt"
        },
        "modality": "raman",
        "model_name": "figure2"
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
