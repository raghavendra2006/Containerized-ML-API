"""
tests/test_api.py — Automated Unit Tests for ML Prediction API

Tests verify API functionality WITHOUT loading the real TensorFlow model.
The prediction function is mocked to return deterministic results,
ensuring tests run in milliseconds.
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image


# ============================================
# Mock the model loading BEFORE importing the app
# ============================================
# This prevents TensorFlow from being loaded during testing,
# which would be extremely slow and memory-intensive.
with patch("src.model.get_model") as mock_get_model:
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    from src.main import app

client = TestClient(app)


# ============================================
# Helper: Create a dummy PNG image in memory
# ============================================
def create_test_image(format="PNG", size=(64, 64)):
    """Generate a dummy image as bytes for testing."""
    image = Image.new("RGB", size, color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


# ============================================
# Test: Health Check Endpoint
# ============================================
class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        """Health response body must contain status: ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_is_json(self):
        """Health response must be valid JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


# ============================================
# Test: Prediction Endpoint — Valid Requests
# ============================================
class TestPredictValidRequests:
    """Tests for POST /predict with valid image uploads."""

    @patch("src.main.predict")
    def test_predict_valid_png(self, mock_predict):
        """Should return 200 with prediction result for a valid PNG."""
        mock_predict.return_value = {
            "class_label": "dog",
            "probabilities": [0.01, 0.02, 0.01, 0.01, 0.02, 0.89, 0.01, 0.01, 0.01, 0.01],
        }
        image = create_test_image("PNG")
        response = client.post(
            "/predict",
            files={"file": ("test_image.png", image, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["class_label"] == "dog"
        assert isinstance(data["probabilities"], list)
        assert len(data["probabilities"]) == 10

    @patch("src.main.predict")
    def test_predict_valid_jpeg(self, mock_predict):
        """Should return 200 with prediction result for a valid JPEG."""
        mock_predict.return_value = {
            "class_label": "cat",
            "probabilities": [0.01, 0.01, 0.01, 0.90, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01],
        }
        image = create_test_image("JPEG")
        response = client.post(
            "/predict",
            files={"file": ("test_photo.jpg", image, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["class_label"] == "cat"

    @patch("src.main.predict")
    def test_predict_response_contains_filename(self, mock_predict):
        """Response should include the original filename."""
        mock_predict.return_value = {
            "class_label": "airplane",
            "probabilities": [0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
        }
        image = create_test_image("PNG")
        response = client.post(
            "/predict",
            files={"file": ("my_plane.png", image, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "my_plane.png"

    @patch("src.main.predict")
    def test_predict_response_contains_timestamp(self, mock_predict):
        """Response should include a UTC timestamp."""
        mock_predict.return_value = {
            "class_label": "ship",
            "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1],
        }
        image = create_test_image("PNG")
        response = client.post(
            "/predict",
            files={"file": ("ship.png", image, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data


# ============================================
# Test: Prediction Endpoint — Invalid Requests
# ============================================
class TestPredictInvalidRequests:
    """Tests for POST /predict with invalid payloads."""

    def test_predict_reject_text_file(self):
        """Should return 400 for non-image files (text)."""
        text_content = io.BytesIO(b"This is not an image file.")
        response = client.post(
            "/predict",
            files={"file": ("document.txt", text_content, "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_reject_pdf_file(self):
        """Should return 400 for PDF uploads."""
        pdf_content = io.BytesIO(b"%PDF-1.4 fake pdf content")
        response = client.post(
            "/predict",
            files={"file": ("report.pdf", pdf_content, "application/pdf")},
        )
        assert response.status_code == 400

    def test_predict_reject_csv_file(self):
        """Should return 400 for CSV uploads."""
        csv_content = io.BytesIO(b"col1,col2\nval1,val2")
        response = client.post(
            "/predict",
            files={"file": ("data.csv", csv_content, "text/csv")},
        )
        assert response.status_code == 400

    def test_predict_error_response_is_json(self):
        """Error response should be structured JSON."""
        text_content = io.BytesIO(b"not an image")
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_content, "text/plain")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data


# ============================================
# Test: Prediction Endpoint — Error Handling
# ============================================
class TestPredictErrorHandling:
    """Tests for error handling in POST /predict."""

    @patch("src.main.predict")
    def test_predict_preprocessing_error_returns_422(self, mock_predict):
        """Should return 422 when image preprocessing fails."""
        mock_predict.side_effect = ValueError("Failed to preprocess image")
        image = create_test_image("PNG")
        response = client.post(
            "/predict",
            files={"file": ("corrupt.png", image, "image/png")},
        )
        assert response.status_code == 422

    @patch("src.main.predict")
    def test_predict_internal_error_returns_500(self, mock_predict):
        """Should return 500 on unexpected internal errors."""
        mock_predict.side_effect = RuntimeError("TensorFlow crashed")
        image = create_test_image("PNG")
        response = client.post(
            "/predict",
            files={"file": ("image.png", image, "image/png")},
        )
        assert response.status_code == 500
