"""Test suite for FastAPI ML serving application."""
import pytest
from fastapi.testclient import TestClient
from serve.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_health_endpoint(client):
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_valid_data(client):
    """Test prediction with valid data."""
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_data(client):
    """Test prediction with invalid data."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_docs_accessible(client):
    """Test API docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
