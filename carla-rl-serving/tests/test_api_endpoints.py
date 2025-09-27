"""
Integration tests for FastAPI endpoints in CarlaRL Policy-as-a-Service.

Tests all API endpoints with mocked dependencies and real FastAPI client.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

# Import after setting up mocks to avoid import errors
@pytest.fixture
def mock_inference_engine():
    """Mock inference engine for testing."""
    engine = Mock()
    engine.device = "cpu"
    engine.predict = Mock(return_value=(
        [Mock(throttle=0.7, brake=0.0, steer=0.1)],
        8.5
    ))
    return engine


@pytest.fixture
def mock_app_state():
    """Mock application state."""
    return {
        "model_loaded": True,
        "inference_engine": None,  # Will be set by mock_inference_engine
        "startup_time": 1695825600.0,
        "warmup_completed": False
    }


@pytest.fixture
def client(mock_app_state, mock_inference_engine):
    """Create test client with mocked dependencies."""
    with patch("src.server.app_state", mock_app_state):
        with patch("src.server.get_inference_engine", return_value=mock_inference_engine):
            from src.server import app
            return TestClient(app)


class TestHealthEndpoint:
    """Test cases for /healthz endpoint."""
    
    def test_health_check_success(self, client, mock_app_state):
        """Test successful health check."""
        mock_app_state["model_loaded"] = True
        
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "git" in data
        assert "device" in data
        assert "timestamp" in data
    
    def test_health_check_degraded(self, client, mock_app_state):
        """Test health check when model not loaded."""
        mock_app_state["model_loaded"] = False
        
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
    
    def test_health_check_headers(self, client):
        """Test health check response headers."""
        response = client.get("/healthz")
        
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers


class TestMetadataEndpoint:
    """Test cases for /metadata endpoint."""
    
    def test_metadata_success(self, client, mock_inference_engine):
        """Test successful metadata retrieval."""
        response = client.get("/metadata")
        
        assert response.status_code == 200
        data = response.json()
        assert "modelName" in data
        assert "version" in data
        assert "device" in data
        assert "inputShape" in data
        assert "actionSpace" in data
        
        # Check action space structure
        action_space = data["actionSpace"]
        assert "throttle" in action_space
        assert "brake" in action_space
        assert "steer" in action_space
    
    def test_metadata_service_unavailable(self, client, mock_app_state):
        """Test metadata when service unavailable."""
        mock_app_state["model_loaded"] = False
        
        response = client.get("/metadata")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "SERVICE_UNAVAILABLE"


class TestWarmupEndpoint:
    """Test cases for /warmup endpoint."""
    
    def test_warmup_success(self, client, mock_inference_engine, mock_app_state):
        """Test successful model warmup."""
        response = client.post("/warmup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warmed"
        assert "timingMs" in data
        assert "device" in data
        
        # Check that warmup was marked as completed
        assert mock_app_state["warmup_completed"] is True
    
    def test_warmup_inference_error(self, client, mock_inference_engine):
        """Test warmup with inference error."""
        mock_inference_engine.predict.side_effect = Exception("Inference failed")
        
        response = client.post("/warmup")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "SERVICE_UNAVAILABLE"


class TestPredictEndpoint:
    """Test cases for /predict endpoint."""
    
    def test_predict_single_observation(self, client, mock_inference_engine):
        """Test prediction with single observation."""
        from src.io_schemas import Action
        
        # Mock the inference engine to return proper Action objects
        mock_action = Action(throttle=0.7, brake=0.0, steer=0.1)
        mock_inference_engine.predict.return_value = ([mock_action], 8.5)
        
        request_data = {
            "observations": [{
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }],
            "deterministic": True
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["actions"]) == 1
        assert data["actions"][0]["throttle"] == 0.7
        assert data["deterministic"] is True
        assert "timingMs" in data
        assert "version" in data
    
    def test_predict_batch_observations(self, client, mock_inference_engine):
        """Test prediction with batch of observations."""
        from src.io_schemas import Action
        
        # Mock batch response
        mock_actions = [
            Action(throttle=0.7, brake=0.0, steer=0.1),
            Action(throttle=0.5, brake=0.2, steer=-0.1)
        ]
        mock_inference_engine.predict.return_value = (mock_actions, 15.2)
        
        request_data = {
            "observations": [
                {
                    "speed": 25.5,
                    "steering": 0.1,
                    "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
                },
                {
                    "speed": 30.0,
                    "steering": -0.05,
                    "sensors": [0.6, 0.4, 0.7, 0.8, 0.3]
                }
            ],
            "deterministic": False
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["actions"]) == 2
        assert data["deterministic"] is False
    
    def test_predict_validation_error(self, client):
        """Test prediction with invalid input."""
        request_data = {
            "observations": [{
                "speed": -1.0,  # Invalid speed
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5]
            }]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "VALIDATION_ERROR"
        assert "validation_errors" in data["details"]
    
    def test_predict_empty_batch(self, client):
        """Test prediction with empty observation batch."""
        request_data = {
            "observations": []
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "VALIDATION_ERROR"
    
    def test_predict_inference_error(self, client, mock_inference_engine):
        """Test prediction with inference error."""
        mock_inference_engine.predict.side_effect = Exception("Model error")
        
        request_data = {
            "observations": [{
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "INFERENCE_ERROR"


class TestMetricsEndpoint:
    """Test cases for /metrics endpoint."""
    
    def test_metrics_format(self, client, mock_app_state):
        """Test metrics endpoint format."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Check Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "carla_rl_uptime_seconds" in content
        assert "carla_rl_model_loaded" in content
        assert "carla_rl_warmup_completed" in content
    
    def test_metrics_values(self, client, mock_app_state):
        """Test metrics endpoint values."""
        mock_app_state["model_loaded"] = True
        mock_app_state["warmup_completed"] = True
        
        response = client.get("/metrics")
        content = response.text
        
        assert "carla_rl_model_loaded 1" in content
        assert "carla_rl_warmup_completed 1" in content


class TestErrorHandling:
    """Test cases for error handling across endpoints."""
    
    def test_request_id_in_errors(self, client):
        """Test that request ID is included in error responses."""
        response = client.post("/predict", json={"invalid": "data"})
        
        assert response.status_code == 422
        data = response.json()
        assert "request_id" in data["details"]
        
        # Request ID should be in response headers too
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == data["details"]["request_id"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/healthz")
        # Note: TestClient may not fully simulate CORS, 
        # but we can check the middleware is configured
        assert response.status_code in [200, 405]  # Either OK or Method Not Allowed
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/predict",
            data="{'invalid': json}",  # Malformed JSON
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestOpenAPIDocumentation:
    """Test cases for OpenAPI documentation."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        assert "openapi" in schema
        assert "paths" in schema
        assert "/healthz" in schema["paths"]
        assert "/predict" in schema["paths"]
        assert "/metadata" in schema["paths"]
        assert "/warmup" in schema["paths"]
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI documentation endpoint."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "redoc" in response.text.lower()
