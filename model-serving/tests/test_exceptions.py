"""
Unit tests for exception handling in CarlaRL Policy-as-a-Service.

Tests custom exceptions, error response creation, and FastAPI integration.
"""

import time
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from src.exceptions import (
    ArtifactValidationError,
    CarlaRLServingException,
    InferenceError,
    ModelLoadingError,
    PreprocessingError,
    ResourceExhaustedError,
    ServiceUnavailableError,
    ValidationError,
    carla_rl_exception_handler,
    create_error_response,
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)


class TestCarlaRLServingException:
    """Test cases for base CarlaRLServingException."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = CarlaRLServingException("Test message")
        assert exc.message == "Test message"
        assert exc.error_code == "CARLA_RL_ERROR"
        assert exc.status_code == 500
        assert isinstance(exc.timestamp, float)
        assert exc.details == {}

    def test_exception_with_details(self):
        """Test exception with custom details."""
        details = {"field": "test", "value": 123}
        exc = CarlaRLServingException(
            message="Custom message", error_code="CUSTOM_ERROR", details=details, status_code=422
        )
        assert exc.message == "Custom message"
        assert exc.error_code == "CUSTOM_ERROR"
        assert exc.status_code == 422
        assert exc.details == details


class TestSpecificExceptions:
    """Test cases for specific exception types."""

    def test_model_loading_error(self):
        """Test ModelLoadingError."""
        exc = ModelLoadingError("Failed to load model")
        assert exc.error_code == "MODEL_LOADING_ERROR"
        assert exc.status_code == 500

    def test_artifact_validation_error(self):
        """Test ArtifactValidationError."""
        exc = ArtifactValidationError("Invalid artifact")
        assert exc.error_code == "ARTIFACT_VALIDATION_ERROR"
        assert exc.status_code == 422

    def test_inference_error(self):
        """Test InferenceError."""
        exc = InferenceError("Inference failed")
        assert exc.error_code == "INFERENCE_ERROR"
        assert exc.status_code == 500

    def test_preprocessing_error(self):
        """Test PreprocessingError."""
        exc = PreprocessingError("Preprocessing failed")
        assert exc.error_code == "PREPROCESSING_ERROR"
        assert exc.status_code == 422

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Validation failed")
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.status_code == 422

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        exc = ServiceUnavailableError("Service unavailable")
        assert exc.error_code == "SERVICE_UNAVAILABLE"
        assert exc.status_code == 503

    def test_resource_exhausted_error(self):
        """Test ResourceExhaustedError."""
        exc = ResourceExhaustedError("Resources exhausted")
        assert exc.error_code == "RESOURCE_EXHAUSTED"
        assert exc.status_code == 429


class TestErrorResponseCreation:
    """Test cases for error response creation."""

    def test_carla_rl_serving_exception_response(self):
        """Test error response for CarlaRLServingException."""
        exc = ModelLoadingError("Failed to load", {"path": "/test"})
        response = create_error_response(exc, "req-123")

        assert response["error"] == "MODEL_LOADING_ERROR"
        assert response["message"] == "Failed to load"
        assert response["details"]["path"] == "/test"
        assert response["details"]["request_id"] == "req-123"
        assert "timestamp" in response

    def test_pydantic_validation_error_response(self):
        """Test error response for PydanticValidationError."""
        try:
            from pydantic import BaseModel, Field

            class TestModel(BaseModel):
                value: int = Field(..., ge=0)

            TestModel(value=-1)
        except PydanticValidationError as exc:
            response = create_error_response(exc, "req-456")

            assert response["error"] == "VALIDATION_ERROR"
            assert response["message"] == "Input validation failed"
            assert "validation_errors" in response["details"]
            assert response["details"]["request_id"] == "req-456"

    def test_http_exception_response(self):
        """Test error response for HTTPException."""
        exc = HTTPException(status_code=404, detail="Not found")
        response = create_error_response(exc)

        assert response["error"] == "HTTP_ERROR"
        assert response["message"] == "Not found"
        assert response["details"]["status_code"] == 404

    def test_generic_exception_response(self):
        """Test error response for generic Exception."""
        exc = ValueError("Generic error")
        response = create_error_response(exc, "req-789")

        assert response["error"] == "INTERNAL_ERROR"
        assert response["message"] == "An unexpected error occurred"
        assert response["details"]["error_type"] == "ValueError"
        assert response["details"]["request_id"] == "req-789"


class TestExceptionHandlers:
    """Test cases for FastAPI exception handlers."""

    @pytest.fixture
    def mock_request(self):
        """Create mock FastAPI request."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.request_id = "test-request-123"
        request.url = Mock()
        request.url.path = "/test"
        request.method = "POST"
        return request

    @pytest.mark.asyncio
    async def test_carla_rl_exception_handler(self, mock_request):
        """Test CarlaRLServingException handler."""
        exc = InferenceError("Test inference error", {"batch_size": 5})

        response = await carla_rl_exception_handler(mock_request, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Check response content
        content = response.body.decode()
        assert "INFERENCE_ERROR" in content
        assert "Test inference error" in content

    @pytest.mark.asyncio
    async def test_validation_exception_handler(self, mock_request):
        """Test PydanticValidationError handler."""
        try:
            from pydantic import BaseModel, Field

            class TestModel(BaseModel):
                value: int = Field(..., ge=0)

            TestModel(value=-1)
        except PydanticValidationError as exc:
            response = await validation_exception_handler(mock_request, exc)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_http_exception_handler(self, mock_request):
        """Test HTTPException handler."""
        exc = HTTPException(status_code=404, detail="Resource not found")

        response = await http_exception_handler(mock_request, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_generic_exception_handler(self, mock_request):
        """Test generic Exception handler."""
        exc = RuntimeError("Unexpected error")

        response = await generic_exception_handler(mock_request, exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500


class TestErrorResponseTimestamps:
    """Test timestamp handling in error responses."""

    def test_timestamp_consistency(self):
        """Test that timestamps are consistent and recent."""
        start_time = time.time()

        exc = ValidationError("Test error")
        response = create_error_response(exc)

        end_time = time.time()

        assert start_time <= response["timestamp"] <= end_time
        assert start_time <= exc.timestamp <= end_time

    def test_exception_timestamp_preservation(self):
        """Test that exception timestamps are preserved in responses."""
        exc = ModelLoadingError("Test error")
        original_timestamp = exc.timestamp

        # Small delay to ensure timestamp would be different if regenerated
        time.sleep(0.01)

        response = create_error_response(exc)

        assert response["timestamp"] == original_timestamp
