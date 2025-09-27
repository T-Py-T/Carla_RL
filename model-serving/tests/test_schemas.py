"""
Unit tests for Pydantic schemas in CarlaRL Policy-as-a-Service.

Tests validation, serialization, and edge cases for all API schemas.
"""

import time

import pytest
from pydantic import ValidationError

from src.io_schemas import (
    Action,
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    Observation,
    PredictRequest,
    PredictResponse,
)


class TestObservation:
    """Test cases for Observation schema."""

    def test_valid_observation(self):
        """Test valid observation creation."""
        obs = Observation(speed=25.5, steering=0.1, sensors=[0.8, 0.2, 0.5, 0.9, 0.1])
        assert obs.speed == 25.5
        assert obs.steering == 0.1
        assert len(obs.sensors) == 5

    def test_speed_validation(self):
        """Test speed field validation."""
        # Valid speeds
        Observation(speed=0.0, steering=0.0, sensors=[1.0])
        Observation(speed=100.0, steering=0.0, sensors=[1.0])
        Observation(speed=200.0, steering=0.0, sensors=[1.0])

        # Invalid speeds
        with pytest.raises(ValidationError):
            Observation(speed=-1.0, steering=0.0, sensors=[1.0])
        with pytest.raises(ValidationError):
            Observation(speed=201.0, steering=0.0, sensors=[1.0])

    def test_steering_validation(self):
        """Test steering field validation."""
        # Valid steering
        Observation(speed=25.0, steering=-1.0, sensors=[1.0])
        Observation(speed=25.0, steering=0.0, sensors=[1.0])
        Observation(speed=25.0, steering=1.0, sensors=[1.0])

        # Invalid steering
        with pytest.raises(ValidationError):
            Observation(speed=25.0, steering=-1.1, sensors=[1.0])
        with pytest.raises(ValidationError):
            Observation(speed=25.0, steering=1.1, sensors=[1.0])

    def test_sensors_validation(self):
        """Test sensors field validation."""
        # Valid sensors
        Observation(speed=25.0, steering=0.0, sensors=[0.5])
        Observation(speed=25.0, steering=0.0, sensors=list(range(100)))

        # Empty sensors (invalid)
        with pytest.raises(ValidationError):
            Observation(speed=25.0, steering=0.0, sensors=[])

        # Out of bounds sensors
        with pytest.raises(ValidationError):
            Observation(speed=25.0, steering=0.0, sensors=[1001.0])
        with pytest.raises(ValidationError):
            Observation(speed=25.0, steering=0.0, sensors=[-1001.0])

    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        obs = Observation(speed=25.5, steering=0.1, sensors=[0.8, 0.2])
        json_str = obs.model_dump_json()
        obs_restored = Observation.model_validate_json(json_str)
        assert obs == obs_restored


class TestAction:
    """Test cases for Action schema."""

    def test_valid_action(self):
        """Test valid action creation."""
        action = Action(throttle=0.7, brake=0.0, steer=0.1)
        assert action.throttle == 0.7
        assert action.brake == 0.0
        assert action.steer == 0.1

    def test_throttle_validation(self):
        """Test throttle field validation."""
        # Valid throttle
        Action(throttle=0.0, brake=0.0, steer=0.0)
        Action(throttle=1.0, brake=0.0, steer=0.0)

        # Invalid throttle
        with pytest.raises(ValidationError):
            Action(throttle=-0.1, brake=0.0, steer=0.0)
        with pytest.raises(ValidationError):
            Action(throttle=1.1, brake=0.0, steer=0.0)

    def test_brake_validation(self):
        """Test brake field validation."""
        # Valid brake
        Action(throttle=0.0, brake=0.0, steer=0.0)
        Action(throttle=0.0, brake=1.0, steer=0.0)

        # Invalid brake
        with pytest.raises(ValidationError):
            Action(throttle=0.0, brake=-0.1, steer=0.0)
        with pytest.raises(ValidationError):
            Action(throttle=0.0, brake=1.1, steer=0.0)

    def test_steer_validation(self):
        """Test steer field validation."""
        # Valid steer
        Action(throttle=0.0, brake=0.0, steer=-1.0)
        Action(throttle=0.0, brake=0.0, steer=1.0)

        # Invalid steer
        with pytest.raises(ValidationError):
            Action(throttle=0.0, brake=0.0, steer=-1.1)
        with pytest.raises(ValidationError):
            Action(throttle=0.0, brake=0.0, steer=1.1)


class TestPredictRequest:
    """Test cases for PredictRequest schema."""

    def test_valid_request(self):
        """Test valid predict request."""
        obs = Observation(speed=25.0, steering=0.0, sensors=[0.5])
        request = PredictRequest(observations=[obs], deterministic=True)
        assert len(request.observations) == 1
        assert request.deterministic is True

    def test_default_deterministic(self):
        """Test default deterministic value."""
        obs = Observation(speed=25.0, steering=0.0, sensors=[0.5])
        request = PredictRequest(observations=[obs])
        assert request.deterministic is False

    def test_batch_size_limits(self):
        """Test batch size validation."""
        obs = Observation(speed=25.0, steering=0.0, sensors=[0.5])

        # Empty batch (invalid)
        with pytest.raises(ValidationError):
            PredictRequest(observations=[])

        # Large batch (at limit)
        large_batch = [obs] * 1000
        request = PredictRequest(observations=large_batch)
        assert len(request.observations) == 1000

        # Too large batch
        with pytest.raises(ValidationError):
            PredictRequest(observations=[obs] * 1001)


class TestPredictResponse:
    """Test cases for PredictResponse schema."""

    def test_valid_response(self):
        """Test valid predict response."""
        action = Action(throttle=0.7, brake=0.0, steer=0.1)
        response = PredictResponse(
            actions=[action], version="v0.1.0", timingMs=8.5, deterministic=True
        )
        assert len(response.actions) == 1
        assert response.version == "v0.1.0"
        assert response.timingMs == 8.5
        assert response.deterministic is True

    def test_timing_validation(self):
        """Test timing field validation."""
        action = Action(throttle=0.7, brake=0.0, steer=0.1)

        # Valid timing
        PredictResponse(actions=[action], version="v0.1.0", timingMs=0.0, deterministic=False)
        PredictResponse(actions=[action], version="v0.1.0", timingMs=1000.0, deterministic=False)

        # Invalid timing
        with pytest.raises(ValidationError):
            PredictResponse(actions=[action], version="v0.1.0", timingMs=-1.0, deterministic=False)


class TestHealthResponse:
    """Test cases for HealthResponse schema."""

    def test_valid_health_response(self):
        """Test valid health response."""
        response = HealthResponse(status="ok", version="v0.1.0", git="abc123", device="cpu")
        assert response.status == "ok"
        assert response.version == "v0.1.0"
        assert response.git == "abc123"
        assert response.device == "cpu"
        assert isinstance(response.timestamp, float)

    def test_timestamp_auto_generation(self):
        """Test automatic timestamp generation."""
        start_time = time.time()
        response = HealthResponse(status="ok", version="v0.1.0", git="abc123", device="cpu")
        end_time = time.time()

        assert start_time <= response.timestamp <= end_time


class TestMetadataResponse:
    """Test cases for MetadataResponse schema."""

    def test_valid_metadata_response(self):
        """Test valid metadata response."""
        response = MetadataResponse(
            modelName="carla-ppo",
            version="v0.1.0",
            device="cpu",
            inputShape=[5],
            actionSpace={"throttle": [0.0, 1.0], "brake": [0.0, 1.0], "steer": [-1.0, 1.0]},
        )
        assert response.modelName == "carla-ppo"
        assert response.inputShape == [5]
        assert len(response.actionSpace) == 3


class TestErrorResponse:
    """Test cases for ErrorResponse schema."""

    def test_valid_error_response(self):
        """Test valid error response."""
        response = ErrorResponse(
            error="VALIDATION_ERROR", message="Invalid input", details={"field": "sensors"}
        )
        assert response.error == "VALIDATION_ERROR"
        assert response.message == "Invalid input"
        assert response.details["field"] == "sensors"
        assert isinstance(response.timestamp, float)

    def test_optional_details(self):
        """Test error response without details."""
        response = ErrorResponse(error="INTERNAL_ERROR", message="Something went wrong")
        assert response.details is None


class TestSchemaExamples:
    """Test that schema examples are valid."""

    def test_observation_example(self):
        """Test Observation schema example."""
        example = Observation.model_config["json_schema_extra"]["example"]
        obs = Observation(**example)
        assert obs.speed == 25.5

    def test_action_example(self):
        """Test Action schema example."""
        example = Action.model_config["json_schema_extra"]["example"]
        action = Action(**example)
        assert action.throttle == 0.7

    def test_predict_request_example(self):
        """Test PredictRequest schema example."""
        example = PredictRequest.model_config["json_schema_extra"]["example"]
        request = PredictRequest(**example)
        assert len(request.observations) == 1

    def test_predict_response_example(self):
        """Test PredictResponse schema example."""
        example = PredictResponse.model_config["json_schema_extra"]["example"]
        response = PredictResponse(**example)
        assert len(response.actions) == 1
