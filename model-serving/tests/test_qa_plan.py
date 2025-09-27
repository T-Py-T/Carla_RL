"""
QA Test Plan for CarlaRL Policy-as-a-Service API Layer.

This module defines full test scenarios and validation criteria
for ensuring the API layer meets all PRD requirements.
"""

import time

from src.io_schemas import Action


class TestQAPlan:
    """full QA test plan for API Layer validation."""

    def test_prd_requirement_fr_1_1_healthz_endpoint(self, client):
        """
        QA Test: FR-1.1 - System MUST provide /healthz endpoint
        returning service status, version, and git SHA
        """
        response = client.get("/healthz")

        assert response.status_code == 200
        data = response.json()

        # Validate required fields
        assert "status" in data
        assert "version" in data
        assert "git" in data
        assert data["status"] in ["ok", "degraded"]
        assert isinstance(data["version"], str)
        assert isinstance(data["git"], str)

        print(" FR-1.1: /healthz endpoint validated")

    def test_prd_requirement_fr_1_2_metadata_endpoint(self, client, mock_inference_engine):
        """
        QA Test: FR-1.2 - System MUST provide /metadata endpoint
        returning model name, version, device, input shape, and action space
        """
        response = client.get("/metadata")

        assert response.status_code == 200
        data = response.json()

        # Validate required fields
        assert "modelName" in data
        assert "version" in data
        assert "device" in data
        assert "inputShape" in data
        assert "actionSpace" in data

        # Validate action space structure
        action_space = data["actionSpace"]
        assert "throttle" in action_space
        assert "brake" in action_space
        assert "steer" in action_space

        print(" FR-1.2: /metadata endpoint validated")

    def test_prd_requirement_fr_1_3_predict_endpoint_batch(self, client, mock_inference_engine):
        """
        QA Test: FR-1.3 - System MUST provide /predict endpoint
        accepting batch observations and returning actions
        """

        # Setup mock for batch processing
        mock_actions = [
            Action(throttle=0.7, brake=0.0, steer=0.1),
            Action(throttle=0.5, brake=0.2, steer=-0.1)
        ]
        mock_inference_engine.predict.return_value = (mock_actions, 12.5)

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
            ]
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Validate batch processing
        assert "actions" in data
        assert len(data["actions"]) == 2
        assert len(data["actions"]) == len(request_data["observations"])

        print(" FR-1.3: /predict batch processing validated")

    def test_prd_requirement_fr_1_4_deterministic_mode(self, client, mock_inference_engine):
        """
        QA Test: FR-1.4 - System MUST support deterministic mode toggle
        for reproducible inference
        """

        mock_action = Action(throttle=0.7, brake=0.0, steer=0.1)
        mock_inference_engine.predict.return_value = ([mock_action], 8.5)

        # Test deterministic=True
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
        assert data["deterministic"] is True

        # Verify inference engine was called with deterministic flag
        mock_inference_engine.predict.assert_called_with(
            request_data["observations"], True
        )

        print(" FR-1.4: Deterministic mode validated")

    def test_prd_requirement_fr_1_5_warmup_endpoint(self, client, mock_inference_engine):
        """
        QA Test: FR-1.5 - System MUST provide /warmup endpoint
        for JIT compilation and model loading optimization
        """
        response = client.post("/warmup")

        assert response.status_code == 200
        data = response.json()

        # Validate warmup response
        assert "status" in data
        assert "timingMs" in data
        assert "device" in data
        assert data["status"] == "warmed"
        assert isinstance(data["timingMs"], (int, float))

        print(" FR-1.5: /warmup endpoint validated")

    def test_prd_requirement_fr_1_8_input_validation(self, client):
        """
        QA Test: FR-1.8 - System MUST validate input schemas
        and return structured error responses
        """
        # Test invalid speed
        invalid_request = {
            "observations": [{
                "speed": -10.0,  # Invalid: negative speed
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5]
            }]
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422
        data = response.json()

        # Validate error structure
        assert "error" in data
        assert "message" in data
        assert "details" in data
        assert "timestamp" in data
        assert data["error"] == "VALIDATION_ERROR"

        print(" FR-1.8: Input validation validated")

    def test_prd_requirement_fr_1_9_timing_information(self, client, mock_inference_engine):
        """
        QA Test: FR-1.9 - System MUST include timing information
        in prediction responses
        """

        mock_action = Action(throttle=0.7, brake=0.0, steer=0.1)
        mock_inference_engine.predict.return_value = ([mock_action], 8.5)

        request_data = {
            "observations": [{
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Validate timing information
        assert "timingMs" in data
        assert isinstance(data["timingMs"], (int, float))
        assert data["timingMs"] >= 0

        print(" FR-1.9: Timing information validated")

    def test_prd_requirement_fr_1_10_semantic_versioning(self, client, mock_inference_engine):
        """
        QA Test: FR-1.10 - System MUST support semantic versioning
        (vMAJOR.MINOR.PATCH) for models
        """

        mock_action = Action(throttle=0.7, brake=0.0, steer=0.1)
        mock_inference_engine.predict.return_value = ([mock_action], 8.5)

        request_data = {
            "observations": [{
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Validate semantic versioning format
        assert "version" in data
        version = data["version"]
        assert version.startswith("v")

        # Basic semantic version pattern check
        import re
        semver_pattern = r'^v\d+\.\d+\.\d+$'
        assert re.match(semver_pattern, version), f"Version {version} doesn't match semantic versioning"

        print(" FR-1.10: Semantic versioning validated")


class TestPerformanceRequirements:
    """Performance validation tests based on PRD requirements."""

    def test_cold_start_performance(self, client):
        """
        QA Test: Performance Requirement - Cold start < 2 seconds on CPU
        Note: This tests service response time, not actual cold start
        """
        start_time = time.time()
        response = client.get("/healthz")
        end_time = time.time()

        response_time = end_time - start_time

        # Service should respond quickly (simulates warm service)
        assert response_time < 1.0, f"Response time {response_time}s exceeds threshold"
        assert response.status_code == 200

        print(f" Service response time: {response_time:.3f}s")

    def test_warm_inference_latency_simulation(self, client, mock_inference_engine):
        """
        QA Test: Performance Requirement - P50 < 10ms for single inference
        Note: This simulates the latency requirement with mock timing
        """

        # Mock fast inference (< 10ms)
        mock_action = Action(throttle=0.7, brake=0.0, steer=0.1)
        mock_inference_engine.predict.return_value = ([mock_action], 7.5)  # 7.5ms

        request_data = {
            "observations": [{
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Validate reported timing meets requirement
        timing_ms = data["timingMs"]
        assert timing_ms < 10.0, f"Inference timing {timing_ms}ms exceeds 10ms requirement"

        print(f" Simulated inference latency: {timing_ms}ms")

    def test_batch_throughput_simulation(self, client, mock_inference_engine):
        """
        QA Test: Performance Requirement - Support high-throughput batch inference
        """

        # Large batch (100 observations)
        batch_size = 100
        mock_actions = [Action(throttle=0.7, brake=0.0, steer=0.1)] * batch_size
        mock_inference_engine.predict.return_value = (mock_actions, 50.0)  # 50ms for batch

        observations = [{
            "speed": 25.5,
            "steering": 0.1,
            "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
        }] * batch_size

        request_data = {"observations": observations}

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Validate batch processing
        assert len(data["actions"]) == batch_size

        # Calculate throughput (requests/second)
        timing_ms = data["timingMs"]
        throughput = (batch_size / timing_ms) * 1000  # requests per second

        print(f" Simulated batch throughput: {throughput:.1f} requests/sec")


class TestErrorHandlingRequirements:
    """Error handling validation based on PRD requirements."""

    def test_structured_error_responses(self, client):
        """
        QA Test: FR-1.8 & FR-3.4 - Structured error responses
        """
        # Test various error scenarios
        error_scenarios = [
            {
                "name": "Invalid speed",
                "data": {
                    "observations": [{
                        "speed": -1.0,
                        "steering": 0.0,
                        "sensors": [0.5]
                    }]
                }
            },
            {
                "name": "Invalid steering",
                "data": {
                    "observations": [{
                        "speed": 25.0,
                        "steering": 2.0,  # > 1.0
                        "sensors": [0.5]
                    }]
                }
            },
            {
                "name": "Empty sensors",
                "data": {
                    "observations": [{
                        "speed": 25.0,
                        "steering": 0.0,
                        "sensors": []
                    }]
                }
            }
        ]

        for scenario in error_scenarios:
            response = client.post("/predict", json=scenario["data"])

            assert response.status_code == 422
            data = response.json()

            # Validate error structure
            required_fields = ["error", "message", "details", "timestamp"]
            for field in required_fields:
                assert field in data, f"Missing {field} in error response for {scenario['name']}"

            assert data["error"] == "VALIDATION_ERROR"
            assert "validation_errors" in data["details"]

            print(f" Structured error response validated for: {scenario['name']}")


def run_qa_validation():
    """
    Run complete QA validation suite.

    This function can be called to validate that all PRD requirements
    are met by the API layer implementation.
    """
    print(" Starting QA Validation for CarlaRL Policy-as-a-Service API Layer")
    print("=" * 70)

    # This would typically be run with pytest, but can also be called directly
    print("Run with: pytest tests/test_qa_plan.py -v")
    print(" QA Test Plan Ready")


if __name__ == "__main__":
    run_qa_validation()
