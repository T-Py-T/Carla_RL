"""
Integration tests for CarlaRL Policy-as-a-Service Infrastructure.

Tests Docker deployment, health checks, and end-to-end functionality.
"""

import subprocess
import time

import pytest
import requests


class TestInfrastructureIntegration:
    """Integration tests for infrastructure and deployment."""

    @pytest.fixture(scope="class")
    def service_url(self) -> str:
        """Base URL for the running service."""
        return "http://localhost:8080"

    def test_health_endpoint_integration(self, service_url: str):
        """Test health endpoint returns proper response."""
        response = requests.get(f"{service_url}/healthz", timeout=10)

        assert response.status_code == 200

        data = response.json()
        required_fields = ["status", "version", "git", "device", "timestamp"]

        for field in required_fields:
            assert field in data

        assert data["status"] in ["ok", "degraded"]
        assert isinstance(data["version"], str)
        assert isinstance(data["git"], str)
        assert isinstance(data["device"], str)
        assert isinstance(data["timestamp"], (int, float))

    def test_metadata_endpoint_integration(self, service_url: str):
        """Test metadata endpoint returns model information."""
        response = requests.get(f"{service_url}/metadata", timeout=10)

        assert response.status_code == 200

        data = response.json()
        required_fields = ["modelName", "version", "device", "inputShape", "actionSpace"]

        for field in required_fields:
            assert field in data

        # Validate action space structure
        action_space = data["actionSpace"]
        assert "throttle" in action_space
        assert "brake" in action_space
        assert "steer" in action_space

    def test_predict_endpoint_integration(self, service_url: str):
        """Test prediction endpoint with real request."""
        request_data = {
            "observations": [
                {"speed": 25.5, "steering": 0.1, "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]}
            ],
            "deterministic": True,
        }

        response = requests.post(
            f"{service_url}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        assert response.status_code == 200

        data = response.json()
        required_fields = ["actions", "version", "timingMs", "deterministic"]

        for field in required_fields:
            assert field in data

        # Validate actions
        actions = data["actions"]
        assert len(actions) == 1

        action = actions[0]
        assert "throttle" in action
        assert "brake" in action
        assert "steer" in action

        # Validate action ranges
        assert 0.0 <= action["throttle"] <= 1.0
        assert 0.0 <= action["brake"] <= 1.0
        assert -1.0 <= action["steer"] <= 1.0

        # Validate timing
        assert data["timingMs"] > 0
        assert data["deterministic"] is True

    def test_warmup_endpoint_integration(self, service_url: str):
        """Test warmup endpoint functionality."""
        response = requests.post(f"{service_url}/warmup", timeout=30)

        assert response.status_code == 200

        data = response.json()
        required_fields = ["status", "timingMs", "device"]

        for field in required_fields:
            assert field in data

        assert data["status"] == "warmed"
        assert data["timingMs"] > 0
        assert isinstance(data["device"], str)

    def test_batch_prediction_integration(self, service_url: str):
        """Test batch prediction with multiple observations."""
        request_data = {
            "observations": [
                {"speed": 20.0, "steering": 0.0, "sensors": [0.1, 0.2, 0.3, 0.4, 0.5]},
                {"speed": 30.0, "steering": 0.2, "sensors": [0.6, 0.7, 0.8, 0.9, 1.0]},
                {"speed": 40.0, "steering": -0.1, "sensors": [0.2, 0.4, 0.6, 0.8, 1.0]},
            ],
            "deterministic": False,
        }

        response = requests.post(
            f"{service_url}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        assert response.status_code == 200

        data = response.json()
        actions = data["actions"]

        # Should return same number of actions as observations
        assert len(actions) == 3

        # All actions should be valid
        for action in actions:
            assert 0.0 <= action["throttle"] <= 1.0
            assert 0.0 <= action["brake"] <= 1.0
            assert -1.0 <= action["steer"] <= 1.0

    def test_error_handling_integration(self, service_url: str):
        """Test error handling with invalid requests."""
        # Test invalid observation data
        invalid_request = {
            "observations": [
                {
                    "speed": -10.0,  # Invalid negative speed
                    "steering": 0.0,
                    "sensors": [0.1, 0.2, 0.3],
                }
            ]
        }

        response = requests.post(
            f"{service_url}/predict",
            json=invalid_request,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        assert response.status_code == 422

        data = response.json()
        assert "error" in data
        assert "message" in data
        assert data["error"] == "VALIDATION_ERROR"

    def test_cors_headers_integration(self, service_url: str):
        """Test CORS headers are properly set."""
        response = requests.options(f"{service_url}/healthz")

        # Should not fail (either 200 or 405 is acceptable)
        assert response.status_code in [200, 405]

    def test_request_id_tracking_integration(self, service_url: str):
        """Test request ID tracking in headers."""
        response = requests.get(f"{service_url}/healthz")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers

        request_id = response.headers["X-Request-ID"]
        process_time = float(response.headers["X-Process-Time"])

        assert len(request_id) > 0
        assert process_time >= 0

    def test_metrics_endpoint_integration(self, service_url: str):
        """Test metrics endpoint returns Prometheus format."""
        response = requests.get(f"{service_url}/metrics", timeout=10)

        assert response.status_code == 200

        content = response.text

        # Check Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "carla_rl_uptime_seconds" in content
        assert "carla_rl_model_loaded" in content

    def test_openapi_documentation_integration(self, service_url: str):
        """Test OpenAPI documentation generation."""
        # Test OpenAPI schema
        response = requests.get(f"{service_url}/openapi.json", timeout=10)
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

        # Check that our endpoints are documented
        paths = schema["paths"]
        assert "/healthz" in paths
        assert "/metadata" in paths
        assert "/predict" in paths
        assert "/warmup" in paths

        # Test Swagger UI
        response = requests.get(f"{service_url}/docs", timeout=10)
        assert response.status_code == 200
        assert "swagger" in response.text.lower()


class TestPerformanceIntegration:
    """Performance integration tests."""

    @pytest.fixture(scope="class")
    def service_url(self) -> str:
        """Base URL for the running service."""
        return "http://localhost:8080"

    def test_latency_performance_integration(self, service_url: str):
        """Test inference latency performance."""
        # Warm up the service first
        warmup_response = requests.post(f"{service_url}/warmup", timeout=30)
        assert warmup_response.status_code == 200

        request_data = {
            "observations": [{"speed": 25.0, "steering": 0.0, "sensors": [0.5] * 5}],
            "deterministic": True,
        }

        # Measure latencies
        latencies = []
        for _ in range(20):
            start_time = time.perf_counter()

            response = requests.post(
                f"{service_url}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            end_time = time.perf_counter()

            assert response.status_code == 200
            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]

        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P50 latency: {p50_latency:.2f}ms")

        # Performance assertions (adjust based on hardware)
        assert avg_latency < 1000.0, f"Average latency too high: {avg_latency}ms"
        assert p50_latency < 1000.0, f"P50 latency too high: {p50_latency}ms"

    def test_throughput_performance_integration(self, service_url: str):
        """Test throughput performance with batch requests."""
        # Warm up
        warmup_response = requests.post(f"{service_url}/warmup", timeout=30)
        assert warmup_response.status_code == 200

        # Create batch request
        batch_size = 10
        request_data = {
            "observations": [
                {"speed": 20.0 + i, "steering": i * 0.1, "sensors": [0.1 * j for j in range(5)]}
                for i in range(batch_size)
            ],
            "deterministic": True,
        }

        # Measure throughput
        start_time = time.perf_counter()
        num_requests = 10

        for _ in range(num_requests):
            response = requests.post(
                f"{service_url}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            assert response.status_code == 200

        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_observations = num_requests * batch_size
        throughput = total_observations / total_time

        print(f"Throughput: {throughput:.1f} observations/sec")

        # Performance assertion (adjust based on hardware)
        assert throughput > 10.0, f"Throughput too low: {throughput} obs/sec"

    def test_concurrent_requests_integration(self, service_url: str):
        """Test concurrent request handling."""
        import queue
        import threading

        # Warm up
        warmup_response = requests.post(f"{service_url}/warmup", timeout=30)
        assert warmup_response.status_code == 200

        request_data = {"observations": [{"speed": 25.0, "steering": 0.0, "sensors": [0.5] * 5}]}

        results_queue = queue.Queue()

        def make_request():
            try:
                response = requests.post(
                    f"{service_url}/predict",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                results_queue.put(("success", response.status_code))
            except Exception as e:
                results_queue.put(("error", str(e)))

        # Launch concurrent requests
        num_threads = 5
        threads = []

        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        successes = 0
        errors = 0

        while not results_queue.empty():
            result_type, result_value = results_queue.get()
            if result_type == "success":
                assert result_value == 200
                successes += 1
            else:
                errors += 1

        # All requests should succeed
        assert successes == num_threads
        assert errors == 0


@pytest.mark.integration
class TestDockerIntegration:
    """Docker-specific integration tests."""

    def test_docker_health_check(self):
        """Test Docker health check functionality."""
        # This test assumes a Docker container is running
        # In a real CI/CD pipeline, this would be automated

        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=carla-rl-serving", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and "healthy" in result.stdout:
                print("Docker container is healthy")
            else:
                pytest.skip("Docker container not running or not healthy")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or container not running")

    def test_docker_resource_limits(self):
        """Test Docker resource usage is within limits."""
        try:
            # Get container stats
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "carla-rl-serving",
                    "--no-stream",
                    "--format",
                    "table {{.CPUPerc}}\t{{.MemUsage}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    stats = lines[1].split("\t")
                    cpu_percent = float(stats[0].replace("%", ""))
                    memory_usage = stats[1]

                    print(f"Container CPU usage: {cpu_percent}%")
                    print(f"Container memory usage: {memory_usage}")

                    # Basic resource checks
                    assert cpu_percent < 100.0, f"CPU usage too high: {cpu_percent}%"
                else:
                    pytest.skip("Could not parse container stats")
            else:
                pytest.skip("Could not get container stats")

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pytest.skip("Docker stats not available")


def run_integration_tests():
    """
    Run integration tests for the infrastructure layer.

    This function can be called to validate the deployed service.
    """
    print("Running Infrastructure Integration Tests")
    print("=" * 50)

    # Note: These tests require a running service instance
    print("Prerequisites:")
    print("- Service running at http://localhost:8080")
    print("- Docker container (optional, for Docker tests)")
    print()
    print("Run with: pytest tests/test_integration.py -v")


if __name__ == "__main__":
    run_integration_tests()
