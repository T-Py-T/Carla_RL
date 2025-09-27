#!/usr/bin/env python3
"""
Cluster validation script for CarlaRL Policy-as-a-Service.

This script performs comprehensive testing on a running cluster deployment
to validate production readiness and performance characteristics.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class TestResult:
    """Container for test results."""
    name: str
    passed: bool
    duration_ms: float
    details: dict[str, Any]
    error: str = ""


class ClusterValidator:
    """Comprehensive cluster validation for CarlaRL serving."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: list[TestResult] = []

    def log_result(self, result: TestResult) -> None:
        """Log test result."""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {result.name} ({result.duration_ms:.1f}ms)")
        if result.error:
            print(f"    Error: {result.error}")
        if result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")
        print()

    def test_service_health(self) -> TestResult:
        """Test basic service health and availability."""
        start_time = time.perf_counter()

        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                details = {
                    "status": data.get("status"),
                    "version": data.get("version"),
                    "device": data.get("device"),
                    "response_time_ms": duration_ms
                }
                return TestResult("Service Health Check", True, duration_ms, details)
            else:
                return TestResult("Service Health Check", False, duration_ms,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult("Service Health Check", False, duration_ms, {}, str(e))

    def test_model_metadata(self) -> TestResult:
        """Test model metadata endpoint."""
        start_time = time.perf_counter()

        try:
            response = requests.get(f"{self.base_url}/metadata", timeout=10)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                required_fields = ["modelName", "version", "device", "inputShape", "actionSpace"]
                missing_fields = [f for f in required_fields if f not in data]

                if not missing_fields:
                    details = {
                        "model_name": data.get("modelName"),
                        "model_version": data.get("version"),
                        "input_shape": data.get("inputShape"),
                        "action_space_keys": list(data.get("actionSpace", {}).keys())
                    }
                    return TestResult("Model Metadata", True, duration_ms, details)
                else:
                    return TestResult("Model Metadata", False, duration_ms,
                                    {"missing_fields": missing_fields},
                                    f"Missing fields: {missing_fields}")
            else:
                return TestResult("Model Metadata", False, duration_ms,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult("Model Metadata", False, duration_ms, {}, str(e))

    def test_single_prediction(self) -> TestResult:
        """Test single observation prediction."""
        start_time = time.perf_counter()

        request_data = {
            "observations": [{
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }],
            "deterministic": True
        }

        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                actions = data.get("actions", [])

                if len(actions) == 1:
                    action = actions[0]
                    valid_action = (
                        0.0 <= action.get("throttle", -1) <= 1.0 and
                        0.0 <= action.get("brake", -1) <= 1.0 and
                        -1.0 <= action.get("steer", -2) <= 1.0
                    )

                    details = {
                        "inference_time_ms": data.get("timingMs"),
                        "total_time_ms": duration_ms,
                        "action": action,
                        "deterministic": data.get("deterministic"),
                        "model_version": data.get("version")
                    }

                    return TestResult("Single Prediction", valid_action, duration_ms, details,
                                    "" if valid_action else "Invalid action values")
                else:
                    return TestResult("Single Prediction", False, duration_ms,
                                    {"action_count": len(actions)},
                                    f"Expected 1 action, got {len(actions)}")
            else:
                return TestResult("Single Prediction", False, duration_ms,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult("Single Prediction", False, duration_ms, {}, str(e))

    def test_batch_prediction(self, batch_size: int = 10) -> TestResult:
        """Test batch prediction with multiple observations."""
        start_time = time.perf_counter()

        request_data = {
            "observations": [
                {
                    "speed": 20.0 + i * 2.0,
                    "steering": (i - 5) * 0.1,
                    "sensors": [0.1 * j + i * 0.05 for j in range(5)]
                }
                for i in range(batch_size)
            ],
            "deterministic": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                actions = data.get("actions", [])

                if len(actions) == batch_size:
                    # Validate all actions
                    valid_actions = all(
                        0.0 <= action.get("throttle", -1) <= 1.0 and
                        0.0 <= action.get("brake", -1) <= 1.0 and
                        -1.0 <= action.get("steer", -2) <= 1.0
                        for action in actions
                    )

                    details = {
                        "batch_size": batch_size,
                        "inference_time_ms": data.get("timingMs"),
                        "total_time_ms": duration_ms,
                        "avg_time_per_obs": duration_ms / batch_size,
                        "throughput_obs_per_sec": batch_size / (duration_ms / 1000)
                    }

                    return TestResult(f"Batch Prediction ({batch_size})", valid_actions,
                                    duration_ms, details,
                                    "" if valid_actions else "Invalid action values")
                else:
                    return TestResult(f"Batch Prediction ({batch_size})", False, duration_ms,
                                    {"expected": batch_size, "actual": len(actions)},
                                    "Action count mismatch")
            else:
                return TestResult(f"Batch Prediction ({batch_size})", False, duration_ms,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult(f"Batch Prediction ({batch_size})", False, duration_ms, {}, str(e))

    def test_warmup(self) -> TestResult:
        """Test model warmup functionality."""
        start_time = time.perf_counter()

        try:
            response = requests.post(f"{self.base_url}/warmup", timeout=60)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                details = {
                    "warmup_status": data.get("status"),
                    "warmup_time_ms": data.get("timingMs"),
                    "total_time_ms": duration_ms,
                    "device": data.get("device")
                }
                return TestResult("Model Warmup", True, duration_ms, details)
            else:
                return TestResult("Model Warmup", False, duration_ms,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult("Model Warmup", False, duration_ms, {}, str(e))

    def test_latency_consistency(self, iterations: int = 20) -> TestResult:
        """Test latency consistency over multiple requests."""
        request_data = {
            "observations": [{
                "speed": 25.0,
                "steering": 0.0,
                "sensors": [0.5] * 5
            }],
            "deterministic": True
        }

        latencies = []
        errors = 0

        start_time = time.perf_counter()

        for _ in range(iterations):
            try:
                req_start = time.perf_counter()
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                req_duration = (time.perf_counter() - req_start) * 1000

                if response.status_code == 200:
                    latencies.append(req_duration)
                else:
                    errors += 1
            except Exception:
                errors += 1

        total_duration = (time.perf_counter() - start_time) * 1000

        if latencies:
            details = {
                "iterations": iterations,
                "successful_requests": len(latencies),
                "failed_requests": errors,
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "success_rate": len(latencies) / iterations
            }

            # Consider test passed if >90% success rate and reasonable latency
            passed = (len(latencies) / iterations > 0.9 and
                     statistics.median(latencies) < 1000)  # < 1 second median

            return TestResult("Latency Consistency", passed, total_duration, details,
                            "" if passed else "High latency or low success rate")
        else:
            return TestResult("Latency Consistency", False, total_duration,
                            {"errors": errors}, "All requests failed")

    def test_concurrent_requests(self, concurrency: int = 5, requests_per_thread: int = 5) -> TestResult:
        """Test concurrent request handling."""
        request_data = {
            "observations": [{
                "speed": 30.0,
                "steering": 0.0,
                "sensors": [0.6] * 5
            }],
            "deterministic": False
        }

        def make_requests() -> list[float]:
            """Make multiple requests and return latencies."""
            latencies = []
            for _ in range(requests_per_thread):
                try:
                    req_start = time.perf_counter()
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    req_duration = (time.perf_counter() - req_start) * 1000

                    if response.status_code == 200:
                        latencies.append(req_duration)
                except Exception:
                    pass
            return latencies

        start_time = time.perf_counter()

        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_requests) for _ in range(concurrency)]
            results = [future.result() for future in futures]

        total_duration = (time.perf_counter() - start_time) * 1000

        # Collect all latencies
        all_latencies = []
        for latency_list in results:
            all_latencies.extend(latency_list)

        total_requests = concurrency * requests_per_thread
        successful_requests = len(all_latencies)

        if all_latencies:
            details = {
                "concurrency": concurrency,
                "requests_per_thread": requests_per_thread,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests,
                "avg_latency_ms": statistics.mean(all_latencies),
                "median_latency_ms": statistics.median(all_latencies),
                "total_time_ms": total_duration,
                "effective_throughput_rps": successful_requests / (total_duration / 1000)
            }

            # Consider passed if >80% success rate under concurrent load
            passed = successful_requests / total_requests > 0.8

            return TestResult("Concurrent Requests", passed, total_duration, details,
                            "" if passed else "Low success rate under concurrent load")
        else:
            return TestResult("Concurrent Requests", False, total_duration,
                            {"total_requests": total_requests}, "All requests failed")

    def test_error_handling(self) -> TestResult:
        """Test error handling with invalid requests."""
        start_time = time.perf_counter()

        # Test with invalid data
        invalid_request = {
            "observations": [{
                "speed": -10.0,  # Invalid negative speed
                "steering": 0.0,
                "sensors": [0.1, 0.2, 0.3]
            }]
        }

        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=invalid_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Should return 422 for validation error
            if response.status_code == 422:
                data = response.json()
                has_error_structure = all(field in data for field in ["error", "message"])

                details = {
                    "status_code": response.status_code,
                    "error_type": data.get("error"),
                    "has_proper_structure": has_error_structure
                }

                return TestResult("Error Handling", has_error_structure, duration_ms, details,
                                "" if has_error_structure else "Missing error structure")
            else:
                return TestResult("Error Handling", False, duration_ms,
                                {"expected_status": 422, "actual_status": response.status_code},
                                f"Expected 422, got {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult("Error Handling", False, duration_ms, {}, str(e))

    def test_metrics_endpoint(self) -> TestResult:
        """Test metrics endpoint availability."""
        start_time = time.perf_counter()

        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                content = response.text
                has_prometheus_format = (
                    "# HELP" in content and
                    "# TYPE" in content and
                    "carla_rl" in content
                )

                details = {
                    "content_length": len(content),
                    "has_prometheus_format": has_prometheus_format,
                    "sample_metrics": content[:200] + "..." if len(content) > 200 else content
                }

                return TestResult("Metrics Endpoint", has_prometheus_format, duration_ms, details,
                                "" if has_prometheus_format else "Invalid Prometheus format")
            else:
                return TestResult("Metrics Endpoint", False, duration_ms,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult("Metrics Endpoint", False, duration_ms, {}, str(e))

    def run_all_tests(self) -> dict[str, Any]:
        """Run all validation tests."""
        print("🧪 Starting Cluster Validation for CarlaRL Policy-as-a-Service")
        print(f"Target: {self.base_url}")
        print("=" * 70)

        # Basic functionality tests
        self.log_result(self.test_service_health())
        self.log_result(self.test_model_metadata())
        self.log_result(self.test_warmup())

        # Prediction tests
        self.log_result(self.test_single_prediction())
        self.log_result(self.test_batch_prediction(5))
        self.log_result(self.test_batch_prediction(20))

        # Performance tests
        self.log_result(self.test_latency_consistency(20))
        self.log_result(self.test_concurrent_requests(3, 5))

        # Error handling and monitoring
        self.log_result(self.test_error_handling())
        self.log_result(self.test_metrics_endpoint())

        # Summary
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        print("=" * 70)
        print("📊 VALIDATION SUMMARY")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
        print(f"Overall Status: {'✅ PRODUCTION READY' if success_rate >= 0.9 else '⚠️  NEEDS ATTENTION'}")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "production_ready": success_rate >= 0.9,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ]
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate CarlaRL Policy-as-a-Service cluster deployment")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the service (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed results (JSON format)"
    )

    args = parser.parse_args()

    validator = ClusterValidator(args.url)
    results = validator.run_all_tests()

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {args.output}")

    # Exit with appropriate code
    return 0 if results["production_ready"] else 1


if __name__ == "__main__":
    exit(main())
