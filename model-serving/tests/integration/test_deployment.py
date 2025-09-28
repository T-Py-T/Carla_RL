"""
Integration tests for Docker Compose and Kubernetes deployments.

This module provides comprehensive testing for both deployment methods,
including health checks, API functionality, performance validation,
and monitoring verification.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a deployment test."""
    test_name: str
    success: bool
    duration_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


class DeploymentTester:
    """
    Comprehensive deployment testing framework.
    
    Supports testing for:
    - Docker Compose deployments
    - Kubernetes deployments
    - Health checks and API functionality
    - Performance validation
    - Monitoring and observability
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 30):
        """Initialize deployment tester."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
    def wait_for_service(self, max_retries: int = 30, retry_interval: int = 2) -> bool:
        """Wait for service to be ready."""
        logger.info(f"Waiting for service at {self.base_url} to be ready...")
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/healthz")
                if response.status_code == 200:
                    logger.info("Service is ready!")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            time.sleep(retry_interval)
        
        logger.error(f"Service failed to become ready after {max_retries * retry_interval} seconds")
        return False
    
    def test_health_endpoint(self) -> TestResult:
        """Test health endpoint functionality."""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/healthz")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                return TestResult(
                    test_name="health_endpoint",
                    success=True,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "response_data": health_data,
                        "response_time_ms": duration_ms
                    }
                )
            else:
                return TestResult(
                    test_name="health_endpoint",
                    success=False,
                    duration_ms=duration_ms,
                    error_message=f"Health check failed with status {response.status_code}",
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="health_endpoint",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    def test_metadata_endpoint(self) -> TestResult:
        """Test metadata endpoint functionality."""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/metadata")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                metadata = response.json()
                return TestResult(
                    test_name="metadata_endpoint",
                    success=True,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "metadata": metadata,
                        "response_time_ms": duration_ms
                    }
                )
            else:
                return TestResult(
                    test_name="metadata_endpoint",
                    success=False,
                    duration_ms=duration_ms,
                    error_message=f"Metadata endpoint failed with status {response.status_code}",
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="metadata_endpoint",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    def test_warmup_endpoint(self) -> TestResult:
        """Test model warmup endpoint."""
        start_time = time.time()
        
        try:
            response = self.session.post(f"{self.base_url}/warmup")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                warmup_data = response.json()
                return TestResult(
                    test_name="warmup_endpoint",
                    success=True,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "warmup_data": warmup_data,
                        "response_time_ms": duration_ms
                    }
                )
            else:
                return TestResult(
                    test_name="warmup_endpoint",
                    success=False,
                    duration_ms=duration_ms,
                    error_message=f"Warmup failed with status {response.status_code}",
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="warmup_endpoint",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    def test_predict_endpoint(self, batch_size: int = 1) -> TestResult:
        """Test prediction endpoint with various batch sizes."""
        start_time = time.time()
        
        try:
            # Create test observations
            observations = []
            for i in range(batch_size):
                observations.append({
                    "speed": 25.0 + i,
                    "steering": 0.0,
                    "sensors": [0.5] * 5
                })
            
            payload = {
                "observations": observations,
                "deterministic": True
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                prediction_data = response.json()
                return TestResult(
                    test_name="predict_endpoint",
                    success=True,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "batch_size": batch_size,
                        "prediction_data": prediction_data,
                        "response_time_ms": duration_ms
                    }
                )
            else:
                return TestResult(
                    test_name="predict_endpoint",
                    success=False,
                    duration_ms=duration_ms,
                    error_message=f"Prediction failed with status {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "batch_size": batch_size
                    }
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="predict_endpoint",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"batch_size": batch_size}
            )
    
    def test_metrics_endpoint(self) -> TestResult:
        """Test metrics endpoint."""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                metrics_text = response.text
                # Basic validation of Prometheus format
                has_help_lines = "# HELP" in metrics_text
                has_type_lines = "# TYPE" in metrics_text
                has_metric_lines = any(line and not line.startswith('#') for line in metrics_text.split('\n'))
                
                return TestResult(
                    test_name="metrics_endpoint",
                    success=has_help_lines and has_type_lines and has_metric_lines,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "content_type": response.headers.get('content-type'),
                        "metrics_length": len(metrics_text),
                        "has_help_lines": has_help_lines,
                        "has_type_lines": has_type_lines,
                        "has_metric_lines": has_metric_lines,
                        "response_time_ms": duration_ms
                    }
                )
            else:
                return TestResult(
                    test_name="metrics_endpoint",
                    success=False,
                    duration_ms=duration_ms,
                    error_message=f"Metrics endpoint failed with status {response.status_code}",
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="metrics_endpoint",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    def test_versions_endpoint(self) -> TestResult:
        """Test versions endpoint."""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/versions")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                versions_data = response.json()
                return TestResult(
                    test_name="versions_endpoint",
                    success=True,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "versions_data": versions_data,
                        "response_time_ms": duration_ms
                    }
                )
            else:
                return TestResult(
                    test_name="versions_endpoint",
                    success=False,
                    duration_ms=duration_ms,
                    error_message=f"Versions endpoint failed with status {response.status_code}",
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="versions_endpoint",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    def test_performance(self, num_requests: int = 10, batch_sizes: List[int] = [1, 4, 8]) -> TestResult:
        """Test performance with multiple requests and batch sizes."""
        start_time = time.time()
        results = []
        
        try:
            for batch_size in batch_sizes:
                batch_results = []
                for i in range(num_requests):
                    result = self.test_predict_endpoint(batch_size)
                    batch_results.append(result)
                    if not result.success:
                        break
                
                if batch_results:
                    success_rate = sum(1 for r in batch_results if r.success) / len(batch_results)
                    avg_duration = sum(r.duration_ms for r in batch_results if r.success) / max(1, sum(1 for r in batch_results if r.success))
                    
                    results.append({
                        "batch_size": batch_size,
                        "success_rate": success_rate,
                        "avg_duration_ms": avg_duration,
                        "total_requests": len(batch_results)
                    })
            
            duration_ms = (time.time() - start_time) * 1000
            overall_success = all(r["success_rate"] > 0.8 for r in results)
            
            return TestResult(
                test_name="performance_test",
                success=overall_success,
                duration_ms=duration_ms,
                details={
                    "batch_results": results,
                    "total_duration_ms": duration_ms,
                    "overall_success": overall_success
                }
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="performance_test",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"batch_results": results}
            )
    
    def test_error_handling(self) -> TestResult:
        """Test error handling with invalid requests."""
        start_time = time.time()
        error_tests = []
        
        try:
            # Test invalid JSON
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    data="invalid json",
                    headers={"Content-Type": "application/json"}
                )
                error_tests.append({
                    "test": "invalid_json",
                    "status_code": response.status_code,
                    "expected_error": True,
                    "got_error": response.status_code >= 400
                })
            except Exception as e:
                error_tests.append({
                    "test": "invalid_json",
                    "error": str(e),
                    "expected_error": True,
                    "got_error": True
                })
            
            # Test missing required fields
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json={"invalid": "data"},
                    headers={"Content-Type": "application/json"}
                )
                error_tests.append({
                    "test": "missing_fields",
                    "status_code": response.status_code,
                    "expected_error": True,
                    "got_error": response.status_code >= 400
                })
            except Exception as e:
                error_tests.append({
                    "test": "missing_fields",
                    "error": str(e),
                    "expected_error": True,
                    "got_error": True
                })
            
            # Test invalid endpoint
            try:
                response = self.session.get(f"{self.base_url}/invalid-endpoint")
                error_tests.append({
                    "test": "invalid_endpoint",
                    "status_code": response.status_code,
                    "expected_error": True,
                    "got_error": response.status_code == 404
                })
            except Exception as e:
                error_tests.append({
                    "test": "invalid_endpoint",
                    "error": str(e),
                    "expected_error": True,
                    "got_error": True
                })
            
            duration_ms = (time.time() - start_time) * 1000
            success = all(test["got_error"] for test in error_tests)
            
            return TestResult(
                test_name="error_handling",
                success=success,
                duration_ms=duration_ms,
                details={"error_tests": error_tests}
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name="error_handling",
                success=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"error_tests": error_tests}
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all deployment tests."""
        logger.info("Starting comprehensive deployment tests...")
        
        # Wait for service to be ready
        if not self.wait_for_service():
            return [TestResult(
                test_name="service_ready",
                success=False,
                duration_ms=0,
                error_message="Service failed to become ready"
            )]
        
        # Run all tests
        tests = [
            self.test_health_endpoint,
            self.test_metadata_endpoint,
            self.test_warmup_endpoint,
            self.test_predict_endpoint,
            self.test_metrics_endpoint,
            self.test_versions_endpoint,
            self.test_performance,
            self.test_error_handling
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                logger.info(f"Test {result.test_name}: {'PASSED' if result.success else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed with exception: {e}")
                results.append(TestResult(
                    test_name=test_func.__name__,
                    success=False,
                    duration_ms=0,
                    error_message=str(e)
                ))
        
        return results
    
    def generate_report(self, results: List[TestResult], output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration_ms for r in results)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration_ms": total_duration,
                "avg_duration_ms": total_duration / total_tests if total_tests > 0 else 0
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "error_message": r.error_message,
                    "details": r.details
                }
                for r in results
            ],
            "timestamp": time.time(),
            "base_url": self.base_url
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to {output_file}")
        
        return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deployment Testing Framework')
    parser.add_argument('--url', default='http://localhost:8080', help='Service URL')
    parser.add_argument('--output', help='Output file for test results')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = DeploymentTester(args.url, args.timeout)
    results = tester.run_all_tests()
    
    # Generate report
    report = tester.generate_report(results, args.output)
    
    # Print summary
    summary = report['summary']
    print("\n=== Deployment Test Results ===")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Total Duration: {summary['total_duration_ms']:.2f}ms")
    
    # Print failed tests
    failed_tests = [r for r in results if not r.success]
    if failed_tests:
        print("\n=== Failed Tests ===")
        for result in failed_tests:
            print(f"- {result.test_name}: {result.error_message}")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
