#!/usr/bin/env python3
"""
Kubernetes Testing Script

This script provides comprehensive testing for Kubernetes deployments
including service validation, performance testing, and monitoring verification.
"""

import subprocess
import sys
import time
import json
import argparse
from typing import Dict, List, Any
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KubernetesTester:
    """Comprehensive Kubernetes testing framework."""
    
    def __init__(self, namespace: str = "default", deployment_file: str = "deploy/k8s/test-deployment.yaml"):
        """Initialize Kubernetes tester."""
        self.namespace = namespace
        self.deployment_file = deployment_file
        self.service_name = "model-serving-test-service"
        self.base_url = None  # Will be set after port forwarding
        
    def run_kubectl(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a kubectl command and return the result."""
        full_command = ["kubectl"] + command
        if self.namespace != "default":
            full_command.extend(["-n", self.namespace])
        
        logger.info(f"Running command: {' '.join(full_command)}")
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
    
    def check_cluster_connection(self) -> bool:
        """Check if kubectl can connect to the cluster."""
        logger.info("Checking cluster connection...")
        try:
            self.run_kubectl(["cluster-info"])
            logger.info("Cluster connection successful")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to connect to cluster")
            return False
    
    def deploy_application(self) -> bool:
        """Deploy the application to Kubernetes."""
        logger.info("Deploying application to Kubernetes...")
        try:
            self.run_kubectl(["apply", "-f", self.deployment_file])
            logger.info("Application deployed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to deploy application")
            return False
    
    def wait_for_deployment(self, deployment_name: str = "model-serving-test", timeout: int = 300) -> bool:
        """Wait for deployment to be ready."""
        logger.info(f"Waiting for deployment {deployment_name} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = self.run_kubectl([
                    "get", "deployment", deployment_name, 
                    "-o", "jsonpath={.status.readyReplicas}"
                ])
                
                ready_replicas = int(result.stdout.strip() or "0")
                if ready_replicas > 0:
                    logger.info(f"Deployment {deployment_name} is ready with {ready_replicas} replicas")
                    return True
                
                time.sleep(5)
            except subprocess.CalledProcessError:
                time.sleep(5)
        
        logger.error(f"Deployment {deployment_name} failed to become ready within {timeout} seconds")
        return False
    
    def setup_port_forward(self, local_port: int = 8080) -> bool:
        """Set up port forwarding to access the service."""
        logger.info(f"Setting up port forwarding to local port {local_port}...")
        
        try:
            # Start port forwarding in background
            self.port_forward_process = subprocess.Popen([
                "kubectl", "port-forward", 
                f"service/{self.service_name}", 
                f"{local_port}:80"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for port forwarding to establish
            time.sleep(5)
            
            self.base_url = f"http://localhost:{local_port}"
            logger.info(f"Port forwarding established: {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up port forwarding: {e}")
            return False
    
    def check_service_health(self) -> Dict[str, Any]:
        """Check the health of the service."""
        if not self.base_url:
            return {"error": "Port forwarding not established"}
        
        logger.info("Checking service health...")
        
        try:
            # Health endpoint
            health_response = requests.get(f"{self.base_url}/healthz", timeout=10)
            health_data = health_response.json() if health_response.status_code == 200 else None
            
            # Metrics endpoint
            metrics_response = requests.get(f"{self.base_url}/metrics", timeout=10)
            metrics_available = metrics_response.status_code == 200
            
            # Metadata endpoint
            metadata_response = requests.get(f"{self.base_url}/metadata", timeout=10)
            metadata_data = metadata_response.json() if metadata_response.status_code == 200 else None
            
            return {
                "status": "healthy" if health_data and metrics_available and metadata_data else "unhealthy",
                "health_endpoint": {
                    "status_code": health_response.status_code,
                    "data": health_data
                },
                "metrics_endpoint": {
                    "status_code": metrics_response.status_code,
                    "available": metrics_available
                },
                "metadata_endpoint": {
                    "status_code": metadata_response.status_code,
                    "data": metadata_data
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests using a Kubernetes job."""
        logger.info("Running integration tests...")
        
        try:
            # Check if test job exists
            result = self.run_kubectl([
                "get", "job", "model-serving-test-job"
            ], check=False)
            
            if result.returncode != 0:
                logger.error("Integration test job not found")
                return {"success": False, "error": "Test job not found"}
            
            # Wait for job to complete
            self.run_kubectl([
                "wait", "--for=condition=complete", 
                "job/model-serving-test-job", 
                "--timeout=300s"
            ])
            
            # Get job logs
            logs_result = self.run_kubectl([
                "logs", "job/model-serving-test-job"
            ])
            
            # Check job status
            status_result = self.run_kubectl([
                "get", "job", "model-serving-test-job", 
                "-o", "jsonpath={.status.conditions[0].type}"
            ])
            
            success = status_result.stdout.strip() == "Complete"
            
            return {
                "success": success,
                "logs": logs_result.stdout,
                "status": status_result.stdout.strip()
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": e.returncode
            }
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run load tests using a Kubernetes job."""
        logger.info("Running load tests...")
        
        try:
            # Check if load test job exists
            result = self.run_kubectl([
                "get", "job", "model-serving-load-test-job"
            ], check=False)
            
            if result.returncode != 0:
                logger.error("Load test job not found")
                return {"success": False, "error": "Load test job not found"}
            
            # Wait for job to complete
            self.run_kubectl([
                "wait", "--for=condition=complete", 
                "job/model-serving-load-test-job", 
                "--timeout=600s"
            ])
            
            # Get job logs
            logs_result = self.run_kubectl([
                "logs", "job/model-serving-load-test-job"
            ])
            
            # Check job status
            status_result = self.run_kubectl([
                "get", "job", "model-serving-load-test-job", 
                "-o", "jsonpath={.status.conditions[0].type}"
            ])
            
            success = status_result.stdout.strip() == "Complete"
            
            return {
                "success": success,
                "logs": logs_result.stdout,
                "status": status_result.stdout.strip()
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": e.returncode
            }
    
    def run_monitoring_tests(self) -> Dict[str, Any]:
        """Run monitoring tests using a Kubernetes job."""
        logger.info("Running monitoring tests...")
        
        try:
            # Check if monitoring test job exists
            result = self.run_kubectl([
                "get", "job", "model-serving-monitoring-test-job"
            ], check=False)
            
            if result.returncode != 0:
                logger.error("Monitoring test job not found")
                return {"success": False, "error": "Monitoring test job not found"}
            
            # Wait for job to complete
            self.run_kubectl([
                "wait", "--for=condition=complete", 
                "job/model-serving-monitoring-test-job", 
                "--timeout=300s"
            ])
            
            # Get job logs
            logs_result = self.run_kubectl([
                "logs", "job/model-serving-monitoring-test-job"
            ])
            
            # Check job status
            status_result = self.run_kubectl([
                "get", "job", "model-serving-monitoring-test-job", 
                "-o", "jsonpath={.status.conditions[0].type}"
            ])
            
            success = status_result.stdout.strip() == "Complete"
            
            return {
                "success": success,
                "logs": logs_result.stdout,
                "status": status_result.stdout.strip()
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": e.returncode
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests using a Kubernetes job."""
        logger.info("Running performance tests...")
        
        try:
            # Check if performance job exists
            result = self.run_kubectl([
                "get", "job", "model-serving-performance-job"
            ], check=False)
            
            if result.returncode != 0:
                logger.error("Performance test job not found")
                return {"success": False, "error": "Performance test job not found"}
            
            # Wait for job to complete
            self.run_kubectl([
                "wait", "--for=condition=complete", 
                "job/model-serving-performance-job", 
                "--timeout=600s"
            ])
            
            # Get job logs
            logs_result = self.run_kubectl([
                "logs", "job/model-serving-performance-job"
            ])
            
            # Check job status
            status_result = self.run_kubectl([
                "get", "job", "model-serving-performance-job", 
                "-o", "jsonpath={.status.conditions[0].type}"
            ])
            
            success = status_result.stdout.strip() == "Complete"
            
            return {
                "success": success,
                "logs": logs_result.stdout,
                "status": status_result.stdout.strip()
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": e.returncode
            }
    
    def get_pod_logs(self, pod_name: str = None) -> str:
        """Get logs from pods."""
        try:
            if pod_name:
                result = self.run_kubectl(["logs", pod_name])
            else:
                result = self.run_kubectl([
                    "logs", "-l", "app=model-serving-test"
                ])
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Failed to get logs: {e.stderr}"
    
    def cleanup(self) -> bool:
        """Clean up Kubernetes resources."""
        logger.info("Cleaning up Kubernetes resources...")
        try:
            self.run_kubectl(["delete", "-f", self.deployment_file])
            logger.info("Resources cleaned up successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to clean up resources")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive Kubernetes test...")
        
        test_results = {
            "start_time": time.time(),
            "cluster_connected": False,
            "application_deployed": False,
            "deployment_ready": False,
            "port_forward_established": False,
            "health_checks": {},
            "integration_tests": {},
            "load_tests": {},
            "monitoring_tests": {},
            "performance_tests": {},
            "logs": {},
            "success": False
        }
        
        try:
            # Check cluster connection
            if not self.check_cluster_connection():
                test_results["error"] = "Failed to connect to cluster"
                return test_results
            
            test_results["cluster_connected"] = True
            
            # Deploy application
            if not self.deploy_application():
                test_results["error"] = "Failed to deploy application"
                return test_results
            
            test_results["application_deployed"] = True
            
            # Wait for deployment to be ready
            if not self.wait_for_deployment():
                test_results["error"] = "Deployment failed to become ready"
                return test_results
            
            test_results["deployment_ready"] = True
            
            # Set up port forwarding
            if not self.setup_port_forward():
                test_results["error"] = "Failed to set up port forwarding"
                return test_results
            
            test_results["port_forward_established"] = True
            
            # Run health checks
            test_results["health_checks"] = self.check_service_health()
            
            # Run integration tests
            test_results["integration_tests"] = self.run_integration_tests()
            
            # Run load tests
            test_results["load_tests"] = self.run_load_tests()
            
            # Run monitoring tests
            test_results["monitoring_tests"] = self.run_monitoring_tests()
            
            # Run performance tests
            test_results["performance_tests"] = self.run_performance_tests()
            
            # Collect logs
            test_results["logs"] = {
                "pods": self.get_pod_logs()
            }
            
            # Determine overall success
            test_results["success"] = (
                test_results["health_checks"].get("status") == "healthy" and
                test_results["integration_tests"].get("success", False)
            )
            
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Test failed with error: {e}")
        
        finally:
            # Clean up
            self.cleanup()
            test_results["end_time"] = time.time()
            test_results["duration"] = test_results["end_time"] - test_results["start_time"]
        
        return test_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Kubernetes Testing Script')
    parser.add_argument('--namespace', default='default', 
                       help='Kubernetes namespace to use')
    parser.add_argument('--deployment-file', default='deploy/k8s/test-deployment.yaml', 
                       help='Kubernetes deployment file to use')
    parser.add_argument('--output', help='Output file for test results')
    parser.add_argument('--deploy-only', action='store_true', 
                       help='Only deploy application, do not run tests')
    parser.add_argument('--cleanup-only', action='store_true', 
                       help='Only clean up resources')
    
    args = parser.parse_args()
    
    # Create tester
    tester = KubernetesTester(args.namespace, args.deployment_file)
    
    if args.cleanup_only:
        success = tester.cleanup()
        sys.exit(0 if success else 1)
    
    if args.deploy_only:
        success = (
            tester.check_cluster_connection() and
            tester.deploy_application() and
            tester.wait_for_deployment() and
            tester.setup_port_forward()
        )
        if success:
            print("Application deployed successfully")
            print(f"Service URL: {tester.base_url}")
        sys.exit(0 if success else 1)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to {args.output}")
    
    # Print summary
    print("\n=== Kubernetes Test Results ===")
    print(f"Cluster Connected: {results['cluster_connected']}")
    print(f"Application Deployed: {results['application_deployed']}")
    print(f"Deployment Ready: {results['deployment_ready']}")
    print(f"Port Forward Established: {results['port_forward_established']}")
    print(f"Overall Success: {results['success']}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    
    # Print health check results
    health_checks = results.get('health_checks', {})
    print(f"Service Health: {health_checks.get('status', 'unknown')}")
    
    # Print test results
    for test_type, test_result in results.items():
        if isinstance(test_result, dict) and 'success' in test_result:
            print(f"{test_type}: {'PASSED' if test_result['success'] else 'FAILED'}")
    
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
