#!/usr/bin/env python3
"""
Docker Compose Testing Script

This script provides comprehensive testing for Docker Compose deployments
including service validation, performance testing, and monitoring verification.
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DockerComposeTester:
    """Comprehensive Docker Compose testing framework."""
    
    def __init__(self, compose_file: str = "deploy/docker/docker-compose.test.yml"):
        """Initialize Docker Compose tester."""
        self.compose_file = compose_file
        self.base_url = "http://localhost:8080"
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"
        
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        logger.info(f"Running command: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
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
    
    def start_services(self, profiles: List[str] = None) -> bool:
        """Start Docker Compose services."""
        logger.info("Starting Docker Compose services...")
        
        command = ["docker-compose", "-f", self.compose_file, "up", "-d"]
        if profiles:
            command.extend(["--profile"] + profiles)
        
        try:
            self.run_command(command)
            logger.info("Services started successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to start services")
            return False
    
    def stop_services(self) -> bool:
        """Stop Docker Compose services."""
        logger.info("Stopping Docker Compose services...")
        
        try:
            self.run_command(["docker-compose", "-f", self.compose_file, "down", "-v"])
            logger.info("Services stopped successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to stop services")
            return False
    
    def wait_for_service(self, url: str, max_retries: int = 30, retry_interval: int = 2) -> bool:
        """Wait for a service to be ready."""
        logger.info(f"Waiting for service at {url} to be ready...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Service at {url} is ready!")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            time.sleep(retry_interval)
        
        logger.error(f"Service at {url} failed to become ready after {max_retries * retry_interval} seconds")
        return False
    
    def check_service_health(self) -> Dict[str, Any]:
        """Check the health of all services."""
        logger.info("Checking service health...")
        
        health_checks = {
            "carla_rl_service": self.check_carla_rl_health(),
            "prometheus": self.check_prometheus_health(),
            "grafana": self.check_grafana_health()
        }
        
        return health_checks
    
    def check_carla_rl_health(self) -> Dict[str, Any]:
        """Check CarlaRL service health."""
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
    
    def check_prometheus_health(self) -> Dict[str, Any]:
        """Check Prometheus health."""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query?query=up", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "data": data
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_grafana_health(self) -> Dict[str, Any]:
        """Check Grafana health."""
        try:
            response = requests.get(f"{self.grafana_url}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "data": data
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_comprehensive_test(self, profiles: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive Docker Compose test...")
        
        test_results = {
            "start_time": time.time(),
            "services_started": False,
            "health_checks": {},
            "success": False
        }
        
        try:
            # Start services
            if not self.start_services(profiles):
                test_results["error"] = "Failed to start services"
                return test_results
            
            test_results["services_started"] = True
            
            # Wait for services to be ready
            if not self.wait_for_service(self.base_url):
                test_results["error"] = "CarlaRL service failed to become ready"
                return test_results
            
            # Run health checks
            test_results["health_checks"] = self.check_service_health()
            
            # Determine overall success
            test_results["success"] = (
                test_results["health_checks"].get("carla_rl_service", {}).get("status") == "healthy"
            )
            
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Test failed with error: {e}")
        
        finally:
            # Stop services
            self.stop_services()
            test_results["end_time"] = time.time()
            test_results["duration"] = test_results["end_time"] - test_results["start_time"]
        
        return test_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Docker Compose Testing Script')
    parser.add_argument('--compose-file', default='deploy/docker/docker-compose.test.yml', 
                       help='Docker Compose file to use')
    parser.add_argument('--profiles', nargs='+', 
                       choices=['testing', 'load-testing', 'monitoring', 'performance'],
                       help='Docker Compose profiles to run')
    parser.add_argument('--output', help='Output file for test results')
    parser.add_argument('--start-only', action='store_true', 
                       help='Only start services, do not run tests')
    parser.add_argument('--stop-only', action='store_true', 
                       help='Only stop services')
    
    args = parser.parse_args()
    
    # Create tester
    tester = DockerComposeTester(args.compose_file)
    
    if args.stop_only:
        success = tester.stop_services()
        sys.exit(0 if success else 1)
    
    if args.start_only:
        success = tester.start_services(args.profiles)
        if success:
            print("Services started successfully")
            print(f"CarlaRL Service: {tester.base_url}")
            print(f"Prometheus: {tester.prometheus_url}")
            print(f"Grafana: {tester.grafana_url}")
        sys.exit(0 if success else 1)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(args.profiles)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to {args.output}")
    
    # Print summary
    print(f"\n=== Docker Compose Test Results ===")
    print(f"Services Started: {results['services_started']}")
    print(f"Overall Success: {results['success']}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    
    # Print health check results
    health_checks = results.get('health_checks', {})
    for service, health in health_checks.items():
        print(f"{service}: {health.get('status', 'unknown')}")
    
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
