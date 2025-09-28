#!/usr/bin/env python3
"""
Comprehensive Deployment Testing Runner

This script provides a unified interface for running tests across
both Docker Compose and Kubernetes deployments.
"""

import argparse
import sys
import json
import time
from typing import Dict, List, Any, Optional
import logging

# Import test modules
from test_docker_compose import DockerComposeTester
from test_kubernetes import KubernetesTester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentTestRunner:
    """Unified deployment testing runner."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.results = {
            "start_time": time.time(),
            "docker_compose": {},
            "kubernetes": {},
            "overall_success": False
        }
    
    def run_docker_compose_tests(self, profiles: List[str] = None) -> Dict[str, Any]:
        """Run Docker Compose tests."""
        logger.info("Starting Docker Compose tests...")
        
        docker_tester = DockerComposeTester()
        results = docker_tester.run_comprehensive_test(profiles)
        
        self.results["docker_compose"] = results
        return results
    
    def run_kubernetes_tests(self, namespace: str = "default") -> Dict[str, Any]:
        """Run Kubernetes tests."""
        logger.info("Starting Kubernetes tests...")
        
        k8s_tester = KubernetesTester(namespace)
        results = k8s_tester.run_comprehensive_test()
        
        self.results["kubernetes"] = results
        return results
    
    def run_all_tests(self, docker_profiles: List[str] = None, k8s_namespace: str = "default") -> Dict[str, Any]:
        """Run all deployment tests."""
        logger.info("Starting comprehensive deployment testing...")
        
        # Run Docker Compose tests
        docker_results = self.run_docker_compose_tests(docker_profiles)
        
        # Run Kubernetes tests
        k8s_results = self.run_kubernetes_tests(k8s_namespace)
        
        # Determine overall success
        self.results["overall_success"] = (
            docker_results.get("success", False) and
            k8s_results.get("success", False)
        )
        
        self.results["end_time"] = time.time()
        self.results["total_duration"] = self.results["end_time"] - self.results["start_time"]
        
        return self.results
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "summary": {
                "overall_success": self.results["overall_success"],
                "total_duration": self.results.get("total_duration", 0),
                "docker_compose_success": self.results.get("docker_compose", {}).get("success", False),
                "kubernetes_success": self.results.get("kubernetes", {}).get("success", False)
            },
            "docker_compose": self.results.get("docker_compose", {}),
            "kubernetes": self.results.get("kubernetes", {}),
            "timestamp": time.time()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Comprehensive test report saved to {output_file}")
        
        return report
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE DEPLOYMENT TEST RESULTS")
        print(f"{'='*60}")
        
        # Overall results
        print(f"Overall Success: {'PASSED' if self.results['overall_success'] else 'FAILED'}")
        print(f"Total Duration: {self.results.get('total_duration', 0):.2f} seconds")
        
        # Docker Compose results
        docker_results = self.results.get("docker_compose", {})
        print("\nDocker Compose Tests:")
        print(f"  Success: {'PASSED' if docker_results.get('success', False) else 'FAILED'}")
        print(f"  Duration: {docker_results.get('duration', 0):.2f} seconds")
        print(f"  Services Started: {docker_results.get('services_started', False)}")
        
        # Health checks
        health_checks = docker_results.get('health_checks', {})
        for service, health in health_checks.items():
            print(f"  {service}: {health.get('status', 'unknown')}")
        
        # Test results
        for test_type, test_result in docker_results.items():
            if isinstance(test_result, dict) and 'success' in test_result:
                print(f"  {test_type}: {'PASSED' if test_result['success'] else 'FAILED'}")
        
        # Kubernetes results
        k8s_results = self.results.get("kubernetes", {})
        print("\nKubernetes Tests:")
        print(f"  Success: {'PASSED' if k8s_results.get('success', False) else 'FAILED'}")
        print(f"  Duration: {k8s_results.get('duration', 0):.2f} seconds")
        print(f"  Cluster Connected: {k8s_results.get('cluster_connected', False)}")
        print(f"  Application Deployed: {k8s_results.get('application_deployed', False)}")
        print(f"  Deployment Ready: {k8s_results.get('deployment_ready', False)}")
        print(f"  Port Forward Established: {k8s_results.get('port_forward_established', False)}")
        
        # Health checks
        k8s_health = k8s_results.get('health_checks', {})
        print(f"  Service Health: {k8s_health.get('status', 'unknown')}")
        
        # Test results
        for test_type, test_result in k8s_results.items():
            if isinstance(test_result, dict) and 'success' in test_result:
                print(f"  {test_type}: {'PASSED' if test_result['success'] else 'FAILED'}")
        
        # Errors
        if 'error' in docker_results:
            print(f"\nDocker Compose Error: {docker_results['error']}")
        
        if 'error' in k8s_results:
            print(f"\nKubernetes Error: {k8s_results['error']}")
        
        print(f"{'='*60}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Comprehensive Deployment Testing Runner')
    
    # Test selection
    parser.add_argument('--docker-only', action='store_true', 
                       help='Run only Docker Compose tests')
    parser.add_argument('--k8s-only', action='store_true', 
                       help='Run only Kubernetes tests')
    
    # Docker Compose options
    parser.add_argument('--docker-profiles', nargs='+', 
                       choices=['testing', 'load-testing', 'monitoring', 'performance'],
                       help='Docker Compose profiles to run')
    parser.add_argument('--docker-compose-file', default='deploy/docker/docker-compose.test.yml',
                       help='Docker Compose file to use')
    
    # Kubernetes options
    parser.add_argument('--k8s-namespace', default='default',
                       help='Kubernetes namespace to use')
    parser.add_argument('--k8s-deployment-file', default='deploy/k8s/test-deployment.yaml',
                       help='Kubernetes deployment file to use')
    
    # Output options
    parser.add_argument('--output', help='Output file for test results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test runner
    runner = DeploymentTestRunner()
    
    try:
        if args.docker_only:
            # Run only Docker Compose tests
            results = runner.run_docker_compose_tests(args.docker_profiles)
            runner.results["overall_success"] = results.get("success", False)
            
        elif args.k8s_only:
            # Run only Kubernetes tests
            results = runner.run_kubernetes_tests(args.k8s_namespace)
            runner.results["overall_success"] = results.get("success", False)
            
        else:
            # Run all tests
            runner.run_all_tests(args.docker_profiles, args.k8s_namespace)
        
        # Generate report
        report = runner.generate_report(args.output)
        
        # Print summary
        runner.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if runner.results["overall_success"] else 1)
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
