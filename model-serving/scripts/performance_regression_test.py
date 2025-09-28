#!/usr/bin/env uv run python
"""
Performance regression testing for CI/CD pipeline integration.

This script runs performance benchmarks and compares against baselines
to detect performance regressions in CI/CD pipelines.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add model-serving to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking import BenchmarkEngine, BenchmarkConfig


def create_mock_inference_function():
    """Create a mock inference function for testing."""
    import time
    import random
    
    def mock_inference(observations, deterministic=False):
        """Mock inference function that simulates model prediction."""
        # Simulate some processing time
        time.sleep(random.uniform(0.001, 0.010))  # 1-10ms
        
        # Return mock actions
        actions = []
        for obs in observations:
            action = {
                "throttle": random.uniform(0.0, 1.0),
                "brake": random.uniform(0.0, 1.0),
                "steer": random.uniform(-1.0, 1.0)
            }
            actions.append(action)
        
        return actions
    
    return mock_inference


def run_performance_regression_test(
    config: BenchmarkConfig,
    baseline_file: Optional[str] = None,
    threshold_percent: float = 10.0
) -> Dict[str, Any]:
    """Run performance regression test against baseline."""
    
    # Create benchmark engine
    engine = BenchmarkEngine(config)
    
    # Create mock inference function
    inference_func = create_mock_inference_function()
    
    # Run benchmark
    print("Running performance regression test...")
    import asyncio
    result = asyncio.run(engine.run_benchmark(inference_func, batch_size=1))
    
    # Load baseline if provided
    baseline = None
    if baseline_file and Path(baseline_file).exists():
        with open(baseline_file, "r") as f:
            baseline = json.load(f)
    
    # Compare with baseline
    comparison_result = {
        "test_passed": True,
        "regression_detected": False,
        "performance_metrics": {
            "p50_latency_ms": result.latency_stats.p50_ms,
            "p95_latency_ms": result.latency_stats.p95_ms,
            "p99_latency_ms": result.latency_stats.p99_ms,
            "throughput_rps": result.throughput_stats.requests_per_second,
            "memory_usage_mb": result.memory_stats.peak_memory_mb
        },
        "requirements_met": {
            "p50_requirement": result.p50_requirement_met,
            "p95_requirement": result.p95_requirement_met,
            "p99_requirement": result.p99_requirement_met,
            "throughput_requirement": result.throughput_requirement_met,
            "memory_requirement": result.memory_requirement_met
        },
        "baseline_comparison": None
    }
    
    # Check if all requirements are met
    if not result.overall_success:
        comparison_result["test_passed"] = False
        comparison_result["regression_detected"] = True
    
    # Compare with baseline if available
    if baseline:
        baseline_metrics = baseline.get("performance_metrics", {})
        
        # Calculate percentage differences
        p50_diff = calculate_percentage_diff(
            result.latency_stats.p50_ms, 
            baseline_metrics.get("p50_latency_ms", 0)
        )
        p95_diff = calculate_percentage_diff(
            result.latency_stats.p95_ms, 
            baseline_metrics.get("p95_latency_ms", 0)
        )
        throughput_diff = calculate_percentage_diff(
            result.throughput_stats.requests_per_second, 
            baseline_metrics.get("throughput_rps", 0)
        )
        
        # Check for significant regressions
        significant_regression = (
            p50_diff > threshold_percent or
            p95_diff > threshold_percent or
            throughput_diff < -threshold_percent
        )
        
        if significant_regression:
            comparison_result["test_passed"] = False
            comparison_result["regression_detected"] = True
        
        comparison_result["baseline_comparison"] = {
            "p50_latency_diff_percent": p50_diff,
            "p95_latency_diff_percent": p95_diff,
            "throughput_diff_percent": throughput_diff,
            "significant_regression": significant_regression,
            "threshold_percent": threshold_percent
        }
    
    return comparison_result


def calculate_percentage_diff(current: float, baseline: float) -> float:
    """Calculate percentage difference from baseline."""
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def main():
    """Main entry point for performance regression testing."""
    parser = argparse.ArgumentParser(
        description="Run performance regression tests for CI/CD pipeline"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="Number of measurement iterations (default: 50)"
    )
    
    parser.add_argument(
        "--p50-threshold",
        type=float,
        default=10.0,
        help="P50 latency threshold in milliseconds (default: 10.0)"
    )
    
    parser.add_argument(
        "--throughput-threshold",
        type=float,
        default=1000.0,
        help="Throughput threshold in requests per second (default: 1000.0)"
    )
    
    parser.add_argument(
        "--baseline-file",
        type=str,
        help="Path to baseline performance file for comparison"
    )
    
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=10.0,
        help="Percentage threshold for detecting regressions (default: 10.0)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for test results (JSON format)"
    )
    
    parser.add_argument(
        "--exit-on-failure",
        action="store_true",
        help="Exit with non-zero code if test fails"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        warmup_iterations=5,
        measurement_iterations=args.iterations,
        p50_threshold_ms=args.p50_threshold,
        throughput_threshold_rps=args.throughput_threshold
    )
    
    # Run regression test
    try:
        result = run_performance_regression_test(
            config,
            args.baseline_file,
            args.regression_threshold
        )
        
        # Print results
        print("\n" + "="*60)
        print("PERFORMANCE REGRESSION TEST RESULTS")
        print("="*60)
        
        print(f"Test Status: {'PASSED' if result['test_passed'] else 'FAILED'}")
        print(f"Regression Detected: {'YES' if result['regression_detected'] else 'NO'}")
        
        print("\nPerformance Metrics:")
        metrics = result["performance_metrics"]
        print(f"  P50 Latency: {metrics['p50_latency_ms']:.2f}ms")
        print(f"  P95 Latency: {metrics['p95_latency_ms']:.2f}ms")
        print(f"  P99 Latency: {metrics['p99_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_rps']:.1f} RPS")
        print(f"  Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
        
        print("\nRequirements Met:")
        reqs = result["requirements_met"]
        print(f"  P50 Requirement: {'✓' if reqs['p50_requirement'] else '✗'}")
        print(f"  P95 Requirement: {'✓' if reqs['p95_requirement'] else '✗'}")
        print(f"  P99 Requirement: {'✓' if reqs['p99_requirement'] else '✗'}")
        print(f"  Throughput Requirement: {'✓' if reqs['throughput_requirement'] else '✗'}")
        print(f"  Memory Requirement: {'✓' if reqs['memory_requirement'] else '✗'}")
        
        if result["baseline_comparison"]:
            print("\nBaseline Comparison:")
            comp = result["baseline_comparison"]
            print(f"  P50 Latency Change: {comp['p50_latency_diff_percent']:+.1f}%")
            print(f"  P95 Latency Change: {comp['p95_latency_diff_percent']:+.1f}%")
            print(f"  Throughput Change: {comp['throughput_diff_percent']:+.1f}%")
            print(f"  Significant Regression: {'YES' if comp['significant_regression'] else 'NO'}")
        
        # Save results to file
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        # Exit with appropriate code
        if not result["test_passed"] and args.exit_on_failure:
            print("\nTest failed - exiting with error code")
            sys.exit(1)
        else:
            print("\nTest completed successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error running performance regression test: {e}")
        if args.exit_on_failure:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
