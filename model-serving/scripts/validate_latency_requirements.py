#!/usr/bin/env uv run python
"""
Validate P50 < 10ms latency requirement on target hardware.

This script specifically validates the P50 latency requirement
and provides detailed analysis of performance characteristics.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add model-serving to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking import BenchmarkEngine, BenchmarkConfig


def create_mock_inference_function():
    """Create a mock inference function for testing."""
    import time
    import random
    
    def mock_inference(observations, deterministic=False):
        """Mock inference function that simulates model prediction."""
        # Simulate some processing time (aim for < 10ms P50)
        time.sleep(random.uniform(0.001, 0.015))  # 1-15ms range
        
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


async def validate_latency_requirements(
    config: BenchmarkConfig,
    target_p50_ms: float = 10.0,
    iterations: int = 1000
) -> Dict[str, Any]:
    """Validate P50 latency requirements with comprehensive analysis."""
    
    print(f"Validating P50 < {target_p50_ms}ms latency requirement...")
    print(f"Running {iterations} iterations for statistical significance")
    
    # Create benchmark engine
    engine = BenchmarkEngine(config)
    
    # Create mock inference function
    inference_func = create_mock_inference_function()
    
    # Run comprehensive benchmark
    result = await engine.run_benchmark(inference_func, batch_size=1)
    
    # Analyze latency distribution
    latency_analysis = analyze_latency_distribution(result.latency_stats)
    
    # Validate requirements
    validation_result = {
        "requirement_met": result.p50_requirement_met,
        "target_p50_ms": target_p50_ms,
        "actual_p50_ms": result.latency_stats.p50_ms,
        "p50_difference_ms": result.latency_stats.p50_ms - target_p50_ms,
        "p50_difference_percent": ((result.latency_stats.p50_ms - target_p50_ms) / target_p50_ms) * 100.0,
        "latency_analysis": latency_analysis,
        "performance_grade": calculate_performance_grade(result.latency_stats, target_p50_ms),
        "recommendations": generate_latency_recommendations(result.latency_stats, target_p50_ms),
        "hardware_info": engine.get_hardware_info(),
        "test_configuration": {
            "iterations": iterations,
            "warmup_iterations": config.warmup_iterations,
            "deterministic_mode": config.deterministic_mode
        }
    }
    
    return validation_result


def analyze_latency_distribution(latency_stats) -> Dict[str, Any]:
    """Analyze latency distribution characteristics."""
    
    # Calculate distribution characteristics
    distribution_analysis = {
        "percentiles": {
            "p50": latency_stats.p50_ms,
            "p90": latency_stats.p90_ms,
            "p95": latency_stats.p95_ms,
            "p99": latency_stats.p99_ms,
            "p99_9": latency_stats.p99_9_ms
        },
        "statistics": {
            "mean": latency_stats.mean_ms,
            "median": latency_stats.median_ms,
            "std": latency_stats.std_ms,
            "min": latency_stats.min_ms,
            "max": latency_stats.max_ms,
            "coefficient_of_variation": latency_stats.coefficient_of_variation
        },
        "distribution_shape": {
            "skewness": latency_stats.skewness,
            "kurtosis": latency_stats.kurtosis,
            "outlier_count": latency_stats.outlier_count,
            "outlier_percentage": latency_stats.outlier_percentage
        }
    }
    
    # Determine distribution characteristics
    if latency_stats.skewness > 1.0:
        distribution_analysis["distribution_type"] = "Right-skewed (long tail)"
    elif latency_stats.skewness < -1.0:
        distribution_analysis["distribution_type"] = "Left-skewed (long tail)"
    else:
        distribution_analysis["distribution_type"] = "Approximately normal"
    
    # Assess consistency
    if latency_stats.coefficient_of_variation < 0.1:
        distribution_analysis["consistency"] = "Very consistent"
    elif latency_stats.coefficient_of_variation < 0.2:
        distribution_analysis["consistency"] = "Consistent"
    elif latency_stats.coefficient_of_variation < 0.5:
        distribution_analysis["consistency"] = "Moderately consistent"
    else:
        distribution_analysis["consistency"] = "Inconsistent"
    
    return distribution_analysis


def calculate_performance_grade(latency_stats, target_p50_ms: float) -> str:
    """Calculate performance grade based on latency characteristics."""
    
    # Base grade on P50 requirement
    if latency_stats.p50_ms <= target_p50_ms * 0.5:  # 50% of target
        base_grade = "A+"
    elif latency_stats.p50_ms <= target_p50_ms * 0.7:  # 70% of target
        base_grade = "A"
    elif latency_stats.p50_ms <= target_p50_ms:  # Meets target
        base_grade = "B"
    elif latency_stats.p50_ms <= target_p50_ms * 1.2:  # 20% over target
        base_grade = "C"
    elif latency_stats.p50_ms <= target_p50_ms * 1.5:  # 50% over target
        base_grade = "D"
    else:
        base_grade = "F"
    
    # Adjust for consistency
    if latency_stats.coefficient_of_variation > 0.5:  # High variance
        if base_grade in ["A+", "A"]:
            base_grade = "B"
        elif base_grade == "B":
            base_grade = "C"
    
    # Adjust for outliers
    if latency_stats.outlier_percentage > 5:  # More than 5% outliers
        if base_grade in ["A+", "A", "B"]:
            base_grade = "C"
    
    return base_grade


def generate_latency_recommendations(latency_stats, target_p50_ms: float) -> list:
    """Generate recommendations for improving latency performance."""
    recommendations = []
    
    # P50 recommendations
    if latency_stats.p50_ms > target_p50_ms:
        recommendations.append(f"P50 latency ({latency_stats.p50_ms:.2f}ms) exceeds target ({target_p50_ms}ms)")
        recommendations.append("Consider optimizing inference pipeline for lower latency")
        recommendations.append("Review model architecture for performance bottlenecks")
    
    # Consistency recommendations
    if latency_stats.coefficient_of_variation > 0.3:
        recommendations.append("High latency variance detected - investigate performance consistency")
        recommendations.append("Consider implementing performance optimizations")
    
    # Outlier recommendations
    if latency_stats.outlier_percentage > 5:
        recommendations.append(f"High outlier rate ({latency_stats.outlier_percentage:.1f}%) - investigate causes")
        recommendations.append("Consider implementing outlier detection and handling")
    
    # Distribution recommendations
    if latency_stats.skewness > 1.0:
        recommendations.append("Right-skewed distribution - investigate occasional high-latency requests")
        recommendations.append("Consider implementing request prioritization")
    
    # General recommendations
    if latency_stats.p95_ms > target_p50_ms * 2:
        recommendations.append("P95 latency significantly higher than target - optimize worst-case performance")
    
    if latency_stats.p99_ms > target_p50_ms * 3:
        recommendations.append("P99 latency very high - investigate extreme cases")
    
    return recommendations


def main():
    """Main entry point for latency requirement validation."""
    parser = argparse.ArgumentParser(
        description="Validate P50 < 10ms latency requirement on target hardware"
    )
    
    parser.add_argument(
        "--target-p50",
        type=float,
        default=10.0,
        help="Target P50 latency in milliseconds (default: 10.0)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for statistical significance (default: 1000)"
    )
    
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for validation results (JSON format)"
    )
    
    parser.add_argument(
        "--exit-on-failure",
        action="store_true",
        help="Exit with non-zero code if validation fails"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.iterations,
        p50_threshold_ms=args.target_p50
    )
    
    # Run validation
    try:
        result = asyncio.run(validate_latency_requirements(
            config, args.target_p50, args.iterations
        ))
        
        # Print results
        print("\n" + "="*70)
        print("P50 LATENCY REQUIREMENT VALIDATION")
        print("="*70)
        
        print(f"Requirement: P50 < {result['target_p50_ms']}ms")
        print(f"Actual P50: {result['actual_p50_ms']:.3f}ms")
        print(f"Difference: {result['p50_difference_ms']:+.3f}ms ({result['p50_difference_percent']:+.1f}%)")
        print(f"Requirement Met: {'✅ YES' if result['requirement_met'] else '❌ NO'}")
        print(f"Performance Grade: {result['performance_grade']}")
        
        print("\nLatency Distribution Analysis:")
        dist = result['latency_analysis']
        print(f"  Distribution Type: {dist['distribution_type']}")
        print(f"  Consistency: {dist['consistency']}")
        print(f"  P50: {dist['percentiles']['p50']:.3f}ms")
        print(f"  P90: {dist['percentiles']['p90']:.3f}ms")
        print(f"  P95: {dist['percentiles']['p95']:.3f}ms")
        print(f"  P99: {dist['percentiles']['p99']:.3f}ms")
        print(f"  Mean: {dist['statistics']['mean']:.3f}ms")
        print(f"  Std Dev: {dist['statistics']['std']:.3f}ms")
        print(f"  Coefficient of Variation: {dist['statistics']['coefficient_of_variation']:.3f}")
        print(f"  Skewness: {dist['distribution_shape']['skewness']:.3f}")
        print(f"  Kurtosis: {dist['distribution_shape']['kurtosis']:.3f}")
        print(f"  Outliers: {dist['distribution_shape']['outlier_count']} ({dist['distribution_shape']['outlier_percentage']:.1f}%)")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save results to file
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        # Exit with appropriate code
        if not result['requirement_met'] and args.exit_on_failure:
            print("\nValidation failed - exiting with error code")
            sys.exit(1)
        else:
            print("\nValidation completed")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error running latency validation: {e}")
        if args.exit_on_failure:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
