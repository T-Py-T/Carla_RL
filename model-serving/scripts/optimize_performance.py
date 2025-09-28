#!/usr/bin/env uv run python
"""
Performance optimization script.

This script analyzes performance bottlenecks and suggests optimizations
to improve throughput and reduce latency.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

# Add model-serving to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking import BenchmarkEngine, BenchmarkConfig


class PerformanceOptimizer:
    """Performance optimization analyzer and optimizer."""
    
    def __init__(self):
        self.optimization_suggestions = []
        self.baseline_metrics = None
    
    def create_optimized_inference_function(self, optimization_level: str = "balanced"):
        """Create inference function with different optimization levels."""
        import random
        
        def mock_inference(observations, deterministic=False):
            """Mock inference function with configurable optimization."""
            
            if optimization_level == "maximum":
                # Maximum optimization - minimal processing
                time.sleep(0.001)  # 1ms
            elif optimization_level == "balanced":
                # Balanced optimization
                time.sleep(0.002)  # 2ms
            elif optimization_level == "conservative":
                # Conservative optimization
                time.sleep(0.005)  # 5ms
            else:
                # Default
                time.sleep(0.010)  # 10ms
            
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
    
    def analyze_performance_bottlenecks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks and suggest optimizations."""
        suggestions = []
        
        # Analyze throughput
        throughput = result.get("throughput_rps", 0)
        if throughput < 200:
            suggestions.append({
                "category": "throughput",
                "priority": "high",
                "issue": f"Low throughput: {throughput:.1f} RPS",
                "suggestions": [
                    "Consider batch processing to increase throughput",
                    "Optimize inference function to reduce processing time",
                    "Use concurrent request processing",
                    "Implement request queuing and batching",
                    "Consider using faster hardware or more CPU cores"
                ]
            })
        
        # Analyze latency
        p50_latency = result.get("p50_latency_ms", 0)
        if p50_latency > 10:
            suggestions.append({
                "category": "latency",
                "priority": "high",
                "issue": f"High P50 latency: {p50_latency:.2f}ms",
                "suggestions": [
                    "Optimize model inference pipeline",
                    "Reduce data preprocessing overhead",
                    "Use model quantization for faster inference",
                    "Implement model caching for repeated requests",
                    "Consider using GPU acceleration"
                ]
            })
        
        p95_latency = result.get("p95_latency_ms", 0)
        if p95_latency > 20:
            suggestions.append({
                "category": "latency",
                "priority": "medium",
                "issue": f"High P95 latency: {p95_latency:.2f}ms",
                "suggestions": [
                    "Investigate tail latency causes",
                    "Implement request timeout handling",
                    "Add circuit breakers for failing requests",
                    "Optimize memory allocation patterns",
                    "Consider load balancing across multiple instances"
                ]
            })
        
        # Analyze memory usage
        memory_usage = result.get("memory_usage_mb", 0)
        if memory_usage > 512:
            suggestions.append({
                "category": "memory",
                "priority": "medium",
                "issue": f"High memory usage: {memory_usage:.1f}MB",
                "suggestions": [
                    "Implement memory pooling for frequent allocations",
                    "Use streaming processing for large datasets",
                    "Optimize data structures to reduce memory footprint",
                    "Implement garbage collection tuning",
                    "Consider using memory-mapped files for large data"
                ]
            })
        
        # Analyze error rate
        error_rate = result.get("error_rate_percent", 0)
        if error_rate > 1.0:
            suggestions.append({
                "category": "reliability",
                "priority": "high",
                "issue": f"High error rate: {error_rate:.2f}%",
                "suggestions": [
                    "Add comprehensive error handling",
                    "Implement retry mechanisms with exponential backoff",
                    "Add input validation and sanitization",
                    "Implement circuit breakers for external dependencies",
                    "Add monitoring and alerting for error conditions"
                ]
            })
        
        return suggestions
    
    def test_optimization_levels(self, baseline_config: BenchmarkConfig) -> Dict[str, Any]:
        """Test different optimization levels to find the best configuration."""
        optimization_levels = ["conservative", "balanced", "maximum"]
        results = {}
        
        for level in optimization_levels:
            print(f"Testing {level} optimization level...")
            
            # Create optimized inference function
            inference_func = self.create_optimized_inference_function(level)
            
            # Run benchmark
            engine = BenchmarkEngine(baseline_config)
            result = asyncio.run(engine.run_benchmark(inference_func, batch_size=1))
            
            results[level] = {
                "throughput_rps": result.throughput_stats.requests_per_second,
                "p50_latency_ms": result.latency_stats.p50_ms,
                "p95_latency_ms": result.latency_stats.p95_ms,
                "p99_latency_ms": result.latency_stats.p99_ms,
                "memory_usage_mb": result.memory_stats.peak_memory_mb,
                "overall_success": result.overall_success
            }
            
            print(f"  Throughput: {result.throughput_stats.requests_per_second:.1f} RPS")
            print(f"  P50 Latency: {result.latency_stats.p50_ms:.2f}ms")
            print(f"  Memory: {result.memory_stats.peak_memory_mb:.1f}MB")
            print(f"  Success: {result.overall_success}")
        
        return results
    
    def find_optimal_batch_size(self, baseline_config: BenchmarkConfig) -> Dict[str, Any]:
        """Find optimal batch size for maximum throughput."""
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = {}
        
        print("Testing different batch sizes...")
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create inference function
            inference_func = self.create_optimized_inference_function("balanced")
            
            # Run benchmark
            engine = BenchmarkEngine(baseline_config)
            result = asyncio.run(engine.run_benchmark(inference_func, batch_size=batch_size))
            
            results[batch_size] = {
                "throughput_rps": result.throughput_stats.requests_per_second,
                "p50_latency_ms": result.latency_stats.p50_ms,
                "p95_latency_ms": result.latency_stats.p95_ms,
                "memory_usage_mb": result.memory_stats.peak_memory_mb,
                "overall_success": result.overall_success,
                "efficiency": result.throughput_stats.requests_per_second / batch_size
            }
            
            print(f"  Throughput: {result.throughput_stats.requests_per_second:.1f} RPS")
            print(f"  Efficiency: {results[batch_size]['efficiency']:.1f} RPS per batch item")
        
        # Find optimal batch size
        optimal_batch_size = max(batch_sizes, key=lambda bs: results[bs]["throughput_rps"])
        
        return {
            "optimal_batch_size": optimal_batch_size,
            "results": results,
            "recommendation": f"Use batch size {optimal_batch_size} for maximum throughput"
        }
    
    def generate_optimization_report(self, 
                                   baseline_result: Dict[str, Any],
                                   optimization_results: Dict[str, Any],
                                   batch_size_results: Dict[str, Any],
                                   suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        # Find best optimization level
        best_level = max(optimization_results.keys(), 
                        key=lambda level: optimization_results[level]["throughput_rps"])
        
        best_throughput = optimization_results[best_level]["throughput_rps"]
        baseline_throughput = baseline_result.get("throughput_rps", 0)
        improvement = ((best_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0
        
        report = {
            "baseline_performance": baseline_result,
            "optimization_levels": optimization_results,
            "best_optimization_level": best_level,
            "batch_size_analysis": batch_size_results,
            "performance_improvement": {
                "baseline_throughput_rps": baseline_throughput,
                "optimized_throughput_rps": best_throughput,
                "improvement_percent": improvement
            },
            "optimization_suggestions": suggestions,
            "recommendations": {
                "use_optimization_level": best_level,
                "use_batch_size": batch_size_results["optimal_batch_size"],
                "expected_throughput_rps": best_throughput,
                "priority_actions": [s for s in suggestions if s["priority"] == "high"]
            }
        }
        
        return report


def main():
    """Main entry point for performance optimization."""
    parser = argparse.ArgumentParser(
        description="Analyze and optimize performance"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="Number of measurement iterations (default: 50)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for optimization report (JSON format)"
    )
    
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline analysis without optimization testing"
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = PerformanceOptimizer()
    
    # Create baseline configuration
    baseline_config = BenchmarkConfig(
        warmup_iterations=5,
        measurement_iterations=args.iterations,
        p50_threshold_ms=10.0,
        p95_threshold_ms=20.0,
        p99_threshold_ms=50.0,
        throughput_threshold_rps=200.0,
        max_memory_usage_mb=1024.0
    )
    
    print("Running performance optimization analysis...")
    print("="*60)
    
    try:
        # Run baseline analysis
        print("1. Running baseline performance analysis...")
        baseline_inference = optimizer.create_optimized_inference_function("conservative")
        baseline_engine = BenchmarkEngine(baseline_config)
        baseline_result = asyncio.run(baseline_engine.run_benchmark(baseline_inference, batch_size=1))
        
        baseline_metrics = {
            "throughput_rps": baseline_result.throughput_stats.requests_per_second,
            "p50_latency_ms": baseline_result.latency_stats.p50_ms,
            "p95_latency_ms": baseline_result.latency_stats.p95_ms,
            "p99_latency_ms": baseline_result.latency_stats.p99_ms,
            "memory_usage_mb": baseline_result.memory_stats.peak_memory_mb,
            "error_rate_percent": baseline_result.throughput_stats.error_rate * 100
        }
        
        print(f"Baseline Throughput: {baseline_metrics['throughput_rps']:.1f} RPS")
        print(f"Baseline P50 Latency: {baseline_metrics['p50_latency_ms']:.2f}ms")
        print(f"Baseline Memory: {baseline_metrics['memory_usage_mb']:.1f}MB")
        
        # Analyze bottlenecks
        print("\n2. Analyzing performance bottlenecks...")
        suggestions = optimizer.analyze_performance_bottlenecks(baseline_metrics)
        
        if suggestions:
            print(f"Found {len(suggestions)} optimization opportunities:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion['issue']} ({suggestion['priority']} priority)")
        else:
            print("No significant bottlenecks found.")
        
        optimization_results = {}
        batch_size_results = {}
        
        if not args.baseline_only:
            # Test optimization levels
            print("\n3. Testing optimization levels...")
            optimization_results = optimizer.test_optimization_levels(baseline_config)
            
            # Test batch sizes
            print("\n4. Testing batch sizes...")
            batch_size_results = optimizer.find_optimal_batch_size(baseline_config)
        
        # Generate report
        print("\n5. Generating optimization report...")
        report = optimizer.generate_optimization_report(
            baseline_metrics, optimization_results, batch_size_results, suggestions
        )
        
        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        if optimization_results:
            best_level = report["best_optimization_level"]
            improvement = report["performance_improvement"]["improvement_percent"]
            print(f"Best optimization level: {best_level}")
            print(f"Performance improvement: {improvement:+.1f}%")
            print(f"Optimized throughput: {report['performance_improvement']['optimized_throughput_rps']:.1f} RPS")
        
        if batch_size_results:
            print(f"Optimal batch size: {batch_size_results['optimal_batch_size']}")
        
        print(f"\nHigh priority actions: {len([s for s in suggestions if s['priority'] == 'high'])}")
        
        # Save report if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nOptimization report saved to: {args.output}")
        
        print("\nOptimization analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during optimization analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
