#!/usr/bin/env uv run python
"""
CLI tool for running performance benchmarks.

This script provides a command-line interface for running comprehensive
performance benchmarks and generating detailed reports.
"""

import argparse
import json
import sys
from pathlib import Path

# Add model-serving to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking import BenchmarkEngine, BenchmarkConfig, HardwareDetector, PerformanceValidator


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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for Policy-as-a-Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic benchmark
  python run_benchmarks.py

  # Run with custom configuration
  python run_benchmarks.py --iterations 200 --batch-sizes 1,4,8,16

  # Run with specific thresholds
  python run_benchmarks.py --p50-threshold 5.0 --throughput-threshold 2000

  # Generate detailed report
  python run_benchmarks.py --output report.json --verbose
        """
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        help="Number of measurement iterations (default: 100)"
    )
    
    parser.add_argument(
        "--warmup-iterations", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated list of batch sizes (default: 1,4,8,16,32)"
    )
    
    # Performance thresholds
    parser.add_argument(
        "--p50-threshold",
        type=float,
        default=10.0,
        help="P50 latency threshold in milliseconds (default: 10.0)"
    )
    
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=20.0,
        help="P95 latency threshold in milliseconds (default: 20.0)"
    )
    
    parser.add_argument(
        "--p99-threshold",
        type=float,
        default=50.0,
        help="P99 latency threshold in milliseconds (default: 50.0)"
    )
    
    parser.add_argument(
        "--throughput-threshold",
        type=float,
        default=1000.0,
        help="Throughput threshold in requests per second (default: 1000.0)"
    )
    
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=1024.0,
        help="Memory usage threshold in MB (default: 1024.0)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for detailed results (JSON format)"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        help="Generate human-readable report to file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Hardware detection
    parser.add_argument(
        "--detect-hardware",
        action="store_true",
        help="Show hardware information and exit"
    )
    
    # Validation options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate against requirements, don't run full benchmark"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    try:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    except ValueError:
        print("Error: Invalid batch sizes format. Use comma-separated integers.")
        sys.exit(1)
    
    # Show hardware information if requested
    if args.detect_hardware:
        detector = HardwareDetector()
        hardware_info = detector.get_hardware_info()
        
        print("Hardware Information:")
        print("=" * 50)
        print(f"CPU: {hardware_info.cpu.model}")
        print(f"  Cores: {hardware_info.cpu.cores}")
        print(f"  Threads: {hardware_info.cpu.threads}")
        print(f"  Frequency: {hardware_info.cpu.frequency_mhz:.0f} MHz")
        print(f"  AVX Support: {hardware_info.cpu.avx_support}")
        print(f"  Intel MKL: {hardware_info.cpu.intel_mkl_available}")
        
        if hardware_info.gpu:
            print(f"GPU: {hardware_info.gpu.model}")
            print(f"  Memory: {hardware_info.gpu.memory_gb:.1f} GB")
            print(f"  Compute Capability: {hardware_info.gpu.compute_capability}")
            print(f"  TensorRT: {hardware_info.gpu.tensorrt_available}")
        else:
            print("GPU: Not available")
        
        print(f"Memory: {hardware_info.memory.total_gb:.1f} GB")
        print(f"Platform: {hardware_info.platform}")
        print(f"Python: {hardware_info.python_version}")
        print(f"PyTorch: {hardware_info.torch_version}")
        
        print("\nOptimization Recommendations:")
        for i, rec in enumerate(hardware_info.optimization_recommendations, 1):
            print(f"  {i}. {rec}")
        
        return
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.iterations,
        batch_sizes=batch_sizes,
        p50_threshold_ms=args.p50_threshold,
        p95_threshold_ms=args.p95_threshold,
        p99_threshold_ms=args.p99_threshold,
        throughput_threshold_rps=args.throughput_threshold,
        max_memory_usage_mb=args.memory_threshold
    )
    
    # Create benchmark engine
    engine = BenchmarkEngine(config)
    
    # Create mock inference function
    inference_func = create_mock_inference_function()
    
    print("Starting Performance Benchmark")
    print("=" * 50)
    print("Configuration:")
    print(f"  Warmup iterations: {config.warmup_iterations}")
    print(f"  Measurement iterations: {config.measurement_iterations}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  P50 threshold: {config.p50_threshold_ms}ms")
    print(f"  P95 threshold: {config.p95_threshold_ms}ms")
    print(f"  P99 threshold: {config.p99_threshold_ms}ms")
    print(f"  Throughput threshold: {config.throughput_threshold_rps} RPS")
    print(f"  Memory threshold: {config.max_memory_usage_mb} MB")
    print("")
    
    try:
        # Run benchmarks
        results = engine.run_batch_size_optimization(inference_func)
        
        # Generate report
        report = engine.generate_report()
        print(report)
        
        # Validate results
        validator = PerformanceValidator()
        validation_results = []
        
        for result in results:
            validation_result = validator.validate_requirements(result)
            validation_results.append(validation_result)
            
            if args.verbose:
                print(f"\nValidation for batch size {result.config.batch_sizes[0] if result.config.batch_sizes else 'Unknown'}:")
                print(f"  Grade: {validation_result.performance_grade}")
                print(f"  Success: {validation_result.overall_success}")
                if validation_result.recommendations:
                    print("  Recommendations:")
                    for rec in validation_result.recommendations:
                        print(f"    - {rec}")
        
        # Save results to file
        if args.output:
            output_data = {
                "config": config.__dict__,
                "results": [result.__dict__ for result in results],
                "validation_results": [vr.__dict__ for vr in validation_results]
            }
            
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"\nDetailed results saved to: {args.output}")
        
        # Generate human-readable report
        if args.report:
            with open(args.report, "w") as f:
                f.write(report)
                f.write("\n\n")
                
                for i, validation_result in enumerate(validation_results, 1):
                    f.write(f"VALIDATION {i}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(validator.generate_validation_report(validation_result))
                    f.write("\n\n")
            
            print(f"Human-readable report saved to: {args.report}")
        
        # Summary
        successful_tests = sum(1 for r in results if r.overall_success)
        total_tests = len(results)
        
        print(f"\nSummary: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("All performance requirements met!")
            sys.exit(0)
        else:
            print("Some performance requirements not met. Check recommendations above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
