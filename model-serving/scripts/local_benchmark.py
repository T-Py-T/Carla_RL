#!/usr/bin/env python3
"""
Local hardware benchmarking script for Policy-as-a-Service.

This script runs comprehensive benchmarks on the local machine to validate
hardware optimizations and measure empirical performance data.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

import psutil

MODEL_NOT_AVAILABLE = "Model not available"

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    dependencies = {
        "torch": False,
        "numpy": False,
        "psutil": False,
        "cuda": False,
        "tensorrt": False,
        "mkl": False,
    }

    try:
        import torch

        dependencies["torch"] = True
        dependencies["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import numpy as np

        dependencies["numpy"] = True
        dependencies["mkl"] = "mkl" in str(np.__config__)
    except ImportError:
        pass

    try:
        import psutil  # noqa: F401

        dependencies["psutil"] = True
    except ImportError:
        pass

    try:
        import tensorrt  # noqa: F401

        dependencies["tensorrt"] = True
    except ImportError:
        pass

    return dependencies


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "used_gb": psutil.virtual_memory().used / (1024**3),
            "swap_gb": psutil.swap_memory().total / (1024**3),
        },
        "dependencies": check_dependencies(),
    }

    # Try to get more detailed CPU info
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "avx" in cpuinfo.lower():
                    info["cpu"]["avx_support"] = True
                if "avx2" in cpuinfo.lower():
                    info["cpu"]["avx2_support"] = True
                if "sse" in cpuinfo.lower():
                    info["cpu"]["sse_support"] = True
    except OSError:
        pass

    # Try to get GPU info
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                "compute_capability": torch.cuda.get_device_capability(0),
                "cuda_version": torch.version.cuda,
            }
        else:
            info["gpu"] = {"available": False}
    except ImportError:
        info["gpu"] = {"available": False, "error": "PyTorch not available"}

    return info


def create_test_model(input_size: int = 10, hidden_size: int = 50, output_size: int = 1) -> Any:
    """Create a test model for benchmarking."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn

        class TestModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, x):
                return self.layers(x)

        return TestModel(input_size, hidden_size, output_size)
    except ImportError:
        print("PyTorch not available, using mock model")
        return None


def benchmark_latency(model: Any, input_tensor: Any, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model latency."""
    if model is None:
        return {"error": MODEL_NOT_AVAILABLE}

    try:
        import torch

        model.eval()
        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        # Benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies.sort()
        return {
            "mean_ms": sum(latencies) / len(latencies),
            "median_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[int(len(latencies) * 0.95)],
            "p99_ms": latencies[int(len(latencies) * 0.99)],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": (
                sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)
            )
            ** 0.5,
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_throughput(
    model: Any, input_tensor: Any, duration_seconds: float = 10.0
) -> Dict[str, float]:
    """Benchmark model throughput."""
    if model is None:
        return {"error": MODEL_NOT_AVAILABLE}

    try:
        import torch

        model.eval()
        num_inferences = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                _ = model(input_tensor)
                num_inferences += 1

        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        return {
            "throughput_rps": num_inferences / actual_duration,
            "total_inferences": num_inferences,
            "duration_seconds": actual_duration,
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_memory_usage(model: Any, input_tensor: Any) -> Dict[str, float]:
    """Benchmark memory usage."""
    if model is None:
        return {"error": MODEL_NOT_AVAILABLE}

    try:
        import gc

        import torch

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Run inference
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)

        # Get peak memory
        peak_memory = process.memory_info().rss
        memory_used = peak_memory - initial_memory

        result = {
            "cpu_memory_mb": memory_used / (1024 * 1024),
            "peak_memory_mb": peak_memory / (1024 * 1024),
        }

        if torch.cuda.is_available():
            result["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)

        return result
    except Exception as e:
        return {"error": str(e)}


def run_optimization_benchmarks() -> Dict[str, Any]:
    """Run benchmarks with different optimization levels."""
    try:
        import torch
        from optimization.optimization_manager import OptimizationManager

        results = {}

        # Create test model
        model = create_test_model()
        if model is None:
            return {"error": "Cannot create test model"}

        # Create test input
        input_tensor = torch.randn(1, 10)

        # Benchmark without optimization
        print("Benchmarking without optimization...")
        results["no_optimization"] = {
            "latency": benchmark_latency(model, input_tensor),
            "throughput": benchmark_throughput(model, input_tensor, duration_seconds=5.0),
            "memory": benchmark_memory_usage(model, input_tensor),
        }

        # Benchmark with optimization
        print("Benchmarking with optimization...")
        manager = OptimizationManager()
        optimizations = manager.auto_optimize()

        optimized_model = manager.optimize_model(model, (10,))

        results["with_optimization"] = {
            "latency": benchmark_latency(optimized_model, input_tensor),
            "throughput": benchmark_throughput(optimized_model, input_tensor, duration_seconds=5.0),
            "memory": benchmark_memory_usage(optimized_model, input_tensor),
            "optimizations_applied": optimizations,
        }

        # Calculate performance improvements
        if (
            "error" not in results["no_optimization"]["latency"]
            and "error" not in results["with_optimization"]["latency"]
        ):
            latency_improvement = (
                results["no_optimization"]["latency"]["median_ms"]
                / results["with_optimization"]["latency"]["median_ms"]
            )
            results["performance_improvement"] = {
                "latency_speedup": latency_improvement,
                "latency_improvement_percent": (latency_improvement - 1) * 100,
            }

        manager.cleanup()
        return results

    except ImportError as e:
        return {"error": f"Optimization modules not available: {e}"}
    except Exception as e:
        return {"error": f"Benchmark error: {e}"}


def run_batch_size_benchmarks() -> Dict[str, Any]:
    """Run benchmarks with different batch sizes."""
    try:
        import torch
        from optimization.optimization_manager import OptimizationManager

        results = {}
        batch_sizes = [1, 4, 8, 16, 32]

        manager = OptimizationManager()
        manager.auto_optimize()

        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")

            model = create_test_model()
            if model is None:
                continue

            input_tensor = torch.randn(batch_size, 10)
            optimized_model = manager.optimize_model(model, (10,))

            results[f"batch_{batch_size}"] = {
                "latency": benchmark_latency(optimized_model, input_tensor, num_runs=50),
                "throughput": benchmark_throughput(
                    optimized_model, input_tensor, duration_seconds=3.0
                ),
                "memory": benchmark_memory_usage(optimized_model, input_tensor),
            }

        manager.cleanup()
        return results

    except ImportError as e:
        return {"error": f"Optimization modules not available: {e}"}
    except Exception as e:
        return {"error": f"Batch benchmark error: {e}"}


def run_hardware_specific_benchmarks() -> Dict[str, Any]:
    """Run hardware-specific benchmarks."""
    try:
        from optimization.optimization_manager import OptimizationManager

        results = {}

        # Test different optimization profiles
        manager = OptimizationManager()
        profiles = manager._optimization_profiles

        for profile_name, profile in profiles.items():
            print(f"Testing profile: {profile_name}")

            # Create a mock hardware info that matches the profile requirements
            from benchmarking.hardware_detector import (
                CPUInfo,
                GPUInfo,
                HardwareInfo,
                MemoryInfo,
            )

            # Create minimal hardware info
            cpu_info = CPUInfo(
                model="Test CPU",
                cores=8,
                threads=16,
                frequency_mhz=3000.0,
                architecture="x86_64",
                features=["avx", "sse"],
                cache_size_mb=16.0,
                avx_support=True,
                sse_support=True,
                intel_mkl_available=True,
            )

            gpu_info = (
                GPUInfo(
                    model="Test GPU",
                    memory_gb=8.0,
                    compute_capability="7.5",
                    cuda_available=True,
                    tensorrt_available=True,
                    driver_version="500.0",
                    cuda_version="11.0",
                )
                if profile_name in ["gpu_accelerated", "balanced"]
                else None
            )

            memory_info = MemoryInfo(
                total_gb=32.0, available_gb=16.0, swap_gb=8.0, memory_type="DDR4"
            )

            HardwareInfo(
                cpu=cpu_info,
                gpu=gpu_info,
                memory=memory_info,
                platform="Test Platform",
                python_version="3.11.0",
                torch_version="2.1.0",
                optimization_recommendations=[],
            )

            # Test profile selection
            selected_profile = manager._select_optimal_profile(
                target_latency_ms=10.0, target_throughput_rps=1000, memory_limit_gb=16.0
            )

            results[profile_name] = {
                "profile_name": selected_profile.name,
                "target_latency_ms": selected_profile.target_latency_ms,
                "target_throughput_rps": selected_profile.target_throughput_rps,
                "memory_limit_gb": selected_profile.memory_limit_gb,
                "suitable": manager._is_profile_suitable(selected_profile, 16.0),
            }

        return results

    except ImportError as e:
        return {"error": f"Optimization modules not available: {e}"}
    except Exception as e:
        return {"error": f"Hardware benchmark error: {e}"}


def create_parser():
    """Create the benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Local hardware benchmarking for Policy-as-a-Service"
    )
    parser.add_argument("--output", "-o", help="Output file for benchmark results (JSON)")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmarks (fewer iterations)"
    )
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive benchmarks")
    parser.add_argument(
        "--system-info-only", action="store_true", help="Only collect system information"
    )
    return parser


def collect_benchmarks(args, system_info):
    """Run the benchmark groups requested by the user."""
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": system_info,
        "benchmarks": {},
    }

    if args.quick or args.comprehensive:
        print("\nRunning optimization benchmarks...")
        benchmark_results["benchmarks"]["optimization"] = run_optimization_benchmarks()

    if args.comprehensive:
        print("\nRunning batch size benchmarks...")
        benchmark_results["benchmarks"]["batch_sizes"] = run_batch_size_benchmarks()
        print("\nRunning hardware-specific benchmarks...")
        benchmark_results["benchmarks"]["hardware_profiles"] = run_hardware_specific_benchmarks()

    return benchmark_results


def print_optimization_results(benchmarks):
    """Print the optimization benchmark summary when available."""
    optimization = benchmarks.get("optimization")
    if not optimization:
        return
    if "error" in optimization:
        print(f"Optimization benchmark error: {optimization['error']}")
        return

    baseline = optimization["no_optimization"]["latency"].get("median_ms")
    optimized = optimization["with_optimization"]["latency"].get("median_ms")
    baseline_text = f"{baseline:.2f}ms" if baseline is not None else "N/A"
    optimized_text = f"{optimized:.2f}ms" if optimized is not None else "N/A"
    print(f"Latency (no optimization): {baseline_text}")
    print(f"Latency (with optimization): {optimized_text}")

    improvement = optimization.get("performance_improvement")
    if improvement:
        print(f"Latency improvement: {improvement['latency_improvement_percent']:.1f}%")
        print(f"Speedup: {improvement['latency_speedup']:.2f}x")


def validate_performance(benchmarks):
    """Print validation results for optimized latency."""
    print("\nPerformance Validation:")
    print("-" * 30)
    optimization = benchmarks.get("optimization", {})
    optimized = optimization.get("with_optimization", {})
    median_ms = optimized.get("latency", {}).get("median_ms")
    if median_ms is None or "error" in optimization:
        return

    if median_ms < 10.0:
        print("✓ P50 latency requirement met (< 10ms)")
    else:
        print(f"✗ P50 latency requirement not met: {median_ms:.2f}ms > 10ms")

    if median_ms < 5.0:
        print("✓ Excellent latency performance (< 5ms)")
    elif median_ms < 10.0:
        print("✓ Good latency performance (< 10ms)")
    else:
        print("⚠ Latency performance needs improvement")


def save_results(path, benchmark_results):
    """Save benchmark results when an output path was supplied."""
    if not path:
        return
    with open(path, "w") as output_file:
        json.dump(benchmark_results, output_file, indent=2)
    print(f"\nResults saved to {path}")


def main():
    """Main benchmarking function."""
    args = create_parser().parse_args()
    print("Policy-as-a-Service Local Benchmarking")
    print("=" * 50)
    print("Collecting system information...")
    system_info = get_system_info()

    if args.system_info_only:
        print("\nSystem Information:")
        print(json.dumps(system_info, indent=2))
        return

    benchmark_results = collect_benchmarks(args, system_info)
    print("\nBenchmark Results:")
    print("=" * 50)
    print_optimization_results(benchmark_results["benchmarks"])
    save_results(args.output, benchmark_results)
    validate_performance(benchmark_results["benchmarks"])
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
