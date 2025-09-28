#!/usr/bin/env python3
"""
CLI tool for hardware optimization management.

This script provides command-line interface for managing hardware-specific
optimizations including CPU, GPU, and memory optimizations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.optimization_manager import OptimizationManager
from optimization.cpu_optimizer import CPUOptimizationConfig
from optimization.gpu_optimizer import GPUOptimizationConfig
from optimization.memory_optimizer import MemoryOptimizationConfig
from benchmarking.hardware_detector import HardwareDetector


def print_hardware_info(hardware_info) -> None:
    """Print hardware information in a formatted way."""
    print("Hardware Information:")
    print("=" * 50)
    
    # CPU Info
    print(f"CPU: {hardware_info.cpu.model}")
    print(f"  Cores: {hardware_info.cpu.cores} physical, {hardware_info.cpu.threads} logical")
    print(f"  Frequency: {hardware_info.cpu.frequency_mhz:.1f} MHz")
    print(f"  Architecture: {hardware_info.cpu.architecture}")
    print(f"  Features: {', '.join(hardware_info.cpu.features)}")
    print(f"  AVX Support: {'Yes' if hardware_info.cpu.avx_support else 'No'}")
    print(f"  SSE Support: {'Yes' if hardware_info.cpu.sse_support else 'No'}")
    print(f"  Intel MKL: {'Yes' if hardware_info.cpu.intel_mkl_available else 'No'}")
    print(f"  Cache Size: {hardware_info.cpu.cache_size_mb:.1f} MB")
    
    # GPU Info
    if hardware_info.gpu:
        print(f"\nGPU: {hardware_info.gpu.model}")
        print(f"  Memory: {hardware_info.gpu.memory_gb:.1f} GB")
        print(f"  Compute Capability: {hardware_info.gpu.compute_capability}")
        print(f"  CUDA Available: {'Yes' if hardware_info.gpu.cuda_available else 'No'}")
        print(f"  TensorRT Available: {'Yes' if hardware_info.gpu.tensorrt_available else 'No'}")
        print(f"  Driver Version: {hardware_info.gpu.driver_version}")
        print(f"  CUDA Version: {hardware_info.gpu.cuda_version}")
    else:
        print("\nGPU: Not available")
    
    # Memory Info
    print(f"\nMemory:")
    print(f"  Total: {hardware_info.memory.total_gb:.1f} GB")
    print(f"  Available: {hardware_info.memory.available_gb:.1f} GB")
    print(f"  Swap: {hardware_info.memory.swap_gb:.1f} GB")
    print(f"  Type: {hardware_info.memory.memory_type}")
    
    # System Info
    print(f"\nSystem:")
    print(f"  Platform: {hardware_info.platform}")
    print(f"  Python Version: {hardware_info.python_version}")
    print(f"  PyTorch Version: {hardware_info.torch_version}")


def print_optimization_recommendations(recommendations: List[str]) -> None:
    """Print optimization recommendations."""
    print("\nOptimization Recommendations:")
    print("=" * 50)
    
    if not recommendations:
        print("No specific recommendations available.")
        return
    
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")


def print_performance_metrics(metrics: Dict) -> None:
    """Print performance metrics."""
    print("\nPerformance Metrics:")
    print("=" * 50)
    
    if not metrics.get("optimization_applied", False):
        print("No optimizations applied.")
        return
    
    # Hardware info
    if metrics.get("hardware_info"):
        print("Hardware detected and optimized.")
    
    # CPU optimizations
    if "cpu_optimizations" in metrics:
        cpu_metrics = metrics["cpu_optimizations"]
        print(f"\nCPU Optimizations:")
        print(f"  Thread Count: {cpu_metrics.get('thread_count', 'N/A')}")
        print(f"  MKL-DNN Enabled: {'Yes' if cpu_metrics.get('mkldnn_enabled') else 'No'}")
        print(f"  MKL Enabled: {'Yes' if cpu_metrics.get('mkl_enabled') else 'No'}")
        print(f"  JIT Compiled Models: {cpu_metrics.get('jit_compiled_models', 0)}")
        print(f"  Thread Pool Available: {'Yes' if cpu_metrics.get('thread_pool_available') else 'No'}")
    
    # GPU optimizations
    if "gpu_optimizations" in metrics:
        gpu_metrics = metrics["gpu_optimizations"]
        print(f"\nGPU Optimizations:")
        print(f"  Device: {gpu_metrics.get('device', 'N/A')}")
        print(f"  cuDNN Enabled: {'Yes' if gpu_metrics.get('cudnn_enabled') else 'No'}")
        print(f"  cuDNN Benchmark: {'Yes' if gpu_metrics.get('cudnn_benchmark') else 'No'}")
        print(f"  JIT Compiled Models: {gpu_metrics.get('jit_compiled_models', 0)}")
        print(f"  TensorRT Engines: {gpu_metrics.get('tensorrt_engines', 0)}")
        print(f"  Mixed Precision: {'Yes' if gpu_metrics.get('mixed_precision_enabled') else 'No'}")
        
        if gpu_metrics.get('gpu_memory_allocated'):
            print(f"  GPU Memory Allocated: {gpu_metrics['gpu_memory_allocated'] / 1024**2:.1f} MB")
        if gpu_metrics.get('gpu_memory_reserved'):
            print(f"  GPU Memory Reserved: {gpu_metrics['gpu_memory_reserved'] / 1024**2:.1f} MB")
    
    # Memory optimizations
    if "memory_optimizations" in metrics:
        memory_metrics = metrics["memory_optimizations"]
        print(f"\nMemory Optimizations:")
        print(f"  Memory Pooling: {'Yes' if memory_metrics.get('memory_pooling_enabled') else 'No'}")
        if memory_metrics.get('total_tensors'):
            print(f"  Pooled Tensors: {memory_metrics['total_tensors']}")
        if memory_metrics.get('total_memory_bytes'):
            print(f"  Pooled Memory: {memory_metrics['total_memory_bytes'] / 1024**2:.1f} MB")
        if memory_metrics.get('python_memory_rss'):
            print(f"  Python RSS Memory: {memory_metrics['python_memory_rss'] / 1024**2:.1f} MB")


def detect_hardware(args) -> None:
    """Detect and display hardware information."""
    print("Detecting hardware...")
    
    detector = HardwareDetector()
    hardware_info = detector.get_hardware_info()
    
    print_hardware_info(hardware_info)
    
    if args.output:
        output_data = {
            "cpu": {
                "model": hardware_info.cpu.model,
                "cores": hardware_info.cpu.cores,
                "threads": hardware_info.cpu.threads,
                "frequency_mhz": hardware_info.cpu.frequency_mhz,
                "architecture": hardware_info.cpu.architecture,
                "features": hardware_info.cpu.features,
                "cache_size_mb": hardware_info.cpu.cache_size_mb,
                "avx_support": hardware_info.cpu.avx_support,
                "sse_support": hardware_info.cpu.sse_support,
                "intel_mkl_available": hardware_info.cpu.intel_mkl_available,
            },
            "gpu": {
                "model": hardware_info.gpu.model if hardware_info.gpu else None,
                "memory_gb": hardware_info.gpu.memory_gb if hardware_info.gpu else None,
                "compute_capability": hardware_info.gpu.compute_capability if hardware_info.gpu else None,
                "cuda_available": hardware_info.gpu.cuda_available if hardware_info.gpu else False,
                "tensorrt_available": hardware_info.gpu.tensorrt_available if hardware_info.gpu else False,
                "driver_version": hardware_info.gpu.driver_version if hardware_info.gpu else None,
                "cuda_version": hardware_info.gpu.cuda_version if hardware_info.gpu else None,
            } if hardware_info.gpu else None,
            "memory": {
                "total_gb": hardware_info.memory.total_gb,
                "available_gb": hardware_info.memory.available_gb,
                "swap_gb": hardware_info.memory.swap_gb,
                "memory_type": hardware_info.memory.memory_type,
            },
            "system": {
                "platform": hardware_info.platform,
                "python_version": hardware_info.python_version,
                "torch_version": hardware_info.torch_version,
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nHardware information saved to {args.output}")


def optimize_hardware(args) -> None:
    """Apply hardware optimizations."""
    print("Applying hardware optimizations...")
    
    manager = OptimizationManager()
    
    # Apply optimizations
    optimizations = manager.auto_optimize(
        target_latency_ms=args.latency,
        target_throughput_rps=args.throughput,
        memory_limit_gb=args.memory_limit
    )
    
    print(f"\nOptimization Profile: {optimizations['profile']['name']}")
    print(f"Description: {optimizations['profile']['description']}")
    print(f"Target Latency: {optimizations['profile']['target_latency_ms']} ms")
    print(f"Target Throughput: {optimizations['profile']['target_throughput_rps']} RPS")
    
    # Get recommendations
    recommendations = manager.get_optimization_recommendations()
    print_optimization_recommendations(recommendations)
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    print_performance_metrics(metrics)
    
    if args.output:
        output_data = {
            "optimizations": optimizations,
            "recommendations": recommendations,
            "metrics": metrics
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nOptimization results saved to {args.output}")
    
    # Cleanup
    manager.cleanup()


def benchmark_optimizations(args) -> None:
    """Benchmark optimization performance."""
    print("Benchmarking optimization performance...")
    
    # This would integrate with the existing benchmarking framework
    print("Benchmarking functionality would be implemented here.")
    print("This would run performance tests to validate optimizations.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hardware optimization management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect hardware
  python optimization_manager.py detect

  # Apply optimizations with custom targets
  python optimization_manager.py optimize --latency 5.0 --throughput 2000

  # Apply optimizations with memory limit
  python optimization_manager.py optimize --memory-limit 8.0

  # Save results to file
  python optimization_manager.py detect --output hardware.json
  python optimization_manager.py optimize --output optimizations.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect hardware information')
    detect_parser.add_argument(
        '--output', '-o',
        help='Output file for hardware information (JSON format)'
    )
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Apply hardware optimizations')
    optimize_parser.add_argument(
        '--latency', '-l',
        type=float,
        default=10.0,
        help='Target latency in milliseconds (default: 10.0)'
    )
    optimize_parser.add_argument(
        '--throughput', '-t',
        type=int,
        default=1000,
        help='Target throughput in requests per second (default: 1000)'
    )
    optimize_parser.add_argument(
        '--memory-limit', '-m',
        type=float,
        help='Memory limit in GB (optional)'
    )
    optimize_parser.add_argument(
        '--output', '-o',
        help='Output file for optimization results (JSON format)'
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark optimization performance')
    benchmark_parser.add_argument(
        '--output', '-o',
        help='Output file for benchmark results (JSON format)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'detect':
            detect_hardware(args)
        elif args.command == 'optimize':
            optimize_hardware(args)
        elif args.command == 'benchmark':
            benchmark_optimizations(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
