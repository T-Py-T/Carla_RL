"""
Hardware-specific performance validation tests.

Tests performance requirements including P50 < 10ms latency
and throughput validation for different hardware configurations.
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

from src.optimization.optimization_manager import OptimizationManager
from src.optimization.cpu_optimizer import CPUOptimizationConfig
from src.optimization.gpu_optimizer import GPUOptimizationConfig
from src.optimization.memory_optimizer import MemoryOptimizationConfig
from src.benchmarking.hardware_detector import HardwareInfo, CPUInfo, GPUInfo, MemoryInfo


class PerformanceValidator:
    """Performance validation utilities for hardware optimizations."""

    @staticmethod
    def measure_latency(
        model: nn.Module, 
        input_tensor: torch.Tensor, 
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Measure inference latency with statistical analysis.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            num_runs: Number of runs for statistical analysis
            
        Returns:
            Dictionary with latency statistics
        """
        model.eval()
        latencies = []
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measurement runs
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        return {
            "mean": sum(latencies) / len(latencies),
            "median": p50,
            "p95": p95,
            "p99": p99,
            "min": min(latencies),
            "max": max(latencies),
            "std": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5
        }

    @staticmethod
    def measure_throughput(
        model: nn.Module, 
        input_tensor: torch.Tensor, 
        duration_seconds: float = 10.0
    ) -> Dict[str, float]:
        """
        Measure inference throughput.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            duration_seconds: Duration of throughput test
            
        Returns:
            Dictionary with throughput statistics
        """
        model.eval()
        num_inferences = 0
        start_time = time.perf_counter()
        
        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                _ = model(input_tensor)
                num_inferences += 1
        
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        throughput_rps = num_inferences / actual_duration
        
        return {
            "throughput_rps": throughput_rps,
            "total_inferences": num_inferences,
            "duration_seconds": actual_duration
        }

    @staticmethod
    def measure_memory_usage(
        model: nn.Module, 
        input_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure memory usage during inference.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            
        Returns:
            Dictionary with memory usage statistics
        """
        import psutil
        import gc
        
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
        
        # GPU memory if available
        gpu_memory_used = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
        
        return {
            "cpu_memory_mb": memory_used / 1024 / 1024,
            "gpu_memory_mb": gpu_memory_used / 1024 / 1024 if torch.cuda.is_available() else 0,
            "peak_memory_mb": peak_memory / 1024 / 1024
        }


class TestPerformanceValidation:
    """Test cases for performance validation."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    @pytest.fixture
    def complex_model(self):
        """Create a more complex model for testing."""
        return nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    @pytest.fixture
    def hardware_info_high_end(self):
        """Create high-end hardware info for testing."""
        cpu_info = CPUInfo(
            model="Intel Core i9-12900K",
            cores=16,
            threads=24,
            frequency_mhz=3200.0,
            architecture="x86_64",
            features=["avx", "avx2", "sse", "sse2", "sse4"],
            cache_size_mb=30.0,
            avx_support=True,
            sse_support=True,
            intel_mkl_available=True
        )
        
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 4090",
            memory_gb=24.0,
            compute_capability="8.9",
            cuda_available=True,
            tensorrt_available=True,
            driver_version="525.60.13",
            cuda_version="12.0"
        )
        
        memory_info = MemoryInfo(
            total_gb=64.0,
            available_gb=32.0,
            swap_gb=16.0,
            memory_type="DDR5"
        )
        
        return HardwareInfo(
            cpu=cpu_info,
            gpu=gpu_info,
            memory=memory_info,
            platform="Linux-5.19.0-32-generic-x86_64-with-glibc2.35",
            python_version="3.11.0",
            torch_version="2.1.0",
            optimization_recommendations=[]
        )

    @pytest.fixture
    def hardware_info_mid_range(self):
        """Create mid-range hardware info for testing."""
        cpu_info = CPUInfo(
            model="Intel Core i7-12700K",
            cores=12,
            threads=20,
            frequency_mhz=3600.0,
            architecture="x86_64",
            features=["avx", "avx2", "sse", "sse2", "sse4"],
            cache_size_mb=25.0,
            avx_support=True,
            sse_support=True,
            intel_mkl_available=True
        )
        
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            memory_gb=10.0,
            compute_capability="8.6",
            cuda_available=True,
            tensorrt_available=True,
            driver_version="525.60.13",
            cuda_version="12.0"
        )
        
        memory_info = MemoryInfo(
            total_gb=32.0,
            available_gb=16.0,
            swap_gb=8.0,
            memory_type="DDR4"
        )
        
        return HardwareInfo(
            cpu=cpu_info,
            gpu=gpu_info,
            memory=memory_info,
            platform="Linux-5.19.0-32-generic-x86_64-with-glibc2.35",
            python_version="3.11.0",
            torch_version="2.1.0",
            optimization_recommendations=[]
        )

    def test_latency_measurement(self, simple_model):
        """Test latency measurement functionality."""
        input_tensor = torch.randn(1, 10)
        
        latencies = PerformanceValidator.measure_latency(simple_model, input_tensor, num_runs=50)
        
        assert "mean" in latencies
        assert "median" in latencies
        assert "p95" in latencies
        assert "p99" in latencies
        assert "min" in latencies
        assert "max" in latencies
        assert "std" in latencies
        
        # Basic sanity checks
        assert latencies["min"] <= latencies["mean"] <= latencies["max"]
        assert latencies["median"] <= latencies["p95"] <= latencies["p99"]
        assert latencies["std"] >= 0

    def test_throughput_measurement(self, simple_model):
        """Test throughput measurement functionality."""
        input_tensor = torch.randn(1, 10)
        
        throughput = PerformanceValidator.measure_throughput(simple_model, input_tensor, duration_seconds=1.0)
        
        assert "throughput_rps" in throughput
        assert "total_inferences" in throughput
        assert "duration_seconds" in throughput
        
        # Basic sanity checks
        assert throughput["throughput_rps"] > 0
        assert throughput["total_inferences"] > 0
        assert throughput["duration_seconds"] > 0

    def test_memory_usage_measurement(self, simple_model):
        """Test memory usage measurement functionality."""
        input_tensor = torch.randn(1, 10)
        
        memory_usage = PerformanceValidator.measure_memory_usage(simple_model, input_tensor)
        
        assert "cpu_memory_mb" in memory_usage
        assert "gpu_memory_mb" in memory_usage
        assert "peak_memory_mb" in memory_usage
        
        # Basic sanity checks
        assert memory_usage["cpu_memory_mb"] >= 0
        assert memory_usage["gpu_memory_mb"] >= 0
        assert memory_usage["peak_memory_mb"] > 0

    def test_performance_validation_high_end_hardware(self, complex_model, hardware_info_high_end):
        """Test performance validation with high-end hardware."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_high_end):
            manager.auto_optimize(target_latency_ms=5.0, target_throughput_rps=5000)
        
        # Optimize model
        optimized_model = manager.optimize_model(complex_model, (100,))
        
        # Test latency
        input_tensor = torch.randn(1, 100)
        latencies = PerformanceValidator.measure_latency(optimized_model, input_tensor, num_runs=100)
        
        # Validate P50 < 10ms requirement
        assert latencies["median"] < 10.0, f"P50 latency {latencies['median']:.2f}ms exceeds 10ms requirement"
        
        # Test throughput
        throughput = PerformanceValidator.measure_throughput(optimized_model, input_tensor, duration_seconds=2.0)
        
        # Should achieve reasonable throughput
        assert throughput["throughput_rps"] > 100, f"Throughput {throughput['throughput_rps']:.1f} RPS too low"
        
        # Test memory usage
        memory_usage = PerformanceValidator.measure_memory_usage(optimized_model, input_tensor)
        
        # Memory usage should be reasonable
        assert memory_usage["cpu_memory_mb"] < 1000, f"CPU memory usage {memory_usage['cpu_memory_mb']:.1f}MB too high"
        
        manager.cleanup()

    def test_performance_validation_mid_range_hardware(self, complex_model, hardware_info_mid_range):
        """Test performance validation with mid-range hardware."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize(target_latency_ms=10.0, target_throughput_rps=1000)
        
        # Optimize model
        optimized_model = manager.optimize_model(complex_model, (100,))
        
        # Test latency
        input_tensor = torch.randn(1, 100)
        latencies = PerformanceValidator.measure_latency(optimized_model, input_tensor, num_runs=100)
        
        # Validate P50 < 10ms requirement
        assert latencies["median"] < 10.0, f"P50 latency {latencies['median']:.2f}ms exceeds 10ms requirement"
        
        # Test throughput
        throughput = PerformanceValidator.measure_throughput(optimized_model, input_tensor, duration_seconds=2.0)
        
        # Should achieve reasonable throughput
        assert throughput["throughput_rps"] > 50, f"Throughput {throughput['throughput_rps']:.1f} RPS too low"
        
        manager.cleanup()

    def test_batch_inference_performance(self, simple_model, hardware_info_mid_range):
        """Test batch inference performance."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 10)
            
            # Measure latency
            latencies = PerformanceValidator.measure_latency(simple_model, input_tensor, num_runs=50)
            
            # Latency should be reasonable even for larger batches
            assert latencies["median"] < 50.0, f"Batch size {batch_size} latency {latencies['median']:.2f}ms too high"
        
        manager.cleanup()

    def test_memory_pooling_performance(self, simple_model, hardware_info_mid_range):
        """Test memory pooling performance benefits."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Test repeated inference with memory pooling
        input_tensor = torch.randn(8, 10)
        
        # Measure memory usage over multiple runs
        memory_usage_before = PerformanceValidator.measure_memory_usage(simple_model, input_tensor)
        
        # Run multiple inferences
        for _ in range(10):
            with torch.no_grad():
                _ = simple_model(input_tensor)
        
        memory_usage_after = PerformanceValidator.measure_memory_usage(simple_model, input_tensor)
        
        # Memory usage should not increase significantly
        memory_increase = memory_usage_after["cpu_memory_mb"] - memory_usage_before["cpu_memory_mb"]
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"
        
        manager.cleanup()

    def test_jit_compilation_performance(self, simple_model, hardware_info_mid_range):
        """Test JIT compilation performance benefits."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Optimize model with JIT
        optimized_model = manager.optimize_model(simple_model, (10,))
        
        input_tensor = torch.randn(1, 10)
        
        # Measure latency
        latencies = PerformanceValidator.measure_latency(optimized_model, input_tensor, num_runs=100)
        
        # JIT compiled model should have reasonable performance
        assert latencies["median"] < 10.0, f"JIT compiled model latency {latencies['median']:.2f}ms too high"
        
        manager.cleanup()

    def test_gpu_optimization_performance(self, simple_model, hardware_info_high_end):
        """Test GPU optimization performance benefits."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_high_end):
            manager.auto_optimize()
        
        # Optimize model for GPU
        optimized_model = manager.optimize_model(simple_model, (10,))
        
        input_tensor = torch.randn(1, 10)
        
        # Measure latency
        latencies = PerformanceValidator.measure_latency(optimized_model, input_tensor, num_runs=100)
        
        # GPU optimized model should have good performance
        assert latencies["median"] < 5.0, f"GPU optimized model latency {latencies['median']:.2f}ms too high"
        
        # Measure throughput
        throughput = PerformanceValidator.measure_throughput(optimized_model, input_tensor, duration_seconds=2.0)
        
        # GPU should provide good throughput
        assert throughput["throughput_rps"] > 200, f"GPU throughput {throughput['throughput_rps']:.1f} RPS too low"
        
        manager.cleanup()

    def test_quantization_performance(self, simple_model, hardware_info_mid_range):
        """Test quantization performance benefits."""
        # Create a manager with quantization enabled
        manager = OptimizationManager()
        
        # Override the profile to enable quantization
        profile = manager._optimization_profiles["memory_constrained"]
        profile.cpu_config.enable_quantization = True
        profile.cpu_config.quantization_bits = 8
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Optimize model with quantization
        optimized_model = manager.optimize_model(simple_model, (10,))
        
        input_tensor = torch.randn(1, 10)
        
        # Measure latency
        latencies = PerformanceValidator.measure_latency(optimized_model, input_tensor, num_runs=100)
        
        # Quantized model should still meet performance requirements
        assert latencies["median"] < 15.0, f"Quantized model latency {latencies['median']:.2f}ms too high"
        
        # Measure memory usage
        memory_usage = PerformanceValidator.measure_memory_usage(optimized_model, input_tensor)
        
        # Quantized model should use less memory
        assert memory_usage["cpu_memory_mb"] < 50, f"Quantized model memory usage {memory_usage['cpu_memory_mb']:.1f}MB too high"
        
        manager.cleanup()

    def test_performance_regression_detection(self, simple_model, hardware_info_mid_range):
        """Test performance regression detection."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Test baseline performance
        input_tensor = torch.randn(1, 10)
        baseline_latencies = PerformanceValidator.measure_latency(simple_model, input_tensor, num_runs=100)
        
        # Optimize model
        optimized_model = manager.optimize_model(simple_model, (10,))
        optimized_latencies = PerformanceValidator.measure_latency(optimized_model, input_tensor, num_runs=100)
        
        # Optimized model should not be significantly slower
        performance_ratio = optimized_latencies["median"] / baseline_latencies["median"]
        assert performance_ratio < 2.0, f"Performance regression detected: {performance_ratio:.2f}x slower"
        
        manager.cleanup()

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_batch_size_performance_scaling(self, simple_model, hardware_info_mid_range, batch_size):
        """Test performance scaling with different batch sizes."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        input_tensor = torch.randn(batch_size, 10)
        
        # Measure latency
        latencies = PerformanceValidator.measure_latency(simple_model, input_tensor, num_runs=50)
        
        # Latency should scale reasonably with batch size
        max_acceptable_latency = 10.0 * batch_size  # Allow some scaling
        assert latencies["median"] < max_acceptable_latency, f"Batch size {batch_size} latency {latencies['median']:.2f}ms exceeds {max_acceptable_latency}ms"
        
        manager.cleanup()
