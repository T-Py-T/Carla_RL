"""
Unit tests for CPU optimizer.

Tests CPU-specific optimizations including AVX, SSE, Intel MKL,
and multi-threading optimizations.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.optimization.cpu_optimizer import CPUOptimizer, CPUOptimizationConfig
from src.benchmarking.hardware_detector import HardwareInfo, CPUInfo, GPUInfo, MemoryInfo


class TestCPUOptimizer:
    """Test cases for CPU optimizer."""

    @pytest.fixture
    def config(self):
        """Create CPU optimization config for testing."""
        return CPUOptimizationConfig(
            enable_avx=True,
            enable_sse=True,
            enable_intel_mkl=True,
            enable_multi_threading=True,
            max_threads=4,
            enable_jit_optimization=True,
            enable_memory_pinning=True,
            batch_size_optimization=True,
            enable_quantization=False,
            quantization_bits=8
        )

    @pytest.fixture
    def optimizer(self, config):
        """Create CPU optimizer for testing."""
        return CPUOptimizer(config)

    @pytest.fixture
    def hardware_info(self):
        """Create mock hardware info for testing."""
        cpu_info = CPUInfo(
            model="Intel Core i7-10700K",
            cores=8,
            threads=16,
            frequency_mhz=3800.0,
            architecture="x86_64",
            features=["avx", "avx2", "sse", "sse2", "sse4"],
            cache_size_mb=16.0,
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
            driver_version="470.63.01",
            cuda_version="11.4"
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
            platform="Linux-5.4.0-74-generic-x86_64-with-glibc2.31",
            python_version="3.11.0",
            torch_version="2.1.0",
            optimization_recommendations=[]
        )

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def test_initialization(self, config):
        """Test CPU optimizer initialization."""
        optimizer = CPUOptimizer(config)
        assert optimizer.config == config
        assert not optimizer._optimization_applied
        assert optimizer._thread_pool is None
        assert len(optimizer._jit_compiled_models) == 0

    def test_optimize_for_hardware(self, optimizer, hardware_info):
        """Test hardware optimization application."""
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "avx" in optimizations
        assert "sse" in optimizations
        assert "intel_mkl" in optimizations
        assert "multi_threading" in optimizations
        assert "memory_pinning" in optimizations
        assert "jit" in optimizations
        
        assert optimizer._optimization_applied

    def test_optimize_for_hardware_no_avx(self, hardware_info):
        """Test optimization with CPU that doesn't support AVX."""
        config = CPUOptimizationConfig(enable_avx=True, enable_sse=True)
        optimizer = CPUOptimizer(config)
        
        # Mock CPU without AVX support
        hardware_info.cpu.avx_support = False
        hardware_info.cpu.sse_support = True
        
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "avx" not in optimizations
        assert "sse" in optimizations

    def test_optimize_model(self, optimizer, simple_model):
        """Test model optimization."""
        input_shape = (10,)
        optimized_model = optimizer.optimize_model(simple_model, input_shape)
        
        assert optimized_model is not None
        assert optimized_model.training is False

    def test_optimize_inference_batch(self, optimizer, simple_model):
        """Test batch inference optimization."""
        # Create test input
        batch_size = 8
        input_tensor = torch.randn(batch_size, 10)
        
        # Optimize model first
        optimized_model = optimizer.optimize_model(simple_model, (10,))
        
        # Test batch inference
        with torch.no_grad():
            result = optimizer.optimize_inference_batch(optimized_model, input_tensor)
        
        assert result.shape == (batch_size, 1)

    def test_optimize_inference_batch_with_custom_batch_size(self, optimizer, simple_model):
        """Test batch inference with custom batch size."""
        input_tensor = torch.randn(16, 10)
        optimized_model = optimizer.optimize_model(simple_model, (10,))
        
        # Test with custom batch size
        result = optimizer.optimize_inference_batch(optimized_model, input_tensor, batch_size=4)
        
        assert result.shape == (16, 1)

    def test_get_optimal_thread_count(self, optimizer, hardware_info):
        """Test optimal thread count calculation."""
        thread_count = optimizer._get_optimal_thread_count(hardware_info)
        
        # Should be min of max_threads (4) and cpu threads (16)
        assert thread_count == 4

    def test_get_optimal_thread_count_no_max(self, hardware_info):
        """Test optimal thread count without max_threads limit."""
        config = CPUOptimizationConfig(max_threads=None)
        optimizer = CPUOptimizer(config)
        
        thread_count = optimizer._get_optimal_thread_count(hardware_info)
        
        # Should be min of cpu threads (16) and 8
        assert thread_count == 8

    def test_get_optimal_batch_size(self, optimizer):
        """Test optimal batch size calculation."""
        # Test different input batch sizes
        assert optimizer._get_optimal_batch_size(2) == 4
        assert optimizer._get_optimal_batch_size(6) == 8
        assert optimizer._get_optimal_batch_size(12) == 16
        assert optimizer._get_optimal_batch_size(24) == 32
        assert optimizer._get_optimal_batch_size(48) == 64

    def test_enable_avx_optimizations(self, optimizer):
        """Test AVX optimization enabling."""
        optimizations = optimizer._enable_avx_optimizations()
        
        assert "mkldnn_enabled" in optimizations
        assert "tf32_enabled" in optimizations

    def test_enable_sse_optimizations(self, optimizer):
        """Test SSE optimization enabling."""
        optimizations = optimizer._enable_sse_optimizations()
        
        assert "mkl_enabled" in optimizations
        assert "blas_optimized" in optimizations

    def test_enable_intel_mkl_optimizations(self, optimizer):
        """Test Intel MKL optimization enabling."""
        optimizations = optimizer._enable_intel_mkl_optimizations()
        
        assert "mkl_threading" in optimizations
        assert "mkl_enabled" in optimizations

    def test_enable_multi_threading(self, optimizer):
        """Test multi-threading optimization enabling."""
        thread_count = 4
        optimizations = optimizer._enable_multi_threading(thread_count)
        
        assert optimizations["thread_count"] == thread_count
        assert optimizations["thread_pool_created"] is True

    def test_enable_memory_pinning(self, optimizer):
        """Test memory pinning optimization enabling."""
        optimizations = optimizer._enable_memory_pinning()
        
        assert optimizations["memory_pinning_enabled"] is True
        assert optimizations["cudnn_benchmark"] is True

    def test_enable_jit_optimizations(self, optimizer):
        """Test JIT optimization enabling."""
        optimizations = optimizer._enable_jit_optimizations()
        
        assert optimizations["jit_fusion_enabled"] is True
        assert optimizations["fusion_strategy"] == "DYNAMIC"

    def test_enable_quantization(self, optimizer):
        """Test quantization optimization enabling."""
        optimizations = optimizer._enable_quantization()
        
        assert optimizations["quantization_bits"] == 8
        assert optimizations["quantization_enabled"] is True

    def test_jit_compile_model(self, optimizer, simple_model):
        """Test JIT model compilation."""
        input_shape = (10,)
        compiled_model = optimizer._jit_compile_model(simple_model, input_shape)
        
        assert compiled_model is not None
        assert isinstance(compiled_model, torch.jit.ScriptModule)
        
        # Test that the same model is returned for same input shape
        compiled_model_2 = optimizer._jit_compile_model(simple_model, input_shape)
        assert compiled_model is compiled_model_2

    def test_quantize_model(self, optimizer, simple_model):
        """Test model quantization."""
        # Test 8-bit quantization
        config = CPUOptimizationConfig(enable_quantization=True, quantization_bits=8)
        optimizer = CPUOptimizer(config)
        
        quantized_model = optimizer._quantize_model(simple_model)
        assert quantized_model is not None

    def test_get_performance_metrics(self, optimizer, hardware_info):
        """Test performance metrics retrieval."""
        # Before optimization
        metrics = optimizer.get_performance_metrics()
        assert metrics["status"] == "No optimizations applied"
        
        # After optimization
        optimizer.optimize_for_hardware(hardware_info)
        metrics = optimizer.get_performance_metrics()
        
        assert "thread_count" in metrics
        assert "mkldnn_enabled" in metrics
        assert "mkl_enabled" in metrics
        assert "jit_compiled_models" in metrics
        assert "thread_pool_available" in metrics

    def test_cleanup(self, optimizer, hardware_info):
        """Test optimizer cleanup."""
        optimizer.optimize_for_hardware(hardware_info)
        
        # Verify optimization was applied
        assert optimizer._optimization_applied
        
        # Cleanup
        optimizer.cleanup()
        
        # Verify cleanup
        assert not optimizer._optimization_applied
        assert optimizer._thread_pool is None
        assert len(optimizer._jit_compiled_models) == 0

    def test_context_manager_cleanup(self, config, hardware_info):
        """Test that cleanup happens on destruction."""
        optimizer = CPUOptimizer(config)
        optimizer.optimize_for_hardware(hardware_info)
        
        # Simulate destruction
        del optimizer
        
        # This should not raise any exceptions
        assert True

    @pytest.mark.parametrize("batch_size,expected", [
        (1, 4),
        (3, 4),
        (5, 8),
        (7, 8),
        (9, 16),
        (15, 16),
        (17, 32),
        (31, 32),
        (33, 64),
        (100, 64),
    ])
    def test_get_optimal_batch_size_parametrized(self, optimizer, batch_size, expected):
        """Test optimal batch size calculation with various inputs."""
        result = optimizer._get_optimal_batch_size(batch_size)
        assert result == expected

    def test_optimize_for_hardware_with_quantization(self, hardware_info):
        """Test hardware optimization with quantization enabled."""
        config = CPUOptimizationConfig(enable_quantization=True, quantization_bits=16)
        optimizer = CPUOptimizer(config)
        
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "quantization" in optimizations
        assert optimizations["quantization"]["quantization_bits"] == 16

    def test_optimize_for_hardware_disabled_features(self, hardware_info):
        """Test hardware optimization with disabled features."""
        config = CPUOptimizationConfig(
            enable_avx=False,
            enable_sse=False,
            enable_intel_mkl=False,
            enable_multi_threading=False,
            enable_jit_optimization=False
        )
        optimizer = CPUOptimizer(config)
        
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "avx" not in optimizations
        assert "sse" not in optimizations
        assert "intel_mkl" not in optimizations
        assert "multi_threading" not in optimizations
        assert "jit" not in optimizations
