"""
Unit tests for GPU optimizer.

Tests GPU-specific optimizations including CUDA, TensorRT,
mixed precision, and memory management.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.optimization.gpu_optimizer import GPUOptimizer, GPUOptimizationConfig
from src.benchmarking.hardware_detector import HardwareInfo, CPUInfo, GPUInfo, MemoryInfo


class TestGPUOptimizer:
    """Test cases for GPU optimizer."""

    @pytest.fixture
    def config(self):
        """Create GPU optimization config for testing."""
        return GPUOptimizationConfig(
            enable_cuda=True,
            enable_tensorrt=True,
            enable_mixed_precision=True,
            enable_memory_optimization=True,
            enable_cudnn_benchmark=True,
            enable_tensor_core_usage=True,
            memory_fraction=0.8,
            enable_jit_optimization=True,
            enable_quantization=False,
            quantization_bits=16,
            enable_dynamic_batching=True,
            max_batch_size=64
        )

    @pytest.fixture
    def optimizer(self, config):
        """Create GPU optimizer for testing."""
        return GPUOptimizer(config)

    @pytest.fixture
    def hardware_info_with_gpu(self):
        """Create mock hardware info with GPU for testing."""
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
    def hardware_info_no_gpu(self):
        """Create mock hardware info without GPU for testing."""
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
        
        memory_info = MemoryInfo(
            total_gb=32.0,
            available_gb=16.0,
            swap_gb=8.0,
            memory_type="DDR4"
        )
        
        return HardwareInfo(
            cpu=cpu_info,
            gpu=None,
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
        """Test GPU optimizer initialization."""
        optimizer = GPUOptimizer(config)
        assert optimizer.config == config
        assert not optimizer._optimization_applied
        assert optimizer._device is None
        assert len(optimizer._jit_compiled_models) == 0
        assert len(optimizer._tensorrt_engines) == 0

    def test_optimize_for_hardware_with_gpu(self, optimizer, hardware_info_with_gpu):
        """Test hardware optimization with GPU available."""
        optimizations = optimizer.optimize_for_hardware(hardware_info_with_gpu)
        
        assert "cuda" in optimizations
        assert "tensorrt" in optimizations
        assert "mixed_precision" in optimizations
        assert "memory" in optimizations
        assert "cudnn" in optimizations
        assert "tensor_cores" in optimizations
        assert "jit" in optimizations
        
        assert optimizer._optimization_applied

    def test_optimize_for_hardware_no_gpu(self, optimizer, hardware_info_no_gpu):
        """Test hardware optimization without GPU."""
        optimizations = optimizer.optimize_for_hardware(hardware_info_no_gpu)
        
        assert "error" in optimizations
        assert optimizations["error"] == "No GPU detected"

    def test_optimize_model(self, optimizer, simple_model):
        """Test model optimization."""
        input_shape = (10,)
        optimized_model = optimizer.optimize_model(simple_model, input_shape)
        
        assert optimized_model is not None
        assert optimized_model.training is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimize_inference_batch_cuda(self, optimizer, simple_model):
        """Test batch inference optimization with CUDA."""
        # Create test input
        batch_size = 8
        input_tensor = torch.randn(batch_size, 10)
        
        # Optimize model first
        optimized_model = optimizer.optimize_model(simple_model, (10,))
        
        # Test batch inference
        with torch.no_grad():
            result = optimizer.optimize_inference_batch(optimized_model, input_tensor)
        
        assert result.shape == (batch_size, 1)
        assert result.device.type == "cuda"

    def test_optimize_inference_batch_cpu(self, optimizer, simple_model):
        """Test batch inference optimization on CPU."""
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

    def test_enable_cuda_optimizations(self, optimizer, hardware_info_with_gpu):
        """Test CUDA optimization enabling."""
        optimizations = optimizer._enable_cuda_optimizations(hardware_info_with_gpu.gpu)
        
        assert "device" in optimizations
        assert "cudnn_enabled" in optimizations
        assert "cudnn_benchmark" in optimizations

    def test_enable_tensorrt_optimizations(self, optimizer, hardware_info_with_gpu):
        """Test TensorRT optimization enabling."""
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_find_spec.return_value = Mock()  # Mock TensorRT availability
            
            optimizations = optimizer._enable_tensorrt_optimizations(hardware_info_with_gpu.gpu)
            
            assert "tensorrt_available" in optimizations

    def test_enable_tensorrt_optimizations_not_available(self, optimizer, hardware_info_with_gpu):
        """Test TensorRT optimization when not available."""
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_find_spec.return_value = None  # Mock TensorRT not available
            
            optimizations = optimizer._enable_tensorrt_optimizations(hardware_info_with_gpu.gpu)
            
            assert optimizations["tensorrt_available"] is False
            assert "error" in optimizations

    def test_enable_mixed_precision(self, optimizer):
        """Test mixed precision optimization enabling."""
        optimizations = optimizer._enable_mixed_precision()
        
        assert optimizations["mixed_precision_enabled"] is True
        assert optimizations["tf32_enabled"] is True
        assert optimizations["scaler_initialized"] is True

    def test_enable_memory_optimizations(self, optimizer, hardware_info_with_gpu):
        """Test memory optimization enabling."""
        optimizations = optimizer._enable_memory_optimizations(hardware_info_with_gpu.gpu)
        
        assert optimizations["memory_fraction"] == 0.8
        assert optimizations["memory_cleared"] is True

    def test_enable_cudnn_optimizations(self, optimizer):
        """Test cuDNN optimization enabling."""
        optimizations = optimizer._enable_cudnn_optimizations()
        
        assert optimizations["cudnn_enabled"] is True
        assert optimizations["cudnn_benchmark"] is True

    def test_enable_tensor_core_usage(self, optimizer):
        """Test Tensor Core usage optimization enabling."""
        optimizations = optimizer._enable_tensor_core_usage()
        
        assert optimizations["tensor_cores_enabled"] is True
        assert optimizations["tf32_enabled"] is True

    def test_enable_jit_optimizations(self, optimizer):
        """Test JIT optimization enabling."""
        optimizations = optimizer._enable_jit_optimizations()
        
        assert optimizations["jit_fusion_enabled"] is True
        assert optimizations["fusion_strategy"] == "DYNAMIC"

    def test_enable_quantization(self, optimizer):
        """Test quantization optimization enabling."""
        optimizations = optimizer._enable_quantization()
        
        assert optimizations["quantization_bits"] == 16
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

    def test_quantize_model_8bit(self, simple_model):
        """Test 8-bit model quantization."""
        config = GPUOptimizationConfig(enable_quantization=True, quantization_bits=8)
        optimizer = GPUOptimizer(config)
        
        quantized_model = optimizer._quantize_model(simple_model)
        assert quantized_model is not None

    def test_quantize_model_16bit(self, simple_model):
        """Test 16-bit model quantization."""
        config = GPUOptimizationConfig(enable_quantization=True, quantization_bits=16)
        optimizer = GPUOptimizer(config)
        
        quantized_model = optimizer._quantize_model(simple_model)
        assert quantized_model is not None

    def test_supports_mixed_precision(self, optimizer, hardware_info_with_gpu):
        """Test mixed precision support detection."""
        # Test with GPU that supports mixed precision
        assert optimizer._supports_mixed_precision(hardware_info_with_gpu.gpu) is True
        
        # Test with GPU that doesn't support mixed precision
        gpu_info_low = GPUInfo(
            model="NVIDIA GeForce GTX 1060",
            memory_gb=6.0,
            compute_capability="6.1",
            cuda_available=True,
            tensorrt_available=False,
            driver_version="470.63.01",
            cuda_version="11.4"
        )
        assert optimizer._supports_mixed_precision(gpu_info_low) is False

    def test_supports_tensor_cores(self, optimizer, hardware_info_with_gpu):
        """Test Tensor Core support detection."""
        # Test with GPU that supports Tensor Cores
        assert optimizer._supports_tensor_cores(hardware_info_with_gpu.gpu) is True
        
        # Test with GPU that doesn't support Tensor Cores
        gpu_info_low = GPUInfo(
            model="NVIDIA GeForce GTX 1060",
            memory_gb=6.0,
            compute_capability="6.1",
            cuda_available=True,
            tensorrt_available=False,
            driver_version="470.63.01",
            cuda_version="11.4"
        )
        assert optimizer._supports_tensor_cores(gpu_info_low) is False

    def test_get_optimal_batch_size(self, optimizer):
        """Test optimal batch size calculation."""
        # Test different input batch sizes
        assert optimizer._get_optimal_batch_size(4) == 8
        assert optimizer._get_optimal_batch_size(12) == 16
        assert optimizer._get_optimal_batch_size(24) == 32
        assert optimizer._get_optimal_batch_size(48) == 64
        assert optimizer._get_optimal_batch_size(100) == 64  # Capped at max_batch_size

    def test_get_performance_metrics(self, optimizer, hardware_info_with_gpu):
        """Test performance metrics retrieval."""
        # Before optimization
        metrics = optimizer.get_performance_metrics()
        assert metrics["status"] == "No optimizations applied"
        
        # After optimization
        optimizer.optimize_for_hardware(hardware_info_with_gpu)
        metrics = optimizer.get_performance_metrics()
        
        assert "device" in metrics
        assert "cudnn_enabled" in metrics
        assert "cudnn_benchmark" in metrics
        assert "jit_compiled_models" in metrics
        assert "tensorrt_engines" in metrics
        assert "mixed_precision_enabled" in metrics

    def test_cleanup(self, optimizer, hardware_info_with_gpu):
        """Test optimizer cleanup."""
        optimizer.optimize_for_hardware(hardware_info_with_gpu)
        
        # Verify optimization was applied
        assert optimizer._optimization_applied
        
        # Cleanup
        optimizer.cleanup()
        
        # Verify cleanup
        assert not optimizer._optimization_applied
        assert optimizer._device is None
        assert len(optimizer._jit_compiled_models) == 0
        assert len(optimizer._tensorrt_engines) == 0

    def test_context_manager_cleanup(self, config, hardware_info_with_gpu):
        """Test that cleanup happens on destruction."""
        optimizer = GPUOptimizer(config)
        optimizer.optimize_for_hardware(hardware_info_with_gpu)
        
        # Simulate destruction
        del optimizer
        
        # This should not raise any exceptions
        assert True

    @pytest.mark.parametrize("batch_size,expected", [
        (1, 8),
        (4, 8),
        (6, 8),
        (10, 16),
        (14, 16),
        (18, 32),
        (30, 32),
        (34, 64),
        (50, 64),
        (100, 64),  # Capped at max_batch_size
    ])
    def test_get_optimal_batch_size_parametrized(self, optimizer, batch_size, expected):
        """Test optimal batch size calculation with various inputs."""
        result = optimizer._get_optimal_batch_size(batch_size)
        assert result == expected

    def test_optimize_for_hardware_with_quantization(self, hardware_info_with_gpu):
        """Test hardware optimization with quantization enabled."""
        config = GPUOptimizationConfig(enable_quantization=True, quantization_bits=8)
        optimizer = GPUOptimizer(config)
        
        optimizations = optimizer.optimize_for_hardware(hardware_info_with_gpu)
        
        assert "quantization" in optimizations
        assert optimizations["quantization"]["quantization_bits"] == 8

    def test_optimize_for_hardware_disabled_features(self, hardware_info_with_gpu):
        """Test hardware optimization with disabled features."""
        config = GPUOptimizationConfig(
            enable_cuda=False,
            enable_tensorrt=False,
            enable_mixed_precision=False,
            enable_memory_optimization=False,
            enable_cudnn_benchmark=False,
            enable_tensor_core_usage=False,
            enable_jit_optimization=False
        )
        optimizer = GPUOptimizer(config)
        
        optimizations = optimizer.optimize_for_hardware(hardware_info_with_gpu)
        
        assert "cuda" not in optimizations
        assert "tensorrt" not in optimizations
        assert "mixed_precision" not in optimizations
        assert "memory" not in optimizations
        assert "cudnn" not in optimizations
        assert "tensor_cores" not in optimizations
        assert "jit" not in optimizations
