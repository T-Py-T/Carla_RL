"""
Unit tests for optimization manager.

Tests automatic hardware detection and optimization selection
based on hardware capabilities and performance requirements.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.optimization.optimization_manager import OptimizationManager, OptimizationProfile
from src.optimization.cpu_optimizer import CPUOptimizationConfig
from src.optimization.gpu_optimizer import GPUOptimizationConfig
from src.optimization.memory_optimizer import MemoryOptimizationConfig
from src.benchmarking.hardware_detector import HardwareInfo, CPUInfo, GPUInfo, MemoryInfo


class TestOptimizationManager:
    """Test cases for OptimizationManager class."""

    @pytest.fixture
    def manager(self):
        """Create optimization manager for testing."""
        return OptimizationManager()

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

    @pytest.fixture
    def hardware_info_low_end(self):
        """Create low-end hardware info for testing."""
        cpu_info = CPUInfo(
            model="Intel Core i5-10400",
            cores=6,
            threads=12,
            frequency_mhz=2900.0,
            architecture="x86_64",
            features=["avx", "avx2", "sse", "sse2", "sse4"],
            cache_size_mb=12.0,
            avx_support=True,
            sse_support=True,
            intel_mkl_available=False
        )
        
        memory_info = MemoryInfo(
            total_gb=16.0,
            available_gb=8.0,
            swap_gb=4.0,
            memory_type="DDR4"
        )
        
        return HardwareInfo(
            cpu=cpu_info,
            gpu=None,
            memory=memory_info,
            platform="Linux-5.19.0-32-generic-x86_64-with-glibc2.35",
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

    def test_initialization(self, manager):
        """Test optimization manager initialization."""
        assert manager._hardware_detector is not None
        assert manager._cpu_optimizer is None
        assert manager._gpu_optimizer is None
        assert manager._memory_optimizer is None
        assert manager._hardware_info is None
        assert not manager._optimization_applied
        assert len(manager._optimization_profiles) == 4

    def test_optimization_profiles_creation(self, manager):
        """Test that optimization profiles are created correctly."""
        profiles = manager._optimization_profiles
        
        assert "high_performance_cpu" in profiles
        assert "gpu_accelerated" in profiles
        assert "memory_constrained" in profiles
        assert "balanced" in profiles
        
        # Check profile structure
        for profile_name, profile in profiles.items():
            assert isinstance(profile, OptimizationProfile)
            assert profile.name
            assert profile.description
            assert isinstance(profile.cpu_config, CPUOptimizationConfig)
            assert isinstance(profile.gpu_config, GPUOptimizationConfig)
            assert isinstance(profile.memory_config, MemoryOptimizationConfig)
            assert profile.target_latency_ms > 0
            assert profile.target_throughput_rps > 0
            assert profile.memory_limit_gb > 0

    def test_auto_optimize_high_end_hardware(self, manager, hardware_info_high_end):
        """Test auto optimization with high-end hardware."""
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_high_end):
            optimizations = manager.auto_optimize(
                target_latency_ms=5.0,
                target_throughput_rps=5000,
                memory_limit_gb=32.0
            )
        
        assert "cpu" in optimizations
        assert "gpu" in optimizations
        assert "memory" in optimizations
        assert "profile" in optimizations
        assert manager._optimization_applied
        assert manager._cpu_optimizer is not None
        assert manager._gpu_optimizer is not None
        assert manager._memory_optimizer is not None

    def test_auto_optimize_mid_range_hardware(self, manager, hardware_info_mid_range):
        """Test auto optimization with mid-range hardware."""
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            optimizations = manager.auto_optimize(
                target_latency_ms=10.0,
                target_throughput_rps=1000,
                memory_limit_gb=16.0
            )
        
        assert "cpu" in optimizations
        assert "gpu" in optimizations
        assert "memory" in optimizations
        assert "profile" in optimizations
        assert manager._optimization_applied

    def test_auto_optimize_low_end_hardware(self, manager, hardware_info_low_end):
        """Test auto optimization with low-end hardware."""
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_low_end):
            optimizations = manager.auto_optimize(
                target_latency_ms=15.0,
                target_throughput_rps=500,
                memory_limit_gb=8.0
            )
        
        assert "cpu" in optimizations
        assert "memory" in optimizations
        assert "profile" in optimizations
        # GPU optimizations should not be present for low-end hardware
        assert "gpu" not in optimizations
        assert manager._optimization_applied

    def test_optimize_model(self, manager, simple_model, hardware_info_mid_range):
        """Test model optimization."""
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        optimized_model = manager.optimize_model(simple_model, (10,))
        
        assert optimized_model is not None
        assert optimized_model.training is False

    def test_optimize_inference(self, manager, simple_model, hardware_info_mid_range):
        """Test inference optimization."""
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Create test input
        input_tensor = torch.randn(8, 10)
        
        # Test inference
        result = manager.optimize_inference(simple_model, input_tensor)
        
        assert result.shape == (8, 1)

    def test_optimize_inference_without_optimization(self, manager, simple_model):
        """Test inference optimization without prior optimization."""
        input_tensor = torch.randn(8, 10)
        
        with pytest.raises(RuntimeError, match="Optimizations not applied"):
            manager.optimize_inference(simple_model, input_tensor)

    def test_get_performance_metrics(self, manager, hardware_info_mid_range):
        """Test performance metrics retrieval."""
        # Before optimization
        metrics = manager.get_performance_metrics()
        assert metrics["optimization_applied"] is False
        assert metrics["hardware_info"] is None
        
        # After optimization
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        metrics = manager.get_performance_metrics()
        assert metrics["optimization_applied"] is True
        assert metrics["hardware_info"] is not None
        assert "cpu_optimizations" in metrics
        assert "gpu_optimizations" in metrics
        assert "memory_optimizations" in metrics

    def test_get_optimization_recommendations(self, manager, hardware_info_high_end):
        """Test optimization recommendations retrieval."""
        # Before optimization
        recommendations = manager.get_optimization_recommendations()
        assert "Run auto_optimize() first" in recommendations[0]
        
        # After optimization
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_high_end):
            manager.auto_optimize()
        
        recommendations = manager.get_optimization_recommendations()
        assert len(recommendations) > 0
        assert any("AVX" in rec for rec in recommendations)
        assert any("Intel MKL" in rec for rec in recommendations)
        assert any("GPU" in rec for rec in recommendations)

    def test_select_optimal_profile_high_performance(self, manager, hardware_info_high_end):
        """Test profile selection for high-performance requirements."""
        manager._hardware_info = hardware_info_high_end
        
        profile = manager._select_optimal_profile(
            target_latency_ms=5.0,
            target_throughput_rps=5000,
            memory_limit_gb=32.0
        )
        
        assert profile.name in ["GPU Accelerated", "High Performance CPU"]

    def test_select_optimal_profile_memory_constrained(self, manager, hardware_info_low_end):
        """Test profile selection for memory-constrained systems."""
        manager._hardware_info = hardware_info_low_end
        
        profile = manager._select_optimal_profile(
            target_latency_ms=15.0,
            target_throughput_rps=500,
            memory_limit_gb=8.0
        )
        
        assert profile.name in ["Memory Constrained", "Balanced"]

    def test_is_profile_suitable(self, manager, hardware_info_high_end):
        """Test profile suitability checking."""
        manager._hardware_info = hardware_info_high_end
        
        # Test suitable profile
        profile = manager._optimization_profiles["gpu_accelerated"]
        assert manager._is_profile_suitable(profile, None) is True
        
        # Test unsuitable profile (memory constraint)
        assert manager._is_profile_suitable(profile, 4.0) is False

    def test_calculate_profile_score(self, manager):
        """Test profile score calculation."""
        profile = manager._optimization_profiles["balanced"]
        
        # Test with matching requirements
        score = manager._calculate_profile_score(profile, 10.0, 1000)
        assert 0.0 <= score <= 1.0
        
        # Test with different requirements
        score = manager._calculate_profile_score(profile, 5.0, 2000)
        assert 0.0 <= score <= 1.0

    def test_should_use_gpu_with_gpu(self, manager, hardware_info_high_end):
        """Test GPU usage decision with GPU available."""
        manager._hardware_info = hardware_info_high_end
        
        assert manager._should_use_gpu() is True

    def test_should_use_gpu_without_gpu(self, manager, hardware_info_low_end):
        """Test GPU usage decision without GPU."""
        manager._hardware_info = hardware_info_low_end
        
        assert manager._should_use_gpu() is False

    def test_should_use_gpu_insufficient_memory(self, manager):
        """Test GPU usage decision with insufficient GPU memory."""
        cpu_info = CPUInfo(
            model="Intel Core i7",
            cores=8,
            threads=16,
            frequency_mhz=3000.0,
            architecture="x86_64",
            features=["avx"],
            cache_size_mb=16.0,
            avx_support=True,
            sse_support=True,
            intel_mkl_available=True
        )
        
        gpu_info = GPUInfo(
            model="NVIDIA GeForce GTX 1060",
            memory_gb=3.0,  # Insufficient memory
            compute_capability="6.1",
            cuda_available=True,
            tensorrt_available=False,
            driver_version="470.63.01",
            cuda_version="11.4"
        )
        
        memory_info = MemoryInfo(
            total_gb=16.0,
            available_gb=8.0,
            swap_gb=4.0,
            memory_type="DDR4"
        )
        
        hardware_info = HardwareInfo(
            cpu=cpu_info,
            gpu=gpu_info,
            memory=memory_info,
            platform="Linux",
            python_version="3.11.0",
            torch_version="2.1.0",
            optimization_recommendations=[]
        )
        
        manager._hardware_info = hardware_info
        
        assert manager._should_use_gpu() is False

    def test_cleanup(self, manager, hardware_info_mid_range):
        """Test optimizer cleanup."""
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Verify optimization was applied
        assert manager._optimization_applied
        assert manager._cpu_optimizer is not None
        assert manager._gpu_optimizer is not None
        assert manager._memory_optimizer is not None
        
        # Cleanup
        manager.cleanup()
        
        # Verify cleanup
        assert not manager._optimization_applied
        assert manager._cpu_optimizer is None
        assert manager._gpu_optimizer is None
        assert manager._memory_optimizer is None

    def test_context_manager_cleanup(self, hardware_info_mid_range):
        """Test that cleanup happens on destruction."""
        manager = OptimizationManager()
        
        with patch.object(manager._hardware_detector, 'get_hardware_info', return_value=hardware_info_mid_range):
            manager.auto_optimize()
        
        # Simulate destruction
        del manager
        
        # This should not raise any exceptions
        assert True

    def test_optimization_profiles_validation(self, manager):
        """Test that all optimization profiles are valid."""
        for profile_name, profile in manager._optimization_profiles.items():
            # Check that all configurations are valid
            assert profile.cpu_config.max_threads is None or profile.cpu_config.max_threads > 0
            assert profile.gpu_config.memory_fraction > 0.0
            assert profile.gpu_config.memory_fraction <= 1.0
            assert profile.memory_config.pool_size_mb > 0
            assert profile.memory_config.max_pool_entries > 0
            assert profile.target_latency_ms > 0
            assert profile.target_throughput_rps > 0
            assert profile.memory_limit_gb > 0
