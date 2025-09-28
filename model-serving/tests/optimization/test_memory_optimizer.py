"""
Unit tests for memory optimizer.

Tests memory optimizations including memory pooling,
allocation strategies, and hardware-specific memory management.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.optimization.memory_optimizer import MemoryOptimizer, MemoryOptimizationConfig, MemoryPool
from src.benchmarking.hardware_detector import HardwareInfo, CPUInfo, GPUInfo, MemoryInfo


class TestMemoryPool:
    """Test cases for MemoryPool class."""

    def test_initialization(self):
        """Test memory pool initialization."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        assert pool.pool_size_bytes == 1024 * 1024 * 1024
        assert pool.max_entries == 100
        assert len(pool.pools) == 0

    def test_get_tensor_new(self):
        """Test getting a new tensor from empty pool."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        tensor = pool.get_tensor((10, 20), torch.float32, torch.device("cpu"))
        
        assert tensor.shape == (10, 20)
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")

    def test_get_tensor_reuse(self):
        """Test getting a tensor from pool with existing tensors."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        # Create and return a tensor
        tensor1 = torch.empty((10, 20), dtype=torch.float32, device=torch.device("cpu"))
        pool.return_tensor(tensor1)
        
        # Get a tensor with same shape/dtype/device
        tensor2 = pool.get_tensor((10, 20), torch.float32, torch.device("cpu"))
        
        assert tensor2.shape == (10, 20)
        assert tensor2.dtype == torch.float32
        assert tensor2.device == torch.device("cpu")

    def test_return_tensor_small(self):
        """Test returning a tensor that should be pooled."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        tensor = torch.empty((100, 100), dtype=torch.float32, device=torch.device("cpu"))
        pool.return_tensor(tensor)
        
        # Check that tensor was added to pool
        key = (tensor.shape, tensor.dtype, tensor.device)
        assert key in pool.pools
        assert len(pool.pools[key]) == 1

    def test_return_tensor_too_small(self):
        """Test returning a tensor that's too small to pool."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        tensor = torch.empty((5, 5), dtype=torch.float32, device=torch.device("cpu"))
        pool.return_tensor(tensor)
        
        # Check that tensor was not added to pool
        assert len(pool.pools) == 0

    def test_return_tensor_too_large(self):
        """Test returning a tensor that's too large to pool."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        tensor = torch.empty((10000, 10000), dtype=torch.float32, device=torch.device("cpu"))
        pool.return_tensor(tensor)
        
        # Check that tensor was not added to pool
        assert len(pool.pools) == 0

    def test_max_entries_limit(self):
        """Test that max_entries limit is respected."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=2)
        
        # Add more tensors than max_entries
        for i in range(5):
            tensor = torch.empty((100, 100), dtype=torch.float32, device=torch.device("cpu"))
            pool.return_tensor(tensor)
        
        # Check that only max_entries tensors are kept
        key = ((100, 100), torch.float32, torch.device("cpu"))
        assert len(pool.pools[key]) == 2

    def test_clear(self):
        """Test clearing the memory pool."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        # Add some tensors
        for i in range(3):
            tensor = torch.empty((100, 100), dtype=torch.float32, device=torch.device("cpu"))
            pool.return_tensor(tensor)
        
        assert len(pool.pools) > 0
        
        # Clear the pool
        pool.clear()
        
        assert len(pool.pools) == 0

    def test_get_stats(self):
        """Test getting memory pool statistics."""
        pool = MemoryPool(pool_size_mb=1024, max_entries=100)
        
        # Add some tensors
        for i in range(3):
            tensor = torch.empty((100, 100), dtype=torch.float32, device=torch.device("cpu"))
            pool.return_tensor(tensor)
        
        stats = pool.get_stats()
        
        assert "total_tensors" in stats
        assert "total_memory_bytes" in stats
        assert "pool_entries" in stats
        assert stats["total_tensors"] == 3
        assert stats["pool_entries"] == 1


class TestMemoryOptimizer:
    """Test cases for MemoryOptimizer class."""

    @pytest.fixture
    def config(self):
        """Create memory optimization config for testing."""
        return MemoryOptimizationConfig(
            enable_memory_pooling=True,
            enable_memory_pinning=True,
            enable_garbage_collection=True,
            enable_memory_mapping=True,
            pool_size_mb=1024,
            max_pool_entries=100,
            enable_shared_memory=True,
            enable_memory_compression=False,
            compression_ratio=0.5,
            enable_memory_prefetching=True,
            prefetch_size_mb=256
        )

    @pytest.fixture
    def optimizer(self, config):
        """Create memory optimizer for testing."""
        return MemoryOptimizer(config)

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
        """Test memory optimizer initialization."""
        optimizer = MemoryOptimizer(config)
        assert optimizer.config == config
        assert not optimizer._optimization_applied
        assert optimizer._memory_pool is None

    def test_optimize_for_hardware(self, optimizer, hardware_info):
        """Test hardware optimization application."""
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "memory_pooling" in optimizations
        assert "memory_pinning" in optimizations
        assert "garbage_collection" in optimizations
        assert "memory_mapping" in optimizations
        assert "shared_memory" in optimizations
        assert "memory_prefetching" in optimizations
        
        assert optimizer._optimization_applied

    def test_optimize_tensor_allocation(self, optimizer):
        """Test optimized tensor allocation."""
        tensor = optimizer.optimize_tensor_allocation(
            shape=(10, 20),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        assert tensor.shape == (10, 20)
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")

    def test_optimize_batch_allocation(self, optimizer):
        """Test optimized batch allocation."""
        tensor = optimizer.optimize_batch_allocation(
            batch_size=8,
            input_shape=(10,),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        assert tensor.shape == (8, 10)
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")

    def test_return_tensor(self, optimizer):
        """Test returning a tensor to the pool."""
        # First optimize for hardware to enable pooling
        hardware_info = Mock()
        hardware_info.memory = Mock()
        hardware_info.memory.available_gb = 16.0
        optimizer.optimize_for_hardware(hardware_info)
        
        tensor = torch.empty((100, 100), dtype=torch.float32, device=torch.device("cpu"))
        optimizer.return_tensor(tensor)
        
        # This should not raise any exceptions
        assert True

    def test_optimize_model_memory(self, optimizer, simple_model):
        """Test model memory optimization."""
        optimized_model = optimizer.optimize_model_memory(simple_model)
        
        assert optimized_model is not None
        assert optimized_model.training is False

    def test_enable_memory_pooling(self, optimizer, hardware_info):
        """Test memory pooling optimization enabling."""
        optimizations = optimizer._enable_memory_pooling(hardware_info.memory)
        
        assert "pool_size_mb" in optimizations
        assert "max_entries" in optimizations
        assert optimizations["memory_pooling_enabled"] is True
        assert optimizer._memory_pool is not None

    def test_enable_memory_pinning(self, optimizer):
        """Test memory pinning optimization enabling."""
        optimizations = optimizer._enable_memory_pinning()
        
        assert optimizations["memory_pinning_enabled"] is True
        assert optimizations["cudnn_benchmark"] is True

    def test_enable_garbage_collection(self, optimizer):
        """Test garbage collection optimization enabling."""
        optimizations = optimizer._enable_garbage_collection()
        
        assert "gc_thresholds" in optimizations
        assert optimizations["garbage_collection_enabled"] is True

    def test_enable_memory_mapping(self, optimizer):
        """Test memory mapping optimization enabling."""
        optimizations = optimizer._enable_memory_mapping()
        
        assert optimizations["memory_mapping_enabled"] is True

    def test_enable_shared_memory(self, optimizer):
        """Test shared memory optimization enabling."""
        optimizations = optimizer._enable_shared_memory()
        
        assert optimizations["shared_memory_enabled"] is True

    def test_enable_memory_compression(self, optimizer):
        """Test memory compression optimization enabling."""
        optimizations = optimizer._enable_memory_compression()
        
        assert optimizations["memory_compression_enabled"] is True
        assert optimizations["compression_ratio"] == 0.5

    def test_enable_memory_prefetching(self, optimizer):
        """Test memory prefetching optimization enabling."""
        optimizations = optimizer._enable_memory_prefetching()
        
        assert optimizations["memory_prefetching_enabled"] is True
        assert optimizations["prefetch_size_mb"] == 256

    def test_get_memory_stats(self, optimizer, hardware_info):
        """Test memory statistics retrieval."""
        # Before optimization
        stats = optimizer.get_memory_stats()
        assert stats["optimization_applied"] is False
        assert stats["memory_pooling_enabled"] is False
        
        # After optimization
        optimizer.optimize_for_hardware(hardware_info)
        stats = optimizer.get_memory_stats()
        
        assert stats["optimization_applied"] is True
        assert stats["memory_pooling_enabled"] is True
        assert "python_memory_rss" in stats
        assert "python_memory_vms" in stats

    def test_cleanup(self, optimizer, hardware_info):
        """Test optimizer cleanup."""
        optimizer.optimize_for_hardware(hardware_info)
        
        # Verify optimization was applied
        assert optimizer._optimization_applied
        
        # Cleanup
        optimizer.cleanup()
        
        # Verify cleanup
        assert not optimizer._optimization_applied
        assert optimizer._memory_pool is None

    def test_context_manager_cleanup(self, config, hardware_info):
        """Test that cleanup happens on destruction."""
        optimizer = MemoryOptimizer(config)
        optimizer.optimize_for_hardware(hardware_info)
        
        # Simulate destruction
        del optimizer
        
        # This should not raise any exceptions
        assert True

    def test_optimize_for_hardware_with_compression(self, hardware_info):
        """Test hardware optimization with compression enabled."""
        config = MemoryOptimizationConfig(enable_memory_compression=True)
        optimizer = MemoryOptimizer(config)
        
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "memory_compression" in optimizations
        assert optimizations["memory_compression"]["memory_compression_enabled"] is True

    def test_optimize_for_hardware_disabled_features(self, hardware_info):
        """Test hardware optimization with disabled features."""
        config = MemoryOptimizationConfig(
            enable_memory_pooling=False,
            enable_memory_pinning=False,
            enable_garbage_collection=False,
            enable_memory_mapping=False,
            enable_shared_memory=False,
            enable_memory_compression=False,
            enable_memory_prefetching=False
        )
        optimizer = MemoryOptimizer(config)
        
        optimizations = optimizer.optimize_for_hardware(hardware_info)
        
        assert "memory_pooling" not in optimizations
        assert "memory_pinning" not in optimizations
        assert "garbage_collection" not in optimizations
        assert "memory_mapping" not in optimizations
        assert "shared_memory" not in optimizations
        assert "memory_compression" not in optimizations
        assert "memory_prefetching" not in optimizations

    def test_memory_pool_integration(self, optimizer, hardware_info):
        """Test memory pool integration with optimizer."""
        # Enable memory pooling
        optimizer.optimize_for_hardware(hardware_info)
        
        # Allocate a tensor
        tensor1 = optimizer.optimize_tensor_allocation(
            shape=(100, 100),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        # Return it to the pool
        optimizer.return_tensor(tensor1)
        
        # Allocate another tensor with same shape/dtype/device
        tensor2 = optimizer.optimize_tensor_allocation(
            shape=(100, 100),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        # Should be able to allocate without issues
        assert tensor2.shape == (100, 100)
        assert tensor2.dtype == torch.float32
        assert tensor2.device == torch.device("cpu")
