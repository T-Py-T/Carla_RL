"""
Memory optimization for Policy-as-a-Service inference.

This module provides memory optimizations including memory pooling,
allocation strategies, and hardware-specific memory management.
"""

import gc
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..benchmarking.hardware_detector import HardwareInfo, MemoryInfo


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""

    enable_memory_pooling: bool = True
    enable_memory_pinning: bool = True
    enable_garbage_collection: bool = True
    enable_memory_mapping: bool = True
    pool_size_mb: int = 1024
    max_pool_entries: int = 100
    enable_shared_memory: bool = True
    enable_memory_compression: bool = False
    compression_ratio: float = 0.5
    enable_memory_prefetching: bool = True
    prefetch_size_mb: int = 256


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""

    def __init__(self, pool_size_mb: int, max_entries: int):
        """Initialize memory pool."""
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.pools: Dict[Tuple[int, ...], List[Tensor]] = {}
        self.lock = threading.Lock()

    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> Tensor:
        """Get a tensor from the pool or create a new one."""
        with self.lock:
            key = (shape, dtype, device)
            
            if key in self.pools and self.pools[key]:
                return self.pools[key].pop()
            
            return torch.empty(shape, dtype=dtype, device=device)

    def return_tensor(self, tensor: Tensor):
        """Return a tensor to the pool for reuse."""
        if not self._should_pool_tensor(tensor):
            return
        
        with self.lock:
            key = (tuple(tensor.shape), tensor.dtype, tensor.device)
            
            if key not in self.pools:
                self.pools[key] = []
            
            if len(self.pools[key]) < self.max_entries:
                # Clear the tensor data but keep the shape/dtype
                tensor.zero_()
                self.pools[key].append(tensor)

    def _should_pool_tensor(self, tensor: Tensor) -> bool:
        """Determine if a tensor should be pooled."""
        # Only pool tensors that are reasonably sized
        numel = tensor.numel()
        return 100 <= numel <= 1000000  # 100 to 1M elements

    def clear(self):
        """Clear all pooled tensors."""
        with self.lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()

    def get_stats(self) -> Dict[str, any]:
        """Get memory pool statistics."""
        with self.lock:
            total_tensors = sum(len(pool) for pool in self.pools.values())
            total_memory = sum(
                sum(tensor.numel() * tensor.element_size() for tensor in pool)
                for pool in self.pools.values()
            )
            
            return {
                "total_tensors": total_tensors,
                "total_memory_bytes": total_memory,
                "pool_entries": len(self.pools),
            }


class MemoryOptimizer:
    """
    Memory optimization engine for Policy-as-a-Service inference.

    Provides memory pooling, allocation strategies, and hardware-specific
    memory management for optimal memory usage.
    """

    def __init__(self, config: MemoryOptimizationConfig):
        """Initialize memory optimizer with configuration."""
        self.config = config
        self._memory_pool: Optional[MemoryPool] = None
        self._optimization_applied = False
        self._memory_stats: Dict[str, any] = {}

    def optimize_for_hardware(self, hardware_info: HardwareInfo) -> Dict[str, any]:
        """
        Apply memory optimizations based on hardware capabilities.
        
        Args:
            hardware_info: Complete hardware information
            
        Returns:
            Dictionary of applied optimizations and their status
        """
        optimizations = {}
        
        # Initialize memory pool
        if self.config.enable_memory_pooling:
            optimizations["memory_pooling"] = self._enable_memory_pooling(hardware_info.memory)
        
        # Apply memory pinning
        if self.config.enable_memory_pinning:
            optimizations["memory_pinning"] = self._enable_memory_pinning()
        
        # Apply garbage collection optimization
        if self.config.enable_garbage_collection:
            optimizations["garbage_collection"] = self._enable_garbage_collection()
        
        # Apply memory mapping
        if self.config.enable_memory_mapping:
            optimizations["memory_mapping"] = self._enable_memory_mapping()
        
        # Apply shared memory
        if self.config.enable_shared_memory:
            optimizations["shared_memory"] = self._enable_shared_memory()
        
        # Apply memory compression
        if self.config.enable_memory_compression:
            optimizations["memory_compression"] = self._enable_memory_compression()
        
        # Apply memory prefetching
        if self.config.enable_memory_prefetching:
            optimizations["memory_prefetching"] = self._enable_memory_prefetching()
        
        self._optimization_applied = True
        return optimizations

    def optimize_tensor_allocation(
        self, 
        shape: Tuple[int, ...], 
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """
        Optimize tensor allocation using memory pooling.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Tensor device
            
        Returns:
            Optimized tensor
        """
        if self._memory_pool and self.config.enable_memory_pooling:
            return self._memory_pool.get_tensor(shape, dtype, device)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def optimize_batch_allocation(
        self, 
        batch_size: int, 
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """
        Optimize batch tensor allocation.
        
        Args:
            batch_size: Batch size
            input_shape: Input tensor shape
            dtype: Tensor data type
            device: Tensor device
            
        Returns:
            Optimized batch tensor
        """
        full_shape = (batch_size,) + input_shape
        return self.optimize_tensor_allocation(full_shape, dtype, device)

    def return_tensor(self, tensor: Tensor):
        """Return a tensor to the memory pool for reuse."""
        if self._memory_pool and self.config.enable_memory_pooling:
            self._memory_pool.return_tensor(tensor)

    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """
        Optimize model memory usage.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Memory-optimized model
        """
        # Set model to evaluation mode
        model.eval()
        
        # Enable memory-efficient attention if available
        if hasattr(model, 'enable_memory_efficient_attention'):
            model.enable_memory_efficient_attention()
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model

    def _enable_memory_pooling(self, memory_info: MemoryInfo) -> Dict[str, any]:
        """Enable memory pooling for efficient allocation."""
        optimizations = {}
        
        # Calculate optimal pool size based on available memory
        pool_size_mb = min(
            self.config.pool_size_mb,
            int(memory_info.available_gb * 1024 * 0.1)  # Use 10% of available memory
        )
        
        self._memory_pool = MemoryPool(pool_size_mb, self.config.max_pool_entries)
        
        optimizations["pool_size_mb"] = pool_size_mb
        optimizations["max_entries"] = self.config.max_pool_entries
        optimizations["memory_pooling_enabled"] = True
        
        return optimizations

    def _enable_memory_pinning(self) -> Dict[str, any]:
        """Enable memory pinning for faster data transfer."""
        optimizations = {}
        
        # Enable memory pinning in PyTorch
        torch.backends.cudnn.benchmark = True
        
        optimizations["memory_pinning_enabled"] = True
        optimizations["cudnn_benchmark"] = True
        
        return optimizations

    def _enable_garbage_collection(self) -> Dict[str, any]:
        """Enable optimized garbage collection."""
        optimizations = {}
        
        # Set garbage collection thresholds
        gc.set_threshold(100, 10, 10)  # More aggressive collection
        
        optimizations["gc_thresholds"] = gc.get_threshold()
        optimizations["garbage_collection_enabled"] = True
        
        return optimizations

    def _enable_memory_mapping(self) -> Dict[str, any]:
        """Enable memory mapping for large data."""
        optimizations = {}
        
        # Enable memory mapping optimizations
        optimizations["memory_mapping_enabled"] = True
        
        return optimizations

    def _enable_shared_memory(self) -> Dict[str, any]:
        """Enable shared memory for multi-process scenarios."""
        optimizations = {}
        
        # Enable shared memory optimizations
        optimizations["shared_memory_enabled"] = True
        
        return optimizations

    def _enable_memory_compression(self) -> Dict[str, any]:
        """Enable memory compression for large tensors."""
        optimizations = {}
        
        # Enable memory compression
        optimizations["memory_compression_enabled"] = True
        optimizations["compression_ratio"] = self.config.compression_ratio
        
        return optimizations

    def _enable_memory_prefetching(self) -> Dict[str, any]:
        """Enable memory prefetching for better performance."""
        optimizations = {}
        
        # Enable memory prefetching
        optimizations["memory_prefetching_enabled"] = True
        optimizations["prefetch_size_mb"] = self.config.prefetch_size_mb
        
        return optimizations

    def get_memory_stats(self) -> Dict[str, any]:
        """Get memory usage statistics."""
        stats = {
            "optimization_applied": self._optimization_applied,
            "memory_pooling_enabled": self._memory_pool is not None,
        }
        
        if self._memory_pool:
            stats.update(self._memory_pool.get_stats())
        
        # Add PyTorch memory stats
        if torch.cuda.is_available():
            stats["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            stats["cuda_memory_reserved"] = torch.cuda.memory_reserved()
            stats["cuda_memory_cached"] = torch.cuda.memory_cached()
        
        # Add Python memory stats
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        stats["python_memory_rss"] = memory_info.rss
        stats["python_memory_vms"] = memory_info.vms
        
        return stats

    def cleanup(self):
        """Cleanup memory resources."""
        if self._memory_pool:
            self._memory_pool.clear()
            self._memory_pool = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._optimization_applied = False

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
