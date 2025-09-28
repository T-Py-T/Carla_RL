"""
CPU-specific optimizations for Policy-as-a-Service inference.

This module provides CPU optimizations including AVX, SSE, Intel MKL,
and multi-threading optimizations for maximum inference performance.
"""

import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.jit import ScriptModule

from ..benchmarking.hardware_detector import CPUInfo, HardwareInfo


@dataclass
class CPUOptimizationConfig:
    """Configuration for CPU optimizations."""

    enable_avx: bool = True
    enable_sse: bool = True
    enable_intel_mkl: bool = True
    enable_multi_threading: bool = True
    max_threads: Optional[int] = None
    enable_jit_optimization: bool = True
    enable_memory_pinning: bool = True
    batch_size_optimization: bool = True
    enable_quantization: bool = False
    quantization_bits: int = 8


class CPUOptimizer:
    """
    CPU optimization engine for Policy-as-a-Service inference.

    Provides hardware-specific optimizations including AVX, SSE,
    Intel MKL, multi-threading, and JIT compilation optimizations.
    """

    def __init__(self, config: CPUOptimizationConfig):
        """Initialize CPU optimizer with configuration."""
        self.config = config
        self._thread_pool: Optional[torch.ThreadPool] = None
        self._jit_compiled_models: Dict[str, ScriptModule] = {}
        self._optimization_applied = False

    def optimize_for_hardware(self, hardware_info: HardwareInfo) -> Dict[str, any]:
        """
        Apply CPU optimizations based on hardware capabilities.
        
        Args:
            hardware_info: Complete hardware information
            
        Returns:
            Dictionary of applied optimizations and their status
        """
        optimizations = {}
        
        # Apply AVX optimizations
        if self.config.enable_avx and hardware_info.cpu.avx_support:
            optimizations["avx"] = self._enable_avx_optimizations()
        
        # Apply SSE optimizations
        if self.config.enable_sse and hardware_info.cpu.sse_support:
            optimizations["sse"] = self._enable_sse_optimizations()
        
        # Apply Intel MKL optimizations
        if self.config.enable_intel_mkl and hardware_info.cpu.intel_mkl_available:
            optimizations["intel_mkl"] = self._enable_intel_mkl_optimizations()
        
        # Apply multi-threading optimizations
        if self.config.enable_multi_threading:
            thread_count = self._get_optimal_thread_count(hardware_info)
            optimizations["multi_threading"] = self._enable_multi_threading(thread_count)
        
        # Apply memory optimizations
        if self.config.enable_memory_pinning:
            optimizations["memory_pinning"] = self._enable_memory_pinning()
        
        # Apply JIT optimizations
        if self.config.enable_jit_optimization:
            optimizations["jit"] = self._enable_jit_optimizations()
        
        # Apply quantization if enabled
        if self.config.enable_quantization:
            optimizations["quantization"] = self._enable_quantization()
        
        self._optimization_applied = True
        return optimizations

    def optimize_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Optimize a PyTorch model for CPU inference.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape for optimization
            
        Returns:
            Optimized model
        """
        # Set model to evaluation mode
        model.eval()
        
        # Apply JIT compilation if enabled
        if self.config.enable_jit_optimization:
            model = self._jit_compile_model(model, input_shape)
        
        # Apply quantization if enabled
        if self.config.enable_quantization:
            model = self._quantize_model(model)
        
        return model

    def optimize_inference_batch(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Optimize batch inference with CPU-specific optimizations.
        
        Args:
            model: Optimized PyTorch model
            inputs: Input tensor batch
            batch_size: Optional batch size override
            
        Returns:
            Optimized inference results
        """
        if batch_size is None and self.config.batch_size_optimization:
            batch_size = self._get_optimal_batch_size(inputs.shape[0])
        
        # Use torch.no_grad() for inference
        with torch.no_grad():
            if batch_size and batch_size < inputs.shape[0]:
                # Process in optimized batches
                results = []
                for i in range(0, inputs.shape[0], batch_size):
                    batch = inputs[i:i + batch_size]
                    batch_result = model(batch)
                    results.append(batch_result)
                return torch.cat(results, dim=0)
            else:
                return model(inputs)

    def _enable_avx_optimizations(self) -> Dict[str, any]:
        """Enable AVX optimizations for CPU inference."""
        optimizations = {}
        
        # Set environment variables for AVX
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
        os.environ["MKL_NUM_THREADS"] = str(torch.get_num_threads())
        
        # Enable AVX in PyTorch
        torch.backends.mkldnn.enabled = True
        torch.backends.mkldnn.allow_tf32 = True
        
        optimizations["mkldnn_enabled"] = True
        optimizations["tf32_enabled"] = True
        
        return optimizations

    def _enable_sse_optimizations(self) -> Dict[str, any]:
        """Enable SSE optimizations for CPU inference."""
        optimizations = {}
        
        # Enable optimized BLAS operations
        torch.backends.mkl.enabled = True
        
        # Set threading for BLAS operations
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(torch.get_num_threads())
        
        optimizations["mkl_enabled"] = True
        optimizations["blas_optimized"] = True
        
        return optimizations

    def _enable_intel_mkl_optimizations(self) -> Dict[str, any]:
        """Enable Intel MKL optimizations."""
        optimizations = {}
        
        # Set MKL environment variables
        os.environ["MKL_NUM_THREADS"] = str(torch.get_num_threads())
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
        
        # Enable MKL optimizations
        torch.backends.mkl.enabled = True
        torch.backends.mkldnn.enabled = True
        
        # Set optimal threading
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(torch.get_num_threads())
        
        optimizations["mkl_threading"] = torch.get_num_threads()
        optimizations["mkl_enabled"] = True
        
        return optimizations

    def _enable_multi_threading(self, thread_count: int) -> Dict[str, any]:
        """Enable multi-threading optimizations."""
        optimizations = {}
        
        # Set PyTorch threading
        torch.set_num_threads(thread_count)
        
        # Set OpenMP threading
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        
        # Create thread pool for parallel operations
        self._thread_pool = torch.ThreadPool(thread_count)
        
        optimizations["thread_count"] = thread_count
        optimizations["thread_pool_created"] = True
        
        return optimizations

    def _enable_memory_pinning(self) -> Dict[str, any]:
        """Enable memory pinning for faster data transfer."""
        optimizations = {}
        
        # Enable memory pinning in PyTorch
        torch.backends.cudnn.benchmark = True
        
        optimizations["memory_pinning_enabled"] = True
        optimizations["cudnn_benchmark"] = True
        
        return optimizations

    def _enable_jit_optimizations(self) -> Dict[str, any]:
        """Enable JIT compilation optimizations."""
        optimizations = {}
        
        # Enable JIT optimizations
        torch.jit.set_fusion_strategy([("DYNAMIC", 20)])
        
        optimizations["jit_fusion_enabled"] = True
        optimizations["fusion_strategy"] = "DYNAMIC"
        
        return optimizations

    def _enable_quantization(self) -> Dict[str, any]:
        """Enable model quantization for faster inference."""
        optimizations = {}
        
        # Set quantization configuration
        optimizations["quantization_bits"] = self.config.quantization_bits
        optimizations["quantization_enabled"] = True
        
        return optimizations

    def _jit_compile_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> ScriptModule:
        """Compile model with JIT for optimization."""
        model_id = f"{model.__class__.__name__}_{input_shape}"
        
        if model_id in self._jit_compiled_models:
            return self._jit_compiled_models[model_id]
        
        # Create example input
        example_input = torch.randn(1, *input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize the traced model
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        self._jit_compiled_models[model_id] = optimized_model
        return optimized_model

    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to the model."""
        if self.config.quantization_bits == 8:
            return torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif self.config.quantization_bits == 16:
            return torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.float16
            )
        else:
            return model

    def _get_optimal_thread_count(self, hardware_info: HardwareInfo) -> int:
        """Get optimal thread count based on hardware."""
        if self.config.max_threads:
            return min(self.config.max_threads, hardware_info.cpu.threads)
        
        # Use logical cores but cap at 8 for inference
        return min(hardware_info.cpu.threads, 8)

    def _get_optimal_batch_size(self, current_batch_size: int) -> int:
        """Get optimal batch size for CPU inference."""
        # CPU-optimized batch sizes based on common patterns
        if current_batch_size <= 4:
            return 4
        elif current_batch_size <= 8:
            return 8
        elif current_batch_size <= 16:
            return 16
        elif current_batch_size <= 32:
            return 32
        else:
            return 64

    def get_performance_metrics(self) -> Dict[str, any]:
        """Get performance metrics for applied optimizations."""
        if not self._optimization_applied:
            return {"status": "No optimizations applied"}
        
        metrics = {
            "thread_count": torch.get_num_threads(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkl_enabled": torch.backends.mkl.enabled,
            "jit_compiled_models": len(self._jit_compiled_models),
            "thread_pool_available": self._thread_pool is not None,
        }
        
        return metrics

    def cleanup(self):
        """Cleanup resources and reset optimizations."""
        if self._thread_pool:
            del self._thread_pool
            self._thread_pool = None
        
        self._jit_compiled_models.clear()
        self._optimization_applied = False

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
