"""
GPU-specific optimizations for Policy-as-a-Service inference.

This module provides GPU optimizations including CUDA, TensorRT,
mixed precision, and memory management for maximum inference performance.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.jit import ScriptModule

from ..benchmarking.hardware_detector import GPUInfo, HardwareInfo


@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU optimizations."""

    enable_cuda: bool = True
    enable_tensorrt: bool = True
    enable_mixed_precision: bool = True
    enable_memory_optimization: bool = True
    enable_cudnn_benchmark: bool = True
    enable_tensor_core_usage: bool = True
    memory_fraction: float = 0.8
    enable_jit_optimization: bool = True
    enable_quantization: bool = False
    quantization_bits: int = 16
    enable_dynamic_batching: bool = True
    max_batch_size: int = 64


class GPUOptimizer:
    """
    GPU optimization engine for Policy-as-a-Service inference.

    Provides hardware-specific optimizations including CUDA, TensorRT,
    mixed precision, and memory management for GPU inference.
    """

    def __init__(self, config: GPUOptimizationConfig):
        """Initialize GPU optimizer with configuration."""
        self.config = config
        self._device: Optional[torch.device] = None
        self._jit_compiled_models: Dict[str, ScriptModule] = {}
        self._tensorrt_engines: Dict[str, any] = {}
        self._optimization_applied = False
        self._scaler: Optional[GradScaler] = None

    def optimize_for_hardware(self, hardware_info: HardwareInfo) -> Dict[str, any]:
        """
        Apply GPU optimizations based on hardware capabilities.
        
        Args:
            hardware_info: Complete hardware information
            
        Returns:
            Dictionary of applied optimizations and their status
        """
        if not hardware_info.gpu:
            return {"error": "No GPU detected"}
        
        optimizations = {}
        
        # Set CUDA device
        if self.config.enable_cuda and hardware_info.gpu.cuda_available:
            optimizations["cuda"] = self._enable_cuda_optimizations(hardware_info.gpu)
        
        # Apply TensorRT optimizations
        if self.config.enable_tensorrt and hardware_info.gpu.tensorrt_available:
            optimizations["tensorrt"] = self._enable_tensorrt_optimizations(hardware_info.gpu)
        
        # Apply mixed precision optimizations
        if self.config.enable_mixed_precision and self._supports_mixed_precision(hardware_info.gpu):
            optimizations["mixed_precision"] = self._enable_mixed_precision()
        
        # Apply memory optimizations
        if self.config.enable_memory_optimization:
            optimizations["memory"] = self._enable_memory_optimizations(hardware_info.gpu)
        
        # Apply cuDNN optimizations
        if self.config.enable_cudnn_benchmark:
            optimizations["cudnn"] = self._enable_cudnn_optimizations()
        
        # Apply Tensor Core optimizations
        if self.config.enable_tensor_core_usage and self._supports_tensor_cores(hardware_info.gpu):
            optimizations["tensor_cores"] = self._enable_tensor_core_usage()
        
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
        Optimize a PyTorch model for GPU inference.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape for optimization
            
        Returns:
            Optimized model
        """
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to GPU
        model = model.to(self._device)
        
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
        Optimize batch inference with GPU-specific optimizations.
        
        Args:
            model: Optimized PyTorch model
            inputs: Input tensor batch
            batch_size: Optional batch size override
            
        Returns:
            Optimized inference results
        """
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move inputs to GPU
        inputs = inputs.to(self._device)
        
        if batch_size is None and self.config.enable_dynamic_batching:
            batch_size = self._get_optimal_batch_size(inputs.shape[0])
        
        # Use torch.no_grad() for inference
        with torch.no_grad():
            if self.config.enable_mixed_precision and self._scaler is not None:
                # Use mixed precision inference
                with autocast():
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
            else:
                # Standard precision inference
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

    def _enable_cuda_optimizations(self, gpu_info: GPUInfo) -> Dict[str, any]:
        """Enable CUDA optimizations for GPU inference."""
        optimizations = {}
        
        # Set CUDA device
        self._device = torch.device("cuda")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory management
        torch.cuda.empty_cache()
        
        optimizations["device"] = str(self._device)
        optimizations["cudnn_enabled"] = True
        optimizations["cudnn_benchmark"] = True
        
        return optimizations

    def _enable_tensorrt_optimizations(self, gpu_info: GPUInfo) -> Dict[str, any]:
        """Enable TensorRT optimizations."""
        optimizations = {}
        
        try:
            # Check if TensorRT is available
            import tensorrt as trt
            
            # Set TensorRT environment variables
            os.environ["TRT_LOGGER_VERBOSITY"] = "1"
            
            optimizations["tensorrt_available"] = True
            optimizations["tensorrt_version"] = trt.__version__
            
        except ImportError:
            optimizations["tensorrt_available"] = False
            optimizations["error"] = "TensorRT not available"
        
        return optimizations

    def _enable_mixed_precision(self) -> Dict[str, any]:
        """Enable mixed precision optimizations."""
        optimizations = {}
        
        # Initialize scaler for mixed precision
        self._scaler = GradScaler()
        
        # Enable automatic mixed precision
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        optimizations["mixed_precision_enabled"] = True
        optimizations["tf32_enabled"] = True
        optimizations["scaler_initialized"] = True
        
        return optimizations

    def _enable_memory_optimizations(self, gpu_info: GPUInfo) -> Dict[str, any]:
        """Enable GPU memory optimizations."""
        optimizations = {}
        
        # Set memory fraction
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        optimizations["memory_fraction"] = self.config.memory_fraction
        optimizations["memory_cleared"] = True
        
        return optimizations

    def _enable_cudnn_optimizations(self) -> Dict[str, any]:
        """Enable cuDNN optimizations."""
        optimizations = {}
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        optimizations["cudnn_enabled"] = True
        optimizations["cudnn_benchmark"] = True
        
        return optimizations

    def _enable_tensor_core_usage(self) -> Dict[str, any]:
        """Enable Tensor Core usage for optimized operations."""
        optimizations = {}
        
        # Enable Tensor Core optimizations
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        optimizations["tensor_cores_enabled"] = True
        optimizations["tf32_enabled"] = True
        
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
        example_input = torch.randn(1, *input_shape).to(self._device)
        
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
            # Use half precision for GPU
            return model.half()
        else:
            return model

    def _supports_mixed_precision(self, gpu_info: GPUInfo) -> bool:
        """Check if GPU supports mixed precision."""
        try:
            compute_capability = float(gpu_info.compute_capability)
            return compute_capability >= 7.0
        except (ValueError, TypeError):
            return False

    def _supports_tensor_cores(self, gpu_info: GPUInfo) -> bool:
        """Check if GPU supports Tensor Cores."""
        try:
            compute_capability = float(gpu_info.compute_capability)
            return compute_capability >= 7.0
        except (ValueError, TypeError):
            return False

    def _get_optimal_batch_size(self, current_batch_size: int) -> int:
        """Get optimal batch size for GPU inference."""
        # GPU-optimized batch sizes
        if current_batch_size <= 8:
            return 8
        elif current_batch_size <= 16:
            return 16
        elif current_batch_size <= 32:
            return 32
        elif current_batch_size <= 64:
            return 64
        else:
            return min(current_batch_size, self.config.max_batch_size)

    def get_performance_metrics(self) -> Dict[str, any]:
        """Get performance metrics for applied optimizations."""
        if not self._optimization_applied:
            return {"status": "No optimizations applied"}
        
        metrics = {
            "device": str(self._device) if self._device else "None",
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "jit_compiled_models": len(self._jit_compiled_models),
            "tensorrt_engines": len(self._tensorrt_engines),
            "mixed_precision_enabled": self._scaler is not None,
        }
        
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        return metrics

    def cleanup(self):
        """Cleanup resources and reset optimizations."""
        if self._scaler:
            del self._scaler
            self._scaler = None
        
        self._jit_compiled_models.clear()
        self._tensorrt_engines.clear()
        self._optimization_applied = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
