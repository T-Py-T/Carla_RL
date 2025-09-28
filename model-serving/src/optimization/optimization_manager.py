"""
Automatic hardware detection and optimization selection.

This module provides intelligent optimization selection based on
hardware capabilities and performance requirements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .cpu_optimizer import CPUOptimizer, CPUOptimizationConfig
from .gpu_optimizer import GPUOptimizer, GPUOptimizationConfig
from .memory_optimizer import MemoryOptimizer, MemoryOptimizationConfig
from ..benchmarking.hardware_detector import HardwareDetector, HardwareInfo


@dataclass
class OptimizationProfile:
    """Optimization profile for specific hardware configurations."""

    name: str
    description: str
    cpu_config: CPUOptimizationConfig
    gpu_config: GPUOptimizationConfig
    memory_config: MemoryOptimizationConfig
    target_latency_ms: float
    target_throughput_rps: int
    memory_limit_gb: float
    hardware_requirements: Dict[str, any]


class OptimizationManager:
    """
    Automatic hardware detection and optimization selection manager.

    Provides intelligent optimization selection based on hardware
    capabilities and performance requirements.
    """

    def __init__(self):
        """Initialize optimization manager."""
        self._hardware_detector = HardwareDetector()
        self._cpu_optimizer: Optional[CPUOptimizer] = None
        self._gpu_optimizer: Optional[GPUOptimizer] = None
        self._memory_optimizer: Optional[MemoryOptimizer] = None
        self._hardware_info: Optional[HardwareInfo] = None
        self._optimization_applied = False
        self._optimization_profiles = self._create_optimization_profiles()

    def auto_optimize(
        self, 
        target_latency_ms: float = 10.0,
        target_throughput_rps: int = 1000,
        memory_limit_gb: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Automatically detect hardware and apply optimal optimizations.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            target_throughput_rps: Target throughput in requests per second
            memory_limit_gb: Optional memory limit in GB
            
        Returns:
            Dictionary of applied optimizations and performance metrics
        """
        # Detect hardware
        self._hardware_info = self._hardware_detector.get_hardware_info()
        
        # Select optimal profile
        profile = self._select_optimal_profile(
            target_latency_ms, target_throughput_rps, memory_limit_gb
        )
        
        # Apply optimizations
        optimizations = self._apply_optimizations(profile)
        
        self._optimization_applied = True
        return optimizations

    def optimize_model(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        target_latency_ms: float = 10.0,
        target_throughput_rps: int = 1000
    ) -> nn.Module:
        """
        Optimize a model for the detected hardware.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape
            target_latency_ms: Target latency in milliseconds
            target_throughput_rps: Target throughput in requests per second
            
        Returns:
            Optimized model
        """
        if not self._optimization_applied:
            self.auto_optimize(target_latency_ms, target_throughput_rps)
        
        # Optimize model with CPU optimizations
        if self._cpu_optimizer:
            model = self._cpu_optimizer.optimize_model(model, input_shape)
        
        # Optimize model with GPU optimizations
        if self._gpu_optimizer:
            model = self._gpu_optimizer.optimize_model(model, input_shape)
        
        # Optimize model memory usage
        if self._memory_optimizer:
            model = self._memory_optimizer.optimize_model_memory(model)
        
        return model

    def optimize_inference(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Optimize inference with hardware-specific optimizations.
        
        Args:
            model: Optimized PyTorch model
            inputs: Input tensor batch
            batch_size: Optional batch size override
            
        Returns:
            Optimized inference results
        """
        if not self._optimization_applied:
            raise RuntimeError("Optimizations not applied. Call auto_optimize() first.")
        
        # Use GPU optimization if available
        if self._gpu_optimizer and self._should_use_gpu():
            return self._gpu_optimizer.optimize_inference_batch(model, inputs, batch_size)
        
        # Use CPU optimization
        if self._cpu_optimizer:
            return self._cpu_optimizer.optimize_inference_batch(model, inputs, batch_size)
        
        # Fallback to standard inference
        with torch.no_grad():
            return model(inputs)

    def get_performance_metrics(self) -> Dict[str, any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "optimization_applied": self._optimization_applied,
            "hardware_info": self._hardware_info.__dict__ if self._hardware_info else None,
        }
        
        if self._cpu_optimizer:
            metrics["cpu_optimizations"] = self._cpu_optimizer.get_performance_metrics()
        
        if self._gpu_optimizer:
            metrics["gpu_optimizations"] = self._gpu_optimizer.get_performance_metrics()
        
        if self._memory_optimizer:
            metrics["memory_optimizations"] = self._memory_optimizer.get_memory_stats()
        
        return metrics

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on hardware."""
        if not self._hardware_info:
            return ["Run auto_optimize() first to get recommendations"]
        
        recommendations = []
        
        # CPU recommendations
        if self._hardware_info.cpu.avx_support:
            recommendations.append("Enable AVX optimizations for better CPU performance")
        if self._hardware_info.cpu.intel_mkl_available:
            recommendations.append("Use Intel MKL for optimized linear algebra operations")
        if self._hardware_info.cpu.cores >= 8:
            recommendations.append("Enable multi-threading for parallel processing")
        
        # GPU recommendations
        if self._hardware_info.gpu:
            if self._hardware_info.gpu.memory_gb >= 8:
                recommendations.append("Use GPU for inference with large batch sizes")
            if self._hardware_info.gpu.tensorrt_available:
                recommendations.append("Use TensorRT for optimized GPU inference")
            if float(self._hardware_info.gpu.compute_capability) >= 7.0:
                recommendations.append("Enable mixed precision inference on GPU")
        
        # Memory recommendations
        if self._hardware_info.memory.total_gb >= 16:
            recommendations.append("Enable memory pooling for better memory efficiency")
        if self._hardware_info.memory.total_gb >= 32:
            recommendations.append("Use larger batch sizes for better throughput")
        
        return recommendations

    def _create_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Create predefined optimization profiles for different hardware configurations."""
        profiles = {}
        
        # High-performance CPU profile
        profiles["high_performance_cpu"] = OptimizationProfile(
            name="High Performance CPU",
            description="Optimized for high-end CPU with AVX and Intel MKL support",
            cpu_config=CPUOptimizationConfig(
                enable_avx=True,
                enable_sse=True,
                enable_intel_mkl=True,
                enable_multi_threading=True,
                max_threads=8,
                enable_jit_optimization=True,
                enable_memory_pinning=True,
                batch_size_optimization=True,
                enable_quantization=False
            ),
            gpu_config=GPUOptimizationConfig(enable_cuda=False),
            memory_config=MemoryOptimizationConfig(
                enable_memory_pooling=True,
                pool_size_mb=2048,
                max_pool_entries=200,
                enable_memory_pinning=True,
                enable_garbage_collection=True
            ),
            target_latency_ms=8.0,
            target_throughput_rps=2000,
            memory_limit_gb=16.0,
            hardware_requirements={
                "cpu_cores": 8,
                "cpu_avx": True,
                "cpu_mkl": True,
                "memory_gb": 16
            }
        )
        
        # GPU-accelerated profile
        profiles["gpu_accelerated"] = OptimizationProfile(
            name="GPU Accelerated",
            description="Optimized for GPU inference with CUDA and TensorRT",
            cpu_config=CPUOptimizationConfig(
                enable_avx=True,
                enable_sse=True,
                enable_intel_mkl=True,
                enable_multi_threading=True,
                max_threads=4,
                enable_jit_optimization=True,
                enable_memory_pinning=True,
                batch_size_optimization=True,
                enable_quantization=False
            ),
            gpu_config=GPUOptimizationConfig(
                enable_cuda=True,
                enable_tensorrt=True,
                enable_mixed_precision=True,
                enable_memory_optimization=True,
                enable_cudnn_benchmark=True,
                enable_tensor_core_usage=True,
                memory_fraction=0.8,
                enable_jit_optimization=True,
                enable_quantization=False,
                enable_dynamic_batching=True,
                max_batch_size=128
            ),
            memory_config=MemoryOptimizationConfig(
                enable_memory_pooling=True,
                pool_size_mb=4096,
                max_pool_entries=500,
                enable_memory_pinning=True,
                enable_garbage_collection=True,
                enable_memory_prefetching=True
            ),
            target_latency_ms=5.0,
            target_throughput_rps=5000,
            memory_limit_gb=32.0,
            hardware_requirements={
                "gpu_memory_gb": 8,
                "gpu_cuda": True,
                "gpu_tensorrt": True,
                "memory_gb": 32
            }
        )
        
        # Memory-constrained profile
        profiles["memory_constrained"] = OptimizationProfile(
            name="Memory Constrained",
            description="Optimized for systems with limited memory",
            cpu_config=CPUOptimizationConfig(
                enable_avx=True,
                enable_sse=True,
                enable_intel_mkl=True,
                enable_multi_threading=True,
                max_threads=4,
                enable_jit_optimization=True,
                enable_memory_pinning=False,
                batch_size_optimization=True,
                enable_quantization=True,
                quantization_bits=8
            ),
            gpu_config=GPUOptimizationConfig(
                enable_cuda=True,
                enable_tensorrt=False,
                enable_mixed_precision=True,
                enable_memory_optimization=True,
                memory_fraction=0.5,
                enable_quantization=True,
                quantization_bits=8,
                max_batch_size=32
            ),
            memory_config=MemoryOptimizationConfig(
                enable_memory_pooling=True,
                pool_size_mb=512,
                max_pool_entries=50,
                enable_memory_pinning=False,
                enable_garbage_collection=True,
                enable_memory_compression=True,
                compression_ratio=0.7
            ),
            target_latency_ms=15.0,
            target_throughput_rps=500,
            memory_limit_gb=8.0,
            hardware_requirements={
                "memory_gb": 8
            }
        )
        
        # Balanced profile
        profiles["balanced"] = OptimizationProfile(
            name="Balanced",
            description="Balanced optimization for general-purpose hardware",
            cpu_config=CPUOptimizationConfig(
                enable_avx=True,
                enable_sse=True,
                enable_intel_mkl=True,
                enable_multi_threading=True,
                max_threads=6,
                enable_jit_optimization=True,
                enable_memory_pinning=True,
                batch_size_optimization=True,
                enable_quantization=False
            ),
            gpu_config=GPUOptimizationConfig(
                enable_cuda=True,
                enable_tensorrt=False,
                enable_mixed_precision=True,
                enable_memory_optimization=True,
                memory_fraction=0.6,
                enable_jit_optimization=True,
                enable_quantization=False,
                max_batch_size=64
            ),
            memory_config=MemoryOptimizationConfig(
                enable_memory_pooling=True,
                pool_size_mb=1024,
                max_pool_entries=100,
                enable_memory_pinning=True,
                enable_garbage_collection=True,
                enable_memory_prefetching=True
            ),
            target_latency_ms=10.0,
            target_throughput_rps=1000,
            memory_limit_gb=16.0,
            hardware_requirements={
                "memory_gb": 16
            }
        )
        
        return profiles

    def _select_optimal_profile(
        self, 
        target_latency_ms: float,
        target_throughput_rps: int,
        memory_limit_gb: Optional[float]
    ) -> OptimizationProfile:
        """Select the optimal optimization profile based on hardware and requirements."""
        if not self._hardware_info:
            raise RuntimeError("Hardware information not available")
        
        # Filter profiles based on hardware capabilities
        suitable_profiles = []
        
        for profile_name, profile in self._optimization_profiles.items():
            if self._is_profile_suitable(profile, memory_limit_gb):
                suitable_profiles.append(profile)
        
        if not suitable_profiles:
            # Fallback to balanced profile
            return self._optimization_profiles["balanced"]
        
        # Select profile based on performance requirements
        best_profile = suitable_profiles[0]
        best_score = self._calculate_profile_score(
            best_profile, target_latency_ms, target_throughput_rps
        )
        
        for profile in suitable_profiles[1:]:
            score = self._calculate_profile_score(
                profile, target_latency_ms, target_throughput_rps
            )
            if score > best_score:
                best_score = score
                best_profile = profile
        
        return best_profile

    def _is_profile_suitable(
        self, 
        profile: OptimizationProfile, 
        memory_limit_gb: Optional[float]
    ) -> bool:
        """Check if a profile is suitable for the current hardware."""
        # Check memory requirements
        if memory_limit_gb and memory_limit_gb < profile.memory_limit_gb:
            return False
        
        if self._hardware_info.memory.total_gb < profile.memory_limit_gb:
            return False
        
        # Check CPU requirements
        cpu_req = profile.hardware_requirements
        if "cpu_cores" in cpu_req and self._hardware_info.cpu.cores < cpu_req["cpu_cores"]:
            return False
        
        if "cpu_avx" in cpu_req and cpu_req["cpu_avx"] and not self._hardware_info.cpu.avx_support:
            return False
        
        if "cpu_mkl" in cpu_req and cpu_req["cpu_mkl"] and not self._hardware_info.cpu.intel_mkl_available:
            return False
        
        # Check GPU requirements
        if "gpu_memory_gb" in cpu_req:
            if not self._hardware_info.gpu or self._hardware_info.gpu.memory_gb < cpu_req["gpu_memory_gb"]:
                return False
        
        if "gpu_cuda" in cpu_req and cpu_req["gpu_cuda"] and not self._hardware_info.gpu:
            return False
        
        if "gpu_tensorrt" in cpu_req and cpu_req["gpu_tensorrt"] and not self._hardware_info.gpu.tensorrt_available:
            return False
        
        return True

    def _calculate_profile_score(
        self, 
        profile: OptimizationProfile, 
        target_latency_ms: float, 
        target_throughput_rps: int
    ) -> float:
        """Calculate a score for a profile based on performance requirements."""
        # Latency score (lower is better)
        latency_score = max(0, 1.0 - abs(profile.target_latency_ms - target_latency_ms) / target_latency_ms)
        
        # Throughput score (higher is better)
        throughput_score = min(1.0, profile.target_throughput_rps / target_throughput_rps)
        
        # Combined score
        return (latency_score + throughput_score) / 2.0

    def _apply_optimizations(self, profile: OptimizationProfile) -> Dict[str, any]:
        """Apply optimizations based on the selected profile."""
        optimizations = {}
        
        # Initialize optimizers
        self._cpu_optimizer = CPUOptimizer(profile.cpu_config)
        self._gpu_optimizer = GPUOptimizer(profile.gpu_config)
        self._memory_optimizer = MemoryOptimizer(profile.memory_config)
        
        # Apply CPU optimizations
        if self._hardware_info:
            cpu_optimizations = self._cpu_optimizer.optimize_for_hardware(self._hardware_info)
            optimizations["cpu"] = cpu_optimizations
        
        # Apply GPU optimizations
        if self._hardware_info and self._hardware_info.gpu:
            gpu_optimizations = self._gpu_optimizer.optimize_for_hardware(self._hardware_info)
            optimizations["gpu"] = gpu_optimizations
        
        # Apply memory optimizations
        if self._hardware_info:
            memory_optimizations = self._memory_optimizer.optimize_for_hardware(self._hardware_info)
            optimizations["memory"] = memory_optimizations
        
        optimizations["profile"] = {
            "name": profile.name,
            "description": profile.description,
            "target_latency_ms": profile.target_latency_ms,
            "target_throughput_rps": profile.target_throughput_rps,
        }
        
        return optimizations

    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used for inference."""
        if not self._hardware_info or not self._hardware_info.gpu:
            return False
        
        # Use GPU if it has sufficient memory and compute capability
        return (
            self._hardware_info.gpu.memory_gb >= 4 and 
            float(self._hardware_info.gpu.compute_capability) >= 6.0
        )

    def cleanup(self):
        """Cleanup all optimizers and resources."""
        if self._cpu_optimizer:
            self._cpu_optimizer.cleanup()
            self._cpu_optimizer = None
        
        if self._gpu_optimizer:
            self._gpu_optimizer.cleanup()
            self._gpu_optimizer = None
        
        if self._memory_optimizer:
            self._memory_optimizer.cleanup()
            self._memory_optimizer = None
        
        self._optimization_applied = False

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
