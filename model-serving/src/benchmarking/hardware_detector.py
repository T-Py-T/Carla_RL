"""
Hardware detection and optimization utilities.

This module provides hardware detection capabilities and optimization
recommendations for different CPU and GPU configurations.
"""

import platform
import sys
from dataclasses import dataclass
from typing import List, Optional

import psutil
import torch


@dataclass
class CPUInfo:
    """CPU hardware information and capabilities."""

    model: str
    cores: int
    threads: int
    frequency_mhz: float
    architecture: str
    features: List[str]
    cache_size_mb: float
    avx_support: bool
    sse_support: bool
    intel_mkl_available: bool


@dataclass
class GPUInfo:
    """GPU hardware information and capabilities."""

    model: str
    memory_gb: float
    compute_capability: str
    cuda_available: bool
    tensorrt_available: bool
    driver_version: str
    cuda_version: str


@dataclass
class MemoryInfo:
    """Memory hardware information."""

    total_gb: float
    available_gb: float
    swap_gb: float
    memory_type: str  # DDR4, DDR5, etc.


@dataclass
class HardwareInfo:
    """Complete hardware information for optimization."""

    cpu: CPUInfo
    gpu: Optional[GPUInfo]
    memory: MemoryInfo
    platform: str
    python_version: str
    torch_version: str
    optimization_recommendations: List[str]


class HardwareDetector:
    """
    Hardware detection and optimization recommendation engine.

    Detects CPU, GPU, and memory capabilities and provides
    optimization recommendations for the Policy-as-a-Service system.
    """

    def __init__(self):
        """Initialize hardware detector."""
        self._cpu_info: Optional[CPUInfo] = None
        self._gpu_info: Optional[GPUInfo] = None
        self._memory_info: Optional[MemoryInfo] = None

    def detect_cpu(self) -> CPUInfo:
        """Detect CPU information and capabilities."""
        if self._cpu_info is not None:
            return self._cpu_info

        # Get basic CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)

        # Get CPU frequency
        cpu_freq = psutil.cpu_freq()
        frequency_mhz = cpu_freq.max if cpu_freq else 0.0

        # Get CPU model and architecture
        try:
            cpu_model = platform.processor()
            if not cpu_model or cpu_model == "unknown":
                # Try alternative method
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":")[1].strip()
                            break
        except (FileNotFoundError, OSError):
            cpu_model = "Unknown"

        # Detect CPU features
        features = self._detect_cpu_features()
        avx_support = "avx" in features or "avx2" in features
        sse_support = "sse" in features or "sse2" in features or "sse4" in features

        # Check for Intel MKL
        intel_mkl_available = self._check_intel_mkl()

        # Estimate cache size (simplified)
        cache_size_mb = self._estimate_cache_size(cpu_count)

        self._cpu_info = CPUInfo(
            model=cpu_model,
            cores=cpu_count,
            threads=cpu_count_logical,
            frequency_mhz=frequency_mhz,
            architecture=platform.machine(),
            features=features,
            cache_size_mb=cache_size_mb,
            avx_support=avx_support,
            sse_support=sse_support,
            intel_mkl_available=intel_mkl_available,
        )

        return self._cpu_info

    def detect_gpu(self) -> Optional[GPUInfo]:
        """Detect GPU information and capabilities."""
        if self._gpu_info is not None:
            return self._gpu_info

        if not torch.cuda.is_available():
            self._gpu_info = None
            return None

        try:
            # Get GPU model
            gpu_model = torch.cuda.get_device_name(0)

            # Get GPU memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024

            # Get compute capability
            compute_capability = torch.cuda.get_device_capability(0)
            compute_capability_str = f"{compute_capability[0]}.{compute_capability[1]}"

            # Get driver and CUDA versions
            driver_version = torch.version.cuda or "Unknown"
            cuda_version = torch.version.cuda or "Unknown"

            # Check for TensorRT
            tensorrt_available = self._check_tensorrt()

            self._gpu_info = GPUInfo(
                model=gpu_model,
                memory_gb=memory_gb,
                compute_capability=compute_capability_str,
                cuda_available=True,
                tensorrt_available=tensorrt_available,
                driver_version=driver_version,
                cuda_version=cuda_version,
            )

        except Exception as e:
            print(f"Warning: Failed to detect GPU information: {e}")
            self._gpu_info = None

        return self._gpu_info

    def detect_memory(self) -> MemoryInfo:
        """Detect memory information."""
        if self._memory_info is not None:
            return self._memory_info

        # Get memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        total_gb = memory.total / 1024 / 1024 / 1024
        available_gb = memory.available / 1024 / 1024 / 1024
        swap_gb = swap.total / 1024 / 1024 / 1024

        # Try to detect memory type (simplified)
        memory_type = self._detect_memory_type()

        self._memory_info = MemoryInfo(
            total_gb=total_gb, available_gb=available_gb, swap_gb=swap_gb, memory_type=memory_type
        )

        return self._memory_info

    def get_hardware_info(self) -> HardwareInfo:
        """Get complete hardware information."""
        cpu_info = self.detect_cpu()
        gpu_info = self.detect_gpu()
        memory_info = self.detect_memory()

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            cpu_info, gpu_info, memory_info
        )

        return HardwareInfo(
            cpu=cpu_info,
            gpu=gpu_info,
            memory=memory_info,
            platform=platform.platform(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch.__version__,
            optimization_recommendations=recommendations,
        )

    def _detect_cpu_features(self) -> List[str]:
        """Detect CPU features using platform-specific methods."""
        features = []

        try:
            # Try to detect CPU features on Linux
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    if "avx" in content.lower():
                        features.append("avx")
                    if "avx2" in content.lower():
                        features.append("avx2")
                    if "sse" in content.lower():
                        features.append("sse")
                    if "sse2" in content.lower():
                        features.append("sse2")
                    if "sse4" in content.lower():
                        features.append("sse4")
                    if "fma" in content.lower():
                        features.append("fma")
        except (FileNotFoundError, OSError):
            pass

        # Fallback: basic feature detection
        if not features:
            features = ["basic"]

        return features

    def _check_intel_mkl(self) -> bool:
        """Check if Intel MKL is available."""
        try:
            import numpy as np

            # Check if numpy was compiled with MKL
            return hasattr(np, "__config__") and "mkl" in str(np.__config__)
        except ImportError:
            return False

    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        import importlib.util

        return importlib.util.find_spec("tensorrt") is not None

    def _estimate_cache_size(self, cores: int) -> float:
        """Estimate CPU cache size based on core count."""
        # Simplified estimation - in practice, this would be more sophisticated
        if cores <= 2:
            return 8.0  # 8MB L3 cache
        elif cores <= 4:
            return 16.0  # 16MB L3 cache
        elif cores <= 8:
            return 32.0  # 32MB L3 cache
        else:
            return 64.0  # 64MB+ L3 cache

    def _detect_memory_type(self) -> str:
        """Detect memory type (DDR4, DDR5, etc.)."""
        try:
            # Try to detect memory type on Linux
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    content = f.read()
                    # This is a simplified detection
                    if "DDR5" in content:
                        return "DDR5"
                    elif "DDR4" in content:
                        return "DDR4"
        except (FileNotFoundError, OSError):
            pass

        # Fallback
        return "Unknown"

    def _generate_optimization_recommendations(
        self, cpu_info: CPUInfo, gpu_info: Optional[GPUInfo], memory_info: MemoryInfo
    ) -> List[str]:
        """Generate optimization recommendations based on hardware."""
        recommendations = []

        # CPU optimizations
        if cpu_info.avx_support:
            recommendations.append("Enable AVX optimizations for CPU inference")
        if cpu_info.intel_mkl_available:
            recommendations.append("Use Intel MKL for optimized linear algebra operations")
        if cpu_info.cores >= 8:
            recommendations.append("Enable multi-threading for batch processing")

        # GPU optimizations
        if gpu_info:
            if gpu_info.memory_gb >= 8:
                recommendations.append("Use GPU for inference with large batch sizes")
            if gpu_info.tensorrt_available:
                recommendations.append("Use TensorRT for optimized GPU inference")
            if gpu_info.compute_capability >= "7.0":
                recommendations.append("Enable mixed precision inference on GPU")

        # Memory optimizations
        if memory_info.total_gb >= 16:
            recommendations.append("Enable memory pinning for faster data transfer")
        if memory_info.total_gb >= 32:
            recommendations.append("Use larger batch sizes for better throughput")

        # General recommendations
        recommendations.append("Enable torch.no_grad() for inference")
        recommendations.append("Use pre-allocated tensors for batch processing")

        return recommendations

    def get_optimal_batch_size(self, hardware_info: HardwareInfo) -> int:
        """Recommend optimal batch size based on hardware."""
        if hardware_info.gpu and hardware_info.gpu.memory_gb >= 8:
            return 32
        elif hardware_info.memory.total_gb >= 16:
            return 16
        elif hardware_info.memory.total_gb >= 8:
            return 8
        else:
            return 4

    def get_optimal_thread_count(self, hardware_info: HardwareInfo) -> int:
        """Recommend optimal thread count based on hardware."""
        # Use logical cores but cap at 8 for inference
        return min(hardware_info.cpu.threads, 8)

    def should_use_gpu(self, hardware_info: HardwareInfo) -> bool:
        """Determine if GPU should be used for inference."""
        if not hardware_info.gpu:
            return False

        # Use GPU if it has sufficient memory and compute capability
        return hardware_info.gpu.memory_gb >= 4 and hardware_info.gpu.compute_capability >= "6.0"
