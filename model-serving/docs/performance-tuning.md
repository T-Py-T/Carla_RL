# Performance Tuning Guide

This guide provides comprehensive performance tuning recommendations and best practices for the Policy-as-a-Service system across different hardware configurations.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [CPU Optimization](#cpu-optimization)
3. [GPU Optimization](#gpu-optimization)
4. [Memory Optimization](#memory-optimization)
5. [Hardware-Specific Profiles](#hardware-specific-profiles)
6. [Performance Validation](#performance-validation)
7. [Troubleshooting](#troubleshooting)

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4 cores, 2.0 GHz, AVX support
- **Memory**: 8 GB RAM
- **Storage**: 10 GB available space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **CPU**: 8+ cores, 3.0+ GHz, AVX2 support, Intel MKL
- **Memory**: 16+ GB RAM
- **GPU**: NVIDIA GPU with 8+ GB VRAM, CUDA 11.0+, TensorRT
- **Storage**: SSD with 50+ GB available space

### High-Performance Requirements

- **CPU**: 16+ cores, 3.5+ GHz, AVX2/AVX-512 support, Intel MKL
- **Memory**: 32+ GB RAM
- **GPU**: NVIDIA RTX 3080+ or A100, 16+ GB VRAM, TensorRT
- **Storage**: NVMe SSD with 100+ GB available space

## CPU Optimization

### AVX and SSE Optimizations

AVX (Advanced Vector Extensions) and SSE (Streaming SIMD Extensions) provide significant performance improvements for vectorized operations.

**Enable AVX optimizations:**
```python
from src.optimization import CPUOptimizer, CPUOptimizationConfig

config = CPUOptimizationConfig(
    enable_avx=True,
    enable_sse=True,
    enable_intel_mkl=True
)

optimizer = CPUOptimizer(config)
optimizations = optimizer.optimize_for_hardware(hardware_info)
```

**Hardware requirements:**
- Intel: Sandy Bridge (2011) or newer
- AMD: Bulldozer (2011) or newer

### Intel MKL Optimization

Intel Math Kernel Library (MKL) provides highly optimized linear algebra routines.

**Check MKL availability:**
```python
import numpy as np
print("MKL available:", "mkl" in str(np.__config__))
```

**Install MKL-optimized PyTorch:**
```bash
# For CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Multi-Threading Configuration

**Optimal thread count:**
- For inference: 4-8 threads
- For training: 8-16 threads
- For data processing: All available threads

```python
config = CPUOptimizationConfig(
    enable_multi_threading=True,
    max_threads=8  # Adjust based on your CPU
)
```

### JIT Compilation

Just-In-Time (JIT) compilation can provide significant performance improvements for repeated operations.

```python
config = CPUOptimizationConfig(
    enable_jit_optimization=True
)
```

## GPU Optimization

### CUDA Setup

**Install CUDA toolkit:**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repository-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo apt-key add /var/cuda-repository-ubuntu2004-12-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# Verify installation
nvcc --version
nvidia-smi
```

**Install PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### TensorRT Optimization

TensorRT provides high-performance inference optimization for NVIDIA GPUs.

**Install TensorRT:**
```bash
# Download from NVIDIA Developer website
# Extract and install
pip install tensorrt
```

**Enable TensorRT:**
```python
from src.optimization import GPUOptimizer, GPUOptimizationConfig

config = GPUOptimizationConfig(
    enable_tensorrt=True,
    enable_cuda=True
)
```

### Mixed Precision

Mixed precision training and inference can provide 1.5-2x speedup with minimal accuracy loss.

```python
config = GPUOptimizationConfig(
    enable_mixed_precision=True,
    enable_tensor_core_usage=True
)
```

**Requirements:**
- GPU with Tensor Cores (RTX 20 series or newer)
- CUDA 10.0+
- PyTorch 1.6+

### Memory Management

**GPU memory optimization:**
```python
config = GPUOptimizationConfig(
    enable_memory_optimization=True,
    memory_fraction=0.8,  # Use 80% of GPU memory
    enable_dynamic_batching=True,
    max_batch_size=128
)
```

## Memory Optimization

### Memory Pooling

Memory pooling reduces allocation overhead and improves performance.

```python
from src.optimization import MemoryOptimizer, MemoryOptimizationConfig

config = MemoryOptimizationConfig(
    enable_memory_pooling=True,
    pool_size_mb=2048,  # 2GB pool
    max_pool_entries=200
)
```

### Memory Pinning

Memory pinning improves data transfer performance between CPU and GPU.

```python
config = MemoryOptimizationConfig(
    enable_memory_pinning=True
)
```

### Garbage Collection

Optimized garbage collection reduces memory fragmentation.

```python
config = MemoryOptimizationConfig(
    enable_garbage_collection=True
)
```

## Hardware-Specific Profiles

### High-Performance CPU Profile

For systems with high-end CPUs and no GPU:

```python
from src.optimization import OptimizationManager

manager = OptimizationManager()
optimizations = manager.auto_optimize(
    target_latency_ms=5.0,
    target_throughput_rps=2000,
    memory_limit_gb=32.0
)
```

**Characteristics:**
- AVX2/AVX-512 optimizations
- Intel MKL acceleration
- Multi-threading (8+ cores)
- Large memory pools
- JIT compilation

### GPU-Accelerated Profile

For systems with powerful GPUs:

```python
optimizations = manager.auto_optimize(
    target_latency_ms=3.0,
    target_throughput_rps=5000,
    memory_limit_gb=64.0
)
```

**Characteristics:**
- CUDA acceleration
- TensorRT optimization
- Mixed precision
- Tensor Core usage
- Large batch sizes

### Memory-Constrained Profile

For systems with limited memory:

```python
optimizations = manager.auto_optimize(
    target_latency_ms=15.0,
    target_throughput_rps=500,
    memory_limit_gb=8.0
)
```

**Characteristics:**
- Model quantization
- Memory compression
- Smaller batch sizes
- Aggressive garbage collection
- Memory pooling

### Balanced Profile

For general-purpose systems:

```python
optimizations = manager.auto_optimize(
    target_latency_ms=10.0,
    target_throughput_rps=1000,
    memory_limit_gb=16.0
)
```

**Characteristics:**
- Moderate optimizations
- CPU and GPU support
- Balanced memory usage
- Good performance/cost ratio

## Performance Validation

### Latency Testing

Test P50 latency requirements:

```python
from tests.optimization.test_performance_validation import PerformanceValidator

# Measure latency
latencies = PerformanceValidator.measure_latency(model, input_tensor, num_runs=100)
print(f"P50 latency: {latencies['median']:.2f}ms")
print(f"P95 latency: {latencies['p95']:.2f}ms")
print(f"P99 latency: {latencies['p99']:.2f}ms")

# Validate requirements
assert latencies['median'] < 10.0, "P50 latency exceeds 10ms requirement"
```

### Throughput Testing

Test throughput performance:

```python
# Measure throughput
throughput = PerformanceValidator.measure_throughput(model, input_tensor, duration_seconds=10.0)
print(f"Throughput: {throughput['throughput_rps']:.1f} RPS")

# Validate requirements
assert throughput['throughput_rps'] > 1000, "Throughput below 1000 RPS requirement"
```

### Memory Usage Testing

Test memory efficiency:

```python
# Measure memory usage
memory_usage = PerformanceValidator.measure_memory_usage(model, input_tensor)
print(f"CPU memory: {memory_usage['cpu_memory_mb']:.1f} MB")
print(f"GPU memory: {memory_usage['gpu_memory_mb']:.1f} MB")

# Validate requirements
assert memory_usage['cpu_memory_mb'] < 1000, "CPU memory usage too high"
```

### Automated Testing

Run comprehensive performance tests:

```bash
# Run all performance tests
python -m pytest tests/optimization/test_performance_validation.py -v

# Run specific test
python -m pytest tests/optimization/test_performance_validation.py::TestPerformanceValidation::test_performance_validation_high_end_hardware -v
```

## Troubleshooting

### Common Issues

#### High Latency

**Symptoms:**
- P50 latency > 10ms
- Slow inference times

**Solutions:**
1. Enable JIT compilation
2. Use GPU acceleration
3. Optimize batch size
4. Enable AVX/SSE optimizations
5. Use Intel MKL

#### Low Throughput

**Symptoms:**
- Throughput < 1000 RPS
- Poor batch processing performance

**Solutions:**
1. Increase batch size
2. Enable multi-threading
3. Use GPU acceleration
4. Optimize memory usage
5. Enable dynamic batching

#### High Memory Usage

**Symptoms:**
- Out of memory errors
- High memory consumption

**Solutions:**
1. Enable memory pooling
2. Reduce batch size
3. Use model quantization
4. Enable memory compression
5. Optimize garbage collection

#### GPU Issues

**Symptoms:**
- CUDA out of memory
- Poor GPU utilization

**Solutions:**
1. Reduce memory fraction
2. Use mixed precision
3. Enable TensorRT
4. Optimize batch size
5. Check GPU compatibility

### Performance Monitoring

**Monitor system resources:**
```bash
# CPU usage
htop

# GPU usage
nvidia-smi -l 1

# Memory usage
free -h

# Disk I/O
iostat -x 1
```

**Monitor application metrics:**
```python
# Get performance metrics
metrics = manager.get_performance_metrics()
print(json.dumps(metrics, indent=2))
```

### Debugging

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run optimization with debug info
manager.auto_optimize()
```

**Profile performance:**
```python
import cProfile

# Profile inference
cProfile.run('model(input_tensor)')
```

### Hardware Compatibility

**Check hardware compatibility:**
```bash
# Run hardware detection
python scripts/optimization_manager.py detect

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"

# Check AVX support
python -c "import cpuinfo; print(cpuinfo.get_cpu_info()['flags'])"
```

## Best Practices

### Development

1. **Start with balanced profile** for initial development
2. **Profile early and often** to identify bottlenecks
3. **Test on target hardware** before deployment
4. **Use version control** for optimization configurations
5. **Document performance requirements** clearly

### Production

1. **Use hardware-specific profiles** for optimal performance
2. **Monitor performance metrics** continuously
3. **Set up alerts** for performance degradation
4. **Regular performance testing** in CI/CD pipeline
5. **Plan for hardware scaling** as load increases

### Maintenance

1. **Update optimization profiles** with new hardware
2. **Review performance metrics** regularly
3. **Test new PyTorch versions** for compatibility
4. **Monitor hardware health** and performance
5. **Keep optimization tools updated**

## Additional Resources

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Intel MKL Documentation](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Memory Management in PyTorch](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
