# Performance Tuning Guide

This guide provides comprehensive performance tuning recommendations and best practices for the CarlaRL Policy-as-a-Service system across different hardware configurations.

## Table of Contents

- [Overview](#overview)
- [Hardware-Specific Optimizations](#hardware-specific-optimizations)
- [CPU Optimizations](#cpu-optimizations)
- [GPU Optimizations](#gpu-optimizations)
- [Memory Optimizations](#memory-optimizations)
- [Network Optimizations](#network-optimizations)
- [Application-Level Optimizations](#application-level-optimizations)
- [Monitoring and Profiling](#monitoring-and-profiling)
- [Benchmarking](#benchmarking)
- [Best Practices](#best-practices)

## Overview

Performance tuning involves optimizing various system components to achieve the best possible performance for your specific hardware and workload. This guide covers optimizations for:

- **Intel/AMD CPUs**: AVX, SSE, Intel MKL optimizations
- **NVIDIA GPUs**: CUDA, TensorRT optimizations
- **Mac M4 Max**: MPS, NEON, Neural Engine optimizations
- **Memory**: Pinning, pooling, allocation strategies
- **Network**: Connection pooling, keep-alive, compression
- **Application**: Batch processing, caching, model optimization

## Hardware-Specific Optimizations

### Intel/AMD CPU Optimizations

#### AVX/SSE Instructions

```bash
# Enable AVX/SSE optimizations
export ENABLE_AVX=true
export ENABLE_SSE=true
export ENABLE_AVX2=true
export ENABLE_AVX512=true

# Set optimal thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

#### Intel MKL Optimization

```bash
# Enable Intel MKL
export ENABLE_MKL=true
export MKL_THREADING_LAYER=GNU
export MKL_INTERFACE_LAYER=LP64

# Set MKL-specific optimizations
export MKL_DYNAMIC=false
export MKL_NUM_THREADS=8
```

#### Configuration

```yaml
# config/intel-optimized.yaml
optimization:
  enable_avx: true
  enable_sse: true
  enable_avx2: true
  enable_avx512: true
  enable_mkl: true
  cpu_threads: 8
  memory_pinning: true
  cache_size: 1000
```

### NVIDIA GPU Optimizations

#### CUDA Optimization

```bash
# Enable CUDA optimizations
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Set CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_MEMORY_POOL_DISABLE=0
```

#### TensorRT Optimization

```bash
# Enable TensorRT
export ENABLE_TENSORRT=true
export TENSORRT_CACHE_PATH=/tmp/tensorrt_cache
export TENSORRT_VERBOSE=1
```

#### Configuration

```yaml
# config/nvidia-optimized.yaml
model:
  device: "cuda"
  precision: "float16"
  optimize: true
  use_tensorrt: true
  batch_size: 16
  max_batch_size: 64

optimization:
  enable_cuda: true
  enable_tensorrt: true
  gpu_memory_fraction: 0.8
  memory_pinning: true
  cache_size: 2000
```

### Mac M4 Max Optimizations

#### Metal Performance Shaders (MPS)

```bash
# Enable MPS optimizations
export MODEL_DEVICE=mps
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export MPS_DEVICE_MEMORY_LIMIT=24GB

# Enable MPS-specific optimizations
export ENABLE_MPS=true
export MPS_ALLOW_TF32=true
export MPS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION=true
```

#### NEON SIMD Optimization

```bash
# Enable NEON SIMD
export ENABLE_NEON=true
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
```

#### Neural Engine Integration

```bash
# Enable Neural Engine
export ENABLE_NEURAL_ENGINE=true
export NEURAL_ENGINE_CACHE_PATH=/tmp/ne_cache
```

#### Configuration

```yaml
# config/mac-m4-max-optimized.yaml
model:
  device: "mps"
  precision: "float16"
  optimize: true
  use_mps: true
  use_neon: true
  use_neural_engine: true
  batch_size: 8
  max_batch_size: 32

optimization:
  enable_mps: true
  enable_neon: true
  enable_neural_engine: true
  memory_efficiency: "high"
  cache_size: 2000
  unified_memory_limit: "24GB"
```

## CPU Optimizations

### 1. Thread Configuration

```bash
# Optimal thread count based on CPU cores
export OMP_NUM_THREADS=8          # OpenMP threads
export MKL_NUM_THREADS=8          # Intel MKL threads
export OPENBLAS_NUM_THREADS=8     # OpenBLAS threads
export NUMEXPR_NUM_THREADS=8      # NumExpr threads
export VECLIB_MAXIMUM_THREADS=8   # VecLib threads (Mac)
```

### 2. CPU Governor

```bash
# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Or use cpupower (if available)
sudo cpupower frequency-set -g performance
```

### 3. CPU Affinity

```python
# Set CPU affinity for worker processes
import os
import psutil

def set_cpu_affinity(pid, cpu_list):
    """Set CPU affinity for a process."""
    process = psutil.Process(pid)
    process.cpu_affinity(cpu_list)

# Example: bind to cores 0-7
set_cpu_affinity(os.getpid(), list(range(8)))
```

### 4. NUMA Optimization

```bash
# Check NUMA topology
numactl --hardware

# Bind to specific NUMA node
numactl --cpunodebind=0 --membind=0 python src/server.py
```

## GPU Optimizations

### 1. CUDA Memory Management

```python
# Optimize CUDA memory allocation
import torch

# Set memory allocation strategy
torch.cuda.set_per_process_memory_fraction(0.8)

# Enable memory caching
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Clear cache when needed
torch.cuda.empty_cache()
```

### 2. TensorRT Optimization

```python
# TensorRT model optimization
import torch
import tensorrt as trt

def optimize_model_with_tensorrt(model, input_shape):
    """Optimize model with TensorRT."""
    # Convert to TensorRT
    trt_model = torch.jit.script(model)
    
    # Set optimization profile
    profile = trt.Profile()
    profile.add_optimization_profile(
        input_shape, input_shape, input_shape
    )
    
    return trt_model
```

### 3. MPS Optimization (Mac M4 Max)

```python
# MPS optimization for Mac M4 Max
import torch

# Enable MPS optimizations
torch.backends.mps.allow_tf32 = True
torch.backends.mps.allow_fp16_reduced_precision_reduction = True

# Set memory management
torch.mps.set_per_process_memory_fraction(0.8)

# Clear MPS cache when needed
torch.mps.empty_cache()
```

## Memory Optimizations

### 1. Memory Pinning

```python
# Enable memory pinning for better performance
import torch

# Pin memory for faster GPU transfers
pinned_memory = torch.cuda.is_available()

# Use pinned memory in DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    pin_memory=pinned_memory,
    num_workers=4
)
```

### 2. Memory Pooling

```python
# Implement memory pooling
class MemoryPool:
    def __init__(self, size, dtype=torch.float32):
        self.pool = torch.empty(size, dtype=dtype)
        self.available = list(range(size))
    
    def get_tensor(self, shape):
        if not self.available:
            raise RuntimeError("Memory pool exhausted")
        
        idx = self.available.pop()
        return self.pool[idx:idx+shape[0]]
    
    def return_tensor(self, tensor):
        # Return tensor to pool
        pass
```

### 3. Memory Pre-allocation

```python
# Pre-allocate tensors for better performance
class TensorCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_tensor(self, shape, dtype):
        key = (shape, dtype)
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                # Remove oldest tensor
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = torch.empty(shape, dtype=dtype)
        
        return self.cache[key]
```

## Network Optimizations

### 1. Connection Pooling

```python
# Implement connection pooling
import asyncio
import aiohttp

class ConnectionPool:
    def __init__(self, max_connections=100):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
```

### 2. Keep-Alive Configuration

```yaml
# Nginx configuration for keep-alive
upstream carla_rl_backend {
    server 127.0.0.1:8080;
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

server {
    location / {
        proxy_pass http://carla_rl_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Compression

```python
# Enable compression for API responses
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## Application-Level Optimizations

### 1. Batch Processing

```python
# Optimize batch processing
class BatchProcessor:
    def __init__(self, max_batch_size=32, timeout=0.1):
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.batch_queue = []
    
    async def process_batch(self, requests):
        """Process requests in batches for better throughput."""
        if len(requests) >= self.max_batch_size:
            return await self._process_batch(requests)
        
        # Wait for more requests or timeout
        await asyncio.sleep(self.timeout)
        if len(requests) > 0:
            return await self._process_batch(requests)
        
        return []
```

### 2. Caching

```python
# Implement intelligent caching
import redis
import json
import hashlib

class ModelCache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get_cache_key(self, input_data):
        """Generate cache key from input data."""
        data_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get(self, input_data):
        """Get cached result."""
        key = self.get_cache_key(input_data)
        result = await self.redis.get(key)
        return json.loads(result) if result else None
    
    async def set(self, input_data, result):
        """Cache result."""
        key = self.get_cache_key(input_data)
        await self.redis.setex(key, self.ttl, json.dumps(result))
```

### 3. Model Optimization

```python
# Model optimization techniques
import torch
import torch.jit

def optimize_model(model, input_shape):
    """Optimize model for inference."""
    # Set to evaluation mode
    model.eval()
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # JIT compile
    if torch.cuda.is_available():
        model = torch.jit.script(model)
    
    # Optimize for inference
    model = torch.optimize_for_inference(model)
    
    return model
```

## Monitoring and Profiling

### 1. Performance Profiling

```python
# CPU profiling
import cProfile
import pstats

def profile_function(func):
    """Profile a function for performance analysis."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

### 2. Memory Profiling

```python
# Memory profiling
import tracemalloc
import psutil

def profile_memory():
    """Profile memory usage."""
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
```

### 3. GPU Profiling

```python
# GPU profiling
import torch

def profile_gpu():
    """Profile GPU usage."""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        print(f"GPU utilization: {torch.cuda.utilization()}%")
```

## Benchmarking

### 1. Latency Benchmarking

```python
# Latency benchmarking
import time
import statistics

def benchmark_latency(func, iterations=1000):
    """Benchmark function latency."""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'p95': sorted(times)[int(0.95 * len(times))],
        'p99': sorted(times)[int(0.99 * len(times))],
        'min': min(times),
        'max': max(times)
    }
```

### 2. Throughput Benchmarking

```python
# Throughput benchmarking
import asyncio
import time

async def benchmark_throughput(func, duration=60, concurrency=100):
    """Benchmark function throughput."""
    start_time = time.time()
    end_time = start_time + duration
    
    async def worker():
        count = 0
        while time.time() < end_time:
            await func()
            count += 1
        return count
    
    # Run concurrent workers
    tasks = [worker() for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    
    total_requests = sum(results)
    throughput = total_requests / duration
    
    return {
        'total_requests': total_requests,
        'duration': duration,
        'throughput': throughput,
        'concurrency': concurrency
    }
```

### 3. Memory Benchmarking

```python
# Memory benchmarking
import psutil
import gc

def benchmark_memory(func, iterations=1000):
    """Benchmark memory usage."""
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run function
    for _ in range(iterations):
        func()
    
    # Final memory
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'baseline_memory': baseline_memory,
        'final_memory': final_memory,
        'memory_growth': final_memory - baseline_memory,
        'memory_per_iteration': (final_memory - baseline_memory) / iterations
    }
```

## Best Practices

### 1. Hardware Selection

- **CPU**: Choose CPUs with AVX2/AVX-512 support for Intel, or high core count for AMD
- **GPU**: Use NVIDIA RTX 3080+ for CUDA, or Mac M4 Max for MPS
- **Memory**: Use high-speed DDR4/DDR5 memory with sufficient capacity
- **Storage**: Use NVMe SSD for model storage and caching

### 2. Configuration Tuning

- **Workers**: Set worker count to CPU core count
- **Batch Size**: Optimize batch size for your hardware (4-16 for most cases)
- **Memory**: Use appropriate memory limits and pinning
- **Caching**: Enable intelligent caching for repeated requests

### 3. Monitoring

- **Metrics**: Monitor latency, throughput, and error rates
- **Resources**: Track CPU, memory, and GPU usage
- **Alerts**: Set up alerts for performance degradation
- **Logging**: Use structured logging for better analysis

### 4. Testing

- **Benchmarks**: Run regular performance benchmarks
- **Load Testing**: Test under realistic load conditions
- **Regression Testing**: Ensure optimizations don't break functionality
- **A/B Testing**: Compare different optimization strategies

### 5. Maintenance

- **Updates**: Keep dependencies and drivers updated
- **Monitoring**: Continuously monitor performance metrics
- **Optimization**: Regularly review and optimize configuration
- **Documentation**: Document optimization strategies and results

## Next Steps

- [Mac M4 Max Deployment Guide](deployment-guides/mac-m4-max-deployment.md)
- [Monitoring Setup Guide](monitoring/monitoring-setup.md)
- [Configuration Reference](configuration-reference.md)
- [Troubleshooting Guide](troubleshooting/troubleshooting.md)
