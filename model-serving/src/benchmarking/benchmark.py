"""
Core benchmarking engine for latency and throughput testing.

This module provides the main benchmarking infrastructure with configurable
test scenarios for validating performance requirements.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import psutil
import torch


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark test scenarios."""
    
    # Test parameters
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    concurrent_requests: int = 1
    
    # Performance thresholds
    p50_threshold_ms: float = 10.0
    p95_threshold_ms: float = 20.0
    p99_threshold_ms: float = 50.0
    throughput_threshold_rps: float = 1000.0
    
    # Hardware constraints
    max_memory_usage_mb: float = 1024.0
    max_cpu_usage_percent: float = 80.0
    
    # Test scenarios
    deterministic_mode: bool = True
    enable_caching: bool = True
    enable_memory_pinning: bool = True


@dataclass
class LatencyStats:
    """Statistical analysis of latency measurements."""
    
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    total_measurements: int


@dataclass
class ThroughputStats:
    """Statistical analysis of throughput measurements."""
    
    requests_per_second: float
    total_requests: int
    total_duration_s: float
    successful_requests: int
    failed_requests: int
    error_rate: float


@dataclass
class MemoryStats:
    """Memory usage statistics during benchmarking."""
    
    peak_memory_mb: float
    average_memory_mb: float
    memory_growth_mb: float
    memory_efficiency: float  # requests per MB


@dataclass
class BenchmarkResult:
    """Complete benchmark test results."""
    
    # Test configuration
    config: BenchmarkConfig
    hardware_info: Dict[str, Any]
    
    # Performance metrics
    latency_stats: LatencyStats
    throughput_stats: ThroughputStats
    memory_stats: MemoryStats
    
    # Validation results
    p50_requirement_met: bool
    p95_requirement_met: bool
    p99_requirement_met: bool
    throughput_requirement_met: bool
    memory_requirement_met: bool
    
    # Overall result
    overall_success: bool
    test_duration_s: float
    timestamp: str


class BenchmarkEngine:
    """
    Core benchmarking engine for performance validation.
    
    Provides configurable test scenarios for validating latency, throughput,
    and memory requirements of the Policy-as-a-Service system.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark engine with configuration."""
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        
    def create_test_observations(self, batch_size: int) -> List[Dict[str, Any]]:
        """Create test observations for benchmarking."""
        observations = []
        for _ in range(batch_size):
            obs = {
                "speed": np.random.uniform(0.0, 200.0),
                "steering": np.random.uniform(-1.0, 1.0),
                "sensors": np.random.uniform(-10.0, 10.0, 3).tolist()
            }
            observations.append(obs)
        return observations
    
    def measure_latency(
        self,
        inference_func: Callable,
        observations: List[Dict[str, Any]],
        iterations: int
    ) -> LatencyStats:
        """Measure inference latency with statistical analysis."""
        latencies = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            try:
                # Perform inference
                result = inference_func(observations, self.config.deterministic_mode)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000.0
                latencies.append(latency_ms)
                
            except Exception as e:
                print(f"Warning: Inference failed during latency measurement: {e}")
                continue
        
        if not latencies:
            raise RuntimeError("No successful latency measurements recorded")
        
        # Calculate statistics
        latencies_array = np.array(latencies)
        
        return LatencyStats(
            p50_ms=float(np.percentile(latencies_array, 50)),
            p95_ms=float(np.percentile(latencies_array, 95)),
            p99_ms=float(np.percentile(latencies_array, 99)),
            mean_ms=float(np.mean(latencies_array)),
            std_ms=float(np.std(latencies_array)),
            min_ms=float(np.min(latencies_array)),
            max_ms=float(np.max(latencies_array)),
            total_measurements=len(latencies)
        )
    
    def measure_throughput(
        self,
        inference_func: Callable,
        observations: List[Dict[str, Any]],
        duration_seconds: int = 10
    ) -> ThroughputStats:
        """Measure throughput (requests per second) over time period."""
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        successful_requests = 0
        failed_requests = 0
        
        while time.perf_counter() < end_time:
            try:
                inference_func(observations, self.config.deterministic_mode)
                successful_requests += 1
            except Exception:
                failed_requests += 1
        
        total_duration = time.perf_counter() - start_time
        total_requests = successful_requests + failed_requests
        requests_per_second = total_requests / total_duration
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        
        return ThroughputStats(
            requests_per_second=requests_per_second,
            total_requests=total_requests,
            total_duration_s=total_duration,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=error_rate
        )
    
    def measure_memory_usage(
        self,
        inference_func: Callable,
        observations: List[Dict[str, Any]],
        iterations: int
    ) -> MemoryStats:
        """Measure memory usage during inference."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [baseline_memory]
        
        for _ in range(iterations):
            # Force garbage collection before measurement
            import gc
            gc.collect()
            
            # Measure memory before inference
            memory_before = process.memory_info().rss / 1024 / 1024
            
            try:
                # Perform inference
                inference_func(observations, self.config.deterministic_mode)
                
                # Measure memory after inference
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_after)
                
            except Exception as e:
                print(f"Warning: Inference failed during memory measurement: {e}")
                continue
        
        if len(memory_samples) < 2:
            raise RuntimeError("No successful memory measurements recorded")
        
        memory_array = np.array(memory_samples)
        peak_memory = np.max(memory_array)
        average_memory = np.mean(memory_array)
        memory_growth = peak_memory - baseline_memory
        
        # Calculate memory efficiency (requests per MB)
        memory_efficiency = iterations / (peak_memory - baseline_memory) if memory_growth > 0 else 0
        
        return MemoryStats(
            peak_memory_mb=float(peak_memory),
            average_memory_mb=float(average_memory),
            memory_growth_mb=float(memory_growth),
            memory_efficiency=float(memory_efficiency)
        )
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information for baseline comparison."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch_version": torch.__version__,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        }
    
    async def run_benchmark(
        self,
        inference_func: Callable,
        batch_size: int = 1
    ) -> BenchmarkResult:
        """Run complete benchmark test for given batch size."""
        print(f"Running benchmark for batch size: {batch_size}")
        
        # Create test observations
        observations = self.create_test_observations(batch_size)
        
        # Get hardware info
        hardware_info = self.get_hardware_info()
        
        # Warmup phase
        print(f"Warming up with {self.config.warmup_iterations} iterations...")
        for _ in range(self.config.warmup_iterations):
            try:
                inference_func(observations, self.config.deterministic_mode)
            except Exception as e:
                print(f"Warning: Warmup failed: {e}")
        
        # Measure latency
        print("Measuring latency...")
        latency_stats = self.measure_latency(
            inference_func, observations, self.config.measurement_iterations
        )
        
        # Measure throughput
        print("Measuring throughput...")
        throughput_stats = self.measure_throughput(
            inference_func, observations, duration_seconds=10
        )
        
        # Measure memory usage
        print("Measuring memory usage...")
        memory_stats = self.measure_memory_usage(
            inference_func, observations, self.config.measurement_iterations
        )
        
        # Validate requirements
        p50_requirement_met = latency_stats.p50_ms <= self.config.p50_threshold_ms
        p95_requirement_met = latency_stats.p95_ms <= self.config.p95_threshold_ms
        p99_requirement_met = latency_stats.p99_ms <= self.config.p99_threshold_ms
        throughput_requirement_met = throughput_stats.requests_per_second >= self.config.throughput_threshold_rps
        memory_requirement_met = memory_stats.peak_memory_mb <= self.config.max_memory_usage_mb
        
        overall_success = (
            p50_requirement_met and
            p95_requirement_met and
            p99_requirement_met and
            throughput_requirement_met and
            memory_requirement_met
        )
        
        result = BenchmarkResult(
            config=self.config,
            hardware_info=hardware_info,
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            memory_stats=memory_stats,
            p50_requirement_met=p50_requirement_met,
            p95_requirement_met=p95_requirement_met,
            p99_requirement_met=p99_requirement_met,
            throughput_requirement_met=throughput_requirement_met,
            memory_requirement_met=memory_requirement_met,
            overall_success=overall_success,
            test_duration_s=0.0,  # Will be set by caller
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
        
        self.results.append(result)
        return result
    
    def run_batch_size_optimization(
        self,
        inference_func: Callable
    ) -> List[BenchmarkResult]:
        """Run benchmarks across different batch sizes to find optimal configuration."""
        print("Running batch size optimization...")
        
        results = []
        for batch_size in self.config.batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            result = asyncio.run(self.run_benchmark(inference_func, batch_size))
            results.append(result)
            
            # Print quick summary
            print(f"  P50: {result.latency_stats.p50_ms:.2f}ms")
            print(f"  P95: {result.latency_stats.p95_ms:.2f}ms")
            print(f"  Throughput: {result.throughput_stats.requests_per_second:.1f} RPS")
            print(f"  Memory: {result.memory_stats.peak_memory_mb:.1f} MB")
            print(f"  Success: {result.overall_success}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("=" * 80)
        report.append("POLICY-AS-A-SERVICE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.overall_success)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report.append(f"SUMMARY:")
        report.append(f"  Total Tests: {total_tests}")
        report.append(f"  Successful: {successful_tests}")
        report.append(f"  Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Detailed results
        for i, result in enumerate(self.results, 1):
            batch_size = result.config.batch_sizes[i-1] if i <= len(result.config.batch_sizes) else "Unknown"
            report.append(f"TEST {i} - Batch Size: {batch_size}")
            report.append("-" * 40)
            report.append(f"  Latency (P50): {result.latency_stats.p50_ms:.2f}ms {'✓' if result.p50_requirement_met else '✗'}")
            report.append(f"  Latency (P95): {result.latency_stats.p95_ms:.2f}ms {'✓' if result.p95_requirement_met else '✗'}")
            report.append(f"  Latency (P99): {result.latency_stats.p99_ms:.2f}ms {'✓' if result.p99_requirement_met else '✗'}")
            report.append(f"  Throughput: {result.throughput_stats.requests_per_second:.1f} RPS {'✓' if result.throughput_requirement_met else '✗'}")
            report.append(f"  Memory: {result.memory_stats.peak_memory_mb:.1f} MB {'✓' if result.memory_requirement_met else '✗'}")
            report.append(f"  Overall: {'PASS' if result.overall_success else 'FAIL'}")
            report.append("")
        
        return "\n".join(report)
