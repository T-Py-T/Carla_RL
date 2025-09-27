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
    
    # Additional statistical measures
    p90_ms: float = 0.0
    p99_9_ms: float = 0.0
    median_ms: float = 0.0
    variance_ms2: float = 0.0
    coefficient_of_variation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    
    def __post_init__(self):
        """Calculate additional statistics after initialization."""
        self.median_ms = self.p50_ms
        self.variance_ms2 = self.std_ms ** 2
        self.coefficient_of_variation = (self.std_ms / self.mean_ms) if self.mean_ms > 0 else 0.0


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
    
    # Additional memory metrics
    baseline_memory_mb: float
    memory_usage_samples: List[float] = field(default_factory=list)
    memory_leak_detected: bool = False
    memory_fragmentation: float = 0.0
    gc_collections: int = 0
    memory_recommendations: List[str] = field(default_factory=list)


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
        """Measure inference latency with comprehensive statistical analysis."""
        latencies = []
        failed_measurements = 0
        
        # Warmup phase to ensure consistent performance
        for _ in range(min(5, iterations // 10)):
            try:
                inference_func(observations, self.config.deterministic_mode)
            except Exception:
                pass
        
        # Main measurement phase
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                # Perform inference
                result = inference_func(observations, self.config.deterministic_mode)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000.0
                latencies.append(latency_ms)
                
            except Exception as e:
                failed_measurements += 1
                if failed_measurements > iterations * 0.1:  # More than 10% failures
                    print(f"Warning: High failure rate during latency measurement: {failed_measurements}/{i+1}")
                continue
        
        if not latencies:
            raise RuntimeError("No successful latency measurements recorded")
        
        # Calculate comprehensive statistics
        latencies_array = np.array(latencies)
        
        # Basic statistics
        mean_ms = float(np.mean(latencies_array))
        std_ms = float(np.std(latencies_array))
        min_ms = float(np.min(latencies_array))
        max_ms = float(np.max(latencies_array))
        
        # Percentile calculations with interpolation
        percentiles = np.percentile(latencies_array, [50, 90, 95, 99, 99.9], interpolation='linear')
        
        # Calculate additional statistics
        skewness = self._calculate_skewness(latencies_array)
        kurtosis = self._calculate_kurtosis(latencies_array)
        
        # Detect outliers
        outlier_count, outlier_percentage = self._detect_outliers(latencies_array)
        
        # Statistical validation
        self._validate_latency_statistics(latencies_array, mean_ms, std_ms)
        
        return LatencyStats(
            p50_ms=float(percentiles[0]),
            p95_ms=float(percentiles[2]),
            p99_ms=float(percentiles[3]),
            p90_ms=float(percentiles[1]),
            p99_9_ms=float(percentiles[4]),
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            total_measurements=len(latencies),
            skewness=skewness,
            kurtosis=kurtosis,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage
        )
    
    def _validate_latency_statistics(self, latencies: np.ndarray, mean: float, std: float) -> None:
        """Validate latency statistics for consistency and reliability."""
        # Check for reasonable latency values (not negative or extremely high)
        if np.any(latencies < 0):
            raise ValueError("Negative latency measurements detected - check timing implementation")
        
        if np.any(latencies > 10000):  # 10 seconds
            print("Warning: Extremely high latency measurements detected - check for performance issues")
        
        # Check for statistical consistency
        if std > mean * 2:  # Standard deviation more than 2x mean
            print("Warning: High variance in latency measurements - results may be unreliable")
        
        # Check for outliers using IQR method
        q1 = np.percentile(latencies, 25)
        q3 = np.percentile(latencies, 75)
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        outliers = latencies[(latencies < q1 - outlier_threshold) | (latencies > q3 + outlier_threshold)]
        
        if len(outliers) > len(latencies) * 0.05:  # More than 5% outliers
            print(f"Warning: {len(outliers)} outliers detected in latency measurements - consider increasing iterations")
    
    def _calculate_skewness(self, latencies: np.ndarray) -> float:
        """Calculate skewness of latency distribution."""
        if len(latencies) < 3:
            return 0.0
        
        mean = np.mean(latencies)
        std = np.std(latencies)
        
        if std == 0:
            return 0.0
        
        # Calculate skewness using the formula: E[(X - μ)³] / σ³
        skewness = np.mean(((latencies - mean) / std) ** 3)
        return float(skewness)
    
    def _calculate_kurtosis(self, latencies: np.ndarray) -> float:
        """Calculate kurtosis of latency distribution."""
        if len(latencies) < 4:
            return 0.0
        
        mean = np.mean(latencies)
        std = np.std(latencies)
        
        if std == 0:
            return 0.0
        
        # Calculate kurtosis using the formula: E[(X - μ)⁴] / σ⁴ - 3
        kurtosis = np.mean(((latencies - mean) / std) ** 4) - 3
        return float(kurtosis)
    
    def _detect_outliers(self, latencies: np.ndarray) -> Tuple[int, float]:
        """Detect outliers using IQR method."""
        if len(latencies) < 4:
            return 0, 0.0
        
        q1 = np.percentile(latencies, 25)
        q3 = np.percentile(latencies, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return 0, 0.0
        
        outlier_threshold = 1.5 * iqr
        outliers = latencies[(latencies < q1 - outlier_threshold) | (latencies > q3 + outlier_threshold)]
        
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(latencies)) * 100.0
        
        return outlier_count, outlier_percentage
    
    def measure_throughput(
        self,
        inference_func: Callable,
        observations: List[Dict[str, Any]],
        duration_seconds: int = 10
    ) -> ThroughputStats:
        """Measure throughput (requests per second) with comprehensive analysis."""
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        successful_requests = 0
        failed_requests = 0
        request_times = []
        error_details = []
        
        # Warmup phase
        warmup_duration = min(1.0, duration_seconds * 0.1)
        warmup_end = start_time + warmup_duration
        while time.perf_counter() < warmup_end:
            try:
                inference_func(observations, self.config.deterministic_mode)
            except Exception:
                pass
        
        # Main measurement phase
        measurement_start = time.perf_counter()
        while time.perf_counter() < end_time:
            request_start = time.perf_counter()
            try:
                inference_func(observations, self.config.deterministic_mode)
                request_end = time.perf_counter()
                successful_requests += 1
                request_times.append(request_end - request_start)
            except Exception as e:
                failed_requests += 1
                error_details.append(str(e))
        
        total_duration = time.perf_counter() - measurement_start
        total_requests = successful_requests + failed_requests
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0.0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        
        # Calculate additional throughput metrics
        if request_times:
            avg_request_time = sum(request_times) / len(request_times)
            max_request_time = max(request_times)
            min_request_time = min(request_times)
        else:
            avg_request_time = 0.0
            max_request_time = 0.0
            min_request_time = 0.0
        
        return ThroughputStats(
            requests_per_second=requests_per_second,
            total_requests=total_requests,
            total_duration_s=total_duration,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=error_rate
        )
    
    def measure_concurrent_throughput(
        self,
        inference_func: Callable,
        observations: List[Dict[str, Any]],
        duration_seconds: int = 10,
        max_workers: int = 4
    ) -> ThroughputStats:
        """Measure throughput with concurrent requests for realistic load testing."""
        import concurrent.futures
        import threading
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        successful_requests = 0
        failed_requests = 0
        request_times = []
        error_details = []
        
        # Thread-safe counters
        success_lock = threading.Lock()
        failure_lock = threading.Lock()
        times_lock = threading.Lock()
        errors_lock = threading.Lock()
        
        def worker():
            nonlocal successful_requests, failed_requests
            while time.perf_counter() < end_time:
                request_start = time.perf_counter()
                try:
                    inference_func(observations, self.config.deterministic_mode)
                    request_end = time.perf_counter()
                    
                    with success_lock:
                        successful_requests += 1
                    with times_lock:
                        request_times.append(request_end - request_start)
                        
                except Exception as e:
                    with failure_lock:
                        failed_requests += 1
                    with errors_lock:
                        error_details.append(str(e))
        
        # Start worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker) for _ in range(max_workers)]
            
            # Wait for all workers to complete
            concurrent.futures.wait(futures)
        
        total_duration = time.perf_counter() - start_time
        total_requests = successful_requests + failed_requests
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0.0
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
        """Measure memory usage during inference with comprehensive profiling."""
        import gc
        
        process = psutil.Process()
        
        # Get baseline memory and force garbage collection
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [baseline_memory]
        
        # Track garbage collection
        initial_gc_count = sum(gc.get_count())
        
        for i in range(iterations):
            # Force garbage collection before each measurement
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
        
        # Calculate final garbage collection count
        final_gc_count = sum(gc.get_count())
        gc_collections = final_gc_count - initial_gc_count
        
        memory_array = np.array(memory_samples)
        peak_memory = np.max(memory_array)
        average_memory = np.mean(memory_array)
        memory_growth = peak_memory - baseline_memory
        
        # Calculate memory efficiency (requests per MB)
        memory_efficiency = iterations / memory_growth if memory_growth > 0 else 0
        
        # Detect memory leaks (continuous growth pattern)
        memory_leak_detected = self._detect_memory_leak(memory_array)
        
        # Calculate memory fragmentation
        memory_fragmentation = self._calculate_memory_fragmentation(memory_array)
        
        # Generate memory optimization recommendations
        recommendations = self._generate_memory_recommendations(
            baseline_memory, peak_memory, memory_growth, memory_efficiency,
            memory_leak_detected, gc_collections
        )
        
        return MemoryStats(
            peak_memory_mb=float(peak_memory),
            average_memory_mb=float(average_memory),
            memory_growth_mb=float(memory_growth),
            memory_efficiency=float(memory_efficiency),
            baseline_memory_mb=float(baseline_memory),
            memory_usage_samples=memory_samples,
            memory_leak_detected=memory_leak_detected,
            memory_fragmentation=memory_fragmentation,
            gc_collections=gc_collections,
            memory_recommendations=recommendations
        )
    
    def _detect_memory_leak(self, memory_samples: np.ndarray) -> bool:
        """Detect potential memory leaks by analyzing memory growth pattern."""
        if len(memory_samples) < 5:
            return False
        
        # Calculate trend using linear regression
        x = np.arange(len(memory_samples))
        slope, _ = np.polyfit(x, memory_samples, 1)
        
        # If memory consistently grows by more than 1MB per iteration, consider it a leak
        return slope > 1.0
    
    def _calculate_memory_fragmentation(self, memory_samples: np.ndarray) -> float:
        """Calculate memory fragmentation based on variance in memory usage."""
        if len(memory_samples) < 2:
            return 0.0
        
        # High variance indicates fragmentation
        variance = np.var(memory_samples)
        mean_memory = np.mean(memory_samples)
        
        # Return fragmentation as percentage of mean memory
        return (variance / mean_memory) * 100.0 if mean_memory > 0 else 0.0
    
    def _generate_memory_recommendations(
        self,
        baseline_memory: float,
        peak_memory: float,
        memory_growth: float,
        memory_efficiency: float,
        memory_leak_detected: bool,
        gc_collections: int
    ) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Memory leak recommendations
        if memory_leak_detected:
            recommendations.append("Memory leak detected - review object lifecycle and cleanup")
            recommendations.append("Consider implementing explicit memory management")
        
        # Memory efficiency recommendations
        if memory_efficiency < 10:  # Less than 10 requests per MB
            recommendations.append("Low memory efficiency - consider batch processing optimization")
            recommendations.append("Review tensor allocation patterns")
        
        # Memory growth recommendations
        if memory_growth > 100:  # More than 100MB growth
            recommendations.append("High memory growth - consider reducing batch size")
            recommendations.append("Implement memory pooling for repeated operations")
        
        # Garbage collection recommendations
        if gc_collections > 50:  # More than 50 GC collections
            recommendations.append("Frequent garbage collection - optimize object creation")
            recommendations.append("Consider using object pooling")
        
        # General recommendations
        if peak_memory > 500:  # More than 500MB peak
            recommendations.append("High peak memory usage - consider memory limits")
            recommendations.append("Review model size and optimization opportunities")
        
        return recommendations
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information for baseline comparison."""
        from .hardware_detector import HardwareDetector
        
        detector = HardwareDetector()
        hardware_info = detector.get_hardware_info()
        
        return {
            "cpu": {
                "model": hardware_info.cpu.model,
                "cores": hardware_info.cpu.cores,
                "threads": hardware_info.cpu.threads,
                "frequency_mhz": hardware_info.cpu.frequency_mhz,
                "architecture": hardware_info.cpu.architecture,
                "features": hardware_info.cpu.features,
                "avx_support": hardware_info.cpu.avx_support,
                "intel_mkl_available": hardware_info.cpu.intel_mkl_available
            },
            "gpu": {
                "model": hardware_info.gpu.model if hardware_info.gpu else None,
                "memory_gb": hardware_info.gpu.memory_gb if hardware_info.gpu else 0.0,
                "compute_capability": hardware_info.gpu.compute_capability if hardware_info.gpu else None,
                "cuda_available": hardware_info.gpu.cuda_available if hardware_info.gpu else False,
                "tensorrt_available": hardware_info.gpu.tensorrt_available if hardware_info.gpu else False
            },
            "memory": {
                "total_gb": hardware_info.memory.total_gb,
                "available_gb": hardware_info.memory.available_gb,
                "memory_type": hardware_info.memory.memory_type
            },
            "platform": hardware_info.platform,
            "python_version": hardware_info.python_version,
            "torch_version": hardware_info.torch_version,
            "optimization_recommendations": hardware_info.optimization_recommendations
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
        
        # Find optimal batch size
        optimal_batch_size = self._find_optimal_batch_size(results)
        print(f"\nOptimal batch size: {optimal_batch_size}")
        
        return results
    
    def _find_optimal_batch_size(self, results: List[BenchmarkResult]) -> int:
        """Find optimal batch size based on performance metrics."""
        if not results:
            return 1
        
        # Score each batch size based on multiple criteria
        scores = []
        for i, result in enumerate(results):
            batch_size = self.config.batch_sizes[i] if i < len(self.config.batch_sizes) else 1
            
            # Calculate composite score
            score = self._calculate_batch_size_score(result, batch_size)
            scores.append((batch_size, score))
        
        # Return batch size with highest score
        optimal_batch_size, _ = max(scores, key=lambda x: x[1])
        return optimal_batch_size
    
    def _calculate_batch_size_score(
        self,
        result: BenchmarkResult,
        batch_size: int
    ) -> float:
        """Calculate composite score for batch size optimization."""
        # Normalize metrics (lower is better for latency, higher is better for throughput)
        latency_score = max(0, 1.0 - (result.latency_stats.p50_ms / self.config.p50_threshold_ms))
        throughput_score = min(1.0, result.throughput_stats.requests_per_second / self.config.throughput_threshold_rps)
        memory_score = max(0, 1.0 - (result.memory_stats.peak_memory_mb / self.config.max_memory_usage_mb))
        
        # Efficiency score (throughput per unit of memory)
        efficiency_score = 0.0
        if result.memory_stats.peak_memory_mb > 0:
            efficiency_score = result.throughput_stats.requests_per_second / result.memory_stats.peak_memory_mb
            efficiency_score = min(1.0, efficiency_score / 10.0)  # Normalize
        
        # Weighted composite score
        composite_score = (
            latency_score * 0.4 +      # 40% weight on latency
            throughput_score * 0.3 +   # 30% weight on throughput
            memory_score * 0.2 +       # 20% weight on memory usage
            efficiency_score * 0.1     # 10% weight on efficiency
        )
        
        return composite_score
    
    def run_dynamic_batch_size_optimization(
        self,
        inference_func: Callable,
        target_latency_ms: float = 10.0,
        max_memory_mb: float = 1024.0
    ) -> Dict[str, Any]:
        """Run dynamic batch size optimization to find the best configuration."""
        print("Running dynamic batch size optimization...")
        
        # Start with small batch size and increase until constraints are violated
        optimal_batch_size = 1
        best_result = None
        best_score = 0.0
        
        # Test batch sizes from 1 to 64 (or until constraints violated)
        for batch_size in range(1, 65):
            print(f"Testing dynamic batch size: {batch_size}")
            
            try:
                result = asyncio.run(self.run_benchmark(inference_func, batch_size))
                
                # Check constraints
                if (result.latency_stats.p50_ms > target_latency_ms or
                    result.memory_stats.peak_memory_mb > max_memory_mb):
                    print(f"  Constraints violated at batch size {batch_size}")
                    break
                
                # Calculate score
                score = self._calculate_batch_size_score(result, batch_size)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    optimal_batch_size = batch_size
                
                print(f"  Score: {score:.3f}, P50: {result.latency_stats.p50_ms:.2f}ms")
                
            except Exception as e:
                print(f"  Error at batch size {batch_size}: {e}")
                break
        
        return {
            "optimal_batch_size": optimal_batch_size,
            "best_result": best_result,
            "best_score": best_score,
            "target_latency_ms": target_latency_ms,
            "max_memory_mb": max_memory_mb
        }
    
    def collect_hardware_baseline(
        self,
        inference_func: Callable,
        save_baseline: bool = True
    ) -> Dict[str, Any]:
        """Collect hardware-specific performance baseline."""
        print("Collecting hardware-specific performance baseline...")
        
        # Get hardware information
        hardware_info = self.get_hardware_info()
        
        # Run comprehensive benchmark
        result = asyncio.run(self.run_benchmark(inference_func, batch_size=1))
        
        # Create baseline data
        baseline = {
            "hardware_signature": self._generate_hardware_signature(hardware_info),
            "hardware_info": hardware_info,
            "performance_metrics": {
                "p50_latency_ms": result.latency_stats.p50_ms,
                "p95_latency_ms": result.latency_stats.p95_ms,
                "p99_latency_ms": result.latency_stats.p99_ms,
                "throughput_rps": result.throughput_stats.requests_per_second,
                "memory_usage_mb": result.memory_stats.peak_memory_mb,
                "memory_efficiency": result.memory_stats.memory_efficiency
            },
            "test_configuration": {
                "batch_size": 1,
                "iterations": self.config.measurement_iterations,
                "warmup_iterations": self.config.warmup_iterations,
                "deterministic_mode": self.config.deterministic_mode
            },
            "timestamp": result.timestamp,
            "torch_version": hardware_info["torch_version"],
            "python_version": hardware_info["python_version"]
        }
        
        # Save baseline if requested
        if save_baseline:
            self._save_hardware_baseline(baseline)
        
        return baseline
    
    def _generate_hardware_signature(self, hardware_info: Dict[str, Any]) -> str:
        """Generate unique hardware signature for baseline identification."""
        cpu_model = hardware_info["cpu"]["model"]
        gpu_model = hardware_info["gpu"]["model"] or "None"
        memory_gb = hardware_info["memory"]["total_gb"]
        
        signature = f"{cpu_model}_{gpu_model}_{memory_gb:.0f}GB"
        return signature.replace(" ", "_").replace("/", "_")
    
    def _save_hardware_baseline(self, baseline: Dict[str, Any]) -> None:
        """Save hardware baseline to file."""
        import json
        from pathlib import Path
        
        baselines_dir = Path("model-serving/baselines")
        baselines_dir.mkdir(exist_ok=True)
        
        filename = f"baseline_{baseline['hardware_signature']}.json"
        filepath = baselines_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(baseline, f, indent=2)
        
        print(f"Hardware baseline saved to: {filepath}")
    
    def load_hardware_baseline(self, hardware_signature: str) -> Optional[Dict[str, Any]]:
        """Load hardware baseline from file."""
        import json
        from pathlib import Path
        
        baselines_dir = Path("model-serving/baselines")
        filename = f"baseline_{hardware_signature}.json"
        filepath = baselines_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading baseline {hardware_signature}: {e}")
            return None
    
    def compare_with_baseline(
        self,
        current_result: BenchmarkResult,
        baseline: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compare current result with hardware baseline."""
        if not baseline:
            hardware_info = self.get_hardware_info()
            hardware_signature = self._generate_hardware_signature(hardware_info)
            baseline = self.load_hardware_baseline(hardware_signature)
        
        if not baseline:
            return {"error": "No baseline available for comparison"}
        
        baseline_metrics = baseline["performance_metrics"]
        
        # Calculate percentage differences
        p50_diff = self._calculate_percentage_diff(
            current_result.latency_stats.p50_ms, baseline_metrics["p50_latency_ms"]
        )
        p95_diff = self._calculate_percentage_diff(
            current_result.latency_stats.p95_ms, baseline_metrics["p95_latency_ms"]
        )
        p99_diff = self._calculate_percentage_diff(
            current_result.latency_stats.p99_ms, baseline_metrics["p99_latency_ms"]
        )
        throughput_diff = self._calculate_percentage_diff(
            current_result.throughput_stats.requests_per_second, baseline_metrics["throughput_rps"]
        )
        memory_diff = self._calculate_percentage_diff(
            current_result.memory_stats.peak_memory_mb, baseline_metrics["memory_usage_mb"]
        )
        
        # Determine if performance improved or degraded
        performance_changes = {
            "p50_latency": {
                "current": current_result.latency_stats.p50_ms,
                "baseline": baseline_metrics["p50_latency_ms"],
                "difference_percent": p50_diff,
                "improved": p50_diff < 0  # Lower latency is better
            },
            "p95_latency": {
                "current": current_result.latency_stats.p95_ms,
                "baseline": baseline_metrics["p95_latency_ms"],
                "difference_percent": p95_diff,
                "improved": p95_diff < 0
            },
            "p99_latency": {
                "current": current_result.latency_stats.p99_ms,
                "baseline": baseline_metrics["p99_latency_ms"],
                "difference_percent": p99_diff,
                "improved": p99_diff < 0
            },
            "throughput": {
                "current": current_result.throughput_stats.requests_per_second,
                "baseline": baseline_metrics["throughput_rps"],
                "difference_percent": throughput_diff,
                "improved": throughput_diff > 0  # Higher throughput is better
            },
            "memory_usage": {
                "current": current_result.memory_stats.peak_memory_mb,
                "baseline": baseline_metrics["memory_usage_mb"],
                "difference_percent": memory_diff,
                "improved": memory_diff < 0  # Lower memory usage is better
            }
        }
        
        # Overall performance assessment
        improvements = sum(1 for metric in performance_changes.values() if metric["improved"])
        total_metrics = len(performance_changes)
        overall_improvement = improvements / total_metrics
        
        return {
            "hardware_signature": baseline["hardware_signature"],
            "baseline_timestamp": baseline["timestamp"],
            "performance_changes": performance_changes,
            "overall_improvement": overall_improvement,
            "summary": {
                "improved_metrics": improvements,
                "total_metrics": total_metrics,
                "performance_trend": "improved" if overall_improvement > 0.5 else "degraded"
            }
        }
    
    def _calculate_percentage_diff(self, current: float, baseline: float) -> float:
        """Calculate percentage difference from baseline."""
        if baseline == 0:
            return 0.0
        return ((current - baseline) / baseline) * 100.0
    
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
