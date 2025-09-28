"""
Unit tests for benchmark engine.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

from src.benchmarking.benchmark import (
    BenchmarkEngine,
    BenchmarkConfig,
    LatencyStats,
    ThroughputStats,
    MemoryStats,
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()

        assert config.warmup_iterations == 10
        assert config.measurement_iterations == 100
        assert config.batch_sizes == [1, 4, 8, 16, 32]
        assert config.p50_threshold_ms == 10.0
        assert config.p95_threshold_ms == 20.0
        assert config.p99_threshold_ms == 50.0
        assert config.throughput_threshold_rps == 1000.0
        assert config.max_memory_usage_mb == 1024.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            warmup_iterations=5,
            measurement_iterations=50,
            batch_sizes=[1, 2, 4],
            p50_threshold_ms=5.0,
        )

        assert config.warmup_iterations == 5
        assert config.measurement_iterations == 50
        assert config.batch_sizes == [1, 2, 4]
        assert config.p50_threshold_ms == 5.0


class TestLatencyStats:
    """Test LatencyStats dataclass."""

    def test_latency_stats_creation(self):
        """Test creating LatencyStats."""
        stats = LatencyStats(
            p50_ms=5.0,
            p95_ms=10.0,
            p99_ms=20.0,
            mean_ms=6.0,
            std_ms=2.0,
            min_ms=1.0,
            max_ms=25.0,
            total_measurements=100,
            p90_ms=8.0,
            p99_9_ms=30.0,
            skewness=0.5,
            kurtosis=2.0,
            outlier_count=2,
            outlier_percentage=2.0,
        )

        assert stats.p50_ms == 5.0
        assert stats.p95_ms == 10.0
        assert stats.p99_ms == 20.0
        assert stats.p90_ms == 8.0
        assert stats.p99_9_ms == 30.0
        assert stats.mean_ms == 6.0
        assert stats.std_ms == 2.0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 25.0
        assert stats.total_measurements == 100
        assert stats.skewness == 0.5
        assert stats.kurtosis == 2.0
        assert stats.outlier_count == 2
        assert stats.outlier_percentage == 2.0

        # Test post-init calculations
        assert stats.median_ms == 5.0  # Should equal p50_ms
        assert stats.variance_ms2 == 4.0  # Should equal std_ms^2
        assert stats.coefficient_of_variation == 2.0 / 6.0  # std/mean


class TestThroughputStats:
    """Test ThroughputStats dataclass."""

    def test_throughput_stats_creation(self):
        """Test creating ThroughputStats."""
        stats = ThroughputStats(
            requests_per_second=1500.0,
            total_requests=1000,
            total_duration_s=10.0,
            successful_requests=950,
            failed_requests=50,
            error_rate=0.05,
        )

        assert stats.requests_per_second == 1500.0
        assert stats.total_requests == 1000
        assert stats.total_duration_s == 10.0
        assert stats.successful_requests == 950
        assert stats.failed_requests == 50
        assert stats.error_rate == 0.05


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_memory_stats_creation(self):
        """Test creating MemoryStats."""
        stats = MemoryStats(
            peak_memory_mb=512.0,
            average_memory_mb=400.0,
            memory_growth_mb=100.0,
            memory_efficiency=10.0,
        )

        assert stats.peak_memory_mb == 512.0
        assert stats.average_memory_mb == 400.0
        assert stats.memory_growth_mb == 100.0
        assert stats.memory_efficiency == 10.0


class TestBenchmarkEngine:
    """Test BenchmarkEngine class."""

    def test_engine_initialization(self):
        """Test benchmark engine initialization."""
        config = BenchmarkConfig(measurement_iterations=50)
        engine = BenchmarkEngine(config)

        assert engine.config == config
        assert engine.results == []

    def test_create_test_observations(self):
        """Test creating test observations."""
        engine = BenchmarkEngine()
        observations = engine.create_test_observations(3)

        assert len(observations) == 3
        for obs in observations:
            assert "speed" in obs
            assert "steering" in obs
            assert "sensors" in obs
            assert isinstance(obs["sensors"], list)
            assert len(obs["sensors"]) == 3

    def test_measure_latency(self):
        """Test latency measurement."""
        engine = BenchmarkEngine()

        # Mock inference function
        def mock_inference(observations, deterministic):
            time.sleep(0.001)  # 1ms
            return [{"throttle": 0.5, "brake": 0.0, "steer": 0.1}] * len(observations)

        observations = engine.create_test_observations(1)
        stats = engine.measure_latency(mock_inference, observations, 10)

        assert isinstance(stats, LatencyStats)
        assert stats.total_measurements == 10
        assert stats.p50_ms > 0
        assert stats.p95_ms > 0
        assert stats.p99_ms > 0
        assert stats.p90_ms > 0
        assert stats.p99_9_ms > 0
        assert stats.mean_ms > 0
        assert stats.skewness is not None
        assert stats.kurtosis is not None
        assert stats.outlier_count >= 0
        assert stats.outlier_percentage >= 0

    def test_statistical_calculations(self):
        """Test statistical calculation methods."""
        engine = BenchmarkEngine()

        # Test with known data
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Test skewness calculation
        skewness = engine._calculate_skewness(test_data)
        assert isinstance(skewness, float)

        # Test kurtosis calculation
        kurtosis = engine._calculate_kurtosis(test_data)
        assert isinstance(kurtosis, float)

        # Test outlier detection
        outlier_count, outlier_percentage = engine._detect_outliers(test_data)
        assert isinstance(outlier_count, int)
        assert isinstance(outlier_percentage, float)
        assert outlier_count >= 0
        assert outlier_percentage >= 0.0

    def test_statistical_validation(self):
        """Test statistical validation methods."""
        engine = BenchmarkEngine()

        # Test with normal data
        normal_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        engine._validate_latency_statistics(normal_data, 3.0, 1.0)  # Should not raise

        # Test with negative values (should raise)
        negative_data = np.array([-1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            engine._validate_latency_statistics(negative_data, 2.6, 1.0)

        # Test with high variance (should warn)
        high_variance_data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        # This should not raise but might print a warning
        engine._validate_latency_statistics(high_variance_data, 22.0, 50.0)

    def test_measure_throughput(self):
        """Test throughput measurement."""
        engine = BenchmarkEngine()

        # Mock inference function
        def mock_inference(observations, deterministic):
            time.sleep(0.001)  # 1ms
            return [{"throttle": 0.5, "brake": 0.0, "steer": 0.1}] * len(observations)

        observations = engine.create_test_observations(1)
        stats = engine.measure_throughput(mock_inference, observations, duration_seconds=1)

        assert isinstance(stats, ThroughputStats)
        assert stats.requests_per_second > 0
        assert stats.total_requests > 0
        assert stats.successful_requests > 0
        assert stats.error_rate >= 0

    @patch("psutil.Process")
    def test_measure_memory_usage(self, mock_process):
        """Test memory usage measurement."""
        # Mock process memory info
        mock_memory = Mock()
        mock_memory.rss = 100 * 1024 * 1024  # 100MB
        mock_process.return_value.memory_info.return_value = mock_memory

        engine = BenchmarkEngine()

        # Mock inference function
        def mock_inference(observations, deterministic):
            time.sleep(0.001)  # 1ms
            return [{"throttle": 0.5, "brake": 0.0, "steer": 0.1}] * len(observations)

        observations = engine.create_test_observations(1)
        stats = engine.measure_memory_usage(mock_inference, observations, 5)

        assert isinstance(stats, MemoryStats)
        assert stats.peak_memory_mb > 0
        assert stats.average_memory_mb > 0
        assert stats.memory_growth_mb >= 0

    def test_get_hardware_info(self):
        """Test hardware information gathering."""
        engine = BenchmarkEngine()
        hardware_info = engine.get_hardware_info()

        assert isinstance(hardware_info, dict)
        assert "cpu_count" in hardware_info
        assert "memory_total_gb" in hardware_info
        assert "torch_cuda_available" in hardware_info
        assert "torch_version" in hardware_info
        assert "python_version" in hardware_info

    @pytest.mark.asyncio
    async def test_run_benchmark(self):
        """Test running complete benchmark."""
        config = BenchmarkConfig(warmup_iterations=2, measurement_iterations=5)
        engine = BenchmarkEngine(config)

        # Mock inference function
        def mock_inference(observations, deterministic):
            time.sleep(0.001)  # 1ms
            return [{"throttle": 0.5, "brake": 0.0, "steer": 0.1}] * len(observations)

        result = await engine.run_benchmark(mock_inference, batch_size=1)

        assert result.config == config
        assert result.p50_requirement_met is not None
        assert result.p95_requirement_met is not None
        assert result.p99_requirement_met is not None
        assert result.throughput_requirement_met is not None
        assert result.memory_requirement_met is not None
        assert result.overall_success is not None
        assert result.timestamp is not None

    def test_run_batch_size_optimization(self):
        """Test batch size optimization."""
        config = BenchmarkConfig(warmup_iterations=1, measurement_iterations=2, batch_sizes=[1, 2])
        engine = BenchmarkEngine(config)

        # Mock inference function
        def mock_inference(observations, deterministic):
            time.sleep(0.001)  # 1ms
            return [{"throttle": 0.5, "brake": 0.0, "steer": 0.1}] * len(observations)

        results = engine.run_batch_size_optimization(mock_inference)

        assert len(results) == 2  # Two batch sizes
        for result in results:
            assert result.config == config
            assert result.latency_stats is not None
            assert result.throughput_stats is not None
            assert result.memory_stats is not None

    def test_generate_report(self):
        """Test report generation."""
        engine = BenchmarkEngine()

        # Add some mock results
        config = BenchmarkConfig()
        result = Mock()
        result.config = config
        result.latency_stats = LatencyStats(
            p50_ms=5.0,
            p95_ms=10.0,
            p99_ms=20.0,
            mean_ms=6.0,
            std_ms=2.0,
            min_ms=1.0,
            max_ms=25.0,
            total_measurements=100,
        )
        result.throughput_stats = ThroughputStats(
            requests_per_second=1500.0,
            total_requests=1000,
            total_duration_s=10.0,
            successful_requests=950,
            failed_requests=50,
            error_rate=0.05,
        )
        result.memory_stats = MemoryStats(
            peak_memory_mb=512.0,
            average_memory_mb=400.0,
            memory_growth_mb=100.0,
            memory_efficiency=10.0,
        )
        result.overall_success = True
        result.p50_requirement_met = True
        result.p95_requirement_met = True
        result.p99_requirement_met = True
        result.throughput_requirement_met = True
        result.memory_requirement_met = True

        engine.results = [result]

        report = engine.generate_report()

        assert isinstance(report, str)
        assert "BENCHMARK REPORT" in report
        assert "SUMMARY:" in report
        assert "TEST 1" in report
        assert "P50" in report
        assert "P95" in report
        assert "P99" in report
        assert "Throughput" in report
        assert "Memory" in report


class TestBenchmarkIntegration:
    """Integration tests for benchmark engine."""

    def test_full_benchmark_workflow(self):
        """Test complete benchmark workflow."""
        config = BenchmarkConfig(warmup_iterations=1, measurement_iterations=3, batch_sizes=[1, 2])
        engine = BenchmarkEngine(config)

        # Mock inference function with realistic timing
        def mock_inference(observations, deterministic):
            time.sleep(0.005)  # 5ms
            return [{"throttle": 0.5, "brake": 0.0, "steer": 0.1}] * len(observations)

        # Run benchmarks
        results = engine.run_batch_size_optimization(mock_inference)

        # Verify results
        assert len(results) == 2
        assert len(engine.results) == 2

        # Generate report
        report = engine.generate_report()
        assert "BENCHMARK REPORT" in report

        # Verify all results have required fields
        for result in results:
            assert hasattr(result, "latency_stats")
            assert hasattr(result, "throughput_stats")
            assert hasattr(result, "memory_stats")
            assert hasattr(result, "overall_success")
            assert hasattr(result, "timestamp")
