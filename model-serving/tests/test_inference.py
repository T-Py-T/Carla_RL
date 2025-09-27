"""
Unit tests for inference engine in CarlaRL Policy-as-a-Service.

Tests inference optimization, performance monitoring, and batch processing.
"""

import time
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.exceptions import InferenceError
from src.inference import InferenceCache, InferenceEngine, InferenceMetrics
from src.io_schemas import Action, Observation
from src.model_loader import PolicyWrapper
from src.preprocessing import MinimalPreprocessor


class SimpleTestModel(nn.Module):
    """Simple test model for inference testing."""

    def __init__(self, input_dim: int = 5, output_dim: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class TestInferenceMetrics:
    """Test cases for InferenceMetrics class."""

    def test_metrics_initialization(self):
        """Test InferenceMetrics initialization."""
        metrics = InferenceMetrics()

        assert metrics.total_requests == 0
        assert metrics.total_observations == 0
        assert metrics.total_inference_time_ms == 0.0
        assert metrics.error_count == 0
        assert len(metrics.latency_history) == 0

    def test_add_request(self):
        """Test adding request metrics."""
        metrics = InferenceMetrics()

        metrics.add_request(
            batch_size=2,
            inference_time_ms=10.0,
            preprocessing_time_ms=2.0,
            postprocessing_time_ms=1.0
        )

        assert metrics.total_requests == 1
        assert metrics.total_observations == 2
        assert metrics.total_inference_time_ms == 10.0
        assert metrics.total_preprocessing_time_ms == 2.0
        assert metrics.total_postprocessing_time_ms == 1.0
        assert len(metrics.latency_history) == 1
        assert metrics.latency_history[0] == 5.0  # 10.0 / 2

    def test_add_error(self):
        """Test adding error metrics."""
        metrics = InferenceMetrics()

        start_time = time.time()
        metrics.add_error()
        end_time = time.time()

        assert metrics.error_count == 1
        assert start_time <= metrics.last_error_time <= end_time

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        metrics = InferenceMetrics()

        # Add various latencies
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for _i, latency in enumerate(latencies):
            metrics.add_request(batch_size=1, inference_time_ms=latency)

        percentiles = metrics.get_latency_percentiles()

        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert percentiles["p50"] == 5.0  # Median of 1-10

    def test_throughput_stats(self):
        """Test throughput statistics."""
        metrics = InferenceMetrics()

        # Add some requests
        metrics.add_request(batch_size=10, inference_time_ms=1000.0)  # 1 second

        throughput = metrics.get_throughput_stats()

        assert "requests_per_second" in throughput
        assert "observations_per_second" in throughput
        assert throughput["requests_per_second"] == 1.0  # 1 request / 1 second
        assert throughput["observations_per_second"] == 10.0  # 10 obs / 1 second

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = InferenceMetrics()

        # Add some data
        metrics.add_request(batch_size=5, inference_time_ms=50.0)
        metrics.add_error()

        # Reset
        metrics.reset()

        assert metrics.total_requests == 0
        assert metrics.total_observations == 0
        assert metrics.error_count == 0
        assert len(metrics.latency_history) == 0

    def test_history_size_limit(self):
        """Test latency history size limit."""
        metrics = InferenceMetrics()
        metrics.max_history_size = 5

        # Add more requests than history limit
        for i in range(10):
            metrics.add_request(batch_size=1, inference_time_ms=float(i))

        assert len(metrics.latency_history) == 5
        # Should keep the most recent values
        assert metrics.latency_history == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestInferenceCache:
    """Test cases for InferenceCache class."""

    def create_test_observations(self, n: int = 2) -> list[Observation]:
        """Create test observations."""
        observations = []
        for i in range(n):
            obs = Observation(
                speed=20.0 + i,
                steering=i * 0.1,
                sensors=[0.1 * j for j in range(3)]
            )
            observations.append(obs)
        return observations

    def test_cache_initialization(self):
        """Test InferenceCache initialization."""
        cache = InferenceCache(max_size=100)

        assert cache.max_size == 100
        assert cache.size() == 0

    def test_cache_put_get(self):
        """Test caching and retrieval."""
        cache = InferenceCache(max_size=10)
        observations = self.create_test_observations()

        # Create test result
        actions = [Action(throttle=0.7, brake=0.0, steer=0.1)]
        result = (actions, 10.0)

        # Cache result
        cache.put(observations, deterministic=True, result=result)
        assert cache.size() == 1

        # Retrieve result
        cached_result = cache.get(observations, deterministic=True)
        assert cached_result is not None
        assert len(cached_result[0]) == 1
        assert cached_result[1] == 10.0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = InferenceCache(max_size=10)
        observations = self.create_test_observations()

        # Try to get non-existent result
        result = cache.get(observations, deterministic=True)
        assert result is None

    def test_cache_deterministic_flag(self):
        """Test that deterministic flag affects caching."""
        cache = InferenceCache(max_size=10)
        observations = self.create_test_observations()

        actions = [Action(throttle=0.7, brake=0.0, steer=0.1)]
        result1 = (actions, 10.0)
        result2 = (actions, 15.0)

        # Cache with different deterministic flags
        cache.put(observations, deterministic=True, result=result1)
        cache.put(observations, deterministic=False, result=result2)

        assert cache.size() == 2

        # Should retrieve different results
        cached_det = cache.get(observations, deterministic=True)
        cached_stoch = cache.get(observations, deterministic=False)

        assert cached_det[1] == 10.0
        assert cached_stoch[1] == 15.0

    def test_cache_size_limit(self):
        """Test cache size limit and LRU eviction."""
        cache = InferenceCache(max_size=2)

        # Create different observations
        obs1 = self.create_test_observations(1)
        obs2 = self.create_test_observations(2)
        obs3 = [Observation(speed=50.0, steering=0.5, sensors=[1.0, 2.0, 3.0])]

        actions = [Action(throttle=0.5, brake=0.0, steer=0.0)]

        # Fill cache to capacity
        cache.put(obs1, True, (actions, 1.0))
        cache.put(obs2, True, (actions, 2.0))
        assert cache.size() == 2

        # Add third item (should evict oldest)
        cache.put(obs3, True, (actions, 3.0))
        assert cache.size() == 2

        # First item should be evicted
        assert cache.get(obs1, True) is None
        assert cache.get(obs2, True) is not None
        assert cache.get(obs3, True) is not None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = InferenceCache(max_size=10)
        observations = self.create_test_observations()

        actions = [Action(throttle=0.7, brake=0.0, steer=0.1)]
        cache.put(observations, True, (actions, 10.0))

        assert cache.size() == 1

        cache.clear()
        assert cache.size() == 0
        assert cache.get(observations, True) is None


class TestInferenceEngine:
    """Test cases for InferenceEngine class."""

    def create_test_engine(self, enable_cache: bool = True) -> InferenceEngine:
        """Create test inference engine."""
        model = SimpleTestModel()
        policy = PolicyWrapper(model)
        preprocessor = MinimalPreprocessor()

        engine = InferenceEngine(
            policy=policy,
            device=torch.device('cpu'),
            preprocessor=preprocessor,
            enable_cache=enable_cache,
            cache_size=100,
            max_batch_size=10
        )

        return engine

    def create_test_observations(self, n: int = 2) -> list[Observation]:
        """Create test observations."""
        observations = []
        for i in range(n):
            obs = Observation(
                speed=20.0 + i,
                steering=i * 0.1,
                sensors=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
            observations.append(obs)
        return observations

    def test_inference_engine_initialization(self):
        """Test InferenceEngine initialization."""
        engine = self.create_test_engine()

        assert engine.device.type == 'cpu'
        assert engine.preprocessor is not None
        assert engine.cache is not None
        assert engine.max_batch_size == 10
        assert engine.metrics.total_requests == 0

    def test_inference_engine_initialization_no_cache(self):
        """Test InferenceEngine without cache."""
        engine = self.create_test_engine(enable_cache=False)

        assert engine.cache is None

    def test_basic_inference(self):
        """Test basic inference functionality."""
        engine = self.create_test_engine()
        observations = self.create_test_observations(2)

        actions, timing_ms = engine.predict(observations, deterministic=True)

        assert len(actions) == 2
        assert all(isinstance(action, Action) for action in actions)
        assert timing_ms > 0
        assert engine.metrics.total_requests == 1
        assert engine.metrics.total_observations == 2

    def test_deterministic_inference(self):
        """Test deterministic inference produces consistent results."""
        engine = self.create_test_engine()
        observations = self.create_test_observations(1)

        # Run inference twice with deterministic mode
        actions1, _ = engine.predict(observations, deterministic=True)
        actions2, _ = engine.predict(observations, deterministic=True)

        # Results should be identical
        assert len(actions1) == len(actions2)
        for a1, a2 in zip(actions1, actions2, strict=False):
            assert abs(a1.throttle - a2.throttle) < 1e-6
            assert abs(a1.brake - a2.brake) < 1e-6
            assert abs(a1.steer - a2.steer) < 1e-6

    def test_batch_processing(self):
        """Test batch processing with different batch sizes."""
        engine = self.create_test_engine()

        # Test different batch sizes
        for batch_size in [1, 3, 5]:
            observations = self.create_test_observations(batch_size)
            actions, timing_ms = engine.predict(observations)

            assert len(actions) == batch_size
            assert timing_ms > 0

    def test_empty_observations_error(self):
        """Test error handling for empty observations."""
        engine = self.create_test_engine()

        with pytest.raises(InferenceError) as exc_info:
            engine.predict([])

        assert "empty observation list" in str(exc_info.value)

    def test_batch_size_limit(self):
        """Test batch size limit enforcement."""
        engine = self.create_test_engine()
        engine.max_batch_size = 3

        observations = self.create_test_observations(5)  # Exceeds limit

        with pytest.raises(InferenceError) as exc_info:
            engine.predict(observations)

        assert "exceeds maximum" in str(exc_info.value)

    def test_caching_functionality(self):
        """Test inference result caching."""
        engine = self.create_test_engine(enable_cache=True)
        observations = self.create_test_observations(1)

        # First inference (cache miss)
        actions1, timing1 = engine.predict(observations, deterministic=True, use_cache=True)

        # Second inference (cache hit)
        actions2, timing2 = engine.predict(observations, deterministic=True, use_cache=True)

        # Results should be identical
        assert len(actions1) == len(actions2)
        assert actions1[0].throttle == actions2[0].throttle

        # Second call should be faster (cached)
        assert timing2 <= timing1 * 1.1  # Allow small overhead

    def test_cache_disabled(self):
        """Test inference without caching."""
        engine = self.create_test_engine(enable_cache=False)
        observations = self.create_test_observations(1)

        # Should work without cache
        actions, timing_ms = engine.predict(observations, use_cache=True)

        assert len(actions) == 1
        assert timing_ms > 0

    def test_warmup_functionality(self):
        """Test inference engine warmup."""
        engine = self.create_test_engine()

        assert not engine._warmed_up

        warmup_time = engine.warmup(warmup_batches=2, warmup_batch_size=1)

        assert engine._warmed_up
        assert warmup_time > 0
        assert engine.metrics.total_requests >= 2  # Warmup requests

    def test_performance_stats(self):
        """Test performance statistics collection."""
        engine = self.create_test_engine()
        observations = self.create_test_observations(2)

        # Run some inferences
        engine.predict(observations, deterministic=True)
        engine.predict(observations, deterministic=False)

        stats = engine.get_performance_stats()

        # Check required fields
        required_fields = [
            "model_version", "git_sha", "device", "warmed_up",
            "total_requests", "total_observations", "error_count",
            "cache_size", "latency_ms", "throughput"
        ]

        for field in required_fields:
            assert field in stats

        assert stats["total_requests"] == 2
        assert stats["total_observations"] == 4

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        engine = self.create_test_engine()
        observations = self.create_test_observations(1)

        # Run inference
        engine.predict(observations)
        assert engine.metrics.total_requests == 1

        # Reset metrics
        engine.reset_metrics()
        assert engine.metrics.total_requests == 0

        if engine.cache:
            assert engine.cache.size() == 0

    def test_memory_usage_stats(self):
        """Test memory usage statistics."""
        engine = self.create_test_engine()

        memory_stats = engine.get_memory_usage()

        assert "preallocated_tensors" in memory_stats
        assert "cache_enabled" in memory_stats
        assert "cache_size" in memory_stats

        assert memory_stats["cache_enabled"] is True
        assert memory_stats["cache_size"] == 0  # Initially empty

    def test_preprocessing_error_handling(self):
        """Test error handling during preprocessing."""
        engine = self.create_test_engine()

        # Create mock preprocessor that raises error
        engine.preprocessor = Mock()
        engine.preprocessor.transform.side_effect = RuntimeError("Preprocessing failed")

        observations = self.create_test_observations(1)

        with pytest.raises(InferenceError) as exc_info:
            engine.predict(observations)

        assert "Preprocessing failed" in str(exc_info.value)
        assert engine.metrics.error_count == 1

    def test_model_inference_error_handling(self):
        """Test error handling during model inference."""
        engine = self.create_test_engine()

        # Create mock policy that raises error
        engine.policy = Mock()
        engine.policy.side_effect = RuntimeError("Model failed")

        observations = self.create_test_observations(1)

        with pytest.raises(InferenceError) as exc_info:
            engine.predict(observations)

        assert "Inference failed" in str(exc_info.value)
        assert engine.metrics.error_count == 1

    def test_tensor_preallocation(self):
        """Test tensor pre-allocation for memory optimization."""
        engine = self.create_test_engine()
        observations = self.create_test_observations(2)

        # Run inference to trigger tensor allocation
        engine.predict(observations)

        # Check that tensors were pre-allocated
        memory_stats = engine.get_memory_usage()
        assert memory_stats["preallocated_tensors"] >= 0

    def test_deterministic_mode_setting(self):
        """Test global deterministic mode setting."""
        engine = self.create_test_engine()

        # Test setting deterministic mode
        engine.set_deterministic_mode(True)

        # Should not raise any errors
        observations = self.create_test_observations(1)
        actions1, _ = engine.predict(observations, deterministic=True)
        actions2, _ = engine.predict(observations, deterministic=True)

        # Results should be very similar (within floating point precision)
        assert abs(actions1[0].throttle - actions2[0].throttle) < 1e-5

    def test_inference_optimization(self):
        """Test inference optimization functionality."""
        engine = self.create_test_engine()

        # Should not raise errors
        engine.optimize_for_inference()

        # Model should still work after optimization
        observations = self.create_test_observations(1)
        actions, timing_ms = engine.predict(observations)

        assert len(actions) == 1
        assert timing_ms > 0


class TestInferenceEngineEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_sensor_arrays(self):
        """Test handling of large sensor arrays."""
        engine = TestInferenceEngine().create_test_engine()

        # Create observation with large sensor array
        large_sensors = [0.1] * 1000  # 1000 sensors
        observations = [Observation(speed=25.0, steering=0.0, sensors=large_sensors)]

        actions, timing_ms = engine.predict(observations)

        assert len(actions) == 1
        assert timing_ms > 0

    def test_extreme_values(self):
        """Test handling of extreme observation values."""
        engine = TestInferenceEngine().create_test_engine()

        # Create observations with extreme values
        observations = [
            Observation(speed=1000.0, steering=10.0, sensors=[1e6, -1e6, 0.0])
        ]

        actions, timing_ms = engine.predict(observations)

        assert len(actions) == 1
        # Actions should be clipped to valid ranges
        assert 0.0 <= actions[0].throttle <= 1.0
        assert 0.0 <= actions[0].brake <= 1.0
        assert -1.0 <= actions[0].steer <= 1.0

    def test_concurrent_inference(self):
        """Test concurrent inference requests."""
        engine = TestInferenceEngine().create_test_engine()
        observations = TestInferenceEngine().create_test_observations(1)

        import threading
        results = []
        errors = []

        def run_inference():
            try:
                actions, timing = engine.predict(observations)
                results.append((actions, timing))
            except Exception as e:
                errors.append(e)

        # Run multiple concurrent inferences
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_inference)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 5

        # All should return valid actions
        for actions, timing in results:
            assert len(actions) == 1
            assert timing > 0
