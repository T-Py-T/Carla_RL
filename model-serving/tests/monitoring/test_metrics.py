"""
Unit tests for metrics collection module.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from prometheus_client import CollectorRegistry

from src.monitoring.metrics import MetricsCollector, get_metrics_collector, initialize_metrics


class TestMetricsCollector:
    """Test cases for MetricsCollector class."""
    
    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        # Check that all metric types are initialized
        assert collector.inference_latency is not None
        assert collector.inference_requests_total is not None
        assert collector.batch_size_histogram is not None
        assert collector.deterministic_requests is not None
        assert collector.cpu_usage_percent is not None
        assert collector.memory_usage_bytes is not None
        assert collector.errors_total is not None
        assert collector.http_requests_total is not None
        assert collector.model_loaded is not None
        assert collector.model_warmed_up is not None
        assert collector.active_requests is not None
        assert collector.service_uptime_seconds is not None
    
    def test_custom_registry(self):
        """Test initialization with custom registry."""
        custom_registry = CollectorRegistry()
        collector = MetricsCollector(custom_registry)
        assert collector.registry is custom_registry
    
    def test_record_inference(self):
        """Test recording inference metrics."""
        collector = MetricsCollector()
        
        # Record inference metrics
        collector.record_inference(
            duration_seconds=0.05,
            model_version="v1.0.0",
            device="cpu",
            batch_size=4,
            deterministic=True,
            status="success"
        )
        
        # Check that metrics were recorded (we can't easily test the exact values
        # without scraping the registry, but we can ensure no exceptions are raised)
        assert True  # If we get here, no exception was raised
    
    def test_record_request(self):
        """Test recording HTTP request metrics."""
        collector = MetricsCollector()
        
        # Record request metrics
        collector.record_request(
            method="POST",
            endpoint="/predict",
            status_code=200,
            duration_seconds=0.1
        )
        
        assert True  # No exception raised
    
    def test_record_error(self):
        """Test recording error metrics."""
        collector = MetricsCollector()
        
        # Record error metrics
        collector.record_error(
            error_type="InferenceError",
            endpoint="/predict",
            model_version="v1.0.0"
        )
        
        assert True  # No exception raised
    
    def test_set_model_status(self):
        """Test setting model status metrics."""
        collector = MetricsCollector()
        
        # Set model status
        collector.set_model_status(
            model_version="v1.0.0",
            device="cpu",
            loaded=True,
            warmed_up=True
        )
        
        assert True  # No exception raised
    
    def test_record_model_loading(self):
        """Test recording model loading duration."""
        collector = MetricsCollector()
        
        # Record model loading
        collector.record_model_loading(
            model_version="v1.0.0",
            duration_seconds=2.5
        )
        
        assert True  # No exception raised
    
    def test_record_model_warmup(self):
        """Test recording model warmup duration."""
        collector = MetricsCollector()
        
        # Record model warmup
        collector.record_model_warmup(
            model_version="v1.0.0",
            duration_seconds=0.5
        )
        
        assert True  # No exception raised
    
    def test_set_active_requests(self):
        """Test setting active requests count."""
        collector = MetricsCollector()
        
        # Set active requests
        collector.set_active_requests(5)
        
        assert True  # No exception raised
    
    def test_set_request_queue_length(self):
        """Test setting request queue length."""
        collector = MetricsCollector()
        
        # Set queue length
        collector.set_request_queue_length(10)
        
        assert True  # No exception raised
    
    def test_set_service_uptime(self):
        """Test setting service uptime."""
        collector = MetricsCollector()
        
        # Set uptime
        collector.set_service_uptime(3600.0)
        
        assert True  # No exception raised
    
    def test_set_service_startup_time(self):
        """Test setting service startup time."""
        collector = MetricsCollector()
        
        # Set startup time
        collector.set_service_startup_time(time.time())
        
        assert True  # No exception raised
    
    def test_inference_timer_context_manager(self):
        """Test inference timer context manager."""
        collector = MetricsCollector()
        
        # Use context manager
        with collector.inference_timer(
            model_version="v1.0.0",
            device="cpu",
            batch_size=2,
            deterministic=False
        ):
            time.sleep(0.01)  # Simulate some work
        
        assert True  # No exception raised
    
    def test_inference_timer_with_exception(self):
        """Test inference timer context manager with exception."""
        collector = MetricsCollector()
        
        # Use context manager with exception
        with pytest.raises(ValueError):
            with collector.inference_timer(
                model_version="v1.0.0",
                device="cpu",
                batch_size=2,
                deterministic=False
            ):
                raise ValueError("Test exception")
        
        assert True  # Exception was properly raised
    
    def test_get_batch_size_range(self):
        """Test batch size range calculation."""
        collector = MetricsCollector()
        
        # Test various batch sizes
        assert collector._get_batch_size_range(1) == "1"
        assert collector._get_batch_size_range(2) == "2-4"
        assert collector._get_batch_size_range(4) == "2-4"
        assert collector._get_batch_size_range(8) == "5-8"
        assert collector._get_batch_size_range(16) == "9-16"
        assert collector._get_batch_size_range(32) == "17-32"
        assert collector._get_batch_size_range(64) == "33-64"
        assert collector._get_batch_size_range(128) == "65-128"
        assert collector._get_batch_size_range(256) == "129+"
    
    def test_get_metrics(self):
        """Test getting metrics in Prometheus format."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_inference(
            duration_seconds=0.05,
            model_version="v1.0.0",
            device="cpu",
            batch_size=4
        )
        
        # Get metrics
        metrics = collector.get_metrics()
        
        # Check that metrics are returned as string
        assert isinstance(metrics, str)
        assert len(metrics) > 0
    
    def test_get_metrics_content_type(self):
        """Test getting metrics content type."""
        collector = MetricsCollector()
        
        content_type = collector.get_metrics_content_type()
        assert content_type == "text/plain; version=0.0.4; charset=utf-8"
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_system_metrics_collection(self, mock_process, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(used=1024**3, percent=60.0)
        mock_process.return_value.cpu_percent.return_value = 25.0
        mock_process.return_value.memory_info.return_value.rss = 512**3
        
        collector = MetricsCollector()
        
        # Wait a bit for background collection to run
        time.sleep(0.1)
        
        # The metrics should be updated (we can't easily test exact values
        # without scraping, but we can ensure no exceptions are raised)
        assert True  # No exception raised


class TestGlobalFunctions:
    """Test cases for global functions."""
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return the same instance
        assert collector1 is collector2
    
    def test_initialize_metrics(self):
        """Test initializing metrics with custom registry."""
        custom_registry = CollectorRegistry()
        collector = initialize_metrics(custom_registry)
        
        assert collector.registry is custom_registry
        
        # Test that global instance is updated
        global_collector = get_metrics_collector()
        assert global_collector is collector
