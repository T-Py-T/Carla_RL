"""
Prometheus metrics collection and exposure.

This module provides comprehensive metrics collection for the CarlaRL serving infrastructure,
including inference latency, throughput, error rates, and system resource utilization.
"""

import time
from typing import Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import psutil
import threading
from contextlib import contextmanager


class MetricsCollector:
    """
    Comprehensive metrics collector for CarlaRL serving infrastructure.
    
    Collects and exposes metrics for:
    - Inference performance (latency, throughput)
    - Error rates and types
    - System resource utilization
    - Model loading and warmup status
    - Request processing statistics
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector with optional custom registry."""
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()
        
        # Initialize all metrics
        self._init_inference_metrics()
        self._init_system_metrics()
        self._init_error_metrics()
        self._init_model_metrics()
        self._init_request_metrics()
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _init_inference_metrics(self):
        """Initialize inference performance metrics."""
        # Inference latency histogram (P50, P95, P99)
        self.inference_latency = Histogram(
            'carla_rl_inference_duration_seconds',
            'Time spent on inference requests',
            ['model_version', 'device', 'batch_size_range'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Inference throughput counter
        self.inference_requests_total = Counter(
            'carla_rl_inference_requests_total',
            'Total number of inference requests',
            ['model_version', 'device', 'status'],
            registry=self.registry
        )
        
        # Batch size distribution
        self.batch_size_histogram = Histogram(
            'carla_rl_batch_size',
            'Distribution of batch sizes for inference requests',
            ['model_version'],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            registry=self.registry
        )
        
        # Deterministic vs non-deterministic inference
        self.deterministic_requests = Counter(
            'carla_rl_deterministic_requests_total',
            'Total number of deterministic inference requests',
            ['model_version'],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system resource utilization metrics."""
        # CPU utilization
        self.cpu_usage_percent = Gauge(
            'carla_rl_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Memory utilization
        self.memory_usage_bytes = Gauge(
            'carla_rl_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'carla_rl_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        # GPU metrics (if available)
        self.gpu_memory_usage_bytes = Gauge(
            'carla_rl_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['device_id'],
            registry=self.registry
        )
        
        self.gpu_utilization_percent = Gauge(
            'carla_rl_gpu_utilization_percent',
            'GPU utilization percentage',
            ['device_id'],
            registry=self.registry
        )
        
        # Process-specific metrics
        self.process_cpu_percent = Gauge(
            'carla_rl_process_cpu_percent',
            'Process CPU usage percentage',
            registry=self.registry
        )
        
        self.process_memory_bytes = Gauge(
            'carla_rl_process_memory_bytes',
            'Process memory usage in bytes',
            registry=self.registry
        )
    
    def _init_error_metrics(self):
        """Initialize error tracking metrics."""
        # Error counters by type
        self.errors_total = Counter(
            'carla_rl_errors_total',
            'Total number of errors',
            ['error_type', 'endpoint', 'model_version'],
            registry=self.registry
        )
        
        # HTTP status codes
        self.http_requests_total = Counter(
            'carla_rl_http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        # Request duration by endpoint
        self.request_duration = Histogram(
            'carla_rl_request_duration_seconds',
            'Request duration by endpoint',
            ['method', 'endpoint', 'status_code'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
    
    def _init_model_metrics(self):
        """Initialize model-specific metrics."""
        # Model loading status
        self.model_loaded = Gauge(
            'carla_rl_model_loaded',
            'Model loading status (1=loaded, 0=not loaded)',
            ['model_version', 'device'],
            registry=self.registry
        )
        
        # Model warmup status
        self.model_warmed_up = Gauge(
            'carla_rl_model_warmed_up',
            'Model warmup status (1=warmed, 0=not warmed)',
            ['model_version'],
            registry=self.registry
        )
        
        # Model loading time
        self.model_loading_duration = Summary(
            'carla_rl_model_loading_duration_seconds',
            'Time spent loading the model',
            ['model_version'],
            registry=self.registry
        )
        
        # Model warmup time
        self.model_warmup_duration = Summary(
            'carla_rl_model_warmup_duration_seconds',
            'Time spent warming up the model',
            ['model_version'],
            registry=self.registry
        )
    
    def _init_request_metrics(self):
        """Initialize request processing metrics."""
        # Active requests
        self.active_requests = Gauge(
            'carla_rl_active_requests',
            'Number of currently active requests',
            registry=self.registry
        )
        
        # Request queue length
        self.request_queue_length = Gauge(
            'carla_rl_request_queue_length',
            'Length of the request queue',
            registry=self.registry
        )
        
        # Service uptime
        self.service_uptime_seconds = Gauge(
            'carla_rl_service_uptime_seconds',
            'Service uptime in seconds',
            registry=self.registry
        )
        
        # Service startup time
        self.service_startup_time = Gauge(
            'carla_rl_service_startup_timestamp_seconds',
            'Service startup timestamp',
            registry=self.registry
        )
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection."""
        def collect_system_metrics():
            while True:
                try:
                    self._update_system_metrics()
                    time.sleep(5)  # Update every 5 seconds
                except Exception:
                    # Continue collecting even if individual updates fail
                    time.sleep(5)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            with self._lock:
                self.cpu_usage_percent.set(cpu_percent)
                self.memory_usage_bytes.set(memory.used)
                self.memory_usage_percent.set(memory.percent)
                self.process_cpu_percent.set(process.cpu_percent())
                self.process_memory_bytes.set(process.memory_info().rss)
            
            # GPU metrics (if available)
            self._update_gpu_metrics()
            
        except Exception:
            # Silently continue if system metrics collection fails
            pass
    
    def _update_gpu_metrics(self):
        """Update GPU metrics if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_id = str(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    torch.cuda.memory_reserved(i)
                    utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                    
                    with self._lock:
                        self.gpu_memory_usage_bytes.labels(device_id=device_id).set(memory_allocated)
                        self.gpu_utilization_percent.labels(device_id=device_id).set(utilization)
        except ImportError:
            # PyTorch not available or no CUDA
            pass
        except Exception:
            # Continue if GPU metrics collection fails
            pass
    
    def record_inference(
        self,
        duration_seconds: float,
        model_version: str,
        device: str,
        batch_size: int,
        deterministic: bool = False,
        status: str = "success"
    ):
        """Record inference metrics."""
        batch_size_range = self._get_batch_size_range(batch_size)
        
        with self._lock:
            self.inference_latency.labels(
                model_version=model_version,
                device=device,
                batch_size_range=batch_size_range
            ).observe(duration_seconds)
            
            self.inference_requests_total.labels(
                model_version=model_version,
                device=device,
                status=status
            ).inc()
            
            self.batch_size_histogram.labels(model_version=model_version).observe(batch_size)
            
            if deterministic:
                self.deterministic_requests.labels(model_version=model_version).inc()
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record HTTP request metrics."""
        with self._lock:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).observe(duration_seconds)
    
    def record_error(self, error_type: str, endpoint: str, model_version: str):
        """Record error metrics."""
        with self._lock:
            self.errors_total.labels(
                error_type=error_type,
                endpoint=endpoint,
                model_version=model_version
            ).inc()
    
    def set_model_status(self, model_version: str, device: str, loaded: bool, warmed_up: bool = False):
        """Set model loading and warmup status."""
        with self._lock:
            self.model_loaded.labels(
                model_version=model_version,
                device=device
            ).set(1 if loaded else 0)
            
            self.model_warmed_up.labels(model_version=model_version).set(1 if warmed_up else 0)
    
    def record_model_loading(self, model_version: str, duration_seconds: float):
        """Record model loading duration."""
        with self._lock:
            self.model_loading_duration.labels(model_version=model_version).observe(duration_seconds)
    
    def record_model_warmup(self, model_version: str, duration_seconds: float):
        """Record model warmup duration."""
        with self._lock:
            self.model_warmup_duration.labels(model_version=model_version).observe(duration_seconds)
    
    def set_active_requests(self, count: int):
        """Set the number of active requests."""
        with self._lock:
            self.active_requests.set(count)
    
    def set_request_queue_length(self, length: int):
        """Set the request queue length."""
        with self._lock:
            self.request_queue_length.set(length)
    
    def set_service_uptime(self, uptime_seconds: float):
        """Set service uptime."""
        with self._lock:
            self.service_uptime_seconds.set(uptime_seconds)
    
    def set_service_startup_time(self, startup_timestamp: float):
        """Set service startup timestamp."""
        with self._lock:
            self.service_startup_time.set(startup_timestamp)
    
    def _get_batch_size_range(self, batch_size: int) -> str:
        """Convert batch size to range string for metrics labels."""
        if batch_size <= 1:
            return "1"
        elif batch_size <= 4:
            return "2-4"
        elif batch_size <= 8:
            return "5-8"
        elif batch_size <= 16:
            return "9-16"
        elif batch_size <= 32:
            return "17-32"
        elif batch_size <= 64:
            return "33-64"
        elif batch_size <= 128:
            return "65-128"
        else:
            return "129+"
    
    @contextmanager
    def inference_timer(self, model_version: str, device: str, batch_size: int, deterministic: bool = False):
        """Context manager for timing inference requests."""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.record_inference(
                duration_seconds=duration,
                model_version=model_version,
                device=device,
                batch_size=batch_size,
                deterministic=deterministic,
                status=status
            )
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def get_metrics_content_type(self) -> str:
        """Get the content type for metrics response."""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics(registry: Optional[CollectorRegistry] = None) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(registry)
    return _metrics_collector
