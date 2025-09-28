"""
Enhanced health check with detailed status information.

This module provides comprehensive health checking capabilities including:
- Model loading and readiness status
- System resource health
- Dependencies and external services
- Performance metrics validation
- Detailed status reporting
"""

import time
import psutil
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: float


class HealthChecker:
    """
    Comprehensive health checker for CarlaRL serving infrastructure.
    
    Provides detailed health status including:
    - Model loading and readiness
    - System resource utilization
    - Dependencies and external services
    - Performance metrics validation
    - Custom health checks
    """
    
    def __init__(self, app_state: Optional[Dict[str, Any]] = None):
        """Initialize health checker with optional app state reference."""
        self.app_state = app_state or {}
        self.checks: List[Tuple[str, Callable[[], HealthCheckResult]]] = []
        self.startup_time = time.time()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.add_check("model_loaded", self._check_model_loaded)
        self.add_check("model_warmed_up", self._check_model_warmed_up)
        self.add_check("system_resources", self._check_system_resources)
        self.add_check("memory_usage", self._check_memory_usage)
        self.add_check("cpu_usage", self._check_cpu_usage)
        self.add_check("disk_space", self._check_disk_space)
        self.add_check("gpu_availability", self._check_gpu_availability)
        self.add_check("service_uptime", self._check_service_uptime)
    
    def add_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Add a custom health check."""
        self.checks.append((name, check_func))
    
    def _check_model_loaded(self) -> HealthCheckResult:
        """Check if model is loaded and ready."""
        start_time = time.time()
        
        model_loaded = self.app_state.get("model_loaded", False)
        inference_engine = self.app_state.get("inference_engine")
        selected_version = self.app_state.get("selected_version", "unknown")
        
        if model_loaded and inference_engine is not None:
            status = HealthStatus.HEALTHY
            message = f"Model {selected_version} loaded and ready"
            details = {
                "model_loaded": True,
                "inference_engine_available": True,
                "model_version": selected_version
            }
        else:
            status = HealthStatus.UNHEALTHY
            message = "Model not loaded or inference engine unavailable"
            details = {
                "model_loaded": model_loaded,
                "inference_engine_available": inference_engine is not None,
                "model_version": selected_version
            }
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="model_loaded",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_model_warmed_up(self) -> HealthCheckResult:
        """Check if model has been warmed up."""
        start_time = time.time()
        
        warmup_completed = self.app_state.get("warmup_completed", False)
        model_loaded = self.app_state.get("model_loaded", False)
        
        if not model_loaded:
            status = HealthStatus.UNKNOWN
            message = "Model not loaded, warmup status unknown"
            details = {"model_loaded": False, "warmup_completed": False}
        elif warmup_completed:
            status = HealthStatus.HEALTHY
            message = "Model warmed up and ready for inference"
            details = {"warmup_completed": True}
        else:
            status = HealthStatus.DEGRADED
            message = "Model loaded but not warmed up (may have higher latency)"
            details = {"warmup_completed": False}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="model_warmed_up",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource health."""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Determine overall status based on resource usage
            if cpu_percent > 90 or memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically high"
            elif cpu_percent > 80 or memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = "System resources high"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
        
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check system resources: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage specifically."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Memory thresholds
            if memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "System memory critically high"
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
                message = "System memory high"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"
            
            details = {
                "system_memory_percent": memory.percent,
                "system_memory_used_gb": memory.used / (1024**3),
                "system_memory_available_gb": memory.available / (1024**3),
                "process_memory_rss_gb": process_memory.rss / (1024**3),
                "process_memory_vms_gb": process_memory.vms / (1024**3)
            }
        
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check memory usage: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="memory_usage",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage specifically."""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            # CPU thresholds
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage critically high"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = "CPU usage high"
            else:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal"
            
            details = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_avg_1min": load_avg[0] if load_avg else None,
                "load_avg_5min": load_avg[1] if load_avg else None,
                "load_avg_15min": load_avg[2] if load_avg else None
            }
        
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check CPU usage: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="cpu_usage",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        start_time = time.time()
        
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            # Disk space thresholds
            if free_percent < 5:
                status = HealthStatus.UNHEALTHY
                message = "Disk space critically low"
            elif free_percent < 10:
                status = HealthStatus.DEGRADED
                message = "Disk space low"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space adequate"
            
            details = {
                "free_percent": free_percent,
                "free_gb": disk_usage.free / (1024**3),
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3)
            }
        
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check disk space: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_gpu_availability(self) -> HealthCheckResult:
        """Check GPU availability and status."""
        start_time = time.time()
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                status = HealthStatus.HEALTHY
                message = "GPU not available, using CPU"
                details = {
                    "gpu_available": False,
                    "gpu_count": 0,
                    "device_type": "cpu"
                }
            else:
                gpu_count = torch.cuda.device_count()
                gpu_details = {}
                
                for i in range(gpu_count):
                    gpu_details[f"gpu_{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory
                    }
                
                status = HealthStatus.HEALTHY
                message = f"GPU available with {gpu_count} device(s)"
                details = {
                    "gpu_available": True,
                    "gpu_count": gpu_count,
                    "device_type": "cuda",
                    "gpu_details": gpu_details
                }
        
        except ImportError:
            status = HealthStatus.HEALTHY
            message = "PyTorch not available, GPU check skipped"
            details = {"gpu_available": False, "error": "PyTorch not installed"}
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check GPU status: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="gpu_availability",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    def _check_service_uptime(self) -> HealthCheckResult:
        """Check service uptime and startup time."""
        start_time = time.time()
        
        current_time = time.time()
        uptime_seconds = current_time - self.startup_time
        
        # Convert to human-readable format
        if uptime_seconds < 60:
            uptime_str = f"{uptime_seconds:.1f} seconds"
        elif uptime_seconds < 3600:
            uptime_str = f"{uptime_seconds/60:.1f} minutes"
        else:
            uptime_str = f"{uptime_seconds/3600:.1f} hours"
        
        status = HealthStatus.HEALTHY
        message = f"Service running for {uptime_str}"
        details = {
            "uptime_seconds": uptime_seconds,
            "uptime_human": uptime_str,
            "startup_timestamp": self.startup_time,
            "current_timestamp": current_time
        }
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="service_uptime",
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
    
    async def run_checks_async(self) -> List[HealthCheckResult]:
        """Run all health checks asynchronously."""
        tasks = []
        
        for name, check_func in self.checks:
            async def run_check(check_name: str, check_func: Callable[[], HealthCheckResult]):
                # Run CPU-bound check in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, check_func)
            
            tasks.append(run_check(name, check_func))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_name = self.checks[i][0]
                processed_results.append(HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(result)}",
                    details={"error": str(result)},
                    duration_ms=0.0,
                    timestamp=time.time()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def run_checks(self) -> List[HealthCheckResult]:
        """Run all health checks synchronously."""
        results = []
        
        for name, check_func in self.checks:
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                    duration_ms=0.0,
                    timestamp=time.time()
                ))
        
        return results
    
    def get_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall health status from individual check results."""
        if not results:
            return HealthStatus.UNKNOWN
        
        # Count statuses
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Determine overall status
        if status_counts.get(HealthStatus.UNHEALTHY, 0) > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        elif status_counts.get(HealthStatus.UNKNOWN, 0) > 0:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        start_time = time.time()
        results = self.run_checks()
        overall_status = self.get_overall_status(results)
        
        # Calculate total check duration
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Group results by status
        status_groups = {}
        for result in results:
            status = result.status.value
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append({
                "name": result.name,
                "message": result.message,
                "duration_ms": result.duration_ms
            })
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "total_duration_ms": total_duration_ms,
            "checks": {
                "total": len(results),
                "by_status": status_groups
            },
            "details": {
                result.name: {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "duration_ms": result.duration_ms
                }
                for result in results
            }
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(app_state: Optional[Dict[str, Any]] = None) -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(app_state)
    return _health_checker


def initialize_health_checker(app_state: Optional[Dict[str, Any]] = None) -> HealthChecker:
    """Initialize the global health checker."""
    global _health_checker
    _health_checker = HealthChecker(app_state)
    return _health_checker
