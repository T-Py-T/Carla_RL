"""
Structured JSON logging with correlation IDs.

This module provides comprehensive logging capabilities including:
- Structured JSON logging with consistent schema
- Request correlation IDs for tracing
- Log level configuration and filtering
- Performance and error logging
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
import threading
from contextvars import ContextVar


class StructuredLogger:
    """
    Structured JSON logger with correlation ID support.
    
    Provides consistent JSON logging format with:
    - Timestamp in ISO format
    - Correlation ID for request tracing
    - Structured fields for different log types
    - Performance metrics logging
    - Error context and stack traces
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        include_correlation_id: bool = True,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger_name: bool = True
    ):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Configuration
        self.include_correlation_id = include_correlation_id
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger_name = include_logger_name
        
        # Context variable for correlation ID
        self.correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
        
        # Thread-local storage for request context
        self._local = threading.local()
        
        # Set up JSON formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter(
                include_correlation_id=include_correlation_id,
                include_timestamp=include_timestamp,
                include_level=include_level,
                include_logger_name=include_logger_name
            ))
            self.logger.addHandler(handler)
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self.correlation_id.set(correlation_id)
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return self.correlation_id.get() or getattr(self._local, 'correlation_id', None)
    
    def clear_correlation_id(self) -> None:
        """Clear correlation ID from current context."""
        self.correlation_id.set(None)
        if hasattr(self._local, 'correlation_id'):
            delattr(self._local, 'correlation_id')
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create structured log entry."""
        entry = {
            "message": message,
            "level": level,
        }
        
        if self.include_timestamp:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if self.include_logger_name:
            entry["logger"] = self.logger.name
        
        if self.include_correlation_id:
            correlation_id = self.get_correlation_id()
            if correlation_id:
                entry["correlation_id"] = correlation_id
        
        # Add any additional fields
        entry.update(kwargs)
        
        return entry
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        entry = self._create_log_entry("INFO", message, **kwargs)
        self.logger.info(json.dumps(entry))
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        entry = self._create_log_entry("DEBUG", message, **kwargs)
        self.logger.debug(json.dumps(entry))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        entry = self._create_log_entry("WARNING", message, **kwargs)
        self.logger.warning(json.dumps(entry))
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self.logger.error(json.dumps(entry))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        entry = self._create_log_entry("CRITICAL", message, **kwargs)
        self.logger.critical(json.dumps(entry))
    
    def log_inference(
        self,
        model_version: str,
        device: str,
        batch_size: int,
        duration_ms: float,
        deterministic: bool = False,
        status: str = "success",
        **kwargs
    ) -> None:
        """Log inference request with performance metrics."""
        self.info(
            "Inference request completed",
            event_type="inference",
            model_version=model_version,
            device=device,
            batch_size=batch_size,
            duration_ms=duration_ms,
            deterministic=deterministic,
            status=status,
            **kwargs
        )
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        user_agent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log HTTP request with performance metrics."""
        self.info(
            "HTTP request completed",
            event_type="http_request",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration_ms=duration_ms,
            user_agent=user_agent,
            **kwargs
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        endpoint: Optional[str] = None,
        model_version: Optional[str] = None,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> None:
        """Log error with context and stack trace."""
        error_data = {
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
        }
        
        if endpoint:
            error_data["endpoint"] = endpoint
        
        if model_version:
            error_data["model_version"] = model_version
        
        if exception:
            error_data["exception_type"] = type(exception).__name__
            error_data["exception_module"] = getattr(exception, '__module__', 'unknown')
        
        error_data.update(kwargs)
        
        self.error("Error occurred", **error_data)
    
    def log_model_loading(
        self,
        model_version: str,
        device: str,
        duration_ms: float,
        status: str = "success",
        **kwargs
    ) -> None:
        """Log model loading event."""
        self.info(
            "Model loading completed",
            event_type="model_loading",
            model_version=model_version,
            device=device,
            duration_ms=duration_ms,
            status=status,
            **kwargs
        )
    
    def log_model_warmup(
        self,
        model_version: str,
        duration_ms: float,
        status: str = "success",
        **kwargs
    ) -> None:
        """Log model warmup event."""
        self.info(
            "Model warmup completed",
            event_type="model_warmup",
            model_version=model_version,
            duration_ms=duration_ms,
            status=status,
            **kwargs
        )
    
    def log_health_check(
        self,
        status: str,
        checks: Dict[str, Any],
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log health check event."""
        self.info(
            "Health check completed",
            event_type="health_check",
            status=status,
            checks=checks,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        memory_bytes: int,
        gpu_metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log system resource metrics."""
        metrics_data = {
            "event_type": "system_metrics",
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_bytes": memory_bytes,
        }
        
        if gpu_metrics:
            metrics_data["gpu_metrics"] = gpu_metrics
        
        metrics_data.update(kwargs)
        
        self.debug("System metrics collected", **metrics_data)
    
    def log_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        severity: str = "warning",
        **kwargs
    ) -> None:
        """Log performance alert."""
        self.warning(
            "Performance alert triggered",
            event_type="performance_alert",
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            severity=severity,
            **kwargs
        )


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(
        self,
        include_correlation_id: bool = True,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger_name: bool = True
    ):
        """Initialize JSON formatter."""
        super().__init__()
        self.include_correlation_id = include_correlation_id
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger_name = include_logger_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        try:
            # Parse existing JSON if it's already structured
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                try:
                    return record.msg  # Already JSON formatted
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Create structured log entry
            log_entry = {
                "message": record.getMessage(),
                "level": record.levelname,
            }
            
            if self.include_timestamp:
                log_entry["timestamp"] = datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat()
            
            if self.include_logger_name:
                log_entry["logger"] = record.name
            
            if self.include_correlation_id and hasattr(record, 'correlation_id'):
                log_entry["correlation_id"] = record.correlation_id
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add any additional attributes
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }:
                    log_entry[key] = value
            
            return json.dumps(log_entry, default=str)
        
        except Exception:
            # Fallback to simple format if JSON formatting fails
            return f"{record.levelname}: {record.getMessage()}"


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str = "carla_rl", **kwargs) -> StructuredLogger:
    """Get or create a structured logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, **kwargs)
    return _loggers[name]


def set_log_level(level: Union[str, int]) -> None:
    """Set log level for all loggers."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    for logger in _loggers.values():
        logger.logger.setLevel(level)


def configure_logging(
    level: Union[str, int] = "INFO",
    format_type: str = "json",
    include_correlation_id: bool = True
) -> None:
    """Configure logging for the application."""
    if format_type == "json":
        # Use structured JSON logging
        get_logger(
            level=level,
            include_correlation_id=include_correlation_id
        )
    else:
        # Use standard Python logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
