"""
Unit tests for structured logging module.
"""

import json
import logging
from unittest.mock import patch, MagicMock

from src.monitoring.logging import (
    StructuredLogger,
    JSONFormatter,
    get_logger,
    set_log_level,
    configure_logging
)


class TestStructuredLogger:
    """Test cases for StructuredLogger class."""
    
    def test_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_logger")
        
        assert logger.logger.name == "test_logger"
        assert logger.include_correlation_id is True
        assert logger.include_timestamp is True
        assert logger.include_level is True
        assert logger.include_logger_name is True
    
    def test_initialization_with_options(self):
        """Test logger initialization with custom options."""
        logger = StructuredLogger(
            "test_logger",
            level=logging.DEBUG,
            include_correlation_id=False,
            include_timestamp=False,
            include_level=False,
            include_logger_name=False
        )
        
        assert logger.include_correlation_id is False
        assert logger.include_timestamp is False
        assert logger.include_level is False
        assert logger.include_logger_name is False
    
    def test_correlation_id_management(self):
        """Test correlation ID management."""
        logger = StructuredLogger("test_logger")
        
        # Test setting correlation ID
        correlation_id = "test-correlation-123"
        logger.set_correlation_id(correlation_id)
        assert logger.get_correlation_id() == correlation_id
        
        # Test clearing correlation ID
        logger.clear_correlation_id()
        assert logger.get_correlation_id() is None
    
    def test_log_methods(self):
        """Test basic logging methods."""
        logger = StructuredLogger("test_logger")
        
        # Test that logging methods don't raise exceptions
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.critical("Test critical message")
        
        assert True  # No exceptions raised
    
    def test_log_with_kwargs(self):
        """Test logging with additional keyword arguments."""
        logger = StructuredLogger("test_logger")
        
        # Test logging with additional fields
        logger.info("Test message", field1="value1", field2=42, field3=True)
        
        assert True  # No exceptions raised
    
    def test_log_inference(self):
        """Test inference logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_inference(
            model_version="v1.0.0",
            device="cpu",
            batch_size=4,
            duration_ms=50.0,
            deterministic=True,
            status="success"
        )
        
        assert True  # No exceptions raised
    
    def test_log_request(self):
        """Test HTTP request logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_request(
            method="POST",
            endpoint="/predict",
            status_code=200,
            duration_ms=100.0,
            user_agent="test-agent"
        )
        
        assert True  # No exceptions raised
    
    def test_log_error(self):
        """Test error logging."""
        logger = StructuredLogger("test_logger")
        
        # Test error logging without exception
        logger.log_error(
            error_type="TestError",
            error_message="Test error message",
            endpoint="/predict",
            model_version="v1.0.0"
        )
        
        # Test error logging with exception
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_error(
                error_type="ValueError",
                error_message="Test error with exception",
                exception=e
            )
        
        assert True  # No exceptions raised
    
    def test_log_model_loading(self):
        """Test model loading logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_model_loading(
            model_version="v1.0.0",
            device="cpu",
            duration_ms=2500.0,
            status="success"
        )
        
        assert True  # No exceptions raised
    
    def test_log_model_warmup(self):
        """Test model warmup logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_model_warmup(
            model_version="v1.0.0",
            duration_ms=500.0,
            status="success"
        )
        
        assert True  # No exceptions raised
    
    def test_log_health_check(self):
        """Test health check logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_health_check(
            status="healthy",
            checks={"model_loaded": True, "memory_ok": True},
            duration_ms=10.0
        )
        
        assert True  # No exceptions raised
    
    def test_log_system_metrics(self):
        """Test system metrics logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_system_metrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_bytes=1024**3,
            gpu_metrics={"gpu_0": {"utilization": 30.0}}
        )
        
        assert True  # No exceptions raised
    
    def test_log_performance_alert(self):
        """Test performance alert logging."""
        logger = StructuredLogger("test_logger")
        
        logger.log_performance_alert(
            metric_name="inference_latency",
            current_value=100.0,
            threshold=50.0,
            severity="warning"
        )
        
        assert True  # No exceptions raised


class TestJSONFormatter:
    """Test cases for JSONFormatter class."""
    
    def test_initialization(self):
        """Test formatter initialization."""
        formatter = JSONFormatter()
        
        assert formatter.include_correlation_id is True
        assert formatter.include_timestamp is True
        assert formatter.include_level is True
        assert formatter.include_logger_name is True
    
    def test_initialization_with_options(self):
        """Test formatter initialization with custom options."""
        formatter = JSONFormatter(
            include_correlation_id=False,
            include_timestamp=False,
            include_level=False,
            include_logger_name=False
        )
        
        assert formatter.include_correlation_id is False
        assert formatter.include_timestamp is False
        assert formatter.include_level is False
        assert formatter.include_logger_name is False
    
    def test_format_record(self):
        """Test formatting log record."""
        formatter = JSONFormatter()
        
        # Create a mock log record
        record = MagicMock()
        record.getMessage.return_value = "Test message"
        record.levelname = "INFO"
        record.name = "test_logger"
        record.created = 1234567890.0
        record.exc_info = None
        record.correlation_id = "test-correlation-123"
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse the JSON
        log_entry = json.loads(formatted)
        
        assert log_entry["message"] == "Test message"
        assert log_entry["level"] == "INFO"
        assert log_entry["logger"] == "test_logger"
        assert log_entry["correlation_id"] == "test-correlation-123"
        assert "timestamp" in log_entry
    
    def test_format_record_with_exception(self):
        """Test formatting log record with exception."""
        formatter = JSONFormatter()
        
        # Create a mock log record with exception
        record = MagicMock()
        record.getMessage.return_value = "Test message"
        record.levelname = "ERROR"
        record.name = "test_logger"
        record.created = 1234567890.0
        record.exc_info = (ValueError, ValueError("Test error"), None)
        record.correlation_id = None
        
        # Mock the formatException method
        formatter.formatException = MagicMock(return_value="ValueError: Test error")
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse the JSON
        log_entry = json.loads(formatted)
        
        assert log_entry["message"] == "Test message"
        assert log_entry["level"] == "ERROR"
        assert "exception" in log_entry
    
    def test_format_record_fallback(self):
        """Test formatter fallback on JSON parsing error."""
        formatter = JSONFormatter()
        
        # Create a mock log record that will cause JSON parsing to fail
        record = MagicMock()
        record.getMessage.return_value = "Test message"
        record.levelname = "INFO"
        record.name = "test_logger"
        record.created = 1234567890.0
        record.exc_info = None
        record.correlation_id = None
        
        # Mock the _create_log_entry method to raise an exception
        with patch.object(formatter, '_create_log_entry', side_effect=Exception("Test error")):
            formatted = formatter.format(record)
            
            # Should fall back to simple format
            assert "INFO: Test message" in formatted


class TestGlobalFunctions:
    """Test cases for global functions."""
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        
        # Should return the same instance
        assert logger1 is logger2
    
    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        # Should return different instances
        assert logger1 is not logger2
    
    def test_set_log_level(self):
        """Test setting log level."""
        logger = get_logger("test_logger")
        
        # Set log level
        set_log_level("DEBUG")
        
        # Check that level was set
        assert logger.logger.level == logging.DEBUG
    
    def test_set_log_level_numeric(self):
        """Test setting log level with numeric value."""
        logger = get_logger("test_logger")
        
        # Set log level with numeric value
        set_log_level(logging.WARNING)
        
        # Check that level was set
        assert logger.logger.level == logging.WARNING
    
    def test_configure_logging_json(self):
        """Test configuring logging with JSON format."""
        configure_logging(level="INFO", format_type="json")
        
        # Should not raise an exception
        assert True
    
    def test_configure_logging_standard(self):
        """Test configuring logging with standard format."""
        configure_logging(level="INFO", format_type="standard")
        
        # Should not raise an exception
        assert True
