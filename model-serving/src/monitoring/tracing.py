"""
Distributed tracing for request flow analysis.

This module provides comprehensive tracing capabilities including:
- Request correlation and tracing
- Performance span tracking
- Error and exception tracing
- Custom span creation and management
- Integration with structured logging
"""

import time
import uuid
from typing import Any, Dict, Optional, List, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
import threading
from enum import Enum


class SpanStatus(Enum):
    """Span execution status."""
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Span:
    """Represents a tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.SUCCESS
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Exception] = None
    
    def finish(self, status: SpanStatus = SpanStatus.SUCCESS, error: Optional[Exception] = None):
        """Finish the span and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        if error:
            self.error = error
            self.add_log("error", {"error": str(error), "error_type": type(error).__name__})
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def add_tags(self, tags: Dict[str, Any]):
        """Add multiple tags to the span."""
        self.tags.update(tags)
    
    def add_log(self, event: str, fields: Dict[str, Any]):
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": time.time(),
            "event": event,
            "fields": fields
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs,
            "error": str(self.error) if self.error else None
        }


class TracingMiddleware:
    """
    Distributed tracing middleware for request flow analysis.
    
    Provides comprehensive tracing capabilities including:
    - Automatic request tracing
    - Performance span tracking
    - Error and exception tracing
    - Custom span creation
    - Integration with logging
    """
    
    def __init__(self, service_name: str = "carla-rl-serving"):
        """Initialize tracing middleware."""
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
        
        # Context variables for current span
        self.current_trace_id: ContextVar[Optional[str]] = ContextVar('current_trace_id', default=None)
        self.current_span_id: ContextVar[Optional[str]] = ContextVar('current_span_id', default=None)
    
    def generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())
    
    def generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())
    
    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new tracing span."""
        if trace_id is None:
            trace_id = self.current_trace_id.get() or self.generate_trace_id()
        
        span_id = self.generate_span_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        # Add service name tag
        span.add_tag("service.name", self.service_name)
        
        # Store span
        with self._lock:
            self.spans[span_id] = span
        
        # Update context
        self.current_trace_id.set(trace_id)
        self.current_span_id.set(span_id)
        
        return span
    
    def finish_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[Exception] = None
    ) -> Optional[Span]:
        """Finish a tracing span."""
        with self._lock:
            span = self.spans.get(span_id)
            if span:
                span.finish(status, error)
                return span
        return None
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        with self._lock:
            return self.spans.get(span_id)
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        current_span_id = self.current_span_id.get()
        if current_span_id:
            return self.get_span(current_span_id)
        return None
    
    def get_trace_spans(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return [span for span in self.spans.values() if span.trace_id == trace_id]
    
    def clear_trace(self, trace_id: str):
        """Clear all spans for a trace."""
        with self._lock:
            spans_to_remove = [span_id for span_id, span in self.spans.items() if span.trace_id == trace_id]
            for span_id in spans_to_remove:
                del self.spans[span_id]
    
    def clear_old_spans(self, max_age_seconds: int = 3600):
        """Clear spans older than specified age."""
        current_time = time.time()
        with self._lock:
            spans_to_remove = [
                span_id for span_id, span in self.spans.items()
                if current_time - span.start_time > max_age_seconds
            ]
            for span_id in spans_to_remove:
                del self.spans[span_id]
    
    @contextmanager
    def span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for creating and managing spans."""
        span = self.start_span(operation_name, trace_id, parent_span_id, tags)
        try:
            yield span
        except Exception as e:
            span.finish(SpanStatus.ERROR, e)
            raise
        else:
            span.finish(SpanStatus.SUCCESS)
    
    def trace_request(
        self,
        method: str,
        endpoint: str,
        request_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs
    ) -> Span:
        """Start a request tracing span."""
        tags = {
            "http.method": method,
            "http.endpoint": endpoint,
            "http.request_id": request_id or str(uuid.uuid4()),
        }
        
        if user_agent:
            tags["http.user_agent"] = user_agent
        
        tags.update(kwargs)
        
        return self.start_span(f"{method} {endpoint}", tags=tags)
    
    def trace_inference(
        self,
        model_version: str,
        device: str,
        batch_size: int,
        deterministic: bool = False,
        **kwargs
    ) -> Span:
        """Start an inference tracing span."""
        tags = {
            "model.version": model_version,
            "model.device": device,
            "inference.batch_size": batch_size,
            "inference.deterministic": deterministic,
        }
        
        tags.update(kwargs)
        
        return self.start_span("inference", tags=tags)
    
    def trace_model_loading(
        self,
        model_version: str,
        device: str,
        **kwargs
    ) -> Span:
        """Start a model loading tracing span."""
        tags = {
            "model.version": model_version,
            "model.device": device,
            "operation": "model_loading",
        }
        
        tags.update(kwargs)
        
        return self.start_span("model_loading", tags=tags)
    
    def trace_model_warmup(
        self,
        model_version: str,
        **kwargs
    ) -> Span:
        """Start a model warmup tracing span."""
        tags = {
            "model.version": model_version,
            "operation": "model_warmup",
        }
        
        tags.update(kwargs)
        
        return self.start_span("model_warmup", tags=tags)
    
    def trace_health_check(
        self,
        check_name: str,
        **kwargs
    ) -> Span:
        """Start a health check tracing span."""
        tags = {
            "health_check.name": check_name,
            "operation": "health_check",
        }
        
        tags.update(kwargs)
        
        return self.start_span(f"health_check.{check_name}", tags=tags)
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of a trace."""
        spans = self.get_trace_spans(trace_id)
        
        if not spans:
            return {"trace_id": trace_id, "spans": [], "summary": {}}
        
        # Calculate trace statistics
        total_duration = max(span.duration_ms or 0 for span in spans)
        span_count = len(spans)
        error_count = sum(1 for span in spans if span.status == SpanStatus.ERROR)
        
        # Group spans by operation
        operations = {}
        for span in spans:
            op_name = span.operation_name
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(span)
        
        return {
            "trace_id": trace_id,
            "spans": [span.to_dict() for span in spans],
            "summary": {
                "total_duration_ms": total_duration,
                "span_count": span_count,
                "error_count": error_count,
                "success_rate": (span_count - error_count) / span_count if span_count > 0 else 0,
                "operations": {
                    op_name: {
                        "count": len(op_spans),
                        "total_duration_ms": sum(span.duration_ms or 0 for span in op_spans),
                        "avg_duration_ms": sum(span.duration_ms or 0 for span in op_spans) / len(op_spans),
                        "error_count": sum(1 for span in op_spans if span.status == SpanStatus.ERROR)
                    }
                    for op_name, op_spans in operations.items()
                }
            }
        }
    
    def export_traces(self, trace_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Export traces in a format suitable for external systems."""
        if trace_ids:
            traces = [self.get_trace_summary(trace_id) for trace_id in trace_ids]
        else:
            # Export all traces
            all_trace_ids = list(set(span.trace_id for span in self.spans.values()))
            traces = [self.get_trace_summary(trace_id) for trace_id in all_trace_ids]
        
        return traces


# Global tracer instance
_tracer: Optional[TracingMiddleware] = None


def get_tracer(service_name: str = "carla-rl-serving") -> TracingMiddleware:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = TracingMiddleware(service_name)
    return _tracer


def initialize_tracer(service_name: str = "carla-rl-serving") -> TracingMiddleware:
    """Initialize the global tracer."""
    global _tracer
    _tracer = TracingMiddleware(service_name)
    return _tracer
