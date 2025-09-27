"""
Inference engine for CarlaRL Policy-as-a-Service.

This module provides high-performance inference with batch processing,
memory optimization, and performance monitoring capabilities.
"""

import time
import threading
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from .io_schemas import Observation, Action
from .model_loader import PolicyWrapper
from .preprocessing import FeaturePreprocessor, to_feature_matrix
from .exceptions import InferenceError
from .version import MODEL_VERSION, GIT_SHA


@dataclass
class InferenceMetrics:
    """Container for inference performance metrics."""
    
    total_requests: int = 0
    total_observations: int = 0
    total_inference_time_ms: float = 0.0
    total_preprocessing_time_ms: float = 0.0
    total_postprocessing_time_ms: float = 0.0
    
    # Latency statistics (P50, P95, P99)
    latency_history: List[float] = field(default_factory=list)
    max_history_size: int = 1000
    
    # Error tracking
    error_count: int = 0
    last_error_time: Optional[float] = None
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_request(
        self, 
        batch_size: int, 
        inference_time_ms: float,
        preprocessing_time_ms: float = 0.0,
        postprocessing_time_ms: float = 0.0
    ) -> None:
        """Add metrics for a completed request."""
        with self._lock:
            self.total_requests += 1
            self.total_observations += batch_size
            self.total_inference_time_ms += inference_time_ms
            self.total_preprocessing_time_ms += preprocessing_time_ms
            self.total_postprocessing_time_ms += postprocessing_time_ms
            
            # Track latency per observation
            latency_per_obs = inference_time_ms / batch_size if batch_size > 0 else inference_time_ms
            self.latency_history.append(latency_per_obs)
            
            # Maintain history size limit
            if len(self.latency_history) > self.max_history_size:
                self.latency_history = self.latency_history[-self.max_history_size:]
    
    def add_error(self) -> None:
        """Record an inference error."""
        with self._lock:
            self.error_count += 1
            self.last_error_time = time.time()
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles."""
        with self._lock:
            if not self.latency_history:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
            sorted_latencies = sorted(self.latency_history)
            n = len(sorted_latencies)
            
            return {
                "p50": sorted_latencies[int(n * 0.5)],
                "p95": sorted_latencies[int(n * 0.95)],
                "p99": sorted_latencies[int(n * 0.99)]
            }
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        with self._lock:
            if self.total_inference_time_ms == 0:
                return {"requests_per_second": 0.0, "observations_per_second": 0.0}
            
            total_time_seconds = self.total_inference_time_ms / 1000.0
            
            return {
                "requests_per_second": self.total_requests / total_time_seconds,
                "observations_per_second": self.total_observations / total_time_seconds
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_requests = 0
            self.total_observations = 0
            self.total_inference_time_ms = 0.0
            self.total_preprocessing_time_ms = 0.0
            self.total_postprocessing_time_ms = 0.0
            self.latency_history.clear()
            self.error_count = 0
            self.last_error_time = None


class InferenceCache:
    """Simple LRU cache for inference results."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[List[Action], float]] = {}
        self.access_order: List[str] = []
        self._lock = threading.Lock()
    
    def _make_key(self, observations: List[Observation], deterministic: bool) -> str:
        """Create cache key from observations and deterministic flag."""
        # Simple hash based on observation values
        obs_str = ""
        for obs in observations:
            obs_str += f"{obs.speed:.3f},{obs.steering:.3f},"
            obs_str += ",".join(f"{s:.3f}" for s in obs.sensors[:5])  # Limit sensor count for key
            obs_str += ";"
        
        return f"{hash(obs_str)}_{deterministic}"
    
    def get(self, observations: List[Observation], deterministic: bool) -> Optional[Tuple[List[Action], float]]:
        """Get cached result if available."""
        key = self._make_key(observations, deterministic)
        
        with self._lock:
            if key in self.cache:
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        
        return None
    
    def put(self, observations: List[Observation], deterministic: bool, result: Tuple[List[Action], float]) -> None:
        """Cache inference result."""
        key = self._make_key(observations, deterministic)
        
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.pop(0)
                self.cache.pop(oldest_key, None)
            
            # Add new entry
            self.cache[key] = result
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)


class InferenceEngine:
    """
    High-performance inference engine for RL policy models.
    
    Provides batch processing, memory optimization, caching, and performance monitoring.
    """
    
    def __init__(
        self,
        policy: PolicyWrapper,
        device: torch.device,
        preprocessor: Optional[FeaturePreprocessor] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        max_batch_size: int = 1000,
        enable_memory_pinning: bool = True
    ):
        self.policy = policy
        self.device = device
        self.preprocessor = preprocessor
        self.max_batch_size = max_batch_size
        self.enable_memory_pinning = enable_memory_pinning
        
        # Performance monitoring
        self.metrics = InferenceMetrics()
        
        # Caching
        self.cache = InferenceCache(cache_size) if enable_cache else None
        
        # Pre-allocated tensors for memory optimization
        self._preallocated_tensors: Dict[Tuple[int, int], torch.Tensor] = {}
        self._tensor_lock = threading.Lock()
        
        # Warmup state
        self._warmed_up = False
        
        # Version tracking
        self.model_version = MODEL_VERSION
        self.git_sha = GIT_SHA
        
        # Ensure policy is in eval mode
        self.policy.eval()
    
    def _get_or_create_tensor(self, batch_size: int, feature_dim: int) -> torch.Tensor:
        """Get or create pre-allocated tensor for given dimensions."""
        key = (batch_size, feature_dim)
        
        with self._tensor_lock:
            if key not in self._preallocated_tensors:
                tensor = torch.zeros(
                    batch_size, feature_dim,
                    device=self.device,
                    dtype=torch.float32
                )
                if self.enable_memory_pinning and self.device.type == 'cpu':
                    tensor = tensor.pin_memory()
                
                self._preallocated_tensors[key] = tensor
            
            return self._preallocated_tensors[key]
    
    def _preprocess_observations(self, observations: List[Observation]) -> Tuple[torch.Tensor, float]:
        """
        Preprocess observations to feature tensor.
        
        Returns:
            Tuple of (feature_tensor, preprocessing_time_ms)
        """
        start_time = time.perf_counter()
        
        try:
            if self.preprocessor:
                features = self.preprocessor.transform(observations)
            else:
                features = to_feature_matrix(observations)
            
            # Convert to tensor
            if isinstance(features, np.ndarray):
                # Try to reuse pre-allocated tensor
                batch_size, feature_dim = features.shape
                if batch_size <= self.max_batch_size:
                    try:
                        tensor = self._get_or_create_tensor(batch_size, feature_dim)
                        tensor[:batch_size, :feature_dim] = torch.from_numpy(features)
                        input_tensor = tensor[:batch_size, :feature_dim]
                    except Exception:
                        # Fallback to direct tensor creation
                        input_tensor = torch.from_numpy(features).to(self.device, dtype=torch.float32)
                else:
                    input_tensor = torch.from_numpy(features).to(self.device, dtype=torch.float32)
            else:
                input_tensor = features.to(self.device, dtype=torch.float32)
            
            preprocessing_time_ms = (time.perf_counter() - start_time) * 1000.0
            return input_tensor, preprocessing_time_ms
            
        except Exception as e:
            raise InferenceError(
                f"Preprocessing failed: {str(e)}",
                details={
                    "num_observations": len(observations),
                    "preprocessor_type": type(self.preprocessor).__name__ if self.preprocessor else "minimal"
                }
            )
    
    def _postprocess_outputs(self, model_outputs: torch.Tensor) -> Tuple[List[Action], float]:
        """
        Postprocess model outputs to Action objects.
        
        Returns:
            Tuple of (actions, postprocessing_time_ms)
        """
        start_time = time.perf_counter()
        
        try:
            # Convert to numpy for processing
            if model_outputs.device != torch.device('cpu'):
                outputs_np = model_outputs.detach().cpu().numpy()
            else:
                outputs_np = model_outputs.detach().numpy()
            
            actions = []
            for output in outputs_np:
                # Assume output format: [throttle, brake, steer]
                # Apply appropriate activation functions and clipping
                throttle = float(np.clip(output[0], 0.0, 1.0))
                brake = float(np.clip(output[1], 0.0, 1.0))
                steer = float(np.clip(output[2], -1.0, 1.0))
                
                actions.append(Action(throttle=throttle, brake=brake, steer=steer))
            
            postprocessing_time_ms = (time.perf_counter() - start_time) * 1000.0
            return actions, postprocessing_time_ms
            
        except Exception as e:
            raise InferenceError(
                f"Postprocessing failed: {str(e)}",
                details={
                    "output_shape": list(model_outputs.shape),
                    "output_device": str(model_outputs.device)
                }
            )
    
    def predict(
        self,
        observations: List[Observation],
        deterministic: bool = False,
        use_cache: bool = True
    ) -> Tuple[List[Action], float]:
        """
        Perform inference on batch of observations.
        
        Args:
            observations: List of observations to process
            deterministic: Whether to use deterministic inference
            use_cache: Whether to use result caching
            
        Returns:
            Tuple of (actions, total_time_ms)
            
        Raises:
            InferenceError: If inference fails
        """
        if not observations:
            raise InferenceError("Cannot perform inference on empty observation list")
        
        batch_size = len(observations)
        if batch_size > self.max_batch_size:
            raise InferenceError(
                f"Batch size {batch_size} exceeds maximum {self.max_batch_size}",
                details={"max_batch_size": self.max_batch_size}
            )
        
        total_start_time = time.perf_counter()
        
        try:
            # Check cache first
            if self.cache and use_cache:
                cached_result = self.cache.get(observations, deterministic)
                if cached_result is not None:
                    actions, cached_time = cached_result
                    # Return cached result with minimal timing overhead
                    total_time_ms = (time.perf_counter() - total_start_time) * 1000.0
                    return actions, total_time_ms
            
            # Preprocess observations
            input_tensor, preprocessing_time_ms = self._preprocess_observations(observations)
            
            # Perform inference
            inference_start_time = time.perf_counter()
            
            with torch.no_grad():
                model_outputs = self.policy(input_tensor, deterministic=deterministic)
            
            inference_time_ms = (time.perf_counter() - inference_start_time) * 1000.0
            
            # Postprocess outputs
            actions, postprocessing_time_ms = self._postprocess_outputs(model_outputs)
            
            total_time_ms = (time.perf_counter() - total_start_time) * 1000.0
            
            # Cache result if enabled
            if self.cache and use_cache:
                self.cache.put(observations, deterministic, (actions, total_time_ms))
            
            # Update metrics
            self.metrics.add_request(
                batch_size, 
                inference_time_ms,
                preprocessing_time_ms,
                postprocessing_time_ms
            )
            
            return actions, total_time_ms
            
        except Exception as e:
            self.metrics.add_error()
            
            if isinstance(e, InferenceError):
                raise
            else:
                raise InferenceError(
                    f"Inference failed: {str(e)}",
                    details={
                        "batch_size": batch_size,
                        "deterministic": deterministic,
                        "device": str(self.device)
                    }
                )
    
    def warmup(self, warmup_batches: int = 3, warmup_batch_size: int = 1) -> float:
        """
        Warm up the inference engine with dummy predictions.
        
        Args:
            warmup_batches: Number of warmup batches to run
            warmup_batch_size: Size of each warmup batch
            
        Returns:
            Total warmup time in milliseconds
        """
        start_time = time.perf_counter()
        
        try:
            # Create dummy observations
            dummy_obs = Observation(
                speed=25.0,
                steering=0.0,
                sensors=[0.5] * 5
            )
            
            for _ in range(warmup_batches):
                dummy_batch = [dummy_obs] * warmup_batch_size
                self.predict(dummy_batch, deterministic=True, use_cache=False)
            
            self._warmed_up = True
            warmup_time_ms = (time.perf_counter() - start_time) * 1000.0
            
            return warmup_time_ms
            
        except Exception as e:
            raise InferenceError(
                f"Warmup failed: {str(e)}",
                details={
                    "warmup_batches": warmup_batches,
                    "warmup_batch_size": warmup_batch_size
                }
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        latency_stats = self.metrics.get_latency_percentiles()
        throughput_stats = self.metrics.get_throughput_stats()
        
        return {
            "model_version": self.model_version,
            "git_sha": self.git_sha,
            "device": str(self.device),
            "warmed_up": self._warmed_up,
            "total_requests": self.metrics.total_requests,
            "total_observations": self.metrics.total_observations,
            "error_count": self.metrics.error_count,
            "cache_size": self.cache.size() if self.cache else 0,
            "latency_ms": latency_stats,
            "throughput": throughput_stats,
            "timing_breakdown_ms": {
                "total_inference": self.metrics.total_inference_time_ms,
                "total_preprocessing": self.metrics.total_preprocessing_time_ms,
                "total_postprocessing": self.metrics.total_postprocessing_time_ms
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics.reset()
        if self.cache:
            self.cache.clear()
    
    def set_deterministic_mode(self, deterministic: bool) -> None:
        """Set global deterministic mode for reproducible inference."""
        if deterministic:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    def optimize_for_inference(self) -> None:
        """Apply inference-specific optimizations."""
        # Set model to eval mode
        self.policy.eval()
        
        # Enable inference optimizations
        torch.backends.cudnn.benchmark = True
        
        # Try to optimize model with TorchScript if not already optimized
        try:
            if not isinstance(self.policy.model, torch.jit.ScriptModule):
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 5, device=self.device)  # Adjust size as needed
                traced_model = torch.jit.trace(self.policy.model, dummy_input)
                self.policy.model = traced_model
                self.policy.model_type = "torchscript"
        except Exception:
            # Optimization failed, continue with original model
            pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            "preallocated_tensors": len(self._preallocated_tensors),
            "cache_enabled": self.cache is not None,
            "cache_size": self.cache.size() if self.cache else 0
        }
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            stats.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
                "gpu_memory_cached_mb": torch.cuda.memory_reserved(self.device) / 1024**2
            })
        
        return stats
