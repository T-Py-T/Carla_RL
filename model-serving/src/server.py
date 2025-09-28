"""
FastAPI server for CarlaRL Policy-as-a-Service.

This module implements the main FastAPI application with all endpoints,
middleware, and error handling for the serving infrastructure.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .exceptions import (
    EXCEPTION_HANDLERS,
    ServiceUnavailableError,
)
from .io_schemas import (
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    VersionInfo,
    VersionsResponse,
    WarmupResponse,
)
from .version import APP_NAME, GIT_SHA, MODEL_NAME, MODEL_VERSION
from .monitoring import (
    get_metrics_collector,
    get_logger,
    get_health_checker,
    get_tracer,
    initialize_metrics,
    configure_logging
)

# Global state for model and inference engine
app_state: dict[str, Any] = {
    "model_loaded": False,
    "inference_engine": None,
    "startup_time": None,
    "warmup_completed": False,
}


async def get_inference_engine():
    """Dependency to get the inference engine with validation."""
    if not app_state["model_loaded"] or app_state["inference_engine"] is None:
        raise ServiceUnavailableError(
            message="Model not loaded or service not ready",
            details={"model_loaded": app_state["model_loaded"]},
        )
    return app_state["inference_engine"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown."""
    # Startup
    app_state["startup_time"] = time.time()
    
    # Initialize monitoring
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    initialize_metrics()
    logger = get_logger("carla_rl_server")
    metrics = get_metrics_collector()
    get_health_checker(app_state)
    tracer = get_tracer("carla-rl-serving")
    
    logger.info("Starting CarlaRL Policy Service", event_type="startup")

    try:
        # Import here to avoid circular imports
        from pathlib import Path

        import torch

        from .inference import InferenceEngine
        from .model_loader import load_artifacts
        from .versioning import VersionSelectionStrategy
        from .versioning.version_selector import get_version_from_environment

        # Configuration from environment
        artifacts_root = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
        use_gpu = os.getenv("USE_GPU", "0") == "1"
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # Select version using intelligent version selection
        print(f"Discovering model versions in: {artifacts_root}")
        selected_version = None
        
        try:
            selected_version = get_version_from_environment(
                artifacts_root,
                env_var="MODEL_VERSION",
                fallback_strategy=VersionSelectionStrategy.LATEST_STABLE
            )
            
            if selected_version is None:
                raise Exception("No suitable model version found")
            
            artifact_dir = artifacts_root / str(selected_version)
            
            print(f"Selected model version: {selected_version}")
            print(f"Loading model artifacts from: {artifact_dir}")
            print(f"Using device: {device}")

        except Exception as e:
            print(f"Version selection failed: {str(e)}")
            # Fallback to old behavior for backward compatibility
            artifact_dir = artifacts_root / MODEL_VERSION
            selected_version = MODEL_VERSION
            print(f"Falling back to static version: {MODEL_VERSION}")
            print(f"Loading model artifacts from: {artifact_dir}")

        # Load model and preprocessor with tracing
        with tracer.trace_model_loading(str(selected_version), str(device)):
            start_time = time.time()
            policy, preprocessor = load_artifacts(artifact_dir, device)
            loading_duration = (time.time() - start_time) * 1000
            
            # Record model loading metrics
            metrics.record_model_loading(str(selected_version), loading_duration / 1000)
            logger.log_model_loading(
                model_version=str(selected_version),
                device=str(device),
                duration_ms=loading_duration,
                status="success"
            )

        # Initialize inference engine
        app_state["inference_engine"] = InferenceEngine(policy, device, preprocessor)
        app_state["model_loaded"] = True
        app_state["selected_version"] = str(selected_version)
        
        # Set model status metrics
        metrics.set_model_status(str(selected_version), str(device), True, False)
        
        # Set service startup time
        metrics.set_service_startup_time(app_state["startup_time"])

        logger.info(
            "Model loaded successfully",
            event_type="model_loaded",
            model_name=MODEL_NAME,
            version=str(selected_version),
            device=str(device),
            loading_duration_ms=loading_duration
        )

    except Exception as e:
        logger.error(
            "Failed to load model",
            event_type="model_loading_error",
            error=str(e),
            error_type=type(e).__name__
        )
        # Don't raise here - let the service start but mark as unavailable
        app_state["model_loaded"] = False
        app_state["inference_engine"] = None

    yield

    # Shutdown
    logger.info("Shutting down CarlaRL Policy Service", event_type="shutdown")
    app_state["inference_engine"] = None
    app_state["model_loaded"] = False


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=MODEL_VERSION,
    description="High-performance serving infrastructure for CarlaRL policies",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(","))


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Get monitoring components
    logger = get_logger("carla_rl_server")
    metrics = get_metrics_collector()
    tracer = get_tracer("carla-rl-serving")
    
    # Set correlation ID for logging
    logger.set_correlation_id(request_id)
    
    # Start request tracing
    span = tracer.trace_request(
        method=request.method,
        endpoint=request.url.path,
        request_id=request_id,
        user_agent=request.headers.get("user-agent")
    )
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record request metrics
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration_seconds=process_time
        )
        
        # Log request
        logger.log_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration_ms=process_time * 1000,
            user_agent=request.headers.get("user-agent")
        )
        
        # Finish tracing span
        tracer.finish_span(span.span_id)
        
    except Exception as e:
        process_time = time.time() - start_time
        
        # Record error metrics
        metrics.record_error(
            error_type=type(e).__name__,
            endpoint=request.url.path,
            model_version=app_state.get("selected_version", "unknown")
        )
        
        # Log error
        logger.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            endpoint=request.url.path,
            model_version=app_state.get("selected_version", "unknown"),
            exception=e
        )
        
        # Finish tracing span with error
        tracer.finish_span(span.span_id, status="error", error=e)
        
        raise
    finally:
        # Clear correlation ID
        logger.clear_correlation_id()

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Register exception handlers
for exception_type, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exception_type, handler)


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, version information, and basic diagnostics.
    """
    import torch
    
    # Get monitoring components
    health_checker = get_health_checker(app_state)
    logger = get_logger("carla_rl_server")
    
    # Run health checks
    start_time = time.time()
    health_summary = health_checker.get_health_summary()
    duration_ms = (time.time() - start_time) * 1000
    
    # Log health check
    logger.log_health_check(
        status=health_summary["status"],
        checks=health_summary["checks"],
        duration_ms=duration_ms
    )
    
    device_str = "cuda" if torch.cuda.is_available() and os.getenv("USE_GPU", "0") == "1" else "cpu"
    current_version = app_state.get("selected_version", MODEL_VERSION)

    return HealthResponse(
        status=health_summary["status"],
        version=current_version,
        git=GIT_SHA,
        device=device_str,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["Model"])
async def get_metadata(inference_engine=Depends(get_inference_engine)) -> MetadataResponse:
    """
    Get model metadata and configuration.

    Returns information about the loaded model including input/output shapes
    and action space bounds.
    """

    device_str = str(inference_engine.device)

    # Get input shape from a dummy forward pass or model inspection
    # This is a simplified version - actual implementation would inspect model
    input_shape = [5]  # speed + steering + 3 sensor values (example)

    action_space = {"throttle": [0.0, 1.0], "brake": [0.0, 1.0], "steer": [-1.0, 1.0]}

    # Use selected version from app state if available
    current_version = app_state.get("selected_version", MODEL_VERSION)
    
    return MetadataResponse(
        modelName=MODEL_NAME,
        version=current_version,
        device=device_str,
        inputShape=input_shape,
        actionSpace=action_space,
    )


@app.post("/warmup", response_model=WarmupResponse, tags=["Model"])
async def warmup_model(inference_engine=Depends(get_inference_engine)) -> WarmupResponse:
    """
    Warm up the model with dummy inference.

    Performs JIT compilation and optimization to reduce cold start latency
    for subsequent inference requests.
    """
    # Get monitoring components
    logger = get_logger("carla_rl_server")
    metrics = get_metrics_collector()
    tracer = get_tracer("carla-rl-serving")
    
    model_version = app_state.get("selected_version", MODEL_VERSION)
    
    # Start warmup tracing
    with tracer.trace_model_warmup(model_version):
        start_time = time.time()

        try:
            # Create dummy observation for warmup
            from .io_schemas import Observation

            dummy_obs = Observation(speed=25.0, steering=0.0, sensors=[0.5] * 5)

            # Perform dummy inference
            _, _ = inference_engine.predict([dummy_obs], deterministic=True)

            app_state["warmup_completed"] = True
            timing_ms = (time.time() - start_time) * 1000.0
            
            # Record warmup metrics
            metrics.record_model_warmup(model_version, timing_ms / 1000)
            metrics.set_model_status(model_version, str(inference_engine.device), True, True)
            
            # Log warmup
            logger.log_model_warmup(
                model_version=model_version,
                duration_ms=timing_ms,
                status="success"
            )

            return WarmupResponse(
                status="warmed", timingMs=timing_ms, device=str(inference_engine.device)
            )

        except Exception as e:
            # Record error metrics
            metrics.record_error(
                error_type=type(e).__name__,
                endpoint="/warmup",
                model_version=model_version
            )
            
            # Log error
            logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                endpoint="/warmup",
                model_version=model_version,
                exception=e
            )
            
            raise ServiceUnavailableError(message="Warmup failed", details={"error": str(e)})


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    request: PredictRequest, inference_engine=Depends(get_inference_engine)
) -> PredictResponse:
    """
    Perform batch inference on observations.

    Takes a batch of observations and returns corresponding actions
    with timing information and model version.
    """
    # Get monitoring components
    logger = get_logger("carla_rl_server")
    metrics = get_metrics_collector()
    tracer = get_tracer("carla-rl-serving")
    
    model_version = app_state.get("selected_version", MODEL_VERSION)
    batch_size = len(request.observations)
    deterministic = request.deterministic or False
    
    # Start inference tracing
    with tracer.trace_inference(
        model_version=model_version,
        device=str(inference_engine.device),
        batch_size=batch_size,
        deterministic=deterministic
    ):
        try:
            # Perform inference with metrics collection
            with metrics.inference_timer(
                model_version=model_version,
                device=str(inference_engine.device),
                batch_size=batch_size,
                deterministic=deterministic
            ):
                actions, timing_ms = inference_engine.predict(
                    request.observations, deterministic
                )

            # Log inference
            logger.log_inference(
                model_version=model_version,
                device=str(inference_engine.device),
                batch_size=batch_size,
                duration_ms=timing_ms,
                deterministic=deterministic,
                status="success"
            )

            return PredictResponse(
                actions=actions,
                version=model_version,
                timingMs=timing_ms,
                deterministic=deterministic,
            )

        except Exception as e:
            # Record error metrics
            metrics.record_error(
                error_type=type(e).__name__,
                endpoint="/predict",
                model_version=model_version
            )
            
            # Log error
            logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                endpoint="/predict",
                model_version=model_version,
                exception=e,
                batch_size=batch_size,
                deterministic=deterministic
            )
            
            from .exceptions import InferenceError

            raise InferenceError(
                message="Inference failed",
                details={
                    "batch_size": batch_size,
                    "deterministic": deterministic,
                    "error": str(e),
                },
            )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics() -> Response:
    """
    Prometheus metrics endpoint for monitoring.

    Returns comprehensive metrics in Prometheus format including:
    - Inference performance metrics
    - System resource utilization
    - Error rates and types
    - Model status and health
    - Request processing statistics
    """
    # Get metrics collector
    metrics_collector = get_metrics_collector()
    
    # Update uptime metric
    uptime = time.time() - (app_state["startup_time"] or time.time())
    metrics_collector.set_service_uptime(uptime)
    
    # Get metrics in Prometheus format
    metrics_data = metrics_collector.get_metrics()
    content_type = metrics_collector.get_metrics_content_type()
    
    return Response(
        content=metrics_data,
        media_type=content_type
    )


@app.get("/versions", response_model=VersionsResponse, tags=["Model"])
async def get_versions() -> VersionsResponse:
    """
    Get available model versions and version selection information.
    
    Returns information about all discovered model versions, their metadata,
    and the currently loaded version with selection strategy details.
    """
    from pathlib import Path
    from .versioning import VersionSelector
    
    # Get artifacts root from environment
    artifacts_root = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    current_version = app_state.get("selected_version", MODEL_VERSION)
    
    try:
        # Create version selector and discover versions
        selector = VersionSelector(artifacts_root)
        available_versions = selector.discover_versions()
        
        # Build version info list
        version_info_list = []
        for version in available_versions:
            try:
                info = selector.get_version_info(version)
                version_info = VersionInfo(
                    version=str(version),
                    is_stable=version.is_stable(),
                    is_current=(str(version) == current_version),
                    performance_metrics=info.get('performance_metrics'),
                    model_card=info.get('model_card')
                )
                version_info_list.append(version_info)
            except Exception:
                # Skip versions with invalid metadata
                continue
        
        return VersionsResponse(
            current_version=current_version,
            available_versions=version_info_list,
            selection_strategy="environment_with_fallback",
            artifacts_root=str(artifacts_root)
        )
    
    except Exception:
        # Return minimal response if version discovery fails
        return VersionsResponse(
            current_version=current_version,
            available_versions=[
                VersionInfo(
                    version=current_version,
                    is_stable=True,
                    is_current=True,
                    performance_metrics=None,
                    model_card=None
                )
            ],
            selection_strategy="fallback",
            artifacts_root=str(artifacts_root)
        )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
