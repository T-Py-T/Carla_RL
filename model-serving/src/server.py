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
from fastapi import Depends, FastAPI, Request
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
    WarmupResponse,
)
from .version import APP_NAME, GIT_SHA, MODEL_NAME, MODEL_VERSION

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

    try:
        # Import here to avoid circular imports
        from pathlib import Path

        import torch

        from .inference import InferenceEngine
        from .model_loader import load_artifacts

        # Configuration from environment
        artifact_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts")) / MODEL_VERSION
        use_gpu = os.getenv("USE_GPU", "0") == "1"
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        print(f"Loading model artifacts from: {artifact_dir}")
        print(f"Using device: {device}")

        # Load model and preprocessor
        policy, preprocessor = load_artifacts(artifact_dir, device)

        # Initialize inference engine
        app_state["inference_engine"] = InferenceEngine(policy, device, preprocessor)
        app_state["model_loaded"] = True

        print(f"Model {MODEL_NAME} v{MODEL_VERSION} loaded successfully")

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        # Don't raise here - let the service start but mark as unavailable
        app_state["model_loaded"] = False
        app_state["inference_engine"] = None

    yield

    # Shutdown
    print("Shutting down CarlaRL Policy Service...")
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

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

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

    device_str = "cuda" if torch.cuda.is_available() and os.getenv("USE_GPU", "0") == "1" else "cpu"

    return HealthResponse(
        status="ok" if app_state["model_loaded"] else "degraded",
        version=MODEL_VERSION,
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

    return MetadataResponse(
        modelName=MODEL_NAME,
        version=MODEL_VERSION,
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
    start_time = time.time()

    try:
        # Create dummy observation for warmup
        from .io_schemas import Observation

        dummy_obs = Observation(speed=25.0, steering=0.0, sensors=[0.5] * 5)

        # Perform dummy inference
        _, _ = inference_engine.predict([dummy_obs], deterministic=True)

        app_state["warmup_completed"] = True
        timing_ms = (time.time() - start_time) * 1000.0

        return WarmupResponse(
            status="warmed", timingMs=timing_ms, device=str(inference_engine.device)
        )

    except Exception as e:
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
    try:
        # Perform inference
        actions, timing_ms = inference_engine.predict(
            request.observations, request.deterministic or False
        )

        return PredictResponse(
            actions=actions,
            version=MODEL_VERSION,
            timingMs=timing_ms,
            deterministic=request.deterministic or False,
        )

    except Exception as e:
        from .exceptions import InferenceError

        raise InferenceError(
            message="Inference failed",
            details={
                "batch_size": len(request.observations),
                "deterministic": request.deterministic,
                "error": str(e),
            },
        )


# Optional metrics endpoint (basic implementation)
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Basic metrics endpoint for monitoring.

    Returns simple text metrics compatible with Prometheus scraping.
    """
    uptime = time.time() - (app_state["startup_time"] or time.time())

    metrics = [
        "# HELP carla_rl_uptime_seconds Service uptime in seconds",
        "# TYPE carla_rl_uptime_seconds counter",
        f"carla_rl_uptime_seconds {uptime:.2f}",
        "",
        "# HELP carla_rl_model_loaded Model loading status (1=loaded, 0=not loaded)",
        "# TYPE carla_rl_model_loaded gauge",
        f"carla_rl_model_loaded {1 if app_state['model_loaded'] else 0}",
        "",
        "# HELP carla_rl_warmup_completed Warmup completion status (1=completed, 0=not completed)",
        "# TYPE carla_rl_warmup_completed gauge",
        f"carla_rl_warmup_completed {1 if app_state['warmup_completed'] else 0}",
    ]

    return "\n".join(metrics)


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
