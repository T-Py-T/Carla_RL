# Implementation Tasks: CarlaRL Policy-as-a-Service

Based on PRD: `prd-carla-rl-serving.md`

## Relevant Files

- `carla-rl-serving/src/io_schemas.py` - Pydantic schemas for API request/response validation with full field validation (Agent 1: API Layer)
- `carla-rl-serving/src/exceptions.py` - Custom exception classes and error handling models with FastAPI integration (Agent 1: API Layer)
- `carla-rl-serving/src/server.py` - FastAPI application with all endpoints, middleware, and error handling (Agent 1: API Layer)
- `carla-rl-serving/src/version.py` - Version management and git metadata utilities (Agent 1: API Layer)
- `carla-rl-serving/tests/test_api_endpoints.py` - Integration tests for all FastAPI endpoints with mocked dependencies (Agent 1: API Layer)
- `carla-rl-serving/tests/test_schemas.py` - full unit tests for Pydantic schemas with validation edge cases (Agent 1: API Layer)
- `carla-rl-serving/tests/test_exceptions.py` - Unit tests for exception handling and error response creation (Agent 1: API Layer)
- `carla-rl-serving/tests/test_qa_plan.py` - QA validation test plan ensuring all PRD requirements are met (Agent 1: API Layer)
- `carla-rl-serving/src/model_loader.py` - Model loading and artifact management utilities (Agent 2: Model Layer)
- `carla-rl-serving/src/preprocessing.py` - Feature preprocessing pipeline (Agent 2: Model Layer)
- `carla-rl-serving/tests/test_model_loader.py` - Unit tests for model loading (Agent 2: Model Layer)
- `carla-rl-serving/tests/test_preprocessing.py` - Unit tests for preprocessing pipeline (Agent 2: Model Layer)
- `carla-rl-serving/src/inference.py` - Inference engine and performance optimization (Agent 3: Inference Layer)
- `carla-rl-serving/src/version.py` - Version management and metadata (Agent 3: Inference Layer)
- `carla-rl-serving/tests/test_inference.py` - Unit tests for inference engine (Agent 3: Inference Layer)
- `carla-rl-serving/tests/test_performance.py` - Performance benchmarking tests (Agent 3: Inference Layer)
- `carla-rl-serving/Dockerfile` - Container definition and build configuration (Agent 4: Infrastructure)
- `carla-rl-serving/pyproject.toml` - Python dependencies and project configuration (Agent 4: Infrastructure)
- `carla-rl-serving/Makefile` - Development and deployment commands (Agent 4: Infrastructure)
- `carla-rl-serving/README.md` - Documentation and usage guide (Agent 4: Infrastructure)
- `carla-rl-serving/artifacts/v0.1.0/model.pt` - Example TorchScript model artifact (Agent 5: Artifacts)
- `carla-rl-serving/artifacts/v0.1.0/model_card.yaml` - Model metadata and configuration (Agent 5: Artifacts)
- `carla-rl-serving/artifacts/v0.1.0/preprocessor.pkl` - Serialized preprocessing pipeline (Agent 5: Artifacts)
- `carla-rl-serving/tests/test_artifacts.py` - Unit tests for artifact validation (Agent 5: Artifacts)

### Notes

- Each agent works on a distinct layer to prevent code conflicts
- Unit tests should be co-located with implementation files where possible
- Use `pytest carla-rl-serving/tests/` to run all tests
- Use `pytest carla-rl-serving/tests/test_[specific_module].py` to run specific test suites
- Integration tests will be handled separately after individual layer completion

## Tasks

- [x] 1.0 **API Layer Development** (Agent 1: FastAPI endpoints, schemas, routing)
  - [x] 1.1 Create Pydantic schemas for Observation, Action, PredictRequest, and PredictResponse with proper validation
  - [x] 1.2 Implement error response schemas and exception handling models
  - [x] 1.3 Create FastAPI application structure with proper middleware and CORS configuration
  - [x] 1.4 Implement `/healthz` endpoint with service status, version, and git SHA reporting
  - [x] 1.5 Implement `/metadata` endpoint returning model info, device, input shape, and action space
  - [x] 1.6 Implement `/predict` endpoint with batch processing and deterministic mode support
  - [x] 1.7 Implement `/warmup` endpoint for JIT compilation and model optimization
  - [x] 1.8 Add full input validation and structured error responses
  - [x] 1.9 Generate OpenAPI/Swagger documentation with proper examples
  - [x] 1.10 Write unit tests for all endpoints with edge cases and error conditions
  - [x] 1.11 Write unit tests for schema validation and serialization/deserialization

- [x] 2.0 **Model Management Layer** (Agent 2: Model loading, preprocessing, artifact handling)
  - [x] 2.1 Create PolicyWrapper class for model encapsulation with deterministic/stochastic modes
  - [x] 2.2 Implement model loading utilities supporting TorchScript and ONNX formats
  - [x] 2.3 Create artifact integrity validation using hash pinning and checksums
  - [x] 2.4 Implement preprocessor loading and serialization (sklearn/custom pipelines)
  - [x] 2.5 Create feature pipeline with train-serve parity validation
  - [x] 2.6 Implement multi-version model support with semantic versioning
  - [x] 2.7 Add device selection logic (CPU/GPU) with automatic fallback
  - [x] 2.8 Create model metadata parsing from model_card.yaml files
  - [x] 2.9 Implement graceful error handling for missing or corrupted artifacts
  - [x] 2.10 Write unit tests for model loading with various artifact configurations
  - [x] 2.11 Write unit tests for preprocessing pipeline with edge cases and validation

- [x] 3.0 **Inference Engine Layer** (Agent 3: Inference optimization, performance, versioning)
  - [x] 3.1 Create InferenceEngine class with batch processing and memory optimization
  - [x] 3.2 Implement tensor pre-allocation and memory pinning for performance
  - [x] 3.3 Add torch.no_grad() context and JIT optimization for inference
  - [x] 3.4 Implement deterministic inference mode with reproducible outputs
  - [x] 3.5 Create performance timing and metrics collection for latency tracking
  - [x] 3.6 Implement batch size optimization and dynamic batching
  - [x] 3.7 Add version management with git SHA tracking and model version consistency
  - [x] 3.8 Create inference result caching for identical inputs (optional optimization)
  - [x] 3.9 Implement graceful degradation and error recovery mechanisms
  - [x] 3.10 Write performance benchmarking tests with latency and throughput validation
  - [x] 3.11 Write unit tests for inference engine with determinism and batch consistency

- [x] 4.0 **Infrastructure & Deployment** (Agent 4: Docker, dependencies, build system)
  - [x] 4.1 Create multi-stage Dockerfile with optimized Python base image
  - [x] 4.2 Configure non-root user and read-only filesystem for security
  - [x] 4.3 Set up pyproject.toml with pinned dependencies and development tools
  - [x] 4.4 Create Makefile with development, testing, and deployment commands
  - [x] 4.5 Configure uvicorn server with production settings and worker management
  - [x] 4.6 Set up environment variable configuration for artifact paths and device selection
  - [x] 4.7 Create health check configuration for container orchestration
  - [x] 4.8 Implement resource limits and memory constraints in container
  - [x] 4.9 Create docker-compose.yml for local development and testing
  - [x] 4.10 Write full README with setup, usage, and deployment instructions
  - [x] 4.11 Create integration tests for containerized deployment

- [x] 5.0 **Artifact Management System** (Agent 5: Model artifacts, validation, export pipeline)
  - [x] 5.1 Create model export utilities from existing CarlaRL training code
  - [x] 5.2 Implement TorchScript model serialization with optimization
  - [x] 5.3 Create preprocessor serialization pipeline maintaining train-serve parity
  - [x] 5.4 Design model_card.yaml schema with metadata, performance metrics, and versioning
  - [x] 5.5 Implement artifact directory structure with semantic versioning
  - [x] 5.6 Create hash-based artifact validation and integrity checking
  - [x] 5.7 Build model registry interface for artifact discovery and management
  - [x] 5.8 Implement artifact migration utilities for version upgrades
  - [x] 5.9 Create example artifacts (v0.1.0) with sample model, preprocessor, and metadata
  - [x] 5.10 Write validation tests for artifact format compliance and integrity
  - [x] 5.11 Write unit tests for export pipeline with various model architectures
