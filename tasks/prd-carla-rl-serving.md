# Product Requirements Document: CarlaRL Policy-as-a-Service

## Introduction/Overview

The CarlaRL Policy-as-a-Service system transforms trained reinforcement learning policies into production-ready microservices for autonomous vehicle inference. This system addresses the critical gap between research model training and production deployment by providing a standardized, high-performance serving infrastructure.

**Problem Statement:** Currently, trained RL policies exist as research artifacts without a standardized way to deploy them for real-time inference in production autonomous vehicle systems. There's no consistent interface for policy serving, model versioning, or performance monitoring.

**Goal:** Create a full MLOps pipeline that enables seamless deployment of trained CarlaRL policies as scalable, monitored microservices with millisecond-latency inference capabilities.

## Goals

1. **Primary Goal:** Deploy trained RL policies as stateless FastAPI microservices with <10ms P50 inference latency on CPU
2. **Model Consistency:** Ensure 100% parity between training and serving feature pipelines to eliminate train-serve skew
3. **Production Readiness:** Provide enterprise-grade serving with health checks, metrics, versioning, and observability
4. **MLOps Integration:** Enable seamless CI/CD pipeline for model training → artifact registry → deployment
5. **Scalability:** Support high-throughput batch inference (1000+ requests/sec) for evaluation workloads
6. **Reliability:** Achieve 99.9% uptime with graceful error handling and automatic recovery

## User Stories

**As a ML Engineer:**
- I want to deploy my trained policy with a single command so that I can quickly test it in production
- I want to version my models semantically so that I can track changes and rollback if needed
- I want to monitor inference performance so that I can optimize model serving

**As a DevOps Engineer:**
- I want standardized Docker containers so that I can deploy policies consistently across environments
- I want Prometheus metrics so that I can monitor system health and set up alerts
- I want health checks so that my load balancer can route traffic appropriately

**As a Research Scientist:**
- I want deterministic inference mode so that I can reproduce experimental results
- I want batch prediction endpoints so that I can efficiently evaluate policies on large datasets
- I want model metadata endpoints so that I can verify which model version is deployed

**As a Product Engineer:**
- I want sub-10ms inference latency so that I can use policies for real-time vehicle control
- I want JSON APIs with clear schemas so that I can integrate with existing systems
- I want graceful error handling so that my applications remain stable

## Functional Requirements

### Phase 1: Core Serving Infrastructure

#### 1. Policy-as-a-Service (Stateless)
1. **FR-1.1:** System MUST provide `/healthz` endpoint returning service status, version, and git SHA
2. **FR-1.2:** System MUST provide `/metadata` endpoint returning model name, version, device, input shape, and action space
3. **FR-1.3:** System MUST provide `/predict` endpoint accepting batch observations and returning actions
4. **FR-1.4:** System MUST support deterministic mode toggle for reproducible inference
5. **FR-1.5:** System MUST provide `/warmup` endpoint for JIT compilation and model loading optimization
6. **FR-1.6:** System MUST load models automatically on startup with configurable artifact directory
7. **FR-1.7:** System MUST support both CPU and GPU inference with automatic device selection
8. **FR-1.8:** System MUST validate input schemas and return structured error responses
9. **FR-1.9:** System MUST include timing information in prediction responses
10. **FR-1.10:** System MUST support semantic versioning (vMAJOR.MINOR.PATCH) for models

#### 2. Standardized Artifacts
11. **FR-2.1:** System MUST support standardized artifact format: `model.pt` + `model_card.yaml` + `preprocessor.pkl`
12. **FR-2.2:** System MUST load preprocessor for feature pipeline parity with training
13. **FR-2.3:** System MUST support TorchScript or ONNX model formats for optimized inference
14. **FR-2.4:** System MUST validate artifact integrity using hash pinning
15. **FR-2.5:** System MUST support multiple model versions simultaneously

#### 3. API Schemas
16. **FR-3.1:** System MUST define Observation schema with speed, steering, and sensor arrays
17. **FR-3.2:** System MUST define Action schema with throttle, brake, and steer values
18. **FR-3.3:** System MUST validate all input/output data against Pydantic schemas
19. **FR-3.4:** System MUST return consistent error format for invalid requests
20. **FR-3.5:** System MUST support OpenAPI/Swagger documentation generation

### Phase 2: Production Features (Future)

#### 4. Episode Orchestrator (Optional Separate Service)
21. **FR-4.1:** System SHOULD provide `/session` endpoints for CARLA episode management
22. **FR-4.2:** System SHOULD provide `/step` endpoint for simulation step execution
23. **FR-4.3:** System SHOULD support WebSocket streaming for real-time telemetry
24. **FR-4.4:** System SHOULD maintain 60 FPS simulation loop on headless GPU

#### 5. Observability
25. **FR-5.1:** System SHOULD expose Prometheus metrics at `/metrics` endpoint
26. **FR-5.2:** System SHOULD log structured events for debugging and monitoring
27. **FR-5.3:** System SHOULD support distributed tracing for request flow analysis

## Non-Goals (Out of Scope)

1. **Training Pipeline:** This system does not include model training capabilities
2. **CARLA Integration:** Phase 1 will not include direct CARLA simulator integration
3. **Model Management UI:** No web interface for model management (CLI/API only)
4. **Multi-tenancy:** Single-tenant deployment model (one service per model)
5. **Data Storage:** No persistent storage of predictions or telemetry data
6. **Authentication:** No built-in authentication (handled by infrastructure layer)
7. **Model Training Orchestration:** No integration with training job scheduling
8. **Custom Model Formats:** Only PyTorch/TorchScript/ONNX support initially

## Design Considerations

### Architecture
- **Microservice Design:** Stateless services for horizontal scalability
- **Container-First:** Docker containers for consistent deployment
- **API-First:** RESTful APIs with OpenAPI specifications
- **Separation of Concerns:** Policy serving separate from simulation orchestration

### Technology Stack
- **Framework:** FastAPI with Pydantic v2 for high-performance APIs
- **ML Runtime:** PyTorch with TorchScript optimization
- **Containerization:** Docker with multi-stage builds
- **Orchestration:** Kubernetes (K3s) for local deployment
- **Monitoring:** Prometheus + Grafana for observability

### Folder Structure
```
carla-rl-serving/
├── src/
│   ├── server.py           # FastAPI application
│   ├── inference.py        # Inference engine
│   ├── io_schemas.py       # Pydantic schemas
│   ├── model_loader.py     # Model loading utilities
│   ├── preprocessing.py    # Feature preprocessing
│   └── version.py          # Version management
├── artifacts/
│   └── v0.1.0/
│       ├── model.pt        # TorchScript model
│       ├── preprocessor.pkl # Feature preprocessor
│       └── model_card.yaml # Model metadata
├── tests/
│   └── test_predict.py     # Unit tests
├── Dockerfile              # Container definition
├── pyproject.toml          # Python dependencies
├── Makefile               # Development commands
└── README.md              # Documentation
```

## Technical Considerations

### Performance Requirements
- **Cold Start:** < 2 seconds on CPU for service initialization
- **Warm Inference:** P50 < 10ms for single prediction on CPU
- **Batch Inference:** Support for batched predictions with linear scaling
- **Memory Usage:** Bounded memory consumption with configurable limits
- **Throughput:** 1000+ requests/sec for batch evaluation workloads

### Dependencies
- **Existing Training Code:** Must integrate with current CarlaRL training pipeline
- **Model Artifacts:** Requires standardized export from training process
- **Infrastructure:** K3s cluster for deployment and service mesh
- **Monitoring Stack:** Prometheus/Grafana for metrics collection

### Security
- **Container Security:** Non-root user, read-only filesystem, minimal base image
- **Input Validation:** Strict schema validation for all API inputs
- **Resource Limits:** CPU/memory limits to prevent resource exhaustion
- **Network Policies:** Restricted network access in Kubernetes

## Success Metrics

### Primary Success Metrics (Model Accuracy/Performance Consistency)
1. **Inference Accuracy:** 100% parity with training environment predictions
2. **Deterministic Reproducibility:** Identical outputs for same inputs in deterministic mode
3. **Model Version Consistency:** Correct model version served 100% of the time
4. **Feature Pipeline Parity:** Zero train-serve skew in preprocessing

### Secondary Success Metrics (Latency Assessment)
1. **P50 Latency:** < 10ms for single inference on CPU
2. **P95 Latency:** < 25ms for single inference on CPU
3. **P99 Latency:** < 50ms for single inference on CPU
4. **Cold Start Time:** < 2 seconds for service initialization
5. **Throughput:** > 1000 requests/sec for batch inference

### Operational Metrics
1. **Service Uptime:** > 99.9% availability
2. **Error Rate:** < 0.1% of requests result in 5xx errors
3. **Resource Utilization:** < 80% CPU/memory usage under normal load
4. **Deployment Success:** > 95% successful deployments without rollback

## Implementation Phases

### Phase 1: Core Serving (Current Priority)
- **Duration:** 2-3 weeks
- **Deliverables:** Stateless policy service, standardized artifacts, basic monitoring
- **Acceptance:** All Phase 1 functional requirements met, performance targets achieved

### Phase 2: Production Hardening
- **Duration:** 2-3 weeks  
- **Deliverables:** Episode orchestrator, advanced monitoring, CI/CD pipeline
- **Acceptance:** Production-ready deployment with full observability

### Phase 3: MLOps Integration
- **Duration:** 3-4 weeks
- **Deliverables:** Automated training-to-deployment pipeline, model registry integration
- **Acceptance:** End-to-end MLOps workflow operational

## Open Questions

1. **Model Export Format:** Should we prioritize TorchScript or ONNX for production inference?
2. **Preprocessing Serialization:** What's the best format for serializing sklearn/custom preprocessors?
3. **GPU Memory Management:** How should we handle GPU memory allocation for concurrent requests?
4. **Model Registry:** Should we integrate with MLflow or build custom artifact management?
5. **Monitoring Granularity:** What level of request/response logging is needed for debugging?
6. **Load Balancing:** How should we handle model warming across multiple service instances?
7. **Rollback Strategy:** What's the safest way to rollback model deployments in production?
8. **Performance Benchmarking:** What's the baseline hardware specification for performance testing?

## Next Steps

1. **Immediate (Week 1):** Export existing trained models to standardized artifact format
2. **Phase 1 Implementation (Weeks 1-3):** Build core serving infrastructure per functional requirements
3. **Testing & Validation (Week 3):** full testing of inference parity and performance
4. **Documentation (Week 4):** Complete API documentation and deployment guides
5. **Phase 2 Planning (Week 4):** Detailed design for episode orchestrator and advanced monitoring

---

**Document Version:** 1.0  
**Created:** September 27, 2025  
**Last Updated:** September 27, 2025  
**Owner:** ML Engineering Team  
**Reviewers:** DevOps, Research Science Teams
