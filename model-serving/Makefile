# Makefile for CarlaRL Policy-as-a-Service
# Provides development, testing, and deployment commands

.PHONY: help install install-dev test test-cov lint format type-check clean build docker-build docker-run docker-push deploy-local health-check

# Default target
help: ## Show this help message
	@echo "CarlaRL Policy-as-a-Service - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-gpu: ## Install with GPU support
	pip install -e ".[gpu]"

install-all: ## Install all dependencies
	pip install -e ".[all]"

# Testing
test: ## Run unit tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-integration: ## Run integration tests
	pytest tests/ -v -m integration

test-performance: ## Run performance tests
	pytest tests/ -v -m slow

test-all: ## Run all tests including slow ones
	pytest tests/ -v --cov=src --cov-report=html

# Code quality
lint: ## Run linting with ruff
	ruff check src/ tests/

lint-fix: ## Fix linting issues automatically
	ruff check --fix src/ tests/

format: ## Format code with black
	black src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/

type-check: ## Run type checking with mypy
	mypy src/

quality: lint format-check type-check ## Run all code quality checks

# Development server
dev: ## Run development server with auto-reload
	ARTIFACT_DIR=artifacts MODEL_VERSION=v0.1.0 uvicorn src.server:app --reload --host 0.0.0.0 --port 8080

dev-gpu: ## Run development server with GPU support
	ARTIFACT_DIR=artifacts MODEL_VERSION=v0.1.0 USE_GPU=1 uvicorn src.server:app --reload --host 0.0.0.0 --port 8080

# Production server
serve: ## Run production server
	ARTIFACT_DIR=artifacts MODEL_VERSION=v0.1.0 uvicorn src.server:app --host 0.0.0.0 --port 8080 --workers 1

serve-multi: ## Run production server with multiple workers
	ARTIFACT_DIR=artifacts MODEL_VERSION=v0.1.0 uvicorn src.server:app --host 0.0.0.0 --port 8080 --workers 4

# Docker operations
docker-build: ## Build Docker image
	docker build -t model-serving:latest .

docker-build-gpu: ## Build Docker image with GPU support
	docker build -t model-serving:gpu --build-arg USE_GPU=1 .

docker-run: ## Run Docker container
	docker run --rm -p 8080:8080 \
		-e ARTIFACT_DIR=/app/artifacts \
		-e MODEL_VERSION=v0.1.0 \
		model-serving:latest

docker-run-gpu: ## Run Docker container with GPU support
	docker run --rm -p 8080:8080 --gpus all \
		-e ARTIFACT_DIR=/app/artifacts \
		-e MODEL_VERSION=v0.1.0 \
		-e USE_GPU=1 \
		model-serving:gpu

docker-run-dev: ## Run Docker container with volume mounts for development
	docker run --rm -p 8080:8080 \
		-v $(PWD)/artifacts:/app/artifacts \
		-v $(PWD)/src:/app/src \
		-e MODEL_VERSION=v0.1.0 \
		model-serving:latest

docker-push: ## Push Docker image to registry
	docker tag model-serving:latest $(REGISTRY)/model-serving:$(VERSION)
	docker push $(REGISTRY)/model-serving:$(VERSION)

docker-compose-up: ## Start services with docker-compose
	docker-compose -f deploy/docker/docker-compose.yml up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose -f deploy/docker/docker-compose.yml down

docker-compose-test: ## Run Docker Compose with testing setup
	docker-compose -f deploy/docker/docker-compose.yml up -d
	@echo "Waiting for service to be ready..."
	@sleep 10
	@echo "Running validation tests..."
	python3 scripts/cluster_validation.py --url http://localhost:8080
	docker-compose -f deploy/docker/docker-compose.yml down

# Health and monitoring
health-check: ## Check service health
	curl -f http://localhost:8080/healthz || (echo "Health check failed" && exit 1)

metrics: ## Get service metrics
	curl -s http://localhost:8080/metrics

metadata: ## Get model metadata
	curl -s http://localhost:8080/metadata | python -m json.tool

warmup: ## Warm up the model
	curl -X POST http://localhost:8080/warmup

# Performance testing
benchmark: ## Run performance benchmarks
	python scripts/benchmark.py --host localhost --port 8080 --requests 1000 --concurrency 10

load-test: ## Run load testing
	python scripts/load_test.py --host localhost --port 8080 --duration 60 --users 50

# Artifact management
create-artifacts: ## Create example model artifacts
	python scripts/create_example_artifacts.py --output artifacts/v0.1.0

validate-artifacts: ## Validate model artifacts
	python scripts/validate_artifacts.py --artifact-dir artifacts/v0.1.0

export-model: ## Export trained model to serving format
	python scripts/export_model.py --model-path $(MODEL_PATH) --output artifacts/$(VERSION)

# Deployment
deploy-local: docker-build ## Deploy locally with Docker
	docker run -d --name model-serving \
		-p 8080:8080 \
		-e ARTIFACT_DIR=/app/artifacts \
		-e MODEL_VERSION=v0.1.0 \
		--restart unless-stopped \
		model-serving:latest

deploy-k8s: ## Deploy to Kubernetes
	kubectl apply -f deploy/k8s/

undeploy-k8s: ## Remove from Kubernetes
	kubectl delete -f deploy/k8s/

# Testing
test-docker: docker-build create-artifacts ## Test with Docker (simple)
	@echo "🧪 Testing Model Serving Service with Docker"
	docker run --rm -d --name model-serving-test -p 8080:8080 \
		-e ARTIFACT_DIR=/app/artifacts -e MODEL_VERSION=v0.1.0 \
		model-serving:latest
	@echo "Waiting for service to start..."
	@sleep 15
	@echo "Running validation tests..."
	python3 scripts/cluster_validation.py --url http://localhost:8080 || true
	@echo "Cleaning up..."
	docker stop model-serving-test || true

test-compose: docker-compose-test ## Test with Docker Compose (full stack)

test-k8s: deploy-k8s ## Test Kubernetes deployment
	@echo "🧪 Testing Kubernetes deployment"
	kubectl wait --for=condition=ready pod -l app=model-serving --timeout=300s
	kubectl port-forward svc/model-serving-service 8080:80 &
	@sleep 10
	python3 scripts/cluster_validation.py --url http://localhost:8080 || true
	@pkill -f "kubectl port-forward" || true

# Cleanup
clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-docker: ## Clean up Docker images and containers
	docker system prune -f
	docker image prune -f

# Documentation
docs: ## Generate documentation
	python scripts/generate_docs.py

docs-serve: ## Serve documentation locally
	python -m http.server 8000 -d docs/

# Environment variables for make commands
REGISTRY ?= your-registry.com
VERSION ?= $(shell git rev-parse --short HEAD)
MODEL_PATH ?= models/latest.pt

# Print environment info
env-info: ## Show environment information
	@echo "Python version: $(shell python --version)"
	@echo "pip version: $(shell pip --version)"
	@echo "Current directory: $(PWD)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'not a git repo')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
	@echo "Registry: $(REGISTRY)"
	@echo "Version: $(VERSION)"
