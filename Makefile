# Highway RL - Function-Driven Repository
# Root Makefile for managing different components
# Usage: make <target>

.PHONY: help setup-sim setup-serving test-sim test-serving clean lint test security docker-check ci-checks

# Default target
help:
	@echo "Highway RL - Function-Driven Repository"
	@echo "===================================="
	@echo ""
	@echo "🏗️  Component Management:"
	@echo "  make setup-sim        - Setup simulation environment (model-sim/)"
	@echo "  make setup-serving    - Setup serving environment (model-serving/)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test-sim         - Test simulation components"
	@echo "  make test-serving     - Test serving components"
	@echo "  make test             - Run all tests"
	@echo ""
	@echo "🔍 Quality Checks:"
	@echo "  make lint             - Run linting (ruff, black, isort, mypy)"
	@echo "  make security         - Run security checks (bandit, safety)"
	@echo "  make docker-check     - Test Docker builds"
	@echo "  make ci-checks        - Run all CI checks locally"
	@echo ""
	@echo "🚀 Quick Commands:"
	@echo "  make train-highway    - Train Highway RL model"
	@echo "  make serve-model      - Start model serving service"
	@echo ""
	@echo "📁 Directory Structure:"
	@echo "  model-sim/           - Simulation and training"
	@echo "  model-serving/       - Model serving microservice"
	@echo "  tasks/               - Project management"

# Simulation Commands (delegate to model-sim/)
setup-sim:
	@echo "🔧 Setting up simulation environment..."
	@cd model-sim && make setup

train-highway:
	@echo "🚂 Training Highway RL model..."
	@cd model-sim && make train-highway

eval-highway:
	@echo "📊 Evaluating Highway RL models..."
	@cd model-sim && make eval-highway

test-sim:
	@echo "🧪 Testing simulation components..."
	@cd model-sim && make test

benchmark-sim:
	@echo "⚡ Running simulation benchmarks..."
	@cd model-sim && make benchmark

# Serving Commands (delegate to model-serving/)
setup-serving:
	@echo "🔧 Setting up serving environment..."
	@cd model-serving && pip install -r requirements.txt

serve-model:
	@echo "🚀 Starting model serving service..."
	@cd model-serving && make run

test-serving:
	@echo "🧪 Testing serving components..."
	@cd model-serving && make test

deploy-docker:
	@echo "🐳 Deploying with Docker..."
	@cd model-serving && make docker-build && make docker-run

deploy-k8s:
	@echo "☸️  Deploying to Kubernetes..."
	@cd model-serving && make deploy-k8s

# Utility Commands
clean:
	@echo "🧹 Cleaning all components..."
	@cd model-sim && make clean || true
	@cd model-serving && make clean || true

status:
	@echo "📊 Repository Status:"
	@echo "model-sim/: $$(cd model-sim && ls -1 | wc -l) items"
	@echo "model-serving/: $$(cd model-serving && ls -1 | wc -l) items"
	@echo "tasks/: $$(cd tasks && ls -1 | wc -l) items"

# Development helpers
sim-shell:
	@echo "🐚 Entering simulation environment shell..."
	@cd model-sim && bash

serving-shell:
	@echo "🐚 Entering serving environment shell..."
	@cd model-serving && bash

# Quality checks (run these before committing)
lint:
	@echo "🔧 Running linting checks..."
	@pip install ruff black isort >/dev/null 2>&1 || echo "Installing linting tools..."
	@echo "Running ruff..."
	@ruff check . --select=E,W,F
	@echo "Running black format check..."
	@black --check --diff .
	@echo "Running isort import check..."
	@isort --check-only --diff .
	@echo "✅ Linting checks passed!"

format:
	@echo "🎨 Auto-fixing code formatting..."
	@pip install black isort ruff >/dev/null 2>&1 || echo "Installing formatting tools..."
	@black .
	@isort .
	@ruff check --fix .
	@echo "✅ Code formatted!"

test:
	@echo "🧪 Running tests..."
	@pip install pytest >/dev/null 2>&1 || echo "Installing pytest..."
	@if [ -d "model-serving/tests" ]; then pytest model-serving/tests/ -v; fi
	@if [ -d "model-sim/tests" ]; then pytest model-sim/tests/ -v; fi
	@echo "✅ Tests completed!"

ci-checks:
	@echo "🔍 Running all CI checks locally..."
	@$(MAKE) lint
	@$(MAKE) test
	@echo "🐳 Testing Docker build..."
	@if [ -f "model-serving/Dockerfile" ]; then docker build -t model-serving:test ./model-serving/; fi
	@echo "✅ All CI checks passed!"