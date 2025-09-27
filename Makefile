# Highway RL - Function-Driven Repository
# Root Makefile for managing different components
# Usage: make <target>

.PHONY: help setup-sim setup-serving test-sim test-serving clean lint test security docker-check ci-checks

# Default target
help:
	@echo "Highway RL - Function-Driven Repository"
	@echo "===================================="
	@echo ""
	@echo "Component Management:"
	@echo "  make setup-sim        - Setup simulation environment (model-sim/)"
	@echo "  make setup-serving    - Setup serving environment (model-serving/)"
	@echo ""
	@echo "Testing:"
	@echo "  make test-sim         - Test simulation components"
	@echo "  make test-serving     - Test serving components"
	@echo "  make test             - Run all tests"
	@echo ""
	@echo "Quality Checks:"
	@echo "  make check            - Run all quality checks (syntax + linting)"
	@echo "  make lint             - Run ruff linting only"
	@echo "  make fix              - Auto-fix code issues"
	@echo "  make format           - Auto-format code (same as fix)"
	@echo "  make diff             - Show what would be fixed"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make train-highway    - Train Highway RL model"
	@echo "  make serve-model      - Start model serving service"
	@echo ""
	@echo "Directory Structure:"
	@echo "  model-sim/           - Simulation and training"
	@echo "  model-serving/       - Model serving microservice"
	@echo "  tasks/               - Project management"

# Simulation Commands (delegate to model-sim/)
setup-sim:
	@echo "Setting up simulation environment..."
	@cd model-sim && make setup

train-highway:
	@echo "Training Highway RL model..."
	@cd model-sim && make train-highway

eval-highway:
	@echo "Evaluating Highway RL models..."
	@cd model-sim && make eval-highway

test-sim:
	@echo "Testing simulation components..."
	@cd model-sim && make test

benchmark-sim:
	@echo "Running simulation benchmarks..."
	@cd model-sim && make benchmark

# Serving Commands (delegate to model-serving/)
setup-serving:
	@echo "Setting up serving environment..."
	@cd model-serving && pip install -r requirements.txt

serve-model:
	@echo "Starting model serving service..."
	@cd model-serving && make run

test-serving:
	@echo "Testing serving components..."
	@cd model-serving && make test

deploy-docker:
	@echo "Deploying with Docker..."
	@cd model-serving && make docker-build && make docker-run

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	@cd model-serving && make deploy-k8s

# Utility Commands
clean:
	@echo "🧹 Cleaning all components..."
	@cd model-sim && make clean || true
	@cd model-serving && make clean || true

status:
	@echo "Repository Status:"
	@echo "model-sim/: $$(cd model-sim && ls -1 | wc -l) items"
	@echo "model-serving/: $$(cd model-serving && ls -1 | wc -l) items"
	@echo "tasks/: $$(cd tasks && ls -1 | wc -l) items"

# Development helpers
sim-shell:
	@echo "Entering simulation environment shell..."
	@cd model-sim && bash

serving-shell:
	@echo "Entering serving environment shell..."
	@cd model-serving && bash

# Quality checks (run these before committing)
check:
	@echo "Running code quality checks..."
	@echo "Checking Python syntax..."
	@find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" -exec python -m py_compile {} \;
	@echo "Python syntax check passed"
	@echo "Running ruff linting..."
	@uv run ruff check --extend-exclude="pyproject.toml,*.toml,*.yaml,*.yml,*.json,*.md" . || (echo "Linting issues found. Run 'make fix' to auto-fix." && exit 1)
	@echo "All checks passed!"

fix:
	@echo "Auto-fixing code issues..."
	@uv run ruff check --fix --extend-exclude="pyproject.toml,*.toml,*.yaml,*.yml,*.json,*.md" .
	@echo "Code issues fixed!"

lint:
	@echo "Running ruff linting only..."
	@uv run ruff check --extend-exclude="pyproject.toml,*.toml,*.yaml,*.yml,*.json,*.md" .

format:
	@echo "Auto-formatting code..."
	@uv run ruff check --fix --extend-exclude="pyproject.toml,*.toml,*.yaml,*.yml,*.json,*.md" .

# Show what would be fixed without actually fixing
diff:
	@echo "Showing what would be fixed..."
	@uv run ruff check --diff --extend-exclude="pyproject.toml,*.toml,*.yaml,*.yml,*.json,*.md" .