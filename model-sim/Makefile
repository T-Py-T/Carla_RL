# Highway RL Makefile
# Apple Silicon optimized autonomous driving simulation
# Usage: make <target>

.PHONY: help setup validate test train play clean install lint format check-gpu monitor

# Default target
help:
	@echo "Highway RL - Apple Silicon Optimized Autonomous Driving"
	@echo "======================================================="
	@echo ""
	@echo "Essential Commands:"
	@echo "  make setup           - Install and configure environment"
	@echo "  make train-highway   - Train RL agent (Apple Silicon optimized)"
	@echo "  make eval-highway    - Evaluate trained models"
	@echo "  make check-gpu       - Verify Metal acceleration"
	@echo ""
	@echo "Development:"
	@echo "  make clean          - Clean temporary files"
	@echo "  make benchmark      - Performance benchmarks"

# Setup and Installation
setup: install validate
	@echo "Setup complete! Ready for RL training"

install:
	@echo "Installing dependencies..."
	uv sync --extra apple-gpu
	@echo "Dependencies installed"

validate:
	@echo "Validating highway RL setup..."
	@uv run python -c "import sys; sys.path.insert(0, 'src'); import highway_env, gymnasium, tensorflow as tf; from highway_rl import HighwayEnvironment, HighwayDQNAgent; env = HighwayEnvironment('highway'); env.close(); gpus = tf.config.list_physical_devices('GPU'); print(f'SUCCESS: System ready - GPU devices: {len(gpus)}')"

# Development Commands
test:
	@echo "Running basic tests..."
	@uv run python -c "import sys; sys.path.insert(0, 'tests/validation'); import sources; print('Sources import: OK')"
	@uv run python -c "import sys; sys.path.insert(0, 'tests/validation'); import sources; env = sources.CarlaEnv(0); print('Environment: OK')"
	@uv run python -c "import sys; sys.path.insert(0, 'tests/validation'); import sources; agent = sources.ARTDQNAgent(id=0); print('Agent: OK')"
	@echo "All basic tests passed"

test-platform:
	@echo "Running cross-platform compatibility tests..."
	@uv run python tests/test_cross_platform.py

lint:
	@echo "Running linter..."
	@if command -v ruff >/dev/null 2>&1; then \
		uv run ruff check src/ scripts/; \
	else \
		echo "WARNING: Ruff not available. Install with: uv add --dev ruff"; \
	fi

format:
	@echo "Formatting code..."
	@if command -v ruff >/dev/null 2>&1; then \
		uv run ruff format src/ scripts/; \
	else \
		echo "WARNING: Ruff not available for formatting"; \
	fi

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf tmp/ 2>/dev/null || true
	@echo "Cleanup complete"

# Platform and Hardware Commands
check-platform:
	@echo "Detecting platform and hardware capabilities..."
	@uv run python scripts/setup_platform.py

setup-carla-docker:
	@echo "Setting up Carla simulator via Docker..."
	@uv run python scripts/setup_carla_docker.py

check-gpu:
	@echo "Checking GPU status and system resources..."
	@uv run python -c "import tensorflow as tf; import psutil; import platform; print('Platform:', platform.system(), platform.machine()); print('TensorFlow:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU devices:', len(gpus)); [print('  -', gpu.name, ':', tf.config.experimental.get_device_details(gpu).get('device_name', 'Unknown')) for gpu in gpus]; print('System Memory: {:.1f} GB'.format(psutil.virtual_memory().total / (1024**3))); print('Available Memory: {:.1f} GB'.format(psutil.virtual_memory().available / (1024**3)))"

# Primary Training Commands (Highway-env)
train-highway:
	@echo "Starting Highway RL training (Apple Silicon optimized)..."
	@echo "Tip: Use 'make check-gpu' to verify GPU acceleration"
	@mkdir -p models/highway training/highway/logs evaluation/highway
	uv run python training/highway/train_highway.py --scenario highway --episodes 1000 --double-dqn --dueling-dqn

train-multi:
	@echo "Starting Multi-Scenario Curriculum Training..."
	@echo "Training across: highway, merge, intersection, parking, racetrack"
	@mkdir -p models/highway training/highway/logs evaluation/highway
	uv run python training/highway/train_highway.py --scenario curriculum --episodes 2000 --double-dqn --dueling-dqn

train-highway-bg:
	@echo "Starting Highway RL training in background..."
	@mkdir -p models/highway training/highway/logs evaluation/highway
	@echo "Logs will be saved to training/highway/logs/training.log"
	nohup uv run python training/highway/train_highway.py --scenario highway --episodes 1000 --double-dqn --dueling-dqn > training/highway/logs/training.log 2>&1 &
	@echo "Training started in background"
	@echo "Use 'make monitor-highway' to watch progress"

eval-highway:
	@echo "Evaluating Highway RL models..."
	@if [ ! -d "models/highway" ] || [ -z "$$(ls -A models/highway 2>/dev/null)" ]; then \
		echo "ERROR: No trained highway models found in models/highway/ directory"; \
		echo "Run 'make train-highway' first to train a model"; \
		exit 1; \
	fi
	@mkdir -p evaluation/highway
	uv run python evaluation/highway/evaluate_models.py

# Legacy CARLA Training Commands
train-carla:
	@echo "Starting Legacy CARLA RL training..."
	@echo "Note: CARLA not supported on macOS Apple Silicon"
	@echo "Use 'make train-highway' for native macOS training"
	@mkdir -p models/carla training/carla/logs
	uv run python scripts/train.py

# Compatibility aliases
train: train-highway
	@echo "Default training now uses highway-env for better Apple Silicon support"
	@echo "Use 'make train-carla' for legacy CARLA training (Windows/Linux)"

play:
	@echo "Starting model playback..."
	@if [ ! -d "models" ] || [ -z "$$(ls -A models 2>/dev/null)" ]; then \
		echo "ERROR: No trained models found in models/ directory"; \
		echo "Run 'make train' first to train a model"; \
		exit 1; \
	fi
	uv run python scripts/play.py

monitor:
	@echo "Monitoring training progress..."
	@if [ -f "training/highway/logs/training.log" ]; then \
		tail -f training/highway/logs/training.log; \
	elif [ -f "logs/training.log" ]; then \
		tail -f logs/training.log; \
	else \
		echo "ERROR: No training log found."; \
		echo "Start training with 'make train-highway-bg' or 'make train-bg'"; \
	fi

monitor-highway:
	@echo "Monitoring highway training progress..."
	@if [ -f "training/highway/logs/training.log" ]; then \
		tail -f training/highway/logs/training.log; \
	else \
		echo "ERROR: No highway training log found. Start training with 'make train-highway-bg'"; \
	fi

# Advanced Commands
benchmark:
	@echo "Running performance benchmarks..."
	@mkdir -p benchmark_results
	@uv run python scripts/benchmark.py

benchmark-modern:
	@echo "Running modern ML stack benchmarks..."
	@echo "Testing TensorFlow 2.20 + Keras 3.x + NumPy 2.x performance"
	@mkdir -p benchmark_results
	@uv run python tests/benchmarks/benchmark_modern_stack.py

debug:
	@echo "Debug mode: Starting training with debug output..."
	@mkdir -p models checkpoint tmp logs
	@echo "Debug mode: Verbose logging enabled"
	TF_CPP_MIN_LOG_LEVEL=0 uv run python scripts/train.py

# Model Management
list-models:
	@echo "Available trained models:"
	@if [ -d "models" ]; then \
		ls -la models/ | grep -E "\.h5$$|\.keras$$" || echo "No models found"; \
	else \
		echo "No models directory found"; \
	fi

backup-models:
	@echo "Backing up trained models..."
	@if [ -d "models" ]; then \
		tar -czf "models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz" models/; \
		echo "Models backed up to models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz"; \
	else \
		echo "ERROR: No models directory to backup"; \
	fi

# Environment Management
create-dirs:
	@mkdir -p models checkpoint tmp logs benchmark_results
	@echo "Created necessary directories"

status:
	@echo "Project Status"
	@echo "=================="
	@echo "Python: $$(uv run python --version)"
	@echo "TensorFlow: $$(uv run python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not available')"
	@echo "GPU: $$(uv run python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices(\"GPU\")))' 2>/dev/null || echo 'Unknown') device(s)"
	@echo "Models: $$(ls models/ 2>/dev/null | wc -l | tr -d ' ') trained models"
	@echo "Checkpoints: $$(ls checkpoint/ 2>/dev/null | wc -l | tr -d ' ') checkpoints"
	@echo ""
	@echo "System Resources:"
	@uv run python -c "import psutil; mem=psutil.virtual_memory(); print(f'Memory: {mem.used/(1024**3):.1f}GB / {mem.total/(1024**3):.1f}GB ({mem.percent:.1f}% used)')"

# Quick start for new users
quickstart: setup
	@echo ""
	@echo "Highway RL Ready!"
	@echo "=================="
	@echo ""
	@echo "Next: make train-highway"
