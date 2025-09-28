#!/bin/bash
"""
Setup script for local hardware benchmarking.

This script sets up the environment and dependencies needed for
comprehensive hardware benchmarking on the local machine.
"""

set -e  # Exit on any error

echo "Policy-as-a-Service Benchmarking Setup"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the model-serving directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install basic dependencies
echo "Installing basic dependencies..."
pip install numpy psutil

# Try to install PyTorch (CPU version first)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Check if CUDA is available and install GPU version if needed
echo "Checking for CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('CUDA is available!')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA is not available, using CPU version')
"

# Install additional optimization dependencies
echo "Installing optimization dependencies..."
pip install py-cpuinfo

# Try to install TensorRT if available
echo "Checking for TensorRT..."
python3 -c "
try:
    import tensorrt
    print('TensorRT is available!')
    print(f'TensorRT version: {tensorrt.__version__}')
except ImportError:
    print('TensorRT not available (optional)')
" || echo "TensorRT not available (optional)"

# Install development dependencies for testing
echo "Installing development dependencies..."
pip install pytest pytest-cov

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py

# Run system information collection
echo "Collecting system information..."
python3 scripts/local_benchmark.py --system-info-only > system_info.json

echo ""
echo "Setup complete!"
echo "=============="
echo ""
echo "Next steps:"
echo "1. Run quick benchmarks: python3 scripts/local_benchmark.py --quick"
echo "2. Run comprehensive benchmarks: python3 scripts/local_benchmark.py --comprehensive"
echo "3. View system info: cat system_info.json"
echo ""
echo "Available commands:"
echo "- python3 scripts/local_benchmark.py --help"
echo "- python3 scripts/optimization_manager.py detect"
echo "- python3 scripts/optimization_manager.py optimize"
echo ""
echo "To activate the environment in the future:"
echo "source venv/bin/activate"
