#!/bin/bash
"""
Minimal setup script for testing hardware optimizations.

This script installs only the minimal dependencies needed to test
the optimization modules without requiring PyTorch or other heavy dependencies.
"""

set -e  # Exit on any error

echo "Minimal Hardware Optimization Setup"
echo "==================================="

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

# Install minimal dependencies
echo "Installing minimal dependencies..."
pip install psutil py-cpuinfo

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py

# Run minimal tests
echo "Running minimal tests..."
python3 scripts/test_optimizations_minimal.py

echo ""
echo "Minimal setup complete!"
echo "======================"
echo ""
echo "Next steps:"
echo "1. For full benchmarking: ./scripts/setup_benchmarking.sh"
echo "2. For quick test: python3 scripts/local_benchmark.py --system-info-only"
echo ""
echo "To activate the environment in the future:"
echo "source venv/bin/activate"
