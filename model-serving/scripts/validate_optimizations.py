#!/usr/bin/env python3
"""
Validation script for hardware optimizations.

This script validates that all optimization modules can be imported
and basic functionality works without requiring external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_imports():
    """Validate that all optimization modules can be imported."""
    print("Validating optimization module imports...")
    
    try:
        # Test basic imports
        from optimization.cpu_optimizer import CPUOptimizer, CPUOptimizationConfig  # noqa: F401
        print("✓ CPU optimizer imports successful")
        
        from optimization.gpu_optimizer import GPUOptimizer, GPUOptimizationConfig  # noqa: F401
        print("✓ GPU optimizer imports successful")
        
        from optimization.memory_optimizer import MemoryOptimizer, MemoryOptimizationConfig  # noqa: F401
        print("✓ Memory optimizer imports successful")
        
        from optimization.optimization_manager import OptimizationManager, OptimizationProfile  # noqa: F401
        print("✓ Optimization manager imports successful")
        
        from optimization import __all__ as optimization_exports
        print(f"✓ Optimization module exports: {optimization_exports}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def validate_configurations():
    """Validate that configuration classes can be instantiated."""
    print("\nValidating configuration classes...")
    
    try:
        from optimization.cpu_optimizer import CPUOptimizationConfig
        from optimization.gpu_optimizer import GPUOptimizationConfig
        from optimization.memory_optimizer import MemoryOptimizationConfig
        
        # Test CPU config
        cpu_config = CPUOptimizationConfig()
        assert cpu_config.enable_avx is True
        assert cpu_config.enable_sse is True
        assert cpu_config.max_threads is None
        print("✓ CPU configuration validation successful")
        
        # Test GPU config
        gpu_config = GPUOptimizationConfig()
        assert gpu_config.enable_cuda is True
        assert gpu_config.enable_tensorrt is True
        assert gpu_config.memory_fraction == 0.8
        print("✓ GPU configuration validation successful")
        
        # Test Memory config
        memory_config = MemoryOptimizationConfig()
        assert memory_config.enable_memory_pooling is True
        assert memory_config.pool_size_mb == 1024
        assert memory_config.max_pool_entries == 100
        print("✓ Memory configuration validation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        return False

def validate_optimization_profiles():
    """Validate that optimization profiles are properly defined."""
    print("\nValidating optimization profiles...")
    
    try:
        from optimization.optimization_manager import OptimizationManager
        
        manager = OptimizationManager()
        profiles = manager._optimization_profiles
        
        # Check that all expected profiles exist
        expected_profiles = ["high_performance_cpu", "gpu_accelerated", "memory_constrained", "balanced"]
        for profile_name in expected_profiles:
            assert profile_name in profiles, f"Profile {profile_name} not found"
        
        # Validate profile structure
        for profile_name, profile in profiles.items():
            assert hasattr(profile, 'name')
            assert hasattr(profile, 'description')
            assert hasattr(profile, 'cpu_config')
            assert hasattr(profile, 'gpu_config')
            assert hasattr(profile, 'memory_config')
            assert hasattr(profile, 'target_latency_ms')
            assert hasattr(profile, 'target_throughput_rps')
            assert hasattr(profile, 'memory_limit_gb')
            assert hasattr(profile, 'hardware_requirements')
            
            # Validate values
            assert profile.target_latency_ms > 0
            assert profile.target_throughput_rps > 0
            assert profile.memory_limit_gb > 0
        
        print(f"✓ Found {len(profiles)} optimization profiles")
        print("✓ Optimization profiles validation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimization profiles validation error: {e}")
        return False

def validate_file_structure():
    """Validate that all required files exist."""
    print("\nValidating file structure...")
    
    base_path = Path(__file__).parent.parent
    
    required_files = [
        "src/optimization/__init__.py",
        "src/optimization/cpu_optimizer.py",
        "src/optimization/gpu_optimizer.py",
        "src/optimization/memory_optimizer.py",
        "src/optimization/optimization_manager.py",
        "tests/optimization/__init__.py",
        "tests/optimization/test_cpu_optimizer.py",
        "tests/optimization/test_gpu_optimizer.py",
        "tests/optimization/test_memory_optimizer.py",
        "tests/optimization/test_optimization_manager.py",
        "tests/optimization/test_performance_validation.py",
        "scripts/optimization_manager.py",
        "docs/performance-tuning.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def main():
    """Main validation function."""
    print("Hardware Optimization Validation")
    print("=" * 40)
    
    all_passed = True
    
    # Run validations
    all_passed &= validate_file_structure()
    all_passed &= validate_imports()
    all_passed &= validate_configurations()
    all_passed &= validate_optimization_profiles()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All validations passed!")
        print("\nHardware optimization implementation is complete and ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -e .[optimization]")
        print("2. Run tests: python -m pytest tests/optimization/")
        print("3. Use optimization manager: python scripts/optimization_manager.py detect")
        return 0
    else:
        print("✗ Some validations failed!")
        print("\nPlease fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
