#!/usr/bin/env python3
"""
Minimal test script for hardware optimizations.

This script tests the optimization modules without requiring
external dependencies like PyTorch.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_config_classes():
    """Test that configuration classes can be instantiated."""
    print("Testing configuration classes...")
    
    try:
        from optimization.cpu_optimizer import CPUOptimizationConfig
        from optimization.gpu_optimizer import GPUOptimizationConfig
        from optimization.memory_optimizer import MemoryOptimizationConfig
        
        # Test CPU config
        cpu_config = CPUOptimizationConfig()
        assert cpu_config.enable_avx is True
        assert cpu_config.enable_sse is True
        print("✓ CPUOptimizationConfig works")
        
        # Test GPU config
        gpu_config = GPUOptimizationConfig()
        assert gpu_config.enable_cuda is True
        assert gpu_config.enable_tensorrt is True
        print("✓ GPUOptimizationConfig works")
        
        # Test Memory config
        memory_config = MemoryOptimizationConfig()
        assert memory_config.enable_memory_pooling is True
        assert memory_config.pool_size_mb == 1024
        print("✓ MemoryOptimizationConfig works")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_optimization_profiles():
    """Test that optimization profiles are properly defined."""
    print("Testing optimization profiles...")
    
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
        return True
        
    except Exception as e:
        print(f"✗ Optimization profiles test failed: {e}")
        return False

def test_hardware_detection():
    """Test hardware detection without external dependencies."""
    print("Testing hardware detection...")
    
    try:
        from benchmarking.hardware_detector import HardwareDetector
        
        detector = HardwareDetector()
        
        # Test CPU detection
        cpu_info = detector.detect_cpu()
        assert hasattr(cpu_info, 'model')
        assert hasattr(cpu_info, 'cores')
        assert hasattr(cpu_info, 'threads')
        print("✓ CPU detection works")
        
        # Test memory detection
        memory_info = detector.detect_memory()
        assert hasattr(memory_info, 'total_gb')
        assert hasattr(memory_info, 'available_gb')
        print("✓ Memory detection works")
        
        # Test GPU detection (may not be available)
        gpu_info = detector.detect_gpu()
        if gpu_info is not None:
            assert hasattr(gpu_info, 'model')
            assert hasattr(gpu_info, 'memory_gb')
            print("✓ GPU detection works")
        else:
            print("✓ GPU detection works (no GPU available)")
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware detection test failed: {e}")
        return False

def test_memory_pool():
    """Test memory pool functionality."""
    print("Testing memory pool...")
    
    try:
        from optimization.memory_optimizer import MemoryPool
        
        # Create memory pool
        pool = MemoryPool(pool_size_mb=100, max_entries=10)
        assert pool.pool_size_bytes == 100 * 1024 * 1024
        assert pool.max_entries == 10
        print("✓ MemoryPool creation works")
        
        # Test stats
        stats = pool.get_stats()
        assert 'total_tensors' in stats
        assert 'total_memory_bytes' in stats
        assert 'pool_entries' in stats
        print("✓ MemoryPool stats work")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory pool test failed: {e}")
        return False

def main():
    """Run all minimal tests."""
    print("Hardware Optimization Minimal Tests")
    print("=" * 40)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_config_classes()
    all_passed &= test_optimization_profiles()
    all_passed &= test_hardware_detection()
    all_passed &= test_memory_pool()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All minimal tests passed!")
        print("\nHardware optimization modules are ready for benchmarking.")
        print("\nNext steps:")
        print("1. Run: ./scripts/setup_benchmarking.sh")
        print("2. Run: ./scripts/run_full_benchmark_suite.sh")
        return 0
    else:
        print("✗ Some tests failed!")
        print("\nPlease fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
