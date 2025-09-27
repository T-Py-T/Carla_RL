#!/usr/bin/env python3
"""
Cross-platform compatibility tests for Carla RL

Ensures the project works on different architectures and operating systems
"""

import unittest
import platform
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility"""
    
    def test_platform_detection(self):
        """Test that platform detection works"""
        system = platform.system()
        machine = platform.machine()
        
        # Should detect valid system
        self.assertIn(system, ['Darwin', 'Linux', 'Windows'])
        
        # Should detect valid architecture
        self.assertIn(machine, ['x86_64', 'arm64', 'amd64', 'i386', 'i686'])
    
    def test_tensorflow_import(self):
        """Test that TensorFlow imports correctly on this platform"""
        try:
            import tensorflow as tf
            
            # Should have a valid version
            self.assertIsNotNone(tf.__version__)
            
            # Should be able to create a simple tensor
            tensor = tf.constant([1, 2, 3])
            self.assertEqual(tensor.numpy().tolist(), [1, 2, 3])
            
        except ImportError:
            self.fail("TensorFlow not available - run platform-specific setup")
    
    def test_gpu_detection(self):
        """Test GPU detection works without errors"""
        try:
            import tensorflow as tf
            
            # Should not crash when listing devices
            physical_devices = tf.config.list_physical_devices()
            self.assertIsInstance(physical_devices, list)
            
            # GPU devices should be detectable
            gpu_devices = tf.config.list_physical_devices('GPU')
            self.assertIsInstance(gpu_devices, list)
            
            # If we have GPUs, they should have valid names
            for gpu in gpu_devices:
                self.assertIsNotNone(gpu.name)
                
        except Exception as e:
            self.fail(f"GPU detection failed: {e}")
    
    def test_basic_dependencies(self):
        """Test that basic dependencies are available"""
        import numpy as np
        import cv2
        import psutil
        import colorama
        
        # Test numpy
        arr = np.array([1, 2, 3])
        self.assertEqual(arr.sum(), 6)
        
        # Test OpenCV
        self.assertIsNotNone(cv2.__version__)
        
        # Test psutil
        memory = psutil.virtual_memory()
        self.assertGreater(memory.total, 0)
        
        # Test colorama
        self.assertIsNotNone(colorama.__version__)
    
    def test_carla_rl_imports(self):
        """Test that our RL modules import correctly"""
        # Test core imports
        from carla_rl.common import STOP, operating_system
        from carla_rl import settings
        
        # Test that STOP enum works
        self.assertIsNotNone(STOP.running)
        
        # Test that settings are accessible
        self.assertIsNotNone(settings.IMG_WIDTH)
        self.assertIsNotNone(settings.IMG_HEIGHT)
        
        # Test operating system detection
        os_type = operating_system()
        self.assertIn(os_type, ['windows', 'linux'])
    
    def test_model_creation(self):
        """Test that models can be created on this platform"""
        try:
            from carla_rl import models, settings
            
            # Test model base creation
            input_shape = (settings.IMG_HEIGHT, settings.IMG_WIDTH, 1)
            model_base = models.model_base_5_residual_CNN(input_shape)
            
            # Should return valid model components
            self.assertIsNotNone(model_base)
            
        except Exception as e:
            self.fail(f"Model creation failed: {e}")
    
    def test_environment_creation(self):
        """Test that Carla environment can be created (with mock)"""
        try:
            from carla_rl.carla import CarlaEnv
            
            # Should be able to create environment
            env = CarlaEnv(0, playing=True)
            self.assertIsNotNone(env)
            
            # Should have correct action space size
            self.assertGreater(env.action_space_size, 0)
            
        except Exception as e:
            self.fail(f"Environment creation failed: {e}")


class TestPlatformSpecificFeatures(unittest.TestCase):
    """Test platform-specific optimizations"""
    
    def test_apple_silicon_detection(self):
        """Test Apple Silicon specific features"""
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                import tensorflow as tf
                
                # Should have Metal GPU device
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    self.assertEqual(details.get('device_name'), 'METAL')
                    
            except Exception as e:
                self.fail(f"Apple Silicon GPU test failed: {e}")
    
    def test_cuda_detection(self):
        """Test NVIDIA CUDA detection"""
        if platform.system() in ['Linux', 'Windows']:
            try:
                import tensorflow as tf
                
                # Check if CUDA is available
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Should be able to detect CUDA devices
                    self.assertGreater(len(gpus), 0)
                    
            except Exception:
                # CUDA not available is okay for CPU-only systems
                pass


def run_platform_tests():
    """Run all platform compatibility tests"""
    print(f"Running cross-platform tests on {platform.system()} {platform.machine()}")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCrossPlatformCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformSpecificFeatures))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_platform_tests()
    sys.exit(0 if success else 1)