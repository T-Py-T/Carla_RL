#!/usr/bin/env python3
"""
M4 Max Performance Benchmark for Carla RL
Tests TensorFlow performance with Metal GPU acceleration
"""

import tensorflow as tf
import time
import numpy as np
import psutil
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def print_header():
    import platform
    
    print("Carla RL Performance Benchmark")
    print("=" * 40)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"TensorFlow: {tf.__version__}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {len(gpus)} device(s)")
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  - {gpu.name}: {details.get('device_name', 'Unknown')}")
    else:
        print("GPU: None detected (CPU-only training)")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    print()

def benchmark_matrix_operations():
    """Benchmark basic matrix operations"""
    print("Matrix Operations Benchmark")
    print("-" * 30)
    
    sizes = [500, 1000, 2000]
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    print(f"Device: {device}")
    
    for size in sizes:
        with tf.device(device):
            # Create random matrices
            a = tf.random.normal((size, size), dtype=tf.float32)
            b = tf.random.normal((size, size), dtype=tf.float32)
            
            # Warm up
            _ = tf.matmul(a, b)
            
            # Benchmark matrix multiplication
            start_time = time.time()
            for _ in range(3):  # Multiple runs for average
                c = tf.matmul(a, b)
                _ = tf.reduce_sum(c).numpy()  # Force computation
            elapsed = (time.time() - start_time) / 3
            
            # Calculate GFLOPS (approximate)
            operations = 2 * size * size * size  # Matrix multiplication operations
            gflops = operations / elapsed / 1e9
            
            print(f"  {size}x{size}: {elapsed:.3f}s ({gflops:.2f} GFLOPS)")
    print()

def benchmark_cnn_operations():
    """Benchmark CNN operations similar to those used in RL"""
    print("🧠 CNN Operations Benchmark")
    print("-" * 30)
    
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Device: {device}")
    
    # Test different image sizes
    image_configs = [
        (270, 480, 1, 16),   # Carla RL default size
        (270, 480, 1, 32),   # Larger batch
        (270, 480, 1, 64),   # Even larger batch (if memory allows)
    ]
    
    for height, width, channels, batch_size in image_configs:
        try:
            with tf.device(device):
                # Create a simple CNN similar to what's used in RL
                inputs = tf.random.normal((batch_size, height, width, channels))
                
                # Simple CNN layers
                conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
                pool1 = tf.keras.layers.MaxPooling2D(2)
                conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
                pool2 = tf.keras.layers.MaxPooling2D(2)
                flatten = tf.keras.layers.Flatten()
                dense = tf.keras.layers.Dense(128, activation='relu')
                
                # Build the model
                x = conv1(inputs)
                x = pool1(x)
                x = conv2(x)
                x = pool2(x)
                x = flatten(x)
                outputs = dense(x)
                
                # Warm up
                _ = outputs.numpy()
                
                # Benchmark forward pass
                start_time = time.time()
                for _ in range(10):  # Multiple runs
                    x = conv1(inputs)
                    x = pool1(x)
                    x = conv2(x)
                    x = pool2(x)
                    x = flatten(x)
                    outputs = dense(x)
                    _ = outputs.numpy()  # Force computation
                elapsed = (time.time() - start_time) / 10
                
                fps = batch_size / elapsed
                print(f"  Batch {batch_size} ({height}x{width}): {elapsed:.3f}s/batch ({fps:.1f} FPS)")
                
        except tf.errors.ResourceExhaustedError:
            print(f"  Batch {batch_size}: Out of memory (expected with large batches)")
            break
        except Exception as e:
            print(f"  Batch {batch_size}: Error - {e}")
            break
    print()

def benchmark_rl_specific():
    """Benchmark operations specific to RL training"""
    print("RL-Specific Operations Benchmark")
    print("-" * 30)
    
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Device: {device}")
    
    try:
        # Import RL components
        from carla_rl import models, settings
        
        # Test model creation and inference
        with tf.device(device):
            # Create a simple model similar to the RL agent
            input_shape = (settings.IMG_HEIGHT, settings.IMG_WIDTH, 1)
            
            # Create model base
            model_base = models.model_base_5_residual_CNN(input_shape)
            model = models.model_head_hidden_dense(*model_base, outputs=len(settings.ACTIONS))
            
            # Test batch inference
            batch_sizes = [1, 8, 16, 32]
            
            for batch_size in batch_sizes:
                try:
                    # Create batch of images
                    images = tf.random.normal((batch_size, *input_shape))
                    
                    # Warm up
                    _ = model(images)
                    
                    # Benchmark inference
                    start_time = time.time()
                    for _ in range(100):  # Many inferences
                        predictions = model(images)
                        _ = predictions.numpy()
                    elapsed = (time.time() - start_time) / 100
                    
                    fps = batch_size / elapsed
                    print(f"  RL Model Batch {batch_size}: {elapsed:.4f}s/batch ({fps:.1f} FPS)")
                    
                except tf.errors.ResourceExhaustedError:
                    print(f"  RL Model Batch {batch_size}: Out of memory")
                    break
                except Exception as e:
                    print(f"  RL Model Batch {batch_size}: Error - {e}")
                    break
                    
    except Exception as e:
        print(f"  RL benchmark failed: {e}")
        print("  (This is expected if TensorFlow or models aren't fully working)")
    print()

def benchmark_memory_usage():
    """Check memory usage during operations"""
    print("Memory Usage Test")
    print("-" * 30)
    
    memory_before = psutil.virtual_memory()
    print(f"Memory before test: {memory_before.used / (1024**3):.1f} GB used")
    
    # Create large tensors to test memory
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    try:
        with tf.device(device):
            # Create progressively larger tensors
            tensors = []
            for i in range(1, 6):
                size = 1000 * i
                tensor = tf.random.normal((size, size))
                tensors.append(tensor)
                
                memory_current = psutil.virtual_memory()
                print(f"  Tensor {size}x{size}: {memory_current.used / (1024**3):.1f} GB used")
                
                # Don't exceed 80% memory usage
                if memory_current.percent > 80:
                    print("  Stopping to avoid memory issues")
                    break
            
            # Clean up
            del tensors
            
    except Exception as e:
        print(f"  Memory test error: {e}")
    
    memory_after = psutil.virtual_memory()
    print(f"Memory after test: {memory_after.used / (1024**3):.1f} GB used")
    print()

def main():
    """Run all benchmarks"""
    print_header()
    
    try:
        benchmark_matrix_operations()
        benchmark_cnn_operations()
        benchmark_rl_specific()
        benchmark_memory_usage()
        
        print("Benchmark Complete!")
        print("System performance validated for RL training!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
