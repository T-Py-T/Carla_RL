#!/usr/bin/env python3
"""
Platform-specific setup for Carla RL
Detects architecture and installs appropriate TensorFlow packages
"""

import platform
import subprocess
import sys

def get_platform_info():
    """Get detailed platform information"""
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'is_apple_silicon': platform.system() == 'Darwin' and platform.machine() == 'arm64',
        'is_intel_mac': platform.system() == 'Darwin' and platform.machine() == 'x86_64',
        'is_linux': platform.system() == 'Linux',
        'is_windows': platform.system() == 'Windows'
    }

def detect_gpu_support():
    """Detect available GPU support"""
    gpu_info = {
        'nvidia': False,
        'amd': False,
        'apple_metal': False,
        'intel': False
    }
    
    system_info = get_platform_info()
    
    if system_info['is_apple_silicon']:
        gpu_info['apple_metal'] = True
        
    elif system_info['is_linux'] or system_info['is_windows']:
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info['nvidia'] = True
        except FileNotFoundError:
            pass
            
        # Check for AMD GPU (basic detection)
        if system_info['is_linux']:
            try:
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    gpu_info['amd'] = True
            except FileNotFoundError:
                pass
    
    return gpu_info

def get_tensorflow_recommendation():
    """Get TensorFlow installation recommendation based on platform"""
    system_info = get_platform_info()
    gpu_info = detect_gpu_support()
    
    recommendations = []
    
    if system_info['is_apple_silicon']:
        recommendations.extend([
            "# Apple Silicon (M1/M2/M3/M4) detected",
            "uv add tensorflow-macos tensorflow-metal",
            "",
            "# Alternative: Use the optional dependency group",
            "uv sync --extra apple-gpu"
        ])
        
    elif gpu_info['nvidia']:
        recommendations.extend([
            "# NVIDIA GPU detected", 
            "uv add tensorflow[and-cuda]",
            "",
            "# Alternative: Use the optional dependency group",
            "uv sync --extra nvidia-gpu"
        ])
        
    else:
        recommendations.extend([
            "# CPU-only setup (no compatible GPU detected)",
            "uv add tensorflow",
            "",
            "# Standard installation should work"
        ])
    
    return recommendations

def detect_carla_support():
    """Detect Carla simulator support options"""
    import subprocess
    import shutil
    
    system_info = get_platform_info()
    carla_info = {
        'native_support': False,
        'docker_available': False,
        'recommended_setup': 'mock'
    }
    
    # Check native Carla support
    if system_info['system'] in ['Linux', 'Windows']:
        carla_info['native_support'] = True
        carla_info['recommended_setup'] = 'native'
    
    # Check Docker availability
    if shutil.which('docker'):
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                carla_info['docker_available'] = True
                if not carla_info['native_support']:
                    carla_info['recommended_setup'] = 'docker'
        except:
            pass
    
    return carla_info

def print_platform_report():
    """Print full platform report"""
    system_info = get_platform_info()
    gpu_info = detect_gpu_support()
    tf_rec = get_tensorflow_recommendation()
    carla_info = detect_carla_support()
    
    print("Platform Detection Report")
    print("=" * 50)
    print(f"Operating System: {system_info['system']}")
    print(f"Architecture: {system_info['machine']}")
    print(f"Processor: {system_info['processor']}")
    print(f"Python Version: {system_info['python_version']}")
    print()
    
    print("GPU Support Detection")
    print("-" * 30)
    if gpu_info['apple_metal']:
        print("PASS: Apple Metal GPU support available")
    if gpu_info['nvidia']:
        print("PASS: NVIDIA CUDA GPU detected")
    if gpu_info['amd']:
        print("WARNING: AMD GPU detected (limited TensorFlow support)")
    if gpu_info['intel']:
        print("INFO: Intel GPU detected")
    if not any(gpu_info.values()):
        print("CPU-only setup (no compatible GPU detected)")
    print()
    
    print("Carla Simulator Support")
    print("-" * 30)
    if carla_info['native_support']:
        print("PASS: Native Carla support available")
        print("- Can run Carla simulator directly on this OS")
    else:
        print("WARNING: No native Carla support for macOS")
        print("- Carla simulator requires Linux or Windows")
    
    if carla_info['docker_available']:
        print("PASS: Docker available for Carla containerization")
        print("- Can run Carla via Docker container")
    else:
        print("WARNING: Docker not available")
        print("- Install Docker to run Carla on macOS")
    
    print(f"Recommended setup: {carla_info['recommended_setup'].upper()}")
    print()
    
    print("TensorFlow Installation Recommendations")
    print("-" * 50)
    for line in tf_rec:
        print(line)
    print()
    
    print("Next Steps")
    print("-" * 20)
    print("1. Run the recommended TensorFlow installation command above")
    print("2. Run 'make validate' to verify setup")
    print("3. Run 'make check-gpu' to verify GPU acceleration")
    print("4. Start training with 'make train'")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        import json
        system_info = get_platform_info()
        gpu_info = detect_gpu_support()
        result = {
            'platform': system_info,
            'gpu': gpu_info,
            'tensorflow_recommendations': get_tensorflow_recommendation()
        }
        print(json.dumps(result, indent=2))
    else:
        print_platform_report()

if __name__ == "__main__":
    main()
