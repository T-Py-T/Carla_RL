#!/usr/bin/env python3
"""
System information collection script.

This script collects comprehensive system information for benchmarking
without requiring PyTorch or other heavy dependencies.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

def check_dependencies():
    """Check which dependencies are available."""
    dependencies = {
        "torch": False,
        "numpy": False,
        "psutil": False,
        "cuda": False,
        "tensorrt": False,
        "mkl": False
    }
    
    try:
        import torch
        dependencies["torch"] = True
        dependencies["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import numpy as np
        dependencies["numpy"] = True
        dependencies["mkl"] = "mkl" in str(np.__config__)
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        pass
    
    try:
        import tensorrt
        dependencies["tensorrt"] = True
    except ImportError:
        pass
    
    return dependencies

def get_system_info():
    """Get comprehensive system information."""
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "dependencies": check_dependencies()
    }
    
    # Get CPU info if psutil is available
    try:
        import psutil
        info["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
        
        # Try to get more detailed CPU info
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "avx" in cpuinfo.lower():
                        info["cpu"]["avx_support"] = True
                    if "avx2" in cpuinfo.lower():
                        info["cpu"]["avx2_support"] = True
                    if "sse" in cpuinfo.lower():
                        info["cpu"]["sse_support"] = True
            except (FileNotFoundError, OSError):
                pass
        
        # Get memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        info["memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "swap_gb": swap.total / (1024**3),
        }
        
    except ImportError:
        info["cpu"] = {"error": "psutil not available"}
        info["memory"] = {"error": "psutil not available"}
    
    # Get GPU info if PyTorch is available
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                "compute_capability": torch.cuda.get_device_capability(0),
                "cuda_version": torch.version.cuda,
            }
        else:
            info["gpu"] = {"available": False}
    except ImportError:
        info["gpu"] = {"available": False, "error": "PyTorch not available"}
    
    return info

def analyze_system_capabilities(system_info):
    """Analyze system capabilities and provide recommendations."""
    analysis = {
        "hardware_grade": "Unknown",
        "recommendations": [],
        "capabilities": {}
    }
    
    # Analyze CPU
    cpu = system_info.get("cpu", {})
    if "error" not in cpu:
        physical_cores = cpu.get("physical_cores", 0)
        logical_cores = cpu.get("logical_cores", 0)
        max_freq = cpu.get("max_frequency", 0)
        
        if physical_cores >= 16:
            analysis["hardware_grade"] = "High-End"
            analysis["capabilities"]["cpu_performance"] = "Excellent"
        elif physical_cores >= 8:
            analysis["hardware_grade"] = "Mid-Range"
            analysis["capabilities"]["cpu_performance"] = "Good"
        elif physical_cores >= 4:
            analysis["hardware_grade"] = "Entry-Level"
            analysis["capabilities"]["cpu_performance"] = "Adequate"
        else:
            analysis["hardware_grade"] = "Low-End"
            analysis["capabilities"]["cpu_performance"] = "Limited"
        
        # CPU recommendations
        if physical_cores < 8:
            analysis["recommendations"].append("Consider upgrading to 8+ core CPU for better performance")
        
        if max_freq < 3000:
            analysis["recommendations"].append("Consider higher frequency CPU for better single-threaded performance")
    else:
        analysis["recommendations"].append("Install psutil to get detailed CPU information")
    
    # Analyze memory
    memory = system_info.get("memory", {})
    if "error" not in memory:
        total_gb = memory.get("total_gb", 0)
        
        if total_gb >= 32:
            analysis["capabilities"]["memory_capacity"] = "Excellent"
        elif total_gb >= 16:
            analysis["capabilities"]["memory_capacity"] = "Good"
        elif total_gb >= 8:
            analysis["capabilities"]["memory_capacity"] = "Adequate"
        else:
            analysis["capabilities"]["memory_capacity"] = "Limited"
            analysis["recommendations"].append("Consider upgrading to 16+ GB RAM for better performance")
    else:
        analysis["recommendations"].append("Install psutil to get detailed memory information")
    
    # Analyze GPU
    gpu = system_info.get("gpu", {})
    if gpu.get("available", False):
        memory_gb = gpu.get("memory_total_gb", 0)
        if memory_gb >= 16:
            analysis["capabilities"]["gpu_performance"] = "Excellent"
        elif memory_gb >= 8:
            analysis["capabilities"]["gpu_performance"] = "Good"
        elif memory_gb >= 4:
            analysis["capabilities"]["gpu_performance"] = "Adequate"
        else:
            analysis["capabilities"]["gpu_performance"] = "Limited"
            analysis["recommendations"].append("Consider upgrading to GPU with 8+ GB VRAM")
    else:
        analysis["capabilities"]["gpu_performance"] = "Not Available"
        analysis["recommendations"].append("Consider adding GPU for significant performance improvement")
    
    # Analyze dependencies
    deps = system_info.get("dependencies", {})
    if not deps.get("torch", False):
        analysis["recommendations"].append("Install PyTorch for model inference capabilities")
    if not deps.get("numpy", False):
        analysis["recommendations"].append("Install NumPy for numerical computations")
    if not deps.get("psutil", False):
        analysis["recommendations"].append("Install psutil for system monitoring")
    if not deps.get("cuda", False) and gpu.get("available", False):
        analysis["recommendations"].append("Install CUDA-enabled PyTorch for GPU acceleration")
    
    return analysis

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect system information for benchmarking")
    parser.add_argument("--output", "-o", help="Output file for system information (JSON)")
    parser.add_argument("--analysis", action="store_true", help="Include system analysis and recommendations")
    
    args = parser.parse_args()
    
    print("Collecting system information...")
    system_info = get_system_info()
    
    if args.analysis:
        print("Analyzing system capabilities...")
        analysis = analyze_system_capabilities(system_info)
        system_info["analysis"] = analysis
    
    # Print summary
    print("\nSystem Information Summary:")
    print("=" * 40)
    print(f"Platform: {system_info['platform']['system']} {system_info['platform']['release']}")
    print(f"Architecture: {system_info['platform']['machine']}")
    print(f"Python: {system_info['platform']['python_version']}")
    
    if "cpu" in system_info and "error" not in system_info["cpu"]:
        cpu = system_info["cpu"]
        print(f"CPU: {cpu['physical_cores']} cores, {cpu['logical_cores']} threads")
        if cpu.get("max_frequency", 0) > 0:
            print(f"Max Frequency: {cpu['max_frequency']:.1f} MHz")
    
    if "memory" in system_info and "error" not in system_info["memory"]:
        memory = system_info["memory"]
        print(f"Memory: {memory['total_gb']:.1f} GB total, {memory['available_gb']:.1f} GB available")
    
    if "gpu" in system_info:
        gpu = system_info["gpu"]
        if gpu.get("available", False):
            print(f"GPU: {gpu['device_name']} ({gpu['memory_total_gb']:.1f} GB)")
        else:
            print("GPU: Not available")
    
    # Dependencies
    deps = system_info["dependencies"]
    print(f"\nDependencies:")
    print(f"  PyTorch: {'✓' if deps['torch'] else '✗'}")
    print(f"  NumPy: {'✓' if deps['numpy'] else '✗'}")
    print(f"  psutil: {'✓' if deps['psutil'] else '✗'}")
    print(f"  CUDA: {'✓' if deps['cuda'] else '✗'}")
    print(f"  TensorRT: {'✓' if deps['tensorrt'] else '✗'}")
    print(f"  Intel MKL: {'✓' if deps['mkl'] else '✗'}")
    
    if args.analysis and "analysis" in system_info:
        analysis = system_info["analysis"]
        print(f"\nHardware Grade: {analysis['hardware_grade']}")
        print(f"CPU Performance: {analysis['capabilities'].get('cpu_performance', 'Unknown')}")
        print(f"Memory Capacity: {analysis['capabilities'].get('memory_capacity', 'Unknown')}")
        print(f"GPU Performance: {analysis['capabilities'].get('gpu_performance', 'Unknown')}")
        
        if analysis["recommendations"]:
            print(f"\nRecommendations:")
            for rec in analysis["recommendations"]:
                print(f"  • {rec}")
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(system_info, f, indent=2)
        print(f"\nSystem information saved to {args.output}")
    
    print("\nSystem information collection complete!")

if __name__ == "__main__":
    main()
