#!/usr/bin/env python3
"""
System information collection script.

This script collects comprehensive system information for benchmarking
without requiring PyTorch or other heavy dependencies.
"""

import argparse
import json
import platform
import time

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
        import psutil  # noqa: F401
        dependencies["psutil"] = True
    except ImportError:
        pass
    
    try:
        import tensorrt  # noqa: F401
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
            except OSError:
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

def grade_capacity(value, excellent, good, adequate):
    """Grade a numeric hardware capacity against descending thresholds."""
    if value >= excellent:
        return "Excellent"
    if value >= good:
        return "Good"
    if value >= adequate:
        return "Adequate"
    return "Limited"


def analyze_cpu(cpu, analysis):
    """Add CPU capabilities and recommendations to an analysis."""
    if "error" not in cpu:
        physical_cores = cpu.get("physical_cores", 0)
        max_freq = cpu.get("max_frequency", 0)
        analysis["capabilities"]["cpu_performance"] = grade_capacity(
            physical_cores, 16, 8, 4
        )
        if physical_cores >= 16:
            analysis["hardware_grade"] = "High-End"
        elif physical_cores >= 8:
            analysis["hardware_grade"] = "Mid-Range"
        elif physical_cores >= 4:
            analysis["hardware_grade"] = "Entry-Level"
        else:
            analysis["hardware_grade"] = "Low-End"

        if physical_cores < 8:
            analysis["recommendations"].append("Consider upgrading to 8+ core CPU for better performance")
        if max_freq < 3000:
            analysis["recommendations"].append("Consider higher frequency CPU for better single-threaded performance")
    else:
        analysis["recommendations"].append("Install psutil to get detailed CPU information")


def analyze_memory(memory, analysis):
    """Add memory capabilities and recommendations to an analysis."""
    if "error" not in memory:
        total_gb = memory.get("total_gb", 0)
        analysis["capabilities"]["memory_capacity"] = grade_capacity(
            total_gb, 32, 16, 8
        )
        if total_gb < 8:
            analysis["recommendations"].append("Consider upgrading to 16+ GB RAM for better performance")
    else:
        analysis["recommendations"].append("Install psutil to get detailed memory information")


def analyze_gpu(gpu, analysis):
    """Add GPU capabilities and recommendations to an analysis."""
    if gpu.get("available", False):
        memory_gb = gpu.get("memory_total_gb", 0)
        analysis["capabilities"]["gpu_performance"] = grade_capacity(
            memory_gb, 16, 8, 4
        )
        if memory_gb < 4:
            analysis["recommendations"].append("Consider upgrading to GPU with 8+ GB VRAM")
    else:
        analysis["capabilities"]["gpu_performance"] = "Not Available"
        analysis["recommendations"].append("Consider adding GPU for significant performance improvement")


def analyze_dependencies(dependencies, gpu, analysis):
    """Add dependency recommendations to an analysis."""
    requirements = {
        "torch": "Install PyTorch for model inference capabilities",
        "numpy": "Install NumPy for numerical computations",
        "psutil": "Install psutil for system monitoring",
    }
    for dependency, recommendation in requirements.items():
        if not dependencies.get(dependency, False):
            analysis["recommendations"].append(recommendation)
    if not dependencies.get("cuda", False) and gpu.get("available", False):
        analysis["recommendations"].append("Install CUDA-enabled PyTorch for GPU acceleration")


def analyze_system_capabilities(system_info):
    """Analyze system capabilities and provide recommendations."""
    analysis = {
        "hardware_grade": "Unknown",
        "recommendations": [],
        "capabilities": {},
    }
    gpu = system_info.get("gpu", {})
    analyze_cpu(system_info.get("cpu", {}), analysis)
    analyze_memory(system_info.get("memory", {}), analysis)
    analyze_gpu(gpu, analysis)
    analyze_dependencies(system_info.get("dependencies", {}), gpu, analysis)
    return analysis


def print_hardware_summary(system_info):
    """Print platform, CPU, memory, and GPU information."""
    platform_info = system_info["platform"]
    print(f"Platform: {platform_info['system']} {platform_info['release']}")
    print(f"Architecture: {platform_info['machine']}")
    print(f"Python: {platform_info['python_version']}")

    cpu = system_info.get("cpu", {})
    if "error" not in cpu and cpu:
        print(f"CPU: {cpu['physical_cores']} cores, {cpu['logical_cores']} threads")
        if cpu.get("max_frequency", 0) > 0:
            print(f"Max Frequency: {cpu['max_frequency']:.1f} MHz")

    memory = system_info.get("memory", {})
    if "error" not in memory and memory:
        print(f"Memory: {memory['total_gb']:.1f} GB total, {memory['available_gb']:.1f} GB available")

    gpu = system_info.get("gpu", {})
    if gpu.get("available", False):
        print(f"GPU: {gpu['device_name']} ({gpu['memory_total_gb']:.1f} GB)")
    elif gpu:
        print("GPU: Not available")


def print_dependencies(dependencies):
    """Print dependency availability."""
    print("\nDependencies:")
    labels = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "psutil": "psutil",
        "cuda": "CUDA",
        "tensorrt": "TensorRT",
        "mkl": "Intel MKL",
    }
    for dependency, label in labels.items():
        status = "✓" if dependencies[dependency] else "✗"
        print(f"  {label}: {status}")


def print_analysis(analysis):
    """Print hardware grades and recommendations."""
    print(f"\nHardware Grade: {analysis['hardware_grade']}")
    capabilities = analysis["capabilities"]
    print(f"CPU Performance: {capabilities.get('cpu_performance', 'Unknown')}")
    print(f"Memory Capacity: {capabilities.get('memory_capacity', 'Unknown')}")
    print(f"GPU Performance: {capabilities.get('gpu_performance', 'Unknown')}")
    if analysis["recommendations"]:
        print("\nRecommendations:")
        for recommendation in analysis["recommendations"]:
            print(f"  • {recommendation}")

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
    
    print("\nSystem Information Summary:")
    print("=" * 40)
    print_hardware_summary(system_info)
    print_dependencies(system_info["dependencies"])
    if args.analysis:
        print_analysis(system_info["analysis"])

    if args.output:
        with open(args.output, "w") as output_file:
            json.dump(system_info, output_file, indent=2)
        print(f"\nSystem information saved to {args.output}")
    
    print("\nSystem information collection complete!")

if __name__ == "__main__":
    main()
