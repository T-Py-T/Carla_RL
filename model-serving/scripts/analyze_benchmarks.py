#!/usr/bin/env python3
"""
Benchmark analysis and reporting script.

This script analyzes benchmark results and generates comprehensive
performance reports with recommendations.
"""

import argparse
import json
import sys
from typing import Dict, Any

def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark results file '{file_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in benchmark results file: {e}")
        sys.exit(1)

def analyze_system_info(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze system information and provide recommendations."""
    analysis = {
        "hardware_grade": "Unknown",
        "recommendations": [],
        "capabilities": {}
    }
    
    # Analyze CPU
    cpu = system_info.get("cpu", {})
    physical_cores = cpu.get("physical_cores", 0)
    cpu.get("logical_cores", 0)
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
    
    # Analyze memory
    memory = system_info.get("memory", {})
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
    if not deps.get("cuda", False) and gpu.get("available", False):
        analysis["recommendations"].append("Install CUDA-enabled PyTorch for GPU acceleration")
    
    return analysis

def analyze_latency_performance(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze latency performance and validate requirements."""
    analysis = {
        "p50_requirement_met": False,
        "p50_latency_ms": None,
        "performance_grade": "Unknown",
        "recommendations": []
    }
    
    if "optimization" not in benchmarks:
        analysis["recommendations"].append("Run optimization benchmarks to analyze latency")
        return analysis
    
    opt_results = benchmarks["optimization"]
    if "error" in opt_results:
        analysis["recommendations"].append(f"Optimization benchmark error: {opt_results['error']}")
        return analysis
    
    # Analyze optimized performance
    if "with_optimization" in opt_results:
        latency_data = opt_results["with_optimization"].get("latency", {})
        if "median_ms" in latency_data:
            p50_latency = latency_data["median_ms"]
            analysis["p50_latency_ms"] = p50_latency
            analysis["p50_requirement_met"] = p50_latency < 10.0
            
            # Performance grading
            if p50_latency < 5.0:
                analysis["performance_grade"] = "Excellent"
            elif p50_latency < 10.0:
                analysis["performance_grade"] = "Good"
            elif p50_latency < 20.0:
                analysis["performance_grade"] = "Adequate"
            else:
                analysis["performance_grade"] = "Poor"
                analysis["recommendations"].append("Latency performance needs significant improvement")
    
    # Analyze performance improvement
    if "performance_improvement" in opt_results:
        improvement = opt_results["performance_improvement"]
        speedup = improvement.get("latency_speedup", 1.0)
        improvement_percent = improvement.get("latency_improvement_percent", 0.0)
        
        if speedup > 2.0:
            analysis["recommendations"].append(f"Excellent optimization: {improvement_percent:.1f}% improvement")
        elif speedup > 1.5:
            analysis["recommendations"].append(f"Good optimization: {improvement_percent:.1f}% improvement")
        elif speedup > 1.1:
            analysis["recommendations"].append(f"Modest optimization: {improvement_percent:.1f}% improvement")
        else:
            analysis["recommendations"].append("Optimization provided minimal improvement")
    
    return analysis

def analyze_throughput_performance(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze throughput performance."""
    analysis = {
        "throughput_rps": None,
        "throughput_grade": "Unknown",
        "recommendations": []
    }
    
    if "optimization" not in benchmarks:
        return analysis
    
    opt_results = benchmarks["optimization"]
    if "with_optimization" in opt_results:
        throughput_data = opt_results["with_optimization"].get("throughput", {})
        if "throughput_rps" in throughput_data:
            throughput = throughput_data["throughput_rps"]
            analysis["throughput_rps"] = throughput
            
            if throughput > 2000:
                analysis["throughput_grade"] = "Excellent"
            elif throughput > 1000:
                analysis["throughput_grade"] = "Good"
            elif throughput > 500:
                analysis["throughput_grade"] = "Adequate"
            else:
                analysis["throughput_grade"] = "Poor"
                analysis["recommendations"].append("Throughput performance needs improvement")
    
    return analysis

def analyze_memory_performance(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze memory usage performance."""
    analysis = {
        "memory_usage_mb": None,
        "memory_efficiency": "Unknown",
        "recommendations": []
    }
    
    if "optimization" not in benchmarks:
        return analysis
    
    opt_results = benchmarks["optimization"]
    if "with_optimization" in opt_results:
        memory_data = opt_results["with_optimization"].get("memory", {})
        if "cpu_memory_mb" in memory_data:
            memory_usage = memory_data["cpu_memory_mb"]
            analysis["memory_usage_mb"] = memory_usage
            
            if memory_usage < 100:
                analysis["memory_efficiency"] = "Excellent"
            elif memory_usage < 500:
                analysis["memory_efficiency"] = "Good"
            elif memory_usage < 1000:
                analysis["memory_efficiency"] = "Adequate"
            else:
                analysis["memory_efficiency"] = "Poor"
                analysis["recommendations"].append("Memory usage is high, consider optimization")
    
    return analysis

def analyze_batch_performance(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze batch size performance scaling."""
    analysis = {
        "optimal_batch_size": None,
        "scaling_efficiency": "Unknown",
        "recommendations": []
    }
    
    if "batch_sizes" not in benchmarks:
        return analysis
    
    batch_results = benchmarks["batch_sizes"]
    if "error" in batch_results:
        analysis["recommendations"].append(f"Batch size benchmark error: {batch_results['error']}")
        return analysis
    
    # Find optimal batch size (best throughput per latency)
    best_ratio = 0
    optimal_batch = 1
    
    for key, result in batch_results.items():
        if key.startswith("batch_") and "error" not in result:
            batch_size = int(key.split("_")[1])
            latency = result.get("latency", {}).get("median_ms", float('inf'))
            throughput = result.get("throughput", {}).get("throughput_rps", 0)
            
            if latency > 0 and throughput > 0:
                ratio = throughput / latency
                if ratio > best_ratio:
                    best_ratio = ratio
                    optimal_batch = batch_size
    
    analysis["optimal_batch_size"] = optimal_batch
    
    if best_ratio > 100:
        analysis["scaling_efficiency"] = "Excellent"
    elif best_ratio > 50:
        analysis["scaling_efficiency"] = "Good"
    elif best_ratio > 20:
        analysis["scaling_efficiency"] = "Adequate"
    else:
        analysis["scaling_efficiency"] = "Poor"
        analysis["recommendations"].append("Batch scaling efficiency needs improvement")
    
    return analysis

def generate_performance_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive performance report."""
    report = []
    report.append("Policy-as-a-Service Performance Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # System analysis
    system_info = results.get("system_info", {})
    system_analysis = analyze_system_info(system_info)
    
    report.append("SYSTEM ANALYSIS")
    report.append("-" * 20)
    report.append(f"Hardware Grade: {system_analysis['hardware_grade']}")
    report.append(f"CPU Performance: {system_analysis['capabilities'].get('cpu_performance', 'Unknown')}")
    report.append(f"Memory Capacity: {system_analysis['capabilities'].get('memory_capacity', 'Unknown')}")
    report.append(f"GPU Performance: {system_analysis['capabilities'].get('gpu_performance', 'Unknown')}")
    report.append("")
    
    if system_analysis['recommendations']:
        report.append("System Recommendations:")
        for rec in system_analysis['recommendations']:
            report.append(f"  • {rec}")
        report.append("")
    
    # Performance analysis
    benchmarks = results.get("benchmarks", {})
    
    # Latency analysis
    latency_analysis = analyze_latency_performance(benchmarks)
    report.append("LATENCY PERFORMANCE")
    report.append("-" * 20)
    if latency_analysis['p50_latency_ms'] is not None:
        report.append(f"P50 Latency: {latency_analysis['p50_latency_ms']:.2f}ms")
        report.append(f"P50 Requirement Met: {'✓' if latency_analysis['p50_requirement_met'] else '✗'}")
        report.append(f"Performance Grade: {latency_analysis['performance_grade']}")
    else:
        report.append("Latency data not available")
    report.append("")
    
    if latency_analysis['recommendations']:
        report.append("Latency Recommendations:")
        for rec in latency_analysis['recommendations']:
            report.append(f"  • {rec}")
        report.append("")
    
    # Throughput analysis
    throughput_analysis = analyze_throughput_performance(benchmarks)
    report.append("THROUGHPUT PERFORMANCE")
    report.append("-" * 20)
    if throughput_analysis['throughput_rps'] is not None:
        report.append(f"Throughput: {throughput_analysis['throughput_rps']:.1f} RPS")
        report.append(f"Throughput Grade: {throughput_analysis['throughput_grade']}")
    else:
        report.append("Throughput data not available")
    report.append("")
    
    if throughput_analysis['recommendations']:
        report.append("Throughput Recommendations:")
        for rec in throughput_analysis['recommendations']:
            report.append(f"  • {rec}")
        report.append("")
    
    # Memory analysis
    memory_analysis = analyze_memory_performance(benchmarks)
    report.append("MEMORY PERFORMANCE")
    report.append("-" * 20)
    if memory_analysis['memory_usage_mb'] is not None:
        report.append(f"Memory Usage: {memory_analysis['memory_usage_mb']:.1f} MB")
        report.append(f"Memory Efficiency: {memory_analysis['memory_efficiency']}")
    else:
        report.append("Memory data not available")
    report.append("")
    
    if memory_analysis['recommendations']:
        report.append("Memory Recommendations:")
        for rec in memory_analysis['recommendations']:
            report.append(f"  • {rec}")
        report.append("")
    
    # Batch analysis
    batch_analysis = analyze_batch_performance(benchmarks)
    report.append("BATCH PERFORMANCE")
    report.append("-" * 20)
    if batch_analysis['optimal_batch_size'] is not None:
        report.append(f"Optimal Batch Size: {batch_analysis['optimal_batch_size']}")
        report.append(f"Scaling Efficiency: {batch_analysis['scaling_efficiency']}")
    else:
        report.append("Batch performance data not available")
    report.append("")
    
    if batch_analysis['recommendations']:
        report.append("Batch Recommendations:")
        for rec in batch_analysis['recommendations']:
            report.append(f"  • {rec}")
        report.append("")
    
    # Overall assessment
    report.append("OVERALL ASSESSMENT")
    report.append("-" * 20)
    
    if latency_analysis['p50_requirement_met']:
        report.append("✓ P50 latency requirement (10ms) is met")
    else:
        report.append("✗ P50 latency requirement (10ms) is not met")
    
    if throughput_analysis['throughput_rps'] and throughput_analysis['throughput_rps'] > 1000:
        report.append("✓ Throughput performance is good (>1000 RPS)")
    elif throughput_analysis['throughput_rps']:
        report.append("⚠ Throughput performance needs improvement")
    
    if memory_analysis['memory_efficiency'] in ['Excellent', 'Good']:
        report.append("✓ Memory efficiency is good")
    elif memory_analysis['memory_efficiency'] != 'Unknown':
        report.append("⚠ Memory efficiency needs improvement")
    
    report.append("")
    report.append("Report generated on: " + results.get("timestamp", "Unknown"))
    
    return "\n".join(report)

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate performance report")
    parser.add_argument("input_file", help="Input benchmark results JSON file")
    parser.add_argument("--output", "-o", help="Output report file (default: print to stdout)")
    parser.add_argument("--json", action="store_true", help="Output analysis as JSON")
    
    args = parser.parse_args()
    
    # Load benchmark results
    results = load_benchmark_results(args.input_file)
    
    if args.json:
        # Generate JSON analysis
        analysis = {
            "system_analysis": analyze_system_info(results.get("system_info", {})),
            "latency_analysis": analyze_latency_performance(results.get("benchmarks", {})),
            "throughput_analysis": analyze_throughput_performance(results.get("benchmarks", {})),
            "memory_analysis": analyze_memory_performance(results.get("benchmarks", {})),
            "batch_analysis": analyze_batch_performance(results.get("benchmarks", {}))
        }
        
        output = json.dumps(analysis, indent=2)
    else:
        # Generate text report
        output = generate_performance_report(results)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Analysis report saved to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()
