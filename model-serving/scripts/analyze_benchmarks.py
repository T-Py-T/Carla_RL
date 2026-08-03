#!/usr/bin/env python3
"""
Benchmark analysis and reporting script.

This script analyzes benchmark results and generates comprehensive
performance reports with recommendations.
"""

import argparse
import json
import sys
from typing import Any, Dict


def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark results file '{file_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in benchmark results file: {e}")
        sys.exit(1)


def grade_value(value, excellent, good, adequate, *, inclusive=True):
    """Grade a numeric value against descending thresholds."""
    grades = (
        (excellent, "Excellent"),
        (good, "Good"),
        (adequate, "Adequate"),
    )
    for threshold, grade in grades:
        if value > threshold or (inclusive and value == threshold):
            return grade
    return "Limited"


def analyze_cpu(cpu, analysis):
    """Populate CPU analysis fields."""
    physical_cores = cpu.get("physical_cores", 0)
    hardware_grades = ((16, "High-End"), (8, "Mid-Range"), (4, "Entry-Level"))
    analysis["hardware_grade"] = "Low-End"
    for threshold, grade in hardware_grades:
        if physical_cores >= threshold:
            analysis["hardware_grade"] = grade
            break
    analysis["capabilities"]["cpu_performance"] = grade_value(physical_cores, 16, 8, 4)
    if physical_cores < 8:
        analysis["recommendations"].append(
            "Consider upgrading to 8+ core CPU for better performance"
        )
    if cpu.get("max_frequency", 0) < 3000:
        analysis["recommendations"].append(
            "Consider higher frequency CPU for better single-threaded performance"
        )


def analyze_memory(memory, analysis):
    """Populate memory analysis fields."""
    total_gb = memory.get("total_gb", 0)
    analysis["capabilities"]["memory_capacity"] = grade_value(total_gb, 32, 16, 8)
    if total_gb < 8:
        analysis["recommendations"].append(
            "Consider upgrading to 16+ GB RAM for better performance"
        )


def analyze_gpu(gpu, analysis):
    """Populate GPU analysis fields."""
    if not gpu.get("available", False):
        analysis["capabilities"]["gpu_performance"] = "Not Available"
        analysis["recommendations"].append(
            "Consider adding GPU for significant performance improvement"
        )
        return

    memory_gb = gpu.get("memory_total_gb", 0)
    analysis["capabilities"]["gpu_performance"] = grade_value(memory_gb, 16, 8, 4)
    if memory_gb < 4:
        analysis["recommendations"].append("Consider upgrading to GPU with 8+ GB VRAM")


def analyze_dependencies(dependencies, gpu, analysis):
    """Populate software dependency recommendations."""
    requirements = {
        "torch": "Install PyTorch for model inference capabilities",
        "numpy": "Install NumPy for numerical computations",
    }
    for dependency, recommendation in requirements.items():
        if not dependencies.get(dependency, False):
            analysis["recommendations"].append(recommendation)
    if not dependencies.get("cuda", False) and gpu.get("available", False):
        analysis["recommendations"].append("Install CUDA-enabled PyTorch for GPU acceleration")


def analyze_system_info(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze system information and provide recommendations."""
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


def analyze_latency_result(optimization, analysis):
    """Populate latency measurements and grade."""
    latency_data = optimization.get("with_optimization", {}).get("latency", {})
    if "median_ms" not in latency_data:
        return
    p50_latency = latency_data["median_ms"]
    analysis["p50_latency_ms"] = p50_latency
    analysis["p50_requirement_met"] = p50_latency < 10.0
    if p50_latency < 5.0:
        analysis["performance_grade"] = "Excellent"
    elif p50_latency < 10.0:
        analysis["performance_grade"] = "Good"
    elif p50_latency < 20.0:
        analysis["performance_grade"] = "Adequate"
    else:
        analysis["performance_grade"] = "Poor"
        analysis["recommendations"].append("Latency performance needs significant improvement")


def analyze_latency_improvement(optimization, analysis):
    """Describe the measured optimization improvement."""
    improvement = optimization.get("performance_improvement")
    if not improvement:
        return
    speedup = improvement.get("latency_speedup", 1.0)
    improvement_percent = improvement.get("latency_improvement_percent", 0.0)
    if speedup > 2.0:
        label = "Excellent optimization"
    elif speedup > 1.5:
        label = "Good optimization"
    elif speedup > 1.1:
        label = "Modest optimization"
    else:
        analysis["recommendations"].append("Optimization provided minimal improvement")
        return
    analysis["recommendations"].append(f"{label}: {improvement_percent:.1f}% improvement")


def analyze_latency_performance(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze latency performance and validate requirements."""
    analysis = {
        "p50_requirement_met": False,
        "p50_latency_ms": None,
        "performance_grade": "Unknown",
        "recommendations": [],
    }
    optimization = benchmarks.get("optimization")
    if not optimization:
        analysis["recommendations"].append("Run optimization benchmarks to analyze latency")
        return analysis
    if "error" in optimization:
        analysis["recommendations"].append(f"Optimization benchmark error: {optimization['error']}")
        return analysis
    analyze_latency_result(optimization, analysis)
    analyze_latency_improvement(optimization, analysis)
    return analysis


def analyze_throughput_performance(benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze throughput performance."""
    analysis = {"throughput_rps": None, "throughput_grade": "Unknown", "recommendations": []}

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
    analysis = {"memory_usage_mb": None, "memory_efficiency": "Unknown", "recommendations": []}

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
    analysis = {"optimal_batch_size": None, "scaling_efficiency": "Unknown", "recommendations": []}

    if "batch_sizes" not in benchmarks:
        return analysis

    batch_results = benchmarks["batch_sizes"]
    if "error" in batch_results:
        analysis["recommendations"].append(f"Batch size benchmark error: {batch_results['error']}")
        return analysis

    best_ratio, optimal_batch = find_optimal_batch(batch_results)
    analysis["optimal_batch_size"] = optimal_batch
    analysis["scaling_efficiency"] = grade_value(best_ratio, 100, 50, 20, inclusive=False)
    if best_ratio <= 20:
        analysis["scaling_efficiency"] = "Poor"
        analysis["recommendations"].append("Batch scaling efficiency needs improvement")
    return analysis


def find_optimal_batch(batch_results):
    """Return the batch size with the best throughput-to-latency ratio."""
    best_ratio = 0
    optimal_batch = 1
    for key, result in batch_results.items():
        if not key.startswith("batch_") or "error" in result:
            continue
        latency = result.get("latency", {}).get("median_ms", float("inf"))
        throughput = result.get("throughput", {}).get("throughput_rps", 0)
        if latency <= 0 or throughput <= 0:
            continue
        ratio = throughput / latency
        if ratio > best_ratio:
            best_ratio = ratio
            optimal_batch = int(key.split("_")[1])
    return best_ratio, optimal_batch


def append_recommendations(report, title, recommendations):
    """Append a labeled recommendation list when non-empty."""
    if not recommendations:
        return
    report.append(f"{title} Recommendations:")
    report.extend(f"  • {recommendation}" for recommendation in recommendations)
    report.append("")


def format_system_section(analysis):
    """Format the system-analysis report section."""
    capabilities = analysis["capabilities"]
    report = [
        "SYSTEM ANALYSIS",
        "-" * 20,
        f"Hardware Grade: {analysis['hardware_grade']}",
        f"CPU Performance: {capabilities.get('cpu_performance', 'Unknown')}",
        f"Memory Capacity: {capabilities.get('memory_capacity', 'Unknown')}",
        f"GPU Performance: {capabilities.get('gpu_performance', 'Unknown')}",
        "",
    ]
    append_recommendations(report, "System", analysis["recommendations"])
    return report


def format_latency_section(analysis):
    """Format the latency-analysis report section."""
    report = ["LATENCY PERFORMANCE", "-" * 20]
    if analysis["p50_latency_ms"] is None:
        report.append("Latency data not available")
    else:
        status = "✓" if analysis["p50_requirement_met"] else "✗"
        report.extend(
            [
                f"P50 Latency: {analysis['p50_latency_ms']:.2f}ms",
                f"P50 Requirement Met: {status}",
                f"Performance Grade: {analysis['performance_grade']}",
            ]
        )
    report.append("")
    append_recommendations(report, "Latency", analysis["recommendations"])
    return report


def format_throughput_section(analysis):
    """Format the throughput-analysis report section."""
    report = ["THROUGHPUT PERFORMANCE", "-" * 20]
    if analysis["throughput_rps"] is None:
        report.append("Throughput data not available")
    else:
        report.extend(
            [
                f"Throughput: {analysis['throughput_rps']:.1f} RPS",
                f"Throughput Grade: {analysis['throughput_grade']}",
            ]
        )
    report.append("")
    append_recommendations(report, "Throughput", analysis["recommendations"])
    return report


def format_memory_section(analysis):
    """Format the memory-analysis report section."""
    report = ["MEMORY PERFORMANCE", "-" * 20]
    if analysis["memory_usage_mb"] is None:
        report.append("Memory data not available")
    else:
        report.extend(
            [
                f"Memory Usage: {analysis['memory_usage_mb']:.1f} MB",
                f"Memory Efficiency: {analysis['memory_efficiency']}",
            ]
        )
    report.append("")
    append_recommendations(report, "Memory", analysis["recommendations"])
    return report


def format_batch_section(analysis):
    """Format the batch-analysis report section."""
    report = ["BATCH PERFORMANCE", "-" * 20]
    if analysis["optimal_batch_size"] is None:
        report.append("Batch performance data not available")
    else:
        report.extend(
            [
                f"Optimal Batch Size: {analysis['optimal_batch_size']}",
                f"Scaling Efficiency: {analysis['scaling_efficiency']}",
            ]
        )
    report.append("")
    append_recommendations(report, "Batch", analysis["recommendations"])
    return report


def format_overall_assessment(latency, throughput, memory):
    """Format the cross-cutting performance assessment."""
    report = ["OVERALL ASSESSMENT", "-" * 20]
    latency_status = "✓" if latency["p50_requirement_met"] else "✗"
    latency_suffix = "is met" if latency["p50_requirement_met"] else "is not met"
    report.append(f"{latency_status} P50 latency requirement (10ms) {latency_suffix}")

    throughput_rps = throughput["throughput_rps"]
    if throughput_rps and throughput_rps > 1000:
        report.append("✓ Throughput performance is good (>1000 RPS)")
    elif throughput_rps:
        report.append("⚠ Throughput performance needs improvement")

    if memory["memory_efficiency"] in ["Excellent", "Good"]:
        report.append("✓ Memory efficiency is good")
    elif memory["memory_efficiency"] != "Unknown":
        report.append("⚠ Memory efficiency needs improvement")
    return report


def generate_performance_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive performance report."""
    benchmarks = results.get("benchmarks", {})
    system = analyze_system_info(results.get("system_info", {}))
    latency = analyze_latency_performance(benchmarks)
    throughput = analyze_throughput_performance(benchmarks)
    memory = analyze_memory_performance(benchmarks)
    batch = analyze_batch_performance(benchmarks)

    report = ["Policy-as-a-Service Performance Analysis Report", "=" * 50, ""]
    report.extend(format_system_section(system))
    report.extend(format_latency_section(latency))
    report.extend(format_throughput_section(throughput))
    report.extend(format_memory_section(memory))
    report.extend(format_batch_section(batch))
    report.extend(format_overall_assessment(latency, throughput, memory))
    report.extend(["", "Report generated on: " + results.get("timestamp", "Unknown")])
    return "\n".join(report)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate performance report"
    )
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
            "batch_analysis": analyze_batch_performance(results.get("benchmarks", {})),
        }

        output = json.dumps(analysis, indent=2)
    else:
        # Generate text report
        output = generate_performance_report(results)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Analysis report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
