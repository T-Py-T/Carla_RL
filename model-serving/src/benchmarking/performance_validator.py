"""
Performance validation and baseline comparison.

This module provides validation of performance requirements against
baseline measurements and hardware-specific thresholds.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import BenchmarkResult, LatencyStats, ThroughputStats, MemoryStats
from .hardware_detector import HardwareInfo


@dataclass
class PerformanceBaseline:
    """Performance baseline for specific hardware configuration."""

    hardware_signature: str
    cpu_model: str
    gpu_model: Optional[str]
    memory_gb: float

    # Baseline metrics
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_usage_mb: float

    # Test configuration
    batch_size: int
    test_date: str
    torch_version: str
    python_version: str


@dataclass
class ValidationResult:
    """Result of performance validation against requirements."""

    # Requirements validation
    p50_requirement_met: bool
    p95_requirement_met: bool
    p99_requirement_met: bool
    throughput_requirement_met: bool
    memory_requirement_met: bool

    # Performance vs baseline
    p50_vs_baseline: float  # Percentage difference from baseline
    p95_vs_baseline: float
    p99_vs_baseline: float
    throughput_vs_baseline: float
    memory_vs_baseline: float

    # Overall assessment
    overall_success: bool
    performance_grade: str  # A, B, C, D, F
    recommendations: List[str]

    # Detailed metrics
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    hardware_info: HardwareInfo


class PerformanceValidator:
    """
    Performance validation engine for requirement compliance.

    Validates performance results against requirements and compares
    against hardware-specific baselines for performance assessment.
    """

    def __init__(self, baselines_dir: Optional[Path] = None):
        """Initialize performance validator with baselines directory."""
        self.baselines_dir = baselines_dir or Path("model-serving/baselines")
        self.baselines_dir.mkdir(exist_ok=True)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._load_baselines()

    def validate_requirements(
        self, benchmark_result: BenchmarkResult, requirements: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """Validate benchmark results against performance requirements."""

        # Default requirements
        if requirements is None:
            requirements = {
                "p50_latency_ms": 10.0,
                "p95_latency_ms": 20.0,
                "p99_latency_ms": 50.0,
                "throughput_rps": 1000.0,
                "memory_mb": 1024.0,
            }

        # Check requirement compliance
        p50_requirement_met = (
            benchmark_result.latency_stats.p50_ms <= requirements["p50_latency_ms"]
        )
        p95_requirement_met = (
            benchmark_result.latency_stats.p95_ms <= requirements["p95_latency_ms"]
        )
        p99_requirement_met = (
            benchmark_result.latency_stats.p99_ms <= requirements["p99_latency_ms"]
        )
        throughput_requirement_met = (
            benchmark_result.throughput_stats.requests_per_second >= requirements["throughput_rps"]
        )
        memory_requirement_met = (
            benchmark_result.memory_stats.peak_memory_mb <= requirements["memory_mb"]
        )

        # Find matching baseline
        baseline = self._find_matching_baseline(benchmark_result.hardware_info)

        # Calculate performance vs baseline
        if baseline:
            p50_vs_baseline = self._calculate_percentage_diff(
                benchmark_result.latency_stats.p50_ms, baseline.p50_latency_ms
            )
            p95_vs_baseline = self._calculate_percentage_diff(
                benchmark_result.latency_stats.p95_ms, baseline.p95_latency_ms
            )
            p99_vs_baseline = self._calculate_percentage_diff(
                benchmark_result.latency_stats.p99_ms, baseline.p99_latency_ms
            )
            throughput_vs_baseline = self._calculate_percentage_diff(
                benchmark_result.throughput_stats.requests_per_second, baseline.throughput_rps
            )
            memory_vs_baseline = self._calculate_percentage_diff(
                benchmark_result.memory_stats.peak_memory_mb, baseline.memory_usage_mb
            )

            baseline_metrics = {
                "p50_latency_ms": baseline.p50_latency_ms,
                "p95_latency_ms": baseline.p95_latency_ms,
                "p99_latency_ms": baseline.p99_latency_ms,
                "throughput_rps": baseline.throughput_rps,
                "memory_mb": baseline.memory_usage_mb,
            }
        else:
            p50_vs_baseline = 0.0
            p95_vs_baseline = 0.0
            p99_vs_baseline = 0.0
            throughput_vs_baseline = 0.0
            memory_vs_baseline = 0.0
            baseline_metrics = {}

        # Calculate overall success
        overall_success = (
            p50_requirement_met
            and p95_requirement_met
            and p99_requirement_met
            and throughput_requirement_met
            and memory_requirement_met
        )

        # Calculate performance grade
        performance_grade = self._calculate_performance_grade(
            p50_requirement_met,
            p95_requirement_met,
            p99_requirement_met,
            throughput_requirement_met,
            memory_requirement_met,
            p50_vs_baseline,
            throughput_vs_baseline,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(benchmark_result, baseline, requirements)

        # Current metrics
        current_metrics = {
            "p50_latency_ms": benchmark_result.latency_stats.p50_ms,
            "p95_latency_ms": benchmark_result.latency_stats.p95_ms,
            "p99_latency_ms": benchmark_result.latency_stats.p99_ms,
            "throughput_rps": benchmark_result.throughput_stats.requests_per_second,
            "memory_mb": benchmark_result.memory_stats.peak_memory_mb,
        }

        return ValidationResult(
            p50_requirement_met=p50_requirement_met,
            p95_requirement_met=p95_requirement_met,
            p99_requirement_met=p99_requirement_met,
            throughput_requirement_met=throughput_requirement_met,
            memory_requirement_met=memory_requirement_met,
            p50_vs_baseline=p50_vs_baseline,
            p95_vs_baseline=p95_vs_baseline,
            p99_vs_baseline=p99_vs_baseline,
            throughput_vs_baseline=throughput_vs_baseline,
            memory_vs_baseline=memory_vs_baseline,
            overall_success=overall_success,
            performance_grade=performance_grade,
            recommendations=recommendations,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_info=HardwareInfo(
                cpu=benchmark_result.hardware_info.get("cpu", {}),
                gpu=benchmark_result.hardware_info.get("gpu"),
                memory=benchmark_result.hardware_info.get("memory", {}),
                platform=benchmark_result.hardware_info.get("platform", ""),
                python_version=benchmark_result.hardware_info.get("python_version", ""),
                torch_version=benchmark_result.hardware_info.get("torch_version", ""),
                optimization_recommendations=[],
            ),
        )

    def save_baseline(self, benchmark_result: BenchmarkResult) -> None:
        """Save benchmark result as a new baseline."""
        hardware_signature = self._generate_hardware_signature(benchmark_result.hardware_info)

        baseline = PerformanceBaseline(
            hardware_signature=hardware_signature,
            cpu_model=benchmark_result.hardware_info.get("cpu", {}).get("model", "Unknown"),
            gpu_model=benchmark_result.hardware_info.get("gpu", {}).get("model")
            if benchmark_result.hardware_info.get("gpu")
            else None,
            memory_gb=benchmark_result.hardware_info.get("memory", {}).get("total_gb", 0.0),
            p50_latency_ms=benchmark_result.latency_stats.p50_ms,
            p95_latency_ms=benchmark_result.latency_stats.p95_ms,
            p99_latency_ms=benchmark_result.latency_stats.p99_ms,
            throughput_rps=benchmark_result.throughput_stats.requests_per_second,
            memory_usage_mb=benchmark_result.memory_stats.peak_memory_mb,
            batch_size=1,  # Default batch size for baseline
            test_date=benchmark_result.timestamp,
            torch_version=benchmark_result.hardware_info.get("torch_version", ""),
            python_version=benchmark_result.hardware_info.get("python_version", ""),
        )

        self.baselines[hardware_signature] = baseline
        self._save_baseline_to_file(baseline)

    def _load_baselines(self) -> None:
        """Load existing baselines from files."""
        for baseline_file in self.baselines_dir.glob("baseline_*.json"):
            try:
                with open(baseline_file, "r") as f:
                    data = json.load(f)
                    baseline = PerformanceBaseline(**data)
                    self.baselines[baseline.hardware_signature] = baseline
            except Exception as e:
                print(f"Warning: Failed to load baseline {baseline_file}: {e}")

    def _save_baseline_to_file(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to JSON file."""
        filename = f"baseline_{baseline.hardware_signature}.json"
        filepath = self.baselines_dir / filename

        with open(filepath, "w") as f:
            json.dump(baseline.__dict__, f, indent=2)

    def _find_matching_baseline(
        self, hardware_info: Dict[str, Any]
    ) -> Optional[PerformanceBaseline]:
        """Find matching baseline for hardware configuration."""
        hardware_signature = self._generate_hardware_signature(hardware_info)
        return self.baselines.get(hardware_signature)

    def _generate_hardware_signature(self, hardware_info: Dict[str, Any]) -> str:
        """Generate unique signature for hardware configuration."""
        cpu_model = hardware_info.get("cpu", {}).get("model", "Unknown")
        gpu_model = (
            hardware_info.get("gpu", {}).get("model") if hardware_info.get("gpu") else "None"
        )
        memory_gb = hardware_info.get("memory", {}).get("total_gb", 0.0)

        # Create a simple signature
        signature = f"{cpu_model}_{gpu_model}_{memory_gb:.0f}GB"
        return signature.replace(" ", "_").replace("/", "_")

    def _calculate_percentage_diff(self, current: float, baseline: float) -> float:
        """Calculate percentage difference from baseline."""
        if baseline == 0:
            return 0.0
        return ((current - baseline) / baseline) * 100.0

    def _calculate_performance_grade(
        self,
        p50_met: bool,
        p95_met: bool,
        p99_met: bool,
        throughput_met: bool,
        memory_met: bool,
        p50_vs_baseline: float,
        throughput_vs_baseline: float,
    ) -> str:
        """Calculate performance grade based on requirements and baseline comparison."""

        # Count requirements met
        requirements_met = sum([p50_met, p95_met, p99_met, throughput_met, memory_met])

        # Base grade on requirements
        if requirements_met == 5:
            base_grade = "A"
        elif requirements_met >= 4:
            base_grade = "B"
        elif requirements_met >= 3:
            base_grade = "C"
        elif requirements_met >= 2:
            base_grade = "D"
        else:
            base_grade = "F"

        # Adjust based on baseline comparison
        if p50_vs_baseline < -10:  # 10% better than baseline
            base_grade = self._upgrade_grade(base_grade)
        elif p50_vs_baseline > 20:  # 20% worse than baseline
            base_grade = self._downgrade_grade(base_grade)

        return base_grade

    def _upgrade_grade(self, grade: str) -> str:
        """Upgrade performance grade."""
        grade_map = {"F": "D", "D": "C", "C": "B", "B": "A", "A": "A+"}
        return grade_map.get(grade, grade)

    def _downgrade_grade(self, grade: str) -> str:
        """Downgrade performance grade."""
        grade_map = {"A+": "A", "A": "B", "B": "C", "C": "D", "D": "F", "F": "F"}
        return grade_map.get(grade, grade)

    def _generate_recommendations(
        self,
        benchmark_result: BenchmarkResult,
        baseline: Optional[PerformanceBaseline],
        requirements: Dict[str, float],
    ) -> List[str]:
        """Generate optimization recommendations based on validation results."""
        recommendations = []

        # Latency recommendations
        if not benchmark_result.p50_requirement_met:
            recommendations.append(
                "P50 latency exceeds 10ms requirement - consider CPU optimization or smaller batch sizes"
            )
        if not benchmark_result.p95_requirement_met:
            recommendations.append(
                "P95 latency exceeds 20ms requirement - check for performance bottlenecks"
            )
        if not benchmark_result.p99_requirement_met:
            recommendations.append(
                "P99 latency exceeds 50ms requirement - investigate memory allocation patterns"
            )

        # Throughput recommendations
        if not benchmark_result.throughput_requirement_met:
            recommendations.append(
                "Throughput below 1000 RPS requirement - consider batch processing optimization"
            )

        # Memory recommendations
        if not benchmark_result.memory_requirement_met:
            recommendations.append(
                "Memory usage exceeds 1GB limit - optimize tensor allocation and caching"
            )

        # Baseline comparison recommendations
        if baseline:
            if benchmark_result.latency_stats.p50_ms > baseline.p50_latency_ms * 1.2:
                recommendations.append(
                    "P50 latency is 20% worse than baseline - check for performance regressions"
                )
            if (
                benchmark_result.throughput_stats.requests_per_second
                < baseline.throughput_rps * 0.8
            ):
                recommendations.append(
                    "Throughput is 20% worse than baseline - investigate system changes"
                )

        # Hardware-specific recommendations
        if benchmark_result.hardware_info.get("torch_cuda_available"):
            recommendations.append(
                "GPU available - consider enabling GPU inference for better performance"
            )

        if benchmark_result.hardware_info.get("cpu_count", 0) >= 8:
            recommendations.append(
                "Multi-core CPU detected - enable parallel processing for batch inference"
            )

        return recommendations

    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall result
        status = "PASS" if validation_result.overall_success else "FAIL"
        report.append(f"Overall Status: {status}")
        report.append(f"Performance Grade: {validation_result.performance_grade}")
        report.append("")

        # Requirements validation
        report.append("REQUIREMENTS VALIDATION:")
        report.append(
            f"  P50 Latency: {validation_result.current_metrics['p50_latency_ms']:.2f}ms {'✓' if validation_result.p50_requirement_met else '✗'}"
        )
        report.append(
            f"  P95 Latency: {validation_result.current_metrics['p95_latency_ms']:.2f}ms {'✓' if validation_result.p95_requirement_met else '✗'}"
        )
        report.append(
            f"  P99 Latency: {validation_result.current_metrics['p99_latency_ms']:.2f}ms {'✓' if validation_result.p99_requirement_met else '✗'}"
        )
        report.append(
            f"  Throughput: {validation_result.current_metrics['throughput_rps']:.1f} RPS {'✓' if validation_result.throughput_requirement_met else '✗'}"
        )
        report.append(
            f"  Memory: {validation_result.current_metrics['memory_mb']:.1f} MB {'✓' if validation_result.memory_requirement_met else '✗'}"
        )
        report.append("")

        # Baseline comparison
        if validation_result.baseline_metrics:
            report.append("BASELINE COMPARISON:")
            report.append(f"  P50 vs Baseline: {validation_result.p50_vs_baseline:+.1f}%")
            report.append(f"  P95 vs Baseline: {validation_result.p95_vs_baseline:+.1f}%")
            report.append(f"  P99 vs Baseline: {validation_result.p99_vs_baseline:+.1f}%")
            report.append(
                f"  Throughput vs Baseline: {validation_result.throughput_vs_baseline:+.1f}%"
            )
            report.append(f"  Memory vs Baseline: {validation_result.memory_vs_baseline:+.1f}%")
            report.append("")

        # Recommendations
        if validation_result.recommendations:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(validation_result.recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")

        return "\n".join(report)
