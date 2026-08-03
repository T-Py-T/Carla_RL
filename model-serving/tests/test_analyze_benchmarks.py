import pytest
from scripts.analyze_benchmarks import analyze_batch_performance


@pytest.mark.parametrize(
    ("ratio", "expected_grade"),
    [
        (100, "Good"),
        (50, "Adequate"),
        (20, "Poor"),
    ],
)
def test_batch_scaling_grades_use_strict_thresholds(ratio, expected_grade):
    benchmarks = {
        "batch_sizes": {
            "batch_1": {
                "latency": {"median_ms": 1},
                "throughput": {"throughput_rps": ratio},
            }
        }
    }

    analysis = analyze_batch_performance(benchmarks)

    assert analysis["scaling_efficiency"] == expected_grade
