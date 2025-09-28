#!/bin/bash
"""
Complete benchmarking workflow for Policy-as-a-Service.

This script runs a comprehensive benchmarking suite to collect
empirical performance data on the local machine.
"""

set -e  # Exit on any error

echo "Policy-as-a-Service Full Benchmark Suite"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the model-serving directory"
    exit 1
fi

# Create results directory
mkdir -p benchmark_results
cd benchmark_results

# Set up environment if needed
if [ ! -d "../venv" ]; then
    echo "Setting up benchmarking environment..."
    cd ..
    ./scripts/setup_benchmarking.sh
    cd benchmark_results
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="benchmark_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

echo "Running benchmark suite: $RESULTS_DIR"
echo "====================================="

# 1. System Information Collection
echo ""
echo "1. Collecting system information..."
python3 ../../scripts/local_benchmark.py --system-info-only --output system_info.json
echo "✓ System information collected"

# 2. Quick Performance Test
echo ""
echo "2. Running quick performance test..."
python3 ../../scripts/local_benchmark.py --quick --output quick_benchmark.json
echo "✓ Quick benchmark completed"

# 3. Comprehensive Performance Test
echo ""
echo "3. Running comprehensive performance test..."
python3 ../../scripts/local_benchmark.py --comprehensive --output comprehensive_benchmark.json
echo "✓ Comprehensive benchmark completed"

# 4. Hardware Detection Test
echo ""
echo "4. Testing hardware detection..."
python3 ../../scripts/optimization_manager.py detect --output hardware_detection.json
echo "✓ Hardware detection completed"

# 5. Optimization Test
echo ""
echo "5. Testing optimization application..."
python3 ../../scripts/optimization_manager.py optimize --output optimization_test.json
echo "✓ Optimization test completed"

# 6. Analyze Results
echo ""
echo "6. Analyzing benchmark results..."
python3 ../../scripts/analyze_benchmarks.py comprehensive_benchmark.json --output performance_report.txt
echo "✓ Performance analysis completed"

# 7. Generate JSON Analysis
echo ""
echo "7. Generating JSON analysis..."
python3 ../../scripts/analyze_benchmarks.py comprehensive_benchmark.json --json --output analysis.json
echo "✓ JSON analysis completed"

# 8. Create Summary
echo ""
echo "8. Creating benchmark summary..."
cat > benchmark_summary.txt << EOF
Policy-as-a-Service Benchmark Summary
====================================
Timestamp: $(date)
Results Directory: $RESULTS_DIR

Files Generated:
- system_info.json: System hardware and software information
- quick_benchmark.json: Quick performance benchmarks
- comprehensive_benchmark.json: Comprehensive performance benchmarks
- hardware_detection.json: Hardware detection results
- optimization_test.json: Optimization application results
- performance_report.txt: Human-readable performance analysis
- analysis.json: Machine-readable performance analysis

Key Metrics:
EOF

# Extract key metrics from analysis
if [ -f "analysis.json" ]; then
    echo "Extracting key metrics..."
    
    # P50 Latency
    P50_LATENCY=$(python3 -c "
import json
with open('analysis.json', 'r') as f:
    data = json.load(f)
    latency = data.get('latency_analysis', {}).get('p50_latency_ms')
    print(f'{latency:.2f}' if latency else 'N/A')
")
    echo "- P50 Latency: ${P50_LATENCY}ms" >> benchmark_summary.txt
    
    # Throughput
    THROUGHPUT=$(python3 -c "
import json
with open('analysis.json', 'r') as f:
    data = json.load(f)
    throughput = data.get('throughput_analysis', {}).get('throughput_rps')
    print(f'{throughput:.1f}' if throughput else 'N/A')
")
    echo "- Throughput: ${THROUGHPUT} RPS" >> benchmark_summary.txt
    
    # Memory Usage
    MEMORY_USAGE=$(python3 -c "
import json
with open('analysis.json', 'r') as f:
    data = json.load(f)
    memory = data.get('memory_analysis', {}).get('memory_usage_mb')
    print(f'{memory:.1f}' if memory else 'N/A')
")
    echo "- Memory Usage: ${MEMORY_USAGE} MB" >> benchmark_summary.txt
    
    # Hardware Grade
    HARDWARE_GRADE=$(python3 -c "
import json
with open('analysis.json', 'r') as f:
    data = json.load(f)
    grade = data.get('system_analysis', {}).get('hardware_grade', 'Unknown')
    print(grade)
")
    echo "- Hardware Grade: ${HARDWARE_GRADE}" >> benchmark_summary.txt
    
    # Performance Grade
    PERFORMANCE_GRADE=$(python3 -c "
import json
with open('analysis.json', 'r') as f:
    data = json.load(f)
    grade = data.get('latency_analysis', {}).get('performance_grade', 'Unknown')
    print(grade)
")
    echo "- Performance Grade: ${PERFORMANCE_GRADE}" >> benchmark_summary.txt
fi

echo "" >> benchmark_summary.txt
echo "Next Steps:" >> benchmark_summary.txt
echo "1. Review performance_report.txt for detailed analysis" >> benchmark_summary.txt
echo "2. Check if P50 latency requirement (<10ms) is met" >> benchmark_summary.txt
echo "3. Consider optimization recommendations" >> benchmark_summary.txt
echo "4. Run additional tests if needed" >> benchmark_summary.txt

echo "✓ Benchmark summary created"

# 9. Display Results
echo ""
echo "BENCHMARK RESULTS SUMMARY"
echo "========================="
cat benchmark_summary.txt

echo ""
echo "Benchmark suite completed successfully!"
echo "Results saved in: $(pwd)"
echo ""
echo "To view detailed results:"
echo "  cat performance_report.txt"
echo ""
echo "To run additional benchmarks:"
echo "  python3 ../../scripts/local_benchmark.py --help"
echo ""
echo "To test specific optimizations:"
echo "  python3 ../../scripts/optimization_manager.py --help"
